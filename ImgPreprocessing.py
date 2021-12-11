import pydicom
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from multiprocessing.pool import ThreadPool
import logging


# TODO implement logging
# ----- Parameters ----- #
# width, height, depth
image_size = (256, 256, 60)

mri_types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
# TODO right now only prepares FLAIR images in stacks of 60. Need to determine optimal depth for remaining scan types.

project_path = Path.cwd()
label_path = project_path / 'train_labels.csv'
train_data_path = project_path / 'train'
local_submission_path = project_path / 'sample_submission.csv'


def search(filepath: Path) -> list:
    """Return a sorted list of all DCM images in a directory."""
    dcm_file_list = [img for img in filepath.iterdir() if img.suffix == '.dcm']

    sort_key = lambda x: int(re.findall(r'\d+', str(x.name))[0])
    dcm_file_list.sort(key=sort_key)

    return dcm_file_list


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte | Taken from TensorFlow documentation."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double. | Taken from TensorFlow documentation."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint | Taken from TensorFlow documentation."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def load_dicom_image(filepath: Path) -> pydicom.FileDataset:
    """Load and resize a DCM image."""
    data = pydicom.read_file(filepath).pixel_array.astype(dtype='float32', copy=False)
    image = cv2.resize(data, image_size[0:2], interpolation=cv2.INTER_LANCZOS4)  # TODO why interpolate?

    return image


def read_labels(filepath: Path) -> pd.DataFrame:
    """
    Read labels from csv, drop invalid records, format/set index
    :param filepath: pathlib.Path
    :return: labels: pandas.DataFrame
    """
    labels = pd.read_csv(filepath)
    labels['BraTS21ID'] = labels['BraTS21ID'].apply(lambda x: str(x).zfill(5))
    labels = labels.set_index("BraTS21ID")
    # per the competition instructions, exclude these labels corresponding to records 00109, 00123, & 00709.
    labels = labels.drop(labels=['00109', '00123', '00709'], axis=0)

    return labels


def get_file_lists(filepath: Path, val_split: float) -> tuple:
    """Read the labels excel file and split records into training and validation sets using a stratified shuffle
    :return
    returns two dataframes, training and validation, containing filepaths for each patient and labels.
    (Training df, Validation df)"""
    # Create folders for pre-processed training and validation images.
    training_out = filepath.parent / 'train_tr'
    validation_out = filepath.parent / 'val_tr'
    training_out.mkdir(parents=True, exist_ok=True)
    validation_out.mkdir(parents=True, exist_ok=True)

    # Read labels file
    labels = read_labels(label_path)

    # Add input filepaths to  df
    labels['in_path'] = [filepath / x for x in labels.index]

    # Split into training and test sets, Stratified and shuffled
    train_idx, val_idx = train_test_split(labels.index,
                                          test_size=val_split,
                                          random_state=42,  # TODO remove this for actual randomness
                                          shuffle=True,
                                          stratify=labels.MGMT_value)
    # Split df into train and val sets
    train, val = labels.loc[train_idx], labels.loc[val_idx]

    # Add output filepaths
    train['out_path'], val['out_path'] = training_out, validation_out

    return train, val


def stack_images(path_list: list) -> np.ndarray:
    """Load all images from filepaths in path_list
    Discard images where all pixel values are 0
    Stack images into a numpy array."""

    img_byte_list = []

    for file in path_list:
        img = load_dicom_image(file)
        # Omit "warm-up" images and blank images
        # if np.count_nonzero(img) < 4100:
        #     continue
        img_byte_list.append(img)

    stacked = np.stack(img_byte_list, axis=0)

    return stacked


def pad_3dimage(image_3d: np.array, padding: int) -> np.array:
    """Return a numpy array of stacked pixel values that are equally padded on each side."""
    # Determine number of zero layers needed on each side
    top_padding = int(padding / 2)
    bottom_padding = padding - top_padding

    # Create top and bottom zero arrays
    top_zero = np.zeros((top_padding, image_size[0], image_size[1]))
    bottom_zero = np.zeros((bottom_padding, image_size[0], image_size[1]))

    # Append layers on top and bottom of image
    padded_image = np.concatenate((top_zero, image_3d), axis=0).astype(dtype='float32')
    padded_image = np.concatenate((padded_image, bottom_zero), axis=0).astype(dtype='float32')

    return padded_image


def select_n_images(sorted_path_list: list, n: int) -> list:
    """Select the middle n images from a list of filepaths."""
    half = int(n/2)
    # Find the center of the filepath list
    list_center = int(len(sorted_path_list) / 2)

    # Determine increment to slice with, want to span 60% of images bidirectionally
    increment = max(int((list_center * 0.6) / half), 1)

    # Determine the upper and lower indices to slice the list on
    top_idx = list_center + min((n - half) * increment, list_center-1)
    bottom_idx = list_center - min(half * increment, list_center)

    # Slice and return the list
    selected_images = sorted_path_list[bottom_idx: top_idx: increment]

    return selected_images


def preprocess_images(record: tuple) -> int:
    """This function is intended to be called by multiple threads
    Call load_dicom_image to load and stack images into a tensor of dimensions 'img_size'
    If the depth of the tensor is insufficient, pad it with zero-matrices
    Output as a serialized TFRecord object (based on protobuf protocol)."""
    patient_id = record[0]
    file_data = record[1]
    # Create output patient folder if it does not exist
    out_path = file_data['out_path'] / file_data['in_path'].name
    out_path.mkdir(parents=True, exist_ok=True)

    # For one scan type, get a sorted list of all image files in the directory.
    path_list = search(file_data['in_path'] / file_data['MRI_Type'])

    # If the path_list contains too many images (more than specified image depth) select the middle n images.
    if len(path_list) > image_size[2]:
        path_list = select_n_images(path_list, image_size[2])

    # Load all of the images as binary files and stack them like a tensor.
    image3d = stack_images(path_list)

    # All inputs to the model need to be of the same shape.
    # If there are not enough images to reach the desired depth, pad with 0's, then normalize.
    usable_images = image3d.shape[0]
    if image3d.shape[0] < image_size[2]:
        padding = image_size[2] - image3d.shape[0]
        image3d = pad_3dimage(image3d, padding)

    image3d = image3d / np.max(image3d)  # TODO why 4096? Why not max for each batch? Confirm 4096 is max DICOM pixel value
    # Convert to tensor
    image_tensor = tf.convert_to_tensor(image3d)
    # Reshape tensor (batch, image height, image width, image depth)
    image_tensor = tf.reshape(image_tensor, (image_size[0], image_size[1], image_size[2]))
    # Serialize tensor
    image_tensor_ser = tf.io.serialize_tensor(image_tensor)
    # inverse operation is tf.io.parse_tensor(image_tensor_ser, out_type=tf.float32)

    # Extract Patient ID and Scan Type from filepaths
    patient_id = bytes(patient_id, encoding='utf-8')
    scan = bytes(file_data['MRI_Type'], encoding='utf-8')

    # Create feature dictionary for TFRecord file.
    data = {'image': _bytes_feature(image_tensor_ser),
            'image_width': _int64_feature(image_size[0]),
            'image_height': _int64_feature(image_size[1]),
            'image_depth': _int64_feature(image_size[2]),
            'label': _int64_feature(file_data['MGMT_value']),
            'scan_type': _bytes_feature(scan),
            'patient_ID': _bytes_feature(patient_id)
            }

    # Create the binary TFRecord object and serialize the data into a byte string
    bin_data = tf.train.Example(features=tf.train.Features(feature=data))
    bin_data = bin_data.SerializeToString()

    # Compress using Gzip format since the dataset is so large.
    option = tf.io.TFRecordOptions(compression_type="GZIP")

    # Write the files to the output folder.
    with tf.io.TFRecordWriter(str(out_path / f"{file_data['MRI_Type']}.tfrec"), options=option) as writer:
        writer.write(bin_data)

    return usable_images


def compile_tfrecord_files(scan_types: list, filepath: Path, val_split: float) -> None:
    """Preprocess samples from the dataset into a TFRecord file."""

    train_df, val_df = get_file_lists(filepath, val_split)

    for kind in scan_types:
        train_df['MRI_Type'] = kind
        val_df['MRI_Type'] = kind

        # Concurrently process 10 files at a time.
        # Prepare training data
        # image_counts = []
        preprocessed_imgs = ThreadPool(10).imap_unordered(preprocess_images, train_df.iterrows())
        for img_cnt in tqdm(preprocessed_imgs, total=len(train_df), desc=f'Preprocessing Training Data ({kind})'):
            # image_counts.append(img_cnt)
            pass

        # Prepare validation data
        preprocessed_imgs = ThreadPool(10).imap_unordered(preprocess_images, val_df.iterrows())
        for img_cnt in tqdm(preprocessed_imgs, total=len(val_df), desc=f'Preprocessing Validation Data ({kind})'):
            # image_counts.append(img_cnt)
            pass

        # print(kind, max(image_counts))
        # the maximum number of usable patient images for each scan type.
        # FLAIR - 316
        # T1w - 253
        # T1wCE - 265
        # T2w - 306
        pass


def find_usable_images(img_list: list):
    imagedf = pd.DataFrame(index=img_list, columns=['pxl_count', 'max_pxl_value'])
    for path in img_list:
        img = load_dicom_image(path)
        imagedf.loc[path] = [np.count_nonzero(img), np.max(img)]
    pass


if __name__ == "__main__":
    compile_tfrecord_files(mri_types, train_data_path, val_split=0.2)
