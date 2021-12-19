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


class DataPrep:

    def __init__(self,
                 image_size=(260, 260, 10),
                 scan_types=('FLAIR', 'T1w', 'T1wCE', 'T2w')):
        self.scan_types = scan_types
        self.image_size = image_size
        self.label_path = Path.cwd() / 'train_labels.csv'
        self.train_data_path = Path.cwd() / 'train'

        # TODO add logging basic config file here. log all images stored in tfr files.

    @staticmethod
    def search(filepath: Path) -> list:
        """Return a sorted list of all DCM images in a directory."""
        dcm_file_list = [img for img in filepath.iterdir() if img.suffix == '.dcm']

        sort_key = lambda x: int(re.findall(r'\d+', str(x.name))[0])
        dcm_file_list.sort(key=sort_key)

        return dcm_file_list

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte | Taken from TensorFlow documentation."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double. | Taken from TensorFlow documentation."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint | Taken from TensorFlow documentation."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def load_dicom_image(self, filepath: Path) -> pydicom.FileDataset:
        """Load and resize a DCM image."""
        data = pydicom.read_file(filepath).pixel_array.astype(dtype='float32', copy=False)
        image = cv2.resize(data, self.image_size[0:2], interpolation=cv2.INTER_LANCZOS4)  # TODO why interpolate?

        return image

    @staticmethod
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

    def get_file_lists(self, filepath: Path, val_split: float, test_split: float) -> tuple:
        """Read the labels excel file and split records into training and validation sets using a stratified shuffle
        :return
        returns two dataframes, training and validation, containing filepaths for each patient and labels.
        (Training df, Validation df)"""
        # Create folders for pre-processed training and validation images.
        training_out = filepath.parent / 'train_tr'
        validation_out = filepath.parent / 'val_tr'
        test_out = filepath.parent / 'test_tr'
        training_out.mkdir(parents=True, exist_ok=True)
        validation_out.mkdir(parents=True, exist_ok=True)
        test_out.mkdir(parents=True, exist_ok=True)

        # Read labels file
        labels = self.read_labels(self.label_path)

        # Add input file paths to  df
        labels['in_path'] = [filepath / x for x in labels.index]

        # Split into training and validation sets, Stratified and shuffled
        train_idx, val_idx = train_test_split(labels.index,
                                              test_size=val_split,
                                              random_state=42,  # TODO remove this for actual randomness
                                              shuffle=True,
                                              stratify=labels.MGMT_value)
        # Split df into train and val sets
        train1, val = labels.loc[train_idx], labels.loc[val_idx]

        # Split training into training and test sets, Stratified and shuffled
        train_idx, test_idx = train_test_split(train1.index,
                                               test_size=test_split,
                                               random_state=42,  # TODO remove this for actual randomness
                                               shuffle=True,
                                               stratify=train1.MGMT_value)
        # Split df into train and test sets
        train, test = train1.loc[train_idx], train1.loc[test_idx]

        # Add output file paths
        train['out_path'], val['out_path'], test['out_path'] = training_out, validation_out, test_out

        return train, val, test

    def stack_images(self, path_list: list) -> np.ndarray:
        """Load all images from filepaths in path_list
        Discard images where all pixel values are 0
        Stack images into a numpy array."""

        img_byte_list = []

        for file in path_list:
            img = self.load_dicom_image(file)
            # Omit "warm-up" images and blank images
            # if np.count_nonzero(img) < 4100:
            #     continue
            img_byte_list.append(img)

        stacked = np.stack(img_byte_list, axis=0)

        return stacked

    def pad_3dimage(self, image_3d: np.array, padding: int) -> np.array:
        """Return a numpy array of stacked pixel values that are equally padded on each side."""
        # Determine number of zero layers needed on each side
        top_padding = int(padding / 2)
        bottom_padding = padding - top_padding

        # Create top and bottom zero arrays
        top_zero = np.zeros((top_padding, self.image_size[0], self.image_size[1]))
        bottom_zero = np.zeros((bottom_padding, self.image_size[0], self.image_size[1]))

        # Append layers on top and bottom of image
        padded_image = np.concatenate((top_zero, image_3d), axis=0).astype(dtype='float32')
        padded_image = np.concatenate((padded_image, bottom_zero), axis=0).astype(dtype='float32')

        return padded_image

    @staticmethod
    def select_n_images(sorted_path_list: list, n: int) -> list:
        """Select the middle n images from a list of filepaths."""
        half = int(n / 2)
        # Find the center of the filepath list
        list_center = int(len(sorted_path_list) / 2)

        # Determine increment to slice with, want to span 60% of images bidirectionally
        increment = max(int((list_center * 0.6) / half), 1)

        # Determine the upper and lower indices to slice the list on
        top_idx = list_center + min((n - half) * increment, list_center - 1)
        bottom_idx = list_center - min(half * increment, list_center)

        # Slice and return the list
        selected_images = sorted_path_list[bottom_idx: top_idx: increment]

        return selected_images

    @staticmethod
    def write_to_tfr(data: dict, out_path: Path, pat_id: str) -> None:
        # Create the binary TFRecord object and serialize the data into a byte string
        bin_data = tf.train.Example(features=tf.train.Features(feature=data))
        bin_data = bin_data.SerializeToString()

        # Compress using Gzip format since the dataset is so large.
        option = tf.io.TFRecordOptions(compression_type="GZIP")

        # Write the files to the output folder.
        with tf.io.TFRecordWriter(str(out_path / f"{pat_id}.tfrec"), options=option) as writer:
            writer.write(bin_data)

    def preprocess_images(self, record: tuple):
        """This function is intended to be called by multiple threads
        Call load_dicom_image to load and stack images into a tensor of dimensions 'img_size'
        If the depth of the tensor is insufficient, pad it with zero-matrices
        Output as a serialized TFRecord object (based on protobuf protocol)."""

        patient_id = record[0]
        file_data = record[1]
        # Create output patient folder if it does not exist
        out_path = file_data['out_path']  # / file_data['in_path'].name
        out_path.mkdir(parents=True, exist_ok=True)

        data = {}
        for scan in self.scan_types:
            # For one scan type, get a sorted list of all image files in the directory.
            path_list = self.search(file_data['in_path'] / scan)

            # If the path_list contains too many images (more than specified image depth) select the middle n images.
            if len(path_list) > self.image_size[2]:
                path_list = self.select_n_images(path_list, self.image_size[2])

            # Load all of the images as binary files and stack them like a tensor.
            image3d = self.stack_images(path_list)

            # All inputs to the model need to be of the same shape.
            # If there are not enough images to reach the desired depth, pad with 0's, then normalize.
            # usable_images = image3d.shape[0]
            if image3d.shape[0] < self.image_size[2]:
                padding = self.image_size[2] - image3d.shape[0]
                image3d = self.pad_3dimage(image3d, padding)

            image3d = image3d / np.max(image3d)
            # Convert to tensor
            image_tensor = tf.convert_to_tensor(image3d)
            # Reshape tensor (batch, image height, image width, image depth)
            image_tensor = tf.reshape(image_tensor, (self.image_size[0], self.image_size[1], self.image_size[2]))
            # Serialize tensor
            image_tensor_ser = tf.io.serialize_tensor(image_tensor)
            # inverse operation is tf.io.parse_tensor(image_tensor_ser, out_type=tf.float32)

            # Add serialized scan to feature dict for TFR file
            data[scan] = self._bytes_feature(image_tensor_ser)

        # Add additional info for TFR file
        bin_patient_id = bytes(patient_id, encoding='utf-8')

        data['image_width'] = self._int64_feature(self.image_size[0])
        data['image_height'] = self._int64_feature(self.image_size[1])
        data['image_depth'] = self._int64_feature(self.image_size[2])
        data['label'] = self._int64_feature(file_data['MGMT_value'])
        data['patient_ID'] = self._bytes_feature(bin_patient_id)

        self.write_to_tfr(data, out_path, patient_id)

    def compile_tfrecord_files(self, val_split: float, test_split: float) -> None:
        """Preprocess samples from the dataset into a TFRecord file."""

        train_df, val_df, test_df = self.get_file_lists(self.train_data_path, val_split, test_split)

        # Concurrently process 10 files at a time.
        # Prepare training data
        preprocessed_imgs = ThreadPool(10).imap_unordered(self.preprocess_images, train_df.iterrows())
        for img_cnt in tqdm(preprocessed_imgs, total=len(train_df), desc=f'Preprocessing Training Data'):
            pass

        # Prepare validation data
        preprocessed_imgs = ThreadPool(10).imap_unordered(self.preprocess_images, val_df.iterrows())
        for img_cnt in tqdm(preprocessed_imgs, total=len(val_df), desc=f'Preprocessing Validation Data'):
            pass

        # Prepare test data
        preprocessed_imgs = ThreadPool(10).imap_unordered(self.preprocess_images, test_df.iterrows())
        for img_cnt in tqdm(preprocessed_imgs, total=len(test_df), desc=f'Preprocessing Validation Data'):
            pass
