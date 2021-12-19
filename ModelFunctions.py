import tensorflow as tf
from datetime import datetime
import numpy as np
from pathlib import Path


class DataPipe:

    def __init__(self, filepath: Path):
        self.filepath = filepath

    @staticmethod
    def load_dataset(path: Path):
        """Loads a TFRecords dataset."""
        dataset = tf.data.TFRecordDataset(str(path), compression_type='GZIP')
        # Create a dictionary describing the features.
        feat_descriptions = {
            'FLAIR': tf.io.FixedLenFeature([], tf.string),
            'T1w': tf.io.FixedLenFeature([], tf.string),
            'T1wCE': tf.io.FixedLenFeature([], tf.string),
            'T2w': tf.io.FixedLenFeature([], tf.string),
            'image_width': tf.io.FixedLenFeature([], tf.int64),
            'image_height': tf.io.FixedLenFeature([], tf.int64),
            'image_depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'patient_ID': tf.io.FixedLenFeature([], tf.string),
        }

        # parsed_dataset = tf.io.parse_single_example(dataset, feat_descriptions)
        parsed_dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feat_descriptions))

        # Right now each tfr file contains images for one patient.
        for parsed_record in parsed_dataset.take(1):
            flair = tf.io.parse_tensor(parsed_record['FLAIR'], out_type=tf.float32)
            t1w = tf.io.parse_tensor(parsed_record['T1w'], out_type=tf.float32)
            t1wce = tf.io.parse_tensor(parsed_record['T1wCE'], out_type=tf.float32)
            t2w = tf.io.parse_tensor(parsed_record['T2w'], out_type=tf.float32)
            label = parsed_record['label']

        return flair, t1w, t1wce, t2w, label

    def get_file_list(self, filepath: Path) -> list:
        """Recursively loop through records folder and return a list of file path objects for all TFRecord files of a
        specified MRI scan type."""
        # TODO debug this
        total_file_list = []

        for file in filepath.iterdir():
            if file.is_dir():
                total_file_list += self.get_file_list(file)
            elif file.suffix == '.tfrec':
                total_file_list.append(file)

        return total_file_list

    def input_generator(self) -> tuple:
        """Yield a single label and stacked MRI image for the model to train on."""
        # Get a list of all tfrec data files.
        file_path_list = self.get_file_list(self.filepath)
        np.random.shuffle(file_path_list)

        counter = 0
        while True:  # On each loop, one example is generated, batches are generated by calling this several times.
            # When at the end of the list, reset the counter, shuffle file list, and end generator.
            if counter >= len(file_path_list):
                counter = 0
                np.random.shuffle(file_path_list)

                flair, t1w, t1wce, t2w, label = self.load_dataset(file_path_list[counter])
                label = tf.reshape(label, (-1, 1))
                # img = tf.reshape(img, (10, 260, 260))
                return (flair, t1w, t1wce, t2w), label

            else:
                flair, t1w, t1wce, t2w, label = self.load_dataset(file_path_list[counter])
                label = tf.reshape(label, (-1, 1))
                # img = tf.reshape(img, (10, 260, 260))

                yield (flair, t1w, t1wce, t2w), label
                counter += 1
