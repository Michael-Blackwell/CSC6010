from DataPrepFunctions import *


if __name__ == "__main__":

    # Create instance of DataPrep class
    data_class = DataPrep(image_size=(300, 300, 20),
                          scan_types=('FLAIR', 'T1w', 'T1wCE', 'T2w'))

    # Call compile_tfrecord_files
    data_class.compile_tfrecord_files(val_split=0.2, test_split=0.1)
