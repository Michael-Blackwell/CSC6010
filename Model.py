import tensorflow as tf
import tensorboard
from datetime import datetime
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

timestamp = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
scan_types = ['T1w']  # ['FLAIR', 'T1w', 'T1wCE', 'T2w']


# Define, train, and evaluate model
# source: https://keras.io/examples/vision/3D_image_classification/
def build_model(width=256, height=256, depth=60, name='FLAIR'):
    """Build a 3D convolutional neural network model."""

    inputs = tf.keras.Input((width, height, depth, 1))

    x = tf.keras.layers.Conv3D(filters=16, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # x = tf.keras.layers.Conv3D(filters=512, kernel_size=3, activation="relu")(x)
    # x = tf.keras.layers.MaxPool3D(pool_size=3)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=50, activation="relu")(x)

    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = tf.keras.Model(inputs, outputs, name=name)

    # Compile model.

    model.compile(
        loss="binary_crossentropy",
        optimizer='Adam',  # tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    return model


def train_model(train_path: Path, val_path: Path, batch: int, epochs: int):
    for scan_type in scan_types:

        # Build the model
        model = build_model(width=256, height=256, depth=60, name=scan_type)

        # Define callbacks and create folders to save them in
        tb_path = Path.cwd() / f'Callbacks/tensorboard/{timestamp}'
        ckpt_path = Path.cwd() / f'Callbacks/checkpoints/{timestamp}'
        early_stop_path = Path.cwd() / f'Callbacks/earlystopping/{timestamp}'

        ckpt_path.mkdir()
        tb_path.mkdir()
        early_stop_path.mkdir()

        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(tb_path),
                                                        # write_images=True,
                                                        histogram_freq=1,
                                                        )
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), save_best_only=True)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

        # Launch Tensorboard, can be accessed by going to http://localhost:6006 in your browser
        tb = tensorboard.program.TensorBoard()
        tb.configure(argv=[None, '--logdir', str(tb_path)])
        url = tb.launch()

        # Experimenting with these: Create Datasets from Generators
        training_data = tf.data.Dataset.from_generator(lambda: input_generator(train_path, scan_type),
                                                       output_types=(tf.float32, tf.int64),
                                                       output_shapes=((256, 256, 60), (1, 1)))

        validation_data = tf.data.Dataset.from_generator(lambda: input_generator(val_path, scan_type),
                                                         output_types=(tf.float32, tf.int64),
                                                         output_shapes=((256, 256, 60), (1, 1)))

        # Change to 'CPU:0' to use CPU instead of GPU
        # with tf.device('GPU:0'):
        model.fit(
            training_data.batch(batch),  # input_generator(train_path, scan_type, 1),#
            validation_data=validation_data.batch(batch),  # input_generator(val_path, scan_type, 1),   #
            epochs=epochs,
            batch_size=batch,
            # shuffle=True,
            # verbose=2,
            callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb]
        )

        # save model
        # model.save(f'./models/{scan_type}')

        # show metrics
        fig, ax = plt.subplots(1, 2, figsize=(20, 3))
        ax = ax.ravel()

        for i, metric in enumerate(["acc", "loss"]):
            ax[i].plot(model.history.history[metric])
            ax[i].plot(model.history.history["val_" + metric])
            ax[i].set_title("{} Model {}".format(scan_type, metric))
            ax[i].set_xlabel("epochs")
            ax[i].set_ylabel(metric)
            ax[i].legend(["train", "val"])


def load_dataset(path: Path):
    """Loads a TFRecords dataset."""
    # path = '/media/storage/RSNA Brain Tumor Project/val_tr/00000/FLAIR.tfrec'
    dataset = tf.data.TFRecordDataset(str(path), compression_type='GZIP')
    # Create a dictionary describing the features.
    feat_descriptions = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_width': tf.io.FixedLenFeature([], tf.int64),
        'image_height': tf.io.FixedLenFeature([], tf.int64),
        'image_depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'scan_type': tf.io.FixedLenFeature([], tf.string),
        'patient_ID': tf.io.FixedLenFeature([], tf.string),
    }

    # parsed_dataset = tf.io.parse_single_example(dataset, feat_descriptions)
    parsed_dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feat_descriptions))

    # Right now each tfr file contains images for one patient.
    for parsed_record in parsed_dataset.take(1):
        img_tensor = tf.io.parse_tensor(parsed_record['image'], out_type=tf.float32)
        label = parsed_record['label']

    return img_tensor, label


def get_file_list(filepath: Path, scantype: str) -> list:
    """Recursively loop through records folder and return a list of file path objects for all TFRecord files of a
    specified MRI scan type."""

    total_file_list = []

    for file in filepath.iterdir():
        if file.is_dir():
            total_file_list += get_file_list(file, scantype)
        elif file.name == f'{scantype}.tfrec':
            total_file_list.append(file)

    return total_file_list


def input_generator(filepath: Path, scantype: str) -> tuple:
    """Yield a single label and stacked MRI image for the model to train on."""
    # Get a list of all tfrec data files.
    file_path_list = get_file_list(filepath, scantype)

    counter = 0
    while True:  # On each loop, one 'batch' is generated
        # When at the end of the list, reset the counter, shuffle file list, and end generator.
        if counter >= len(file_path_list):
            counter = 0
            np.random.shuffle(file_path_list)

            img, label = load_dataset(file_path_list[counter])
            label = tf.reshape(label, (-1, 1))
            return img, label

        else:
            img, label = load_dataset(file_path_list[counter])
            label = tf.reshape(label, (-1, 1))

            yield img, label
            counter += 1


if __name__ == '__main__':

    train_path = Path('/media/storage/RSNA Brain Tumor Project/train_tr')
    val_path = Path('/media/storage/RSNA Brain Tumor Project/val_tr')
    train_model(train_path, val_path, batch=2, epochs=1)
