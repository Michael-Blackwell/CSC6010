import tensorflow as tf
import pickle
import tensorboard
from Config import *
import numpy as np
import matplotlib.pyplot as plt

timestamp = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
# valpath = '/media/storage/RSNA Brain Tumor Project/val_tr/mri_val.tfrec'
# trainpath = '/media/storage/RSNA Brain Tumor Project/train_tr/mri_train.tfrec'
scan_types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']


def build_model(width=256, height=256, depth=400, name='3dcnn'):
    """Build a 3D convolutional neural network model."""

    inputs = tf.keras.Input((width, height, depth, 1))

    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv3D(filters=512, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Flatten()(x)

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


def train_model():
    for scan_type in scan_types:
        # load train_dataset dataset
        tf_data_path = Path.cwd() / 'train_tr/00002'  # / f'datasets/{scan_type}_train_dataset'
        with open(str(tf_data_path / 'element_spec'), 'rb') as in_:
            es = pickle.load(in_)
        train_dataset = tf.data.experimental.load(str(tf_data_path), es, compression='GZIP')

        # # load validation_dataset
        # tf_data_path = f'./datasets/{scan_type}_validation_dataset'
        # with open(tf_data_path + '/element_spec', 'rb') as in_:
        #     es = pickle.load(in_)
        # validation_dataset = tf.data.experimental.load(str(tf_data_path), es, compression='GZIP')

        # Get Model
        model = build_model(width=image_w, height=image_h, depth=image_d, name=scan_type)

        # Define and create callback folders
        tb_path = Path.cwd() / f'Callbacks/tensorboard/{timestamp}'
        ckpt_path = Path.cwd() / f'Callbacks/checkpoints/{timestamp}'
        # early_stop_path = Path.cwd() / f'Callbacks/earlystopping/{timestamp}'
        ckpt_path.mkdir()
        tb_path.mkdir()
        # early_stop_path.mkdir()
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(tb_path), write_images=True)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), save_best_only=True)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

        # Launch Tensorboard
        tb = tensorboard.program.TensorBoard()
        tb.configure(argv=[None, '--logdir', str(tb_path)])
        url = tb.launch()

        epochs = 10
        # with tf.device('CPU:0'):
        model.fit(
            train_dataset,
            # validation_data=validation_dataset,
            epochs=epochs,
            # batch_size=1,
            # shuffle=True,
            # verbose=2,
            callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb],
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
    path = '/media/storage/RSNA Brain Tumor Project/train_tr/00000/FLAIR.tfrec'
    dataset = tf.data.TFRecordDataset(str(path), compression_type='GZIP')
    # Create a dictionary describing the features.
    feat_descriptions = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_width': tf.io.FixedLenFeature([], tf.int64),
        'image_height': tf.io.FixedLenFeature([], tf.int64),
        'image_depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    # parsed_dataset = tf.io.parse_single_example(dataset, feat_descriptions)
    parsed_dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feat_descriptions))

    return parsed_dataset


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


def input_generator(filepath: Path, scantype: str, batchsize: int) -> tuple:
    """Yield a single label and stacked MRI image for the model to train on."""
    # Get a list of all tfrec data files.
    file_path_list = get_file_list(filepath, scantype)

    i = 0
    while True:  # On each loop, one 'batch' is generated
        if i * batchsize >= len(file_path_list):
            i = 0
            np.random.shuffle(file_path_list)
        else:
            batch = file_path_list[i * batchsize:(i + 1) * batchsize]
            images, labels = [], []
            # Since this project is a binary classifier we do not need to worry about assigning text labels
            label_classes = tf.constant(0, 1)

            for file in batch:  # prepare the batch
                temp = load_dataset(file)
                data.append(temp.values.reshape(32, 32, 1))
                pattern = tf.constant(eval("file[14:21]"))  # This line has changed
                labels.append()

            data = np.asarray(data).reshape(-1, 32, 32, 1)
            labels = np.asarray(labels)

            yield images, labels
            i += 1


if __name__ == '__main__':
    # model = build_model()
    train_model()

