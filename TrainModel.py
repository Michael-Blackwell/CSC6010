from ModelFunctions import *

timestamp = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
# width, height, depth
image_shape = (300, 300, 10)
train_path = Path('/media/storage/RSNA Brain Tumor Project/train_tr')
val_path = Path('/media/storage/RSNA Brain Tumor Project/val_tr')


def build_model(width, height, depth, name='Model'):
    """Build a 3D convolutional neural network model
    There are many ways to do this."""

    flair = tf.keras.Input((width, height, depth, 1), name='FLAIR')
    t1w = tf.keras.Input((width, height, depth, 1), name='t1w')
    t1wce = tf.keras.Input((width, height, depth, 1), name='t1wce')
    t2 = tf.keras.Input((width, height, depth, 1), name='t2')

    # FLAIR input & convolution
    f = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation='relu')(flair)
    f = tf.keras.layers.MaxPool3D(pool_size=3)(f)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(0.2)(f)

    # T1w input & convolution
    t1 = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation='relu')(t1w)
    t1 = tf.keras.layers.MaxPool3D(pool_size=3)(t1)
    t1 = tf.keras.layers.BatchNormalization()(t1)
    t1 = tf.keras.layers.Dropout(0.2)(t1)

    # T1wCE input & convolution
    t1e = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation='relu')(t1wce)
    t1e = tf.keras.layers.MaxPool3D(pool_size=3)(t1e)
    t1e = tf.keras.layers.BatchNormalization()(t1e)
    t1e = tf.keras.layers.Dropout(0.2)(t1e)

    # T2w input & convolution
    t = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation='relu')(t2)
    t = tf.keras.layers.MaxPool3D(pool_size=3)(t)
    t = tf.keras.layers.BatchNormalization()(t)
    t = tf.keras.layers.Dropout(0.2)(t)

    # Concatenate all 4 together
    x = tf.keras.layers.Concatenate(axis=4)([f, t1, t1e, t])

    # Convolve again
    x = tf.keras.layers.Conv3D(filters=256, kernel_size=2, activation='relu')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(3,3,1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Flatten, Dense, Output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=50, activation="relu")(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    # Define the model.
    model = tf.keras.Model(inputs=[flair, t1w, t1wce, t2], outputs=outputs, name=name)

    # Compile model
    model.compile(
        loss="binary_crossentropy",
        optimizer='Adam',  # tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["binary_accuracy"],
    )

    return model


def train_model(train_path: Path, val_path: Path, batch: int, epochs: int):
    # Build the model
    model = build_model(width=image_shape[0],
                        height=image_shape[1],
                        depth=image_shape[2])

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
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="binary_accuracy", patience=15)

    # Launch Tensorboard, can be accessed by going to http://localhost:6006 in your browser
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, '--logdir', str(tb_path)])
    url = tb.launch()

    # Training data input generator
    train_gen = DataPipe(filepath=train_path)
    training_data = tf.data.Dataset.from_generator(train_gen.input_generator,
                                                   output_types=((tf.float32, tf.float32, tf.float32, tf.float32),
                                                                 tf.int64),
                                                   output_shapes=((image_shape, image_shape, image_shape, image_shape),
                                                                  (1, 1)))
    # validation data input generator
    val_gen = DataPipe(filepath=val_path)
    validation_data = tf.data.Dataset.from_generator(val_gen.input_generator,
                                                     output_types=((tf.float32, tf.float32, tf.float32, tf.float32),
                                                                   tf.int64),
                                                     output_shapes=(
                                                         (image_shape, image_shape, image_shape, image_shape),
                                                         (1, 1)))

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
    model.save(f'./models/All_Scans{timestamp}')

    # show metrics
    # fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    # ax = ax.ravel()

    # for i, metric in enumerate(["acc", "loss"]):
    #     ax[i].plot(model.history.history[metric])
    #     ax[i].plot(model.history.history["val_" + metric])
    #     ax[i].set_title("{} Model {}".format(scan_type, metric))
    #     ax[i].set_xlabel("epochs")
    #     ax[i].set_ylabel(metric)
    #     ax[i].legend(["train", "val"])

    return model


if __name__ == '__main__':
    train_model(train_path, val_path, batch=16, epochs=10)
