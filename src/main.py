import datetime
import os
from os import path
from pathlib import Path
from typing import Any, cast

from tensorflow import keras as imported_keras  # type: ignore

from resnet import ResNet


keras = cast(Any, imported_keras)


def validate_resnet():
    data: Any = keras.datasets.cifar10.load_data()
    (training_data, training_labels), (validation_data, validation_labels) = data

    data_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    data_generator.fit(training_data)

    model = ResNet((32, 32), [1, 2, 2, 1], True).model
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    file_name = f'resnet-30-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    checkpoint_directory = path.join('checkpoints', file_name)
    Path(checkpoint_directory).mkdir(parents=True, exist_ok=True)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=path.join(checkpoint_directory, 'cp-{epoch:04d}.ckpt'),
        verbose=1
    )

    callback_directory = path.join('tensorboard_logs', file_name)
    Path(callback_directory).mkdir(parents=True, exist_ok=True)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=callback_directory,
        histogram_freq=1)

    model.fit(
        data_generator.flow(training_data, training_labels, 128),
        epochs=20,
        verbose=1,
        validation_data=(validation_data, validation_labels),
        callbacks=[checkpoint_callback, tensorboard_callback]
    )


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow

    cast(Any, tensorflow).get_logger().setLevel('ERROR')
    cast(Any, tensorflow).autograph.set_verbosity(0)

    validate_resnet()
