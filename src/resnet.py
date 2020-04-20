from typing import Any, List, Tuple, cast

from tensorflow import Tensor
from tensorflow import keras  # type: ignore


layers = cast(Any, keras.layers)
models = cast(Any, keras.models)


class ResNet:
    def __init__(
        self, image_size: Tuple[int, int], block_count: List[int], is_residual: bool = True
    ) -> None:
        self.filter_count = 64

        # Input layer
        input: Tensor = layers.Input(shape=(*image_size, 3))
        output: Tensor = layers.BatchNormalization()(input)

        output = self.__add_convolution(output)
        output = self.__add_relu(output)

        for i, layer_block_count in enumerate(block_count):
            for j in range(layer_block_count):
                is_downsample = j == 0 and i != 0

                residual_output: Tensor = self.__add_convolution(output,
                    strides=(1 if not is_downsample else 2))

                if is_residual:
                    residual_output = self.__add_relu(residual_output)
                    residual_output = self.__add_convolution(residual_output)

                    if is_downsample:
                        output = self.__add_convolution(output, kernel_size=1, strides=2)

                    residual_output = layers.Add()([output, residual_output])

                output = self.__add_relu(residual_output)

            self.filter_count *= 2

        output = layers.AveragePooling2D(4)(output)
        output = layers.Flatten()(output)
        output = layers.Dense(10, activation='softmax')(output)

        model = models.Model(input, output)

        self.model = model

    def __add_relu(self, input: Tensor) -> Tensor:
        output = layers.ReLU()(input)
        output = layers.BatchNormalization()(output)

        return output

    def __add_convolution(
        self, input: Tensor, kernel_size: int = 3, strides: int = 1
    ) -> Tensor:
        output = layers.Conv2D(
            kernel_size=kernel_size,
            strides=strides,
            filters=self.filter_count,
            padding="same"
        )(input)

        return output
