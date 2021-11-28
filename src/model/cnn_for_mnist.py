from itertools import chain

import tensorflow as tf

from layers.mlp_head import MlpHead


def build_cnn_for_mnist(num_classes, mlp_dim, filters, kernel_size):
    convolutions = list(chain.from_iterable(
        ((tf.keras.layers.Conv2D(f, kernel_size, padding='same'),
          tf.keras.layers.MaxPool2D(pool_size=(2, 2))))
        for f in filters
    ))
    return tf.keras.Sequential([
        *convolutions,
        tf.keras.layers.Flatten(),
        MlpHead(mlp_dim, num_classes, final_activation='softmax')
    ])
