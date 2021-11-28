from config import base_train_log_dir, experiment4_dir, base_val_log_dir, \
    base_model_dir
from load_dataset import load_normalized_rgb_and_resized_mnist
from model.transfer_learning_transformer import \
    build_transfer_learning_transformer
from train_util import limit_memory_growth, train_loop
from util import get_params

import tensorflow as tf


def train_for_experiment4(epochs, batch_size, learning_rate, vit_types):
    num_classes = 10
    image_size = (32, 32)
    train_ds, test_ds = load_normalized_rgb_and_resized_mnist()

    def train_model(model, directory):
        train_log_dir = base_train_log_dir / experiment4_dir / directory
        val_log_dir = base_val_log_dir / experiment4_dir / directory
        model_dir = base_model_dir / experiment4_dir / directory

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        train_loop(model, optimizer, loss_fn, train_ds, test_ds, epochs,
                   batch_size,
                   model_dir, train_log_dir, val_log_dir)

    for vit_type in vit_types:
        tf.print('training', vit_type)

        transformer = build_transfer_learning_transformer(image_size, vit_type,
                                                          num_classes)

        train_model(transformer, vit_type)


if __name__ == '__main__':
    limit_memory_growth()
    params = get_params('model', 'experiment4')
    train_for_experiment4(**params)
