from pathlib import Path

import tensorflow as tf

from config import base_train_log_dir, experiment1_dir, base_val_log_dir, \
    base_model_dir, transformer_blocks, patch_sizes
from load_dataset import load_normalized_mnist
from model.naive_transformer import build_naive_transformer
from model.patch_transformer import build_patch_transformer
from train_util import limit_memory_growth, train_loop
from util import get_params


def train_for_experiment1(epochs, batch_size, embed_dim, num_heads, ff_dim):
    image_size = 28
    num_classes = 10

    # load datasets
    train_ds, test_ds = load_normalized_mnist()

    def train_model(model, directory):
        # directories
        train_log_dir = base_train_log_dir / experiment1_dir / directory
        val_log_dir = base_val_log_dir / experiment1_dir / directory
        model_dir = base_model_dir / experiment1_dir / directory

        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        train_loop(model, optimizer, loss_fn, train_ds, test_ds, epochs,
                   batch_size, model_dir, train_log_dir, val_log_dir)

    for num_transformer_blocks in transformer_blocks:
        dir_string = f'naive-{num_transformer_blocks}layers'
        tf.print('training', dir_string)

        image_shape = (image_size, image_size, 1)
        naive = build_naive_transformer(image_shape, embed_dim, num_heads,
                                        ff_dim, num_classes,
                                        num_transformer_blocks)
        train_model(naive, Path(dir_string))

        for patch_size in patch_sizes:
            dir_string = f'patch{patch_size}-{num_transformer_blocks}layers'
            tf.print('training', dir_string)

            patch_transformer = build_patch_transformer(
                patch_size, image_size, embed_dim, num_heads, ff_dim,
                num_classes, num_transformer_blocks)

            train_model(patch_transformer, Path(dir_string))


if __name__ == '__main__':
    limit_memory_growth()
    params = get_params('model', 'experiment1')
    train_for_experiment1(**params)
