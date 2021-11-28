from pathlib import Path

import tensorflow as tf

from config import base_train_log_dir, base_val_log_dir, \
    base_model_dir, experiment2_dir
from load_dataset import load_normalized_mnist
from model.cnn_for_mnist import build_cnn_for_mnist
from model.patch_transformer import build_patch_transformer
from train_util import limit_memory_growth, train_loop
from util import get_params


def train_for_experiment2():
    image_size = 28
    num_classes = 10

    # read epochs, batch_size and patch_size
    params = get_params('model', 'experiment2')
    epochs = params['epochs']
    batch_size = params['batch_size']
    patch_size = params['patch_size']

    train_ds, test_ds = load_normalized_mnist()

    def train_model(model, directory):
        # directories
        train_log_dir = base_train_log_dir / experiment2_dir / directory
        val_log_dir = base_val_log_dir / experiment2_dir / directory
        model_dir = base_model_dir / experiment2_dir / directory

        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        train_loop(model, optimizer, loss_fn, train_ds, test_ds, epochs,
                   batch_size, model_dir, train_log_dir, val_log_dir)

    # train CNNs
    params = get_params('model', 'experiment2', 'cnn')
    for param_key in params.keys():
        dir_string = f'cnn-{param_key}'
        tf.print('training', dir_string)
        cnn = build_cnn_for_mnist(num_classes=num_classes, **params[param_key])
        train_model(cnn, Path(dir_string))

    # train patch transformers
    params = get_params('model', 'experiment2', 'patch')
    for param_key in params.keys():
        dir_string = f'patch-{param_key}'
        tf.print('training', dir_string)
        patch = build_patch_transformer(
            patch_size=patch_size, image_size=image_size,
            num_classes=num_classes, **params[param_key])
        train_model(patch, Path(dir_string))


if __name__ == '__main__':
    # trigger training
    limit_memory_growth()
    train_for_experiment2()
