import tensorflow as tf

from config import base_train_log_dir, base_val_log_dir, \
    base_model_dir, cnn_dir
from load_dataset import load_normalized_mnist
from model.cnn_for_mnist import build_cnn_for_mnist
from train_util import limit_memory_growth, train_loop
from util import get_params


def train_cnn_for_mnist(epochs, batch_size, mlp_dim, filters, kernel_size):
    num_classes = 10

    # load datasets
    train_ds, test_ds = load_normalized_mnist()

    # directories
    train_log_dir = base_train_log_dir / cnn_dir
    val_log_dir = base_val_log_dir / cnn_dir
    model_dir = base_model_dir / cnn_dir

    # model, optimizer and loss
    model = build_cnn_for_mnist(num_classes, mlp_dim, filters, kernel_size)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loop(model, optimizer, loss_fn, train_ds, test_ds, epochs,
               batch_size, model_dir, train_log_dir, val_log_dir)


if __name__ == '__main__':
    limit_memory_growth()
    params = get_params('model', 'cnn')
    train_cnn_for_mnist(**params)
