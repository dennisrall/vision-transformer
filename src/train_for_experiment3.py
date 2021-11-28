import tensorflow as tf

from config import base_train_log_dir, experiment3_dir, base_val_log_dir, \
    base_model_dir
from load_dataset import load_normalized_rgb_and_resized_mnist
from model.hybrid_transformer import build_hybrid_transformer, build_pure_cnn
from train_util import limit_memory_growth, train_loop
from util import get_params


def train_for_experiment3(epochs, batch_size, embed_dim, num_heads, ff_dim):
    num_classes = 10
    train_ds, test_ds = load_normalized_rgb_and_resized_mnist()

    def train_model(model, directory):
        train_log_dir = base_train_log_dir / experiment3_dir / directory
        val_log_dir = base_val_log_dir / experiment3_dir / directory
        model_dir = base_model_dir / experiment3_dir / directory

        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        train_loop(model, optimizer, loss_fn, train_ds, test_ds, epochs,
                   batch_size, model_dir, train_log_dir, val_log_dir)

    for num_cnn_layers in range(1, 20):
        dir_string = f'vgg16-{num_cnn_layers}'
        tf.print('training', dir_string)

        hybrid_transformer = build_hybrid_transformer(num_cnn_layers,
                                                      embed_dim, num_heads,
                                                      ff_dim, num_classes)
        train_model(hybrid_transformer, dir_string)

    dir_string = 'pure_cnn'
    tf.print('training', dir_string)

    pure_cnn = build_pure_cnn(ff_dim, num_classes)
    train_model(pure_cnn, dir_string)


if __name__ == '__main__':
    limit_memory_growth()
    params = get_params('model', 'experiment3')
    train_for_experiment3(**params)
