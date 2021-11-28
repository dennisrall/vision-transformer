import tensorflow as tf

from config import base_train_log_dir, patch_dir, base_val_log_dir, \
    base_model_dir
from load_dataset import load_normalized_mnist
from model.patch_transformer import build_patch_transformer
from train_util import limit_memory_growth, train_loop
from util import get_params


def train_patch_transformer(epochs, batch_size, embed_dim, num_heads,
                            num_transformer_blocks, ff_dim):
    image_size = 28
    patch_size = 7
    num_classes = 10

    # load datasets
    train_ds, test_ds = load_normalized_mnist()

    # directories
    train_log_dir = base_train_log_dir / patch_dir
    val_log_dir = base_val_log_dir / patch_dir
    model_dir = base_model_dir / patch_dir

    # model, optimizer and loss
    model = build_patch_transformer(patch_size, image_size, embed_dim,
                                    num_heads,
                                    ff_dim, num_classes,
                                    num_transformer_blocks)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loop(model, optimizer, loss_fn, train_ds, test_ds, epochs,
               batch_size, model_dir, train_log_dir, val_log_dir)


if __name__ == '__main__':
    limit_memory_growth()
    params = get_params('model', 'patch')
    train_patch_transformer(**params)
