import tensorflow as tf

from load_dataset import load_normalized_mnist
from model.cnn_for_mnist import build_cnn_for_mnist
from model.naive_transformer import build_naive_transformer
from model.patch_transformer import build_patch_transformer
from util import get_params


def build_and_call_cnn_for_mnist(**kwargs):
    params = clean_up_params(get_params('model', 'cnn'), **kwargs)
    cnn_for_mnist = build_cnn_for_mnist(**params)
    return call_model_once(cnn_for_mnist)


def build_and_call_naive_transformer(**kwargs):
    params = clean_up_params(get_params('model', 'naive'), **kwargs)
    naive_transformer = build_naive_transformer(**params)
    return call_model_once(naive_transformer)


def build_and_call_patch_transformer(**kwargs):
    params = clean_up_params(get_params('model', 'patch'), **kwargs)
    patch_transformer = build_patch_transformer(**params)
    return call_model_once(patch_transformer)


def load_model_from_checkpoint(model, model_dir):
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=1)
    checkpoint.restore(manager.latest_checkpoint)
    return model


def clean_up_params(params, **kwargs):
    del params['epochs']
    del params['batch_size']
    params.update(kwargs)
    return params


def call_model_once(model):
    _, test_ds = load_normalized_mnist()
    for batch in test_ds.batch(1).take(1):
        model(batch['image'], training=False)
    return model
