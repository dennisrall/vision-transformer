import tensorflow as tf
from matplotlib import pyplot as plt

from config import base_plot_dir, base_model_dir, experiment1_dir
from evaluate.load_model import clean_up_params, load_model_from_checkpoint
from load_dataset import load_normalized_mnist
from model.patch_transformer import build_patch_transformer
from util import get_params


def visualize_attention(image_num):
    filename = base_plot_dir / 'attention_visualization'
    transformer = load_transformer()
    _, test_ds = load_normalized_mnist()
    image = next(iter(test_ds.batch(1).skip(image_num)))
    prediction = transformer(image['image'])
    print('label:', image['label'].numpy()[0])
    print('prediction:', tf.argmax(prediction[0]).numpy())
    att_scores = transformer.layers[1].get_att_scores()
    rollout_value = attention_rollout(att_scores)
    heatmap = calc_heatmap(rollout_value)
    visualize(image['image'], heatmap, filename)


def load_transformer():
    params = clean_up_params(get_params('model', 'experiment1'), patch_size=2,
                             num_classes=10, image_size=28,
                             num_transformer_blocks=15)
    model = build_patch_transformer(**params)
    model_dir = base_model_dir / experiment1_dir / 'patch2-15layers'
    return load_model_from_checkpoint(model, model_dir)


def aggregate_heads(att_scores):
    att_scores = tf.reduce_mean(att_scores, axis=1)
    att_scores = tf.keras.utils.normalize(att_scores, order=1)
    return att_scores


def add_residual_connection(att_scores):
    n, x, y = att_scores.shape
    return 0.5 * att_scores + 0.5 * tf.eye(x, y, [n])


def rollout(att_scores):
    iterator = iter(att_scores)
    result = next(iterator)
    for att_score in att_scores:
        result = tf.matmul(att_score, result)
    return result[0]


def attention_rollout(att_scores):
    att_scores = tf.concat(att_scores, axis=0)
    att_scores = aggregate_heads(att_scores)
    att_scores = add_residual_connection(att_scores)
    return rollout(att_scores)


def calc_heatmap(rollout_value):
    patch_size = 2
    number_of_patches = 28 // patch_size

    patch_values = rollout_value[1:]

    result = []
    for patch_value in patch_values:
        x = tf.concat([[patch_value] * patch_size], axis=0)
        result = tf.concat([*result, *x], axis=0)

    result = tf.reshape(result, (number_of_patches, 28))

    result = tf.concat([
        [row] * patch_size
        for row in result
    ], axis=0)
    return result


def visualize(image, heatmap, filename):
    plt.cla()
    plt.figure()
    heatmap = tf.expand_dims(heatmap, axis=-1)
    for i in range(10):
        size = 28 + i
        heatmap = tf.image.resize(heatmap, (size, size), method='gaussian')
    heatmap = tf.image.resize(heatmap, (28, 28), method='gaussian')
    plt.imshow(tf.image.grayscale_to_rgb(image[0]))
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(filename)


if __name__ == '__main__':
    visualize_attention(**get_params('evaluate', 'attention_visualization'))
