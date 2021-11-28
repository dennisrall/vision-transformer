from itertools import islice

import tensorflow as tf

from layers.mlp_head import MlpHead
from layers.transformer import ClsTokenTransformerWithEmbeddings
from load_dataset import load_normalized_rgb_and_resized_mnist


def build_hybrid_transformer(num_cnn_layers,
                             embed_dim,
                             num_heads,
                             ff_dim,
                             num_classes):
    pretrained_net = tf.keras.applications.VGG16(input_shape=(32, 32, 3),
                                                 include_top=False,
                                                 weights='imagenet')
    part_of_mobile_net = list(islice(pretrained_net.layers, num_cnn_layers))
    for layer in part_of_mobile_net:
        layer.trainable = False
    num_transformer_blocks = len(pretrained_net.layers) - num_cnn_layers
    output_shape = get_output_shape_of_vgg16(num_cnn_layers)
    new_shape = (output_shape[1] * output_shape[2], output_shape[3])
    reshape = tf.keras.layers.Reshape(new_shape)
    transformer = ClsTokenTransformerWithEmbeddings(
        depth=num_transformer_blocks, dim=embed_dim, num_heads=num_heads,
        mlp_dim=ff_dim, num_tokens=new_shape[0])
    mlp_head = MlpHead(hidden_units=ff_dim, units=num_classes,
                       final_activation='softmax')
    return tf.keras.Sequential([
        *part_of_mobile_net,
        reshape,
        transformer,
        mlp_head,
    ])


def build_pure_cnn(ff_dim, num_classes):
    pretrained_net = tf.keras.applications.VGG16(input_shape=(32, 32, 3),
                                                 include_top=False,
                                                 weights='imagenet')
    pretrained_net.trainable = False
    flatten = tf.keras.layers.Flatten()
    mlp_head = MlpHead(hidden_units=ff_dim, units=num_classes,
                       final_activation='softmax')
    return tf.keras.Sequential([
        pretrained_net,
        flatten,
        mlp_head,
    ])


def get_output_shape_of_vgg16(num_cnn_layers):
    pretrained_net = tf.keras.applications.VGG16(input_shape=(32, 32, 3),
                                                 include_top=False)
    part_of_mobile_net = list(islice(pretrained_net.layers, num_cnn_layers))
    model = tf.keras.Sequential(part_of_mobile_net)
    test_ds, _ = load_normalized_rgb_and_resized_mnist()
    data = next(iter(test_ds.batch(1).take(1)))
    image = data['image']
    output = model(image)
    return output.shape
