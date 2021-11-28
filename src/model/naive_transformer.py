import tensorflow as tf

from layers.mlp_head import MlpHead
from layers.transformer import ClsTokenTransformerWithEmbeddings


def build_naive_transformer(image_shape, embed_dim, num_heads, ff_dim,
                            num_classes, num_transformer_blocks):
    num_pixels = image_shape[0] * image_shape[1]
    reshape = tf.keras.layers.Reshape((num_pixels, 1))
    transformer = ClsTokenTransformerWithEmbeddings(
        depth=num_transformer_blocks, dim=embed_dim, num_heads=num_heads,
        mlp_dim=ff_dim, num_tokens=num_pixels)
    mlp_head = MlpHead(hidden_units=ff_dim, units=num_classes,
                       final_activation='softmax')
    return tf.keras.Sequential([
        reshape,
        transformer,
        mlp_head,
    ])
