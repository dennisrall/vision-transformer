import tensorflow as tf

from layers.extract_patches import ExtractPatches
from layers.mlp_head import MlpHead
from layers.transformer import ClsTokenTransformerWithEmbeddings


def build_patch_transformer(patch_size, image_size, embed_dim, num_heads,
                            ff_dim, num_classes, num_transformer_blocks):
    assert image_size % patch_size == 0, \
        'Image size must be divisible by the patch size'
    num_patches = (image_size // patch_size) ** 2

    extract_patches = ExtractPatches(patch_size)
    transformer = ClsTokenTransformerWithEmbeddings(
        depth=num_transformer_blocks, dim=embed_dim, num_heads=num_heads,
        mlp_dim=ff_dim, num_tokens=num_patches)
    mlp_head = MlpHead(hidden_units=ff_dim, units=num_classes,
                       final_activation='softmax')
    return tf.keras.Sequential([
        extract_patches,
        transformer,
        mlp_head,
    ])
