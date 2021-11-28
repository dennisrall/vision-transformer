from evaluate.load_model import build_and_call_patch_transformer, \
    build_and_call_naive_transformer
from train_util import calculate_trainable_variables


def compare_patch_and_naive_transformer():
    patch_transformer = build_and_call_patch_transformer(
        patch_size=1, image_size=28, num_heads=8, embed_dim=8, num_classes=10,
        ff_dim=32, num_transformer_blocks=64)
    naive_transformer = build_and_call_naive_transformer(
        image_shape=(28, 28, 1), num_heads=8, embed_dim=8, num_classes=10,
        ff_dim=32, num_transformer_blocks=64)
    print(calculate_trainable_variables(patch_transformer))
    print(calculate_trainable_variables(naive_transformer))


if __name__ == '__main__':
    compare_patch_and_naive_transformer()
