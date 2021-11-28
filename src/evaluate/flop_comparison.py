from evaluate.load_model import build_and_call_patch_transformer, \
    build_and_call_naive_transformer
from flops import get_flops


def compare_flops():
    patch_transformer = build_and_call_patch_transformer(
        patch_size=2, image_size=28, num_heads=8, embed_dim=8, num_classes=10,
        ff_dim=32, num_transformer_blocks=64)
    naive_transformer = build_and_call_naive_transformer(
        image_shape=(28, 28, 1), num_heads=8, embed_dim=8, num_classes=10,
        ff_dim=32, num_transformer_blocks=64)
    patch_flops = get_flops(patch_transformer)
    naive_flops = get_flops(naive_transformer)
    print('patch flops', patch_flops)
    print('naive flops', naive_flops)
    print('ratio', naive_flops / patch_flops)
