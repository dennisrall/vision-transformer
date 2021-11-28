from vit_keras import vit

vit_types = {
    'vit_b16': vit.vit_b16,
    'vit_b32': vit.vit_b32,
    'vit_l16': vit.vit_l16,
    'vit_l32': vit.vit_l32,
}


def build_transfer_learning_transformer(image_size, vit_type, num_classes):
    return vit_types[vit_type](image_size=image_size, activation='softmax',
                               pretrained=True, include_top=True,
                               pretrained_top=False, classes=num_classes)
