import tensorflow as tf


class ExtractPatches(tf.keras.layers.Layer):
    """Extract quadratic patches from quadratic images.


    Extract patches of size (patch_size, patch_size, c) from batches of images
    of size (batch_size, image_size, image_size, c).
    """

    def __init__(self, patch_size):
        super(ExtractPatches, self).__init__()
        self.patch_size = patch_size

    def call(self, x, **kwargs):
        x = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID')
        old_shape = x.shape
        new_shape = (old_shape[0], old_shape[1] * old_shape[2], old_shape[3])
        x = tf.reshape(x, new_shape)
        return x

    def get_config(self):
        config = super(ExtractPatches, self).get_config()
        new_config = {
            'patch_size': self.patch_size,
        }
        config.update(new_config)
        return config
