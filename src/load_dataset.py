import tensorflow as tf
import tensorflow_datasets as tfds


def load_normalized_mnist():
    def normalize(record):
        image, label = record['image'], record['label']
        image = tf.cast(image, tf.float32)
        image = tf.math.divide(image, 255.0)
        return {'image': image, 'label': label}

    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'])
    train_ds = train_ds.map(normalize)
    test_ds = test_ds.map(normalize)
    return train_ds, test_ds


def load_normalized_rgb_and_resized_mnist():
    def normalize(record):
        image, label = record['image'], record['label']
        image = tf.cast(image, tf.float32)
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize_with_pad(image, 32, 32)
        image = tf.keras.applications.vgg16.preprocess_input(image)
        return {'image': image, 'label': label}

    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'])
    train_ds = train_ds.map(normalize)
    test_ds = test_ds.map(normalize)
    return train_ds, test_ds
