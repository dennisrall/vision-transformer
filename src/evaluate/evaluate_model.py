import tensorflow as tf

from load_dataset import load_normalized_mnist, \
    load_normalized_rgb_and_resized_mnist


def evaluate_model(model, use_rgb_mnist=False):
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_acc')

    @tf.function
    def val_step(images, labels):
        predictions = model(images, training=False)
        val_accuracy(y_pred=predictions, y_true=labels)

    if not use_rgb_mnist:
        _, test_ds = load_normalized_mnist()
    else:
        _, test_ds = load_normalized_rgb_and_resized_mnist()
    for batch in test_ds.batch(8):
        val_step(batch['image'], batch['label'])
    return val_accuracy.result().numpy()
