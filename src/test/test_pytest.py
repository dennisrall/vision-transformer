def test_tensorflow():
    import tensorflow as tf
    constant = tf.constant('TensorFlow was here!')
    str(constant.numpy())
