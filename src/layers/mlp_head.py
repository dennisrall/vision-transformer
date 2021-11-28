import tensorflow as tf


class MlpHead(tf.keras.layers.Layer):
    """Multi-Layer-Perceptron used as a classification head."""

    def __init__(self, hidden_units, units, final_activation=None):
        super().__init__()
        self.hidden_units = hidden_units
        self.units = units
        self.final_activation = final_activation
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units, activation=final_activation)

    def call(self, x, **kwargs):
        x = self.dense1(x, **kwargs)
        x = self.dense2(x, **kwargs)
        return x

    def get_config(self):
        config = super(MlpHead, self).get_config()
        new_config = {
            'hidden_units': self.hidden_units,
            'units': self.units,
            'final_activation': self.final_activation,
        }
        config.update(new_config)
        return config
