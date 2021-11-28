import tensorflow as tf


class PositionEmbedding(tf.keras.layers.Layer):
    """Position embedding for transformers.

    A learned embedding is added to token sequences of length num_tokens and
    dimension embed_dim.
    """

    def __init__(self, num_tokens, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        self.pos_embedding = \
            self.add_weight('position_embeddings',
                            shape=(num_tokens, embed_dim),
                            initializer=tf.keras.initializers.RandomNormal(),
                            dtype=tf.float32)

    def call(self, x, **kwargs):
        return x + self.pos_embedding

    def get_config(self):
        config = super(PositionEmbedding, self).get_config()
        new_config = {
            'num_tokens': self.num_tokens,
            'embed_dim': self.embed_dim,
        }
        config.update(new_config)
        return config
