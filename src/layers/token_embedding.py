from tensorflow.keras import layers


class TokenEmbedding(layers.Layer):
    """Token embedding for transformers."""

    def __init__(self, embed_dim):
        super(TokenEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = layers.Dense(units=embed_dim, use_bias=False)

    def call(self, x, **kwargs):
        return self.embedding(x, **kwargs)

    def get_config(self):
        config = super(TokenEmbedding, self).get_config()
        new_config = {
            'embed_dim': self.embed_dim,
        }
        config.update(new_config)
        return config
