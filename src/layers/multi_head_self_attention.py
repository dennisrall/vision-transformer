from tensorflow.keras import layers


class MultiHeadSelfAttention(layers.Layer):
    """Self attention wrapper around the (general) attention layer."""

    def __init__(self, num_heads, key_dim):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim)

    def call(self, x, **kwargs):
        return self.multi_head_attention(x, x, return_attention_scores=True)

    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        new_config = {
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
        }
        config.update(new_config)
        return config
