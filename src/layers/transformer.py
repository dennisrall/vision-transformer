import tensorflow as tf

from layers.mlp_head import MlpHead
from layers.multi_head_self_attention import MultiHeadSelfAttention
from layers.position_embedding import PositionEmbedding
from layers.token_embedding import TokenEmbedding


class TransformerBlock(tf.keras.layers.Layer):
    """Basic building block of a transformer."""

    def __init__(self, dim, num_heads, mlp_dim, epsilon=1e-5):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.epsilon = epsilon

        self.normalization1 = tf.keras.layers.LayerNormalization(
            epsilon=epsilon)
        self.attention = MultiHeadSelfAttention(num_heads=num_heads,
                                                key_dim=dim)
        self.normalization2 = tf.keras.layers.LayerNormalization(
            epsilon=epsilon)
        self.mlp_head = MlpHead(hidden_units=mlp_dim, units=dim)

    def call(self, x, **kwargs):
        residual = x
        x = self.normalization1(x, **kwargs)
        x, att_score = self.attention(x, **kwargs)
        x = residual + x

        residual = x
        x = self.normalization2(x, **kwargs)
        x = self.mlp_head(x, **kwargs)
        x = residual + x
        return x, att_score

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        new_config = {
            'dim': self.dim,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
            'epsilon': self.epsilon,
        }
        config.update(new_config)
        return config


class Transformer(tf.keras.layers.Layer):
    """Stack of TransformerBlocks of size depth."""

    def __init__(self, depth, dim, num_heads, mlp_dim):
        super(Transformer, self).__init__()
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        self.transformer_blocks = [
            TransformerBlock(dim=dim, num_heads=num_heads, mlp_dim=mlp_dim)
            for _ in range(depth)
        ]

    def call(self, x, **kwargs):
        att_scores = []
        for transformer_block in self.transformer_blocks:
            x, att_score = transformer_block(x, **kwargs)
            att_scores.append(att_score)
        return x, att_scores

    def get_config(self):
        config = super(Transformer, self).get_config()
        new_config = {
            'depth': self.depth,
            'dim': self.dim,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
        }
        config.update(new_config)
        return config


class ClsTokenTransformer(tf.keras.layers.Layer):
    """Returns only a transformed cls_token.

    A cls_token is added to the sequence before transforming it.
    Afterwards only the transformed cls_token is returned."""

    def __init__(self, depth, dim, num_heads, mlp_dim):
        super(ClsTokenTransformer, self).__init__()
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        self.cls_token = \
            self.add_weight('cls_token', shape=(1, 1, dim),
                            initializer=tf.keras.initializers.RandomNormal(),
                            dtype=tf.float32)
        self.transformer = Transformer(depth, dim, num_heads, mlp_dim)
        self.dim = dim

    def call(self, x, **kwargs):
        shape = tf.shape(x)
        cls_tokens = tf.broadcast_to(self.cls_token, (shape[0], 1, self.dim))
        x = tf.concat((cls_tokens, x), axis=1)
        x, att_scores = self.transformer(x, **kwargs)
        return x[:, 0], att_scores

    def get_config(self):
        config = super(ClsTokenTransformer, self).get_config()
        new_config = {
            'depth': self.depth,
            'dim': self.dim,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
        }
        config.update(new_config)
        return config


class ClsTokenTransformerWithEmbeddings(tf.keras.layers.Layer):
    """A ClsTokenTransformer with Token and Position embeddings."""

    def __init__(self, depth, dim, num_heads, mlp_dim, num_tokens):
        super(ClsTokenTransformerWithEmbeddings, self).__init__()
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.num_tokens = num_tokens

        self.token_embedding = TokenEmbedding(embed_dim=dim)
        self.position_embedding = PositionEmbedding(num_tokens=num_tokens,
                                                    embed_dim=dim)
        self.cls_token_transformer = ClsTokenTransformer(depth, dim, num_heads,
                                                         mlp_dim)
        self.att_scores = None

    def call(self, x, **kwargs):
        x = self.token_embedding(x, **kwargs)
        x = self.position_embedding(x, **kwargs)
        x, self.att_scores = self.cls_token_transformer(x, **kwargs)
        return x

    def get_att_scores(self):
        return self.att_scores

    def get_config(self):
        config = super(ClsTokenTransformerWithEmbeddings, self).get_config()
        new_config = {
            'depth': self.depth,
            'dim': self.dim,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
            'num_tokens': self.num_tokens,
        }
        config.update(new_config)
        return config
