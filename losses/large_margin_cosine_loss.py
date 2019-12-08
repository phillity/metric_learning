import tensorflow as tf
import tensorflow.keras.layers as L


__all__ = [
    "CosFace"
]

"""
CosFace: Large Margin Cosine Loss for Deep Face Recognition (CVPR '18)
Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, Wei Liu
https://arxiv.org/abs/1801.09414

Implementation inspired by repo:
https://github.com/4uiiurz1/keras-arcface

large margin cosine loss:
L = -log(e^(s(cos(theta_{y_i, i}) - m)) / (e^(s(cos(theta_{y_i, i}) - m) + sum(e^(s(cos(theta_{j, i})))))
W = W* / ||W*||
x = x* / ||x*||
cos(theta_{j, i}) = W_j.T * x_i
"""


class CosFace(L.Layer):
    def __init__(self, num_classes=10, scale=30.0, margin=0.35, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

    def build(self, input_shape):
        self.W = self.add_weight(name="W",
                                 shape=(input_shape[0][-1], self.num_classes),
                                 initializer="glorot_uniform",
                                 trainable=True)

    def call(self, inputs):
        # get embeddings and one hot labels from inputs
        embeddings, onehot_labels = inputs
        # normalize final W layer
        W = tf.nn.l2_normalize(self.W, axis=0)
        # get logits from multiplying embeddings (batch_size, embedding_size) and W (embedding_size, num_classes)
        logits = tf.matmul(embeddings, self.W)
        # subtract margin from logits
        target_logits = logits - self.margin
        # get cross entropy
        logits = logits * (1 - onehot_labels) + target_logits * onehot_labels
        # apply scaling
        logits = logits * self.scale
        # get class probability distribution
        predictions = tf.nn.softmax(logits)
        return predictions

    def compute_output_shape(self, input_shape):
        return (None, self.num_classes)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_classes": self.num_classes,
            "scale": self.scale,
            "margin": self.margin})
        return config
