from tensorflow.keras.layers import Layer
import tensorflow as tf

class LegalMoveMask(Layer):
    def __init__(self, **kwargs):
        super(LegalMoveMask, self).__init__(**kwargs)

    def call(self, inputs):
        predictions, legal_moves_mask = inputs
        return predictions * legal_moves_mask

class DropConnect(Layer):
    def __init__(self, rate, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            keep_prob = 1.0 - self.rate
            random_tensor = keep_prob
            random_tensor += tf.random.uniform(tf.shape(inputs), dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = tf.divide(inputs, keep_prob) * binary_tensor
            return output
        return inputs