import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, Flatten, Input, Add
from tensorflow.keras.models import Model

REG_FACTOR = 0.0001
KERNEL_SIZE = (3, 3)
FILTERS = 128
CONV_LAYER_COUNT = 20


class AlphaNet:
    def __init__(self):
        self.model = self._build_model()

    @staticmethod
    def _build_model():
        inp = Input(shape=(6, 7, 3), name='input')

        x = ConvLayer()(inp)
        for _ in range(CONV_LAYER_COUNT):
            x = ResLayer()(x)

        policy_out = PolicyLayer(name='policy_out')(x)
        value_out = ValueLayer(name='value_out')(x)

        model = Model(inp, outputs=[policy_out, value_out])
        model.compile(optimizer='adam',
                      loss={'policy_out': tf.keras.losses.CategoricalCrossentropy(),
                            'value_out': 'mean_squared_error'},
                      # TODO own loss function ?
                      loss_weights=[0.5, 0.5])
        return model


class ConvLayer(Layer):
    def __init__(self):
        super().__init__()
        self.conv2d = Conv2D(FILTERS, KERNEL_SIZE, padding='same',
                             kernel_regularizer=tf.keras.regularizers.l2(REG_FACTOR))
        self.batch_norm = BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.conv2d(inputs)
        x = self.batch_norm(x, training=kwargs.get('training'))
        return tf.nn.relu(x)


class ResLayer(Layer):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = Conv2D(FILTERS, KERNEL_SIZE, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(REG_FACTOR))
        self.batch_norm_1 = BatchNormalization()

        self.conv2d_2 = Conv2D(FILTERS, KERNEL_SIZE, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(REG_FACTOR))
        self.batch_norm_2 = BatchNormalization()
        self.add = Add()

    def call(self, inputs, **kwargs):
        x = self.conv2d_1(inputs)
        x = self.batch_norm_1(x, training=kwargs.get('training'))
        x = tf.nn.relu(x)

        x = self.conv2d_2(x)
        x = self.batch_norm_2(x, training=kwargs.get('training'))

        x = self.add([x, inputs])

        return tf.nn.relu(x)


class PolicyLayer(Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv2d = Conv2D(2, (1, 1), kernel_regularizer=tf.keras.regularizers.l2(REG_FACTOR))
        self.batch_norm = BatchNormalization()
        self.dense = Dense(7, kernel_regularizer=tf.keras.regularizers.l2(REG_FACTOR))
        self.flatten = Flatten()

    def call(self, inputs, **kwargs):
        x = self.conv2d(inputs)
        x = self.batch_norm(x, training=kwargs.get('training'))
        x = tf.nn.relu(x)

        x = self.flatten(x)
        x = self.dense(x)
        return tf.nn.softmax(x)


class ValueLayer(Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv2d = Conv2D(1, (1, 1), kernel_regularizer=tf.keras.regularizers.l2(REG_FACTOR))
        self.batch_norm = BatchNormalization()

        self.flatten = Flatten()
        self.dense_1 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(REG_FACTOR))
        self.dense_2 = Dense(1, activation='tanh')

    def call(self, inputs, **kwargs):
        x = self.conv2d(inputs)
        x = self.batch_norm(x, training=kwargs.get('training'))
        x = tf.nn.relu(x)

        x = self.flatten(x)
        x = self.dense_1(x)
        return self.dense_2(x)


if __name__ == '__main__':
    a = AlphaNet()
    print(a.model.summary())
