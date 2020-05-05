import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, Flatten, Input, Add
from tensorflow.keras.models import Model

# Turn off memory consumption
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

KERNEL_SIZE = (3, 3)

FILTERS = 128


class AlphaNet:
    def __init__(self):
        self.model = self._build_model()

    @staticmethod
    def _build_model():
        inp = Input(shape=(6, 7, 3), name='input')

        x = ConvLayer()(inp)
        for _ in range(20):
            x = ResLayer()(x)

        policy_out = PolicyLayer(name='policy_out')(x)
        value_out = ValueLayer(name='value_out')(x)

        model = Model(inp, outputs=[policy_out, value_out])
        model.compile(optimizer='adam',
                      loss={'policy_layer': 'mean_squared_error', 'value_layer': 'binary_crossentropy'},
                      loss_weights=[0.5, 0.5])
        return model


class ConvLayer(Layer):
    def __init__(self):
        super().__init__()
        self.conv2d = Conv2D(FILTERS, KERNEL_SIZE, padding='same')
        self.batch_norm = BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.conv2d(inputs)
        x = self.batch_norm(x)
        return tf.nn.relu(x)


class ResLayer(Layer):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = Conv2D(FILTERS, KERNEL_SIZE, padding='same')
        self.batch_norm_1 = BatchNormalization()

        self.conv2d_2 = Conv2D(FILTERS, KERNEL_SIZE, padding='same')
        self.batch_norm_2 = BatchNormalization()
        self.add = Add()

    def call(self, inputs, **kwargs):
        x = self.conv2d_1(inputs)
        x = self.batch_norm_1(x)
        x = tf.nn.relu(x)

        x = self.conv2d_2(x)
        x = self.batch_norm_2(x)

        x = self.add([x, inputs])

        return tf.nn.relu(x)


class PolicyLayer(Layer):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.conv2d = Conv2D(2, (1, 1))
        self.batch_norm = BatchNormalization()
        self.dense = Dense(7)
        self.flatten = Flatten()

    def call(self, inputs, **kwargs):
        x = self.conv2d(inputs)
        x = self.batch_norm(x)
        x = tf.nn.relu(x)

        x = self.flatten(x)
        x = self.dense(x)
        return tf.nn.softmax(x)


class ValueLayer(Layer):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.conv2d = Conv2D(1, (1, 1))
        self.batch_norm = BatchNormalization()

        self.flatten = Flatten()
        self.dense_1 = Dense(64, activation='relu')
        self.dense_2 = Dense(1, activation='tanh')

    def call(self, inputs, **kwargs):
        x = self.conv2d(inputs)
        x = self.batch_norm(x)
        x = tf.nn.relu(x)

        x = self.flatten(x)
        x = self.dense_1(x)
        return self.dense_2(x)


if __name__ == '__main__':
    a = AlphaNet()
    print(a.model.summary())
