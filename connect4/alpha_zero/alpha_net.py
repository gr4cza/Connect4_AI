import os

import numpy as np
import tensorflow as tf

# For avoid memmory leak
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, Flatten, Input, Add
from tensorflow.keras.models import Model

REG_FACTOR = 0.0001
KERNEL_SIZE = (3, 3)
FILTERS = 128
CONV_LAYER_COUNT = 20

DATA_PATH = f'{os.path.dirname(__file__)}/training_data/'


class AlphaNet:
    def __init__(self, model_name):
        model_path = self._get_model_path(model_name)

        if self._check_model_exists(model_path):
            self.load_model(model_name)
        else:
            self._model = self._build_model()
            self.save_model(model_name)

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

    def train(self, data, epochs, new_name):
        self._model.fit(data.board, {'policy_out': data.policy, 'value_out': data.value}, epochs=epochs)
        self.save_model(new_name)
        return new_name

    def predict(self, board):
        return self._model.predict_on_batch(np.reshape(board, (-1, 6, 7, 3)))

    def load_model(self, file_name):
        file_path = self._get_model_path(file_name)
        if self._check_model_exists(file_path):
            self._model = tf.keras.models.load_model(file_path)
        else:
            print(f'Saved model "{file_name}" does not exists!')

    def save_model(self, file_name):
        file_path = self._get_model_path(file_name)
        self._model.save(file_path)

    @staticmethod
    def _get_model_path(file_name):
        return DATA_PATH + f'models/{file_name}/'

    @staticmethod
    def _check_model_exists(model_path):
        return os.path.exists(model_path)

    def release(self):
        del self._model
        tf.keras.backend.clear_session()


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
    a = AlphaNet('test')
    print(a._model.summary())  # noqa
