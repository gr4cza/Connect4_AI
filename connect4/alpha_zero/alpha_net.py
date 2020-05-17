import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

# For avoid memory leak
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

from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, Flatten, Input, Add  # noqa
from tensorflow.keras.models import Model  # noqa

REG_FACTOR = 0.0001
KERNEL_SIZE = (3, 3)
FILTERS = 128
CONV_LAYER_COUNT = 20

BASE_DIR = f'{os.path.dirname(__file__)}/training_data/models/'


class AlphaNet:
    def __init__(self, model_name, is_latest=False):
        self.model_name = model_name

        model_path = self._get_model_path(model_name)

        if self._check_model_exists(model_path):
            self.load_model(model_name, is_latest)
        else:
            self._model = self._build_model()
            self.save_model()

    @staticmethod
    def _build_model():
        inp = Input(shape=(6, 7, 3), name='input')

        x = ConvLayer()(inp)
        for _ in range(CONV_LAYER_COUNT):
            x = ResLayer()(x)

        policy_out = PolicyLayer(name='policy_out')(x)
        value_out = ValueLayer(name='value_out')(x)

        model = Model(inp, outputs=[policy_out, value_out])
        AlphaNet._compile(model)
        return model

    @staticmethod
    def _compile(model):
        model.compile(optimizer='adam',
                      loss={'policy_out': tf.nn.softmax_cross_entropy_with_logits,
                            'value_out': 'mean_squared_error'},
                      loss_weights=[0.5, 0.5])

    def train(self, data, epochs):

        # load dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (data.board, {'policy_out': data.policy, 'value_out': data.value}))

        # prepare dataset
        size = int(len(data.value) / 10)
        train_dataset = train_dataset.shuffle(size, reshuffle_each_iteration=True).batch(32) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        # learn on dataset
        history = self._model.fit(train_dataset, epochs=epochs, verbose=2)
        self._save_history(history)

        # save new model
        self.save_model()

    def predict(self, board):
        return self._model.predict_on_batch(np.reshape(board, (-1, 6, 7, 3)))

    def load_model(self, model_name, is_latest):
        file_path = self._get_model_path(model_name)

        if self._check_model_exists(file_path):
            model = self._get_model(file_path, is_latest)
            self._model = tf.keras.models.load_model(model, compile=False)
            self._compile(self._model)
        else:
            print(f'Saved model "{model_name}" does not exists (or corrupted)!')

    def save_model(self):
        file_path = self._get_model_path(self.model_name)
        new_model_path = self._get_new_model_number(file_path)
        self._model.save(new_model_path)

    @staticmethod
    def _get_model_path(model_name):
        return BASE_DIR + f'{model_name}/'

    @staticmethod
    def _check_model_exists(model_path):
        return os.path.exists(model_path)

    @staticmethod
    def _get_model(file_path, is_latest):
        if not os.path.exists(file_path + 'catalog.json'):
            net_number = AlphaNet.create_data_json(file_path)
        else:
            with open(file_path + 'catalog.json', 'r')as f:
                data = json.load(f)
                net_number = data['best_net'] if not is_latest else data['latest_net']
        print(f'Loading net version: {net_number:02}')
        return file_path + f'{net_number:02}/'

    @staticmethod
    def create_data_json(file_path):
        Path(file_path).mkdir(parents=True, exist_ok=True)

        with open(file_path + 'catalog.json', 'w')as f:
            data = {'best_net': 0, 'latest_net': 0}
            f.write(json.dumps(data))
            return data['best_net']

    @staticmethod
    def _get_new_model_number(file_path):
        if not os.path.exists(file_path + 'catalog.json'):
            new_model_number = AlphaNet.create_data_json(file_path)
        else:
            with open(file_path + 'catalog.json', 'r+')as f:
                data = json.load(f)
                f.seek(0)
                data['latest_net'] += 1
                f.write(json.dumps(data))
                new_model_number = data['latest_net']
        return file_path + f'{new_model_number:02}/'

    def release(self):
        del self._model
        tf.keras.backend.clear_session()

    @staticmethod
    def _save_history(history):
        print(history)


def cross_entropy_loss(y_true, y_pred):
    return tf.math.negative(tf.math.reduce_sum(tf.math.multiply_no_nan(tf.math.log(y_pred), y_true)))


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
        return self.dense(x)


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
