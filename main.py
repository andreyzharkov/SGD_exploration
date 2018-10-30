import os
import numpy as np
import cv2
from keras.layers import Input, Lambda, Dense, Layer
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras.models import Model, load_model
import keras.backend as K

class SGDL(SGD):
    """Stochastic gradient descent Langevian optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, gamma=0.1, **kwargs):
        super(SGDL, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.gamma = K.variable(0.1, name="gamma")

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        gamma = self.gamma * (1. / (1. + K.cast_to_floatx(self.iterations)))

        # momentum
        shapes = [K.int_shape(p) for p in params]
        gaussians = [K.random_normal(shape) for shape in shapes]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m, W in zip(params, grads, moments, gaussians):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v
            new_p = new_p + gamma * W

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

class SomeLayer(Layer):
    def __init__(self, initializer='glorot_uniform'):
        self.initializer = initializer

    def build(self, input_shape):
        self.W = self.add_weight(shape=input_shape,
                                 initializer=self.initializer,
                                 name='W')

    def call(self, inputs):
        return self.W * inputs

def build_some_model(input_dim=1):
    input_x = Input(shape=(input_dim,))
    x = input_x


def train_multiple_times(model, optimizer, loss, times=100, **fit_generator_params):
    initial_weights = model.get_weights()
    all_weights = []
    for i in range(times):
        model.compile(optimizer=optimizer, loss=loss)
        model.fit_generator(**fit_generator_params)
        all_weights.append(model.get_weights())
    return all_weights