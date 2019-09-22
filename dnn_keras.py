#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:30:38 2019

@author: dhruv
"""

import tensorflow as tf
import numpy as np

random_seed = 1
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data(path='mnist.npz')
X_train = X_train.reshape((-1,784)) / 256
X_test = X_test.reshape((-1,784)) / 256
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from tensorflow import keras
y_train_enc = keras.utils.to_categorical(y_train)

model = keras.models.Sequential()
model.add(keras.layers.Dense(64, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='tanh'))
model.add(keras.layers.Dense(64, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='tanh'))
model.add(keras.layers.Dense(y_train_enc.shape[1], kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='softmax'))

sgd_optimizer = keras.optimizers.SGD(learning_rate=0.001, decay=1e-7, momentum=0.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
history = model.fit(X_train, y_train_enc, batch_size=128, epochs=50, validation_split=0.1, verbose=1)

y_test_pred = model.predict_classes(X_test, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))



