####################################################################################################
#                                             models.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 17/04/21                                                                                #
#                                                                                                  #
# Purpose: Definition of the architecture of the augmentation for the MUSIC algorithm.             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import Regularizer, L1, L1L2
from tensorflow.keras.utils import plot_model

from scipy import linalg
from scipy import signal

from syntheticEx import *
from utils import *


#********************#
#   initialization   #
#********************#
n = m - d   # number of noise vectors
r = angles.shape[1]   # resolution (i.e. angle grid size)


#***********#
#   model   #
#***********#
def create_model_simpleCNN():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)

    y = GaussianNoise(1)(y)

    y = BatchNormalization()(y)

    y = Conv1D(filters=64, strides=(2,), kernel_size=(5,), activation='relu')(y)
    y = MaxPooling1D(pool_size=2, strides=2)(y)
    y = Conv1D(filters=64, strides=(1,), kernel_size=(3,), activation='relu')(y)
    y = MaxPooling1D(pool_size=2, strides=2)(y)

    y = Permute((2, 1))(y)

    y = BatchNormalization()(y)

    y = Dropout(0.2)(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.2)(y)

    y = BatchNormalization()(y)

    y = Dense(n)(y)

    return x, y


#***********#
#   model   #
#***********#
def create_model_CNNwithEVD():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)

    y = GaussianNoise(1)(y)

    y = BatchNormalization()(y)

    y = Conv1D(filters=64, strides=(2,), kernel_size=(5,), activation='relu')(y)
    y = MaxPooling1D(pool_size=2, strides=2)(y)
    y = Conv1D(filters=64, strides=(1,), kernel_size=(3,), activation='relu')(y)
    y = MaxPooling1D(pool_size=2, strides=2)(y)

    y = Permute((2, 1))(y)

    y = BatchNormalization()(y)

    y = Dropout(0.2)(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.2)(y)

    y = BatchNormalization()(y)

    y = Dense(m)(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    return x, y


#***********#
#   model   #
#***********#
def create_model_E2E_simpleCNN():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)

    y = GaussianNoise(1)(y)

    y = BatchNormalization()(y)

    y = Conv1D(filters=64, strides=(2,), kernel_size=(5,), activation='relu')(y)
    y = MaxPooling1D(pool_size=2, strides=2)(y)
    y = Conv1D(filters=64, strides=(1,), kernel_size=(3,), activation='relu')(y)
    y = MaxPooling1D(pool_size=2, strides=2)(y)

    y = Permute((2, 1))(y)

    y = BatchNormalization()(y)

    y = Dropout(0.2)(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.2)(y)

    y = BatchNormalization()(y)

    y = Flatten()(y)
    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def create_model_RNN():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eigh(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    return x, y


#***********#
#   model   #
#***********#
def create_model_E2E_RNN():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m * m)(y)

    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def create_model_deepRNN():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m, return_sequences=True)(y)
    y = GRU(2 * m, return_sequences=True)(y)
    y = GRU(2 * m)(y)

    y = Dense(2 * m * m)(y)

    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    return x, y


#***********#
#   model   #
#***********#
def create_model_E2E_deepRNN():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m, return_sequences=True)(y)
    y = GRU(2 * m, return_sequences=True)(y)
    y = GRU(2 * m)(y)

    y = Dense(2 * m * m)(y)

    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def create_model_simpleRNN():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m)(y)

    y = Dense(2 * m * m)(y)

    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    return x, y


#***********#
#   model   #
#***********#
def create_model_simpleRNN_noEVD():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m)(y)

    y = Dense(2 * m * n)(y)

    y = Reshape((2 * m, n))(y)

    return x, y


#***********#
#   model   #
#***********#
def create_model_E2E_simpleRNN():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m)(y)

    y = Dense(2 * m * m)(y)

    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def create_model_alternative():
    """
        Deep augmented MUSIC as presented in
        "DEEP AUGMENTED MUSIC ALGORITHM FOR DATA-DRIVEN DOA ESTIMATION".

    """
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m)(y)

    y = Dense(2 * m * m)(y)

    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    y = Lambda(lambda y: calculate_spectrum(y))(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def create_model_E2E_alternative():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m)(y)

    y = Dense(2 * m * m)(y)

    y = Dense(angles.shape[1])(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def create_model_alternative_noEVD():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m)(y)

    y = Dense(2 * m * n)(y)

    y = Reshape((2 * m, n))(y)

    y = Lambda(lambda y: calculate_spectrum(y))(y)

    y = Dense(2 * m, activation='relu')(y)
    y = Dense(2 * m, activation='relu')(y)
    y = Dense(2 * m, activation='relu')(y)

    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def deepMUSIC():
    q = 12

    inp = Input((3, m, m))

    x = BatchNormalization()(inp)
    x = Permute((2, 3, 1))(x)

    out = []
    for i in range(q):
        y = Conv2D(filters=256, strides=(1, 1), kernel_size=(5, 5), activation='relu')(x)
        # y = BatchNormalization()(y)
        y = Conv2D(filters=256, strides=(1, 1), kernel_size=(5, 5), activation='relu', padding='same')(y)
        # y = BatchNormalization()(y)
        y = Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3), activation='relu')(y)
        # y = BatchNormalization()(y)
        y = Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3), activation='relu', padding='same')(y)
        y = BatchNormalization()(y)

        y = Flatten()(y)

        y = Dropout(0.3)(y)

        y = Dense(1024, activation='relu')(y)

        y = Dropout(0.3)(y)

        out.append(Dense(r//q)(y))

    y = Concatenate()(out)
    y = LayerNormalization()(y)

    # y = Dense(2 * m, activation='relu')(y)
    # y = Dense(2 * m, activation='relu')(y)
    # y = Dense(2 * m, activation='relu')(y)
    #
    # y = Dense(d)(y)

    return inp, y


#***********#
#   model   #
#***********#
def deepMUSICSeperate():
    q = 24

    inp = Input((3, m, m))

    x = BatchNormalization()(inp)
    x = Permute((2, 3, 1))(x)

    out = []
    for i in range(q):
        y = Conv2D(filters=256, strides=(1, 1), kernel_size=(5, 5), activation='relu')(x)
        # y = BatchNormalization()(y)
        # y = Conv2D(filters=256, strides=(1, 1), kernel_size=(5, 5), activation='relu', padding='same')(y)
        # y = BatchNormalization()(y)
        y = Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3), activation='relu')(y)
        # y = BatchNormalization()(y)
        # y = Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3), activation='relu', padding='same')(y)

        # y = LayerNormalization()(y)

        y = Flatten()(y)

        y = Dropout(0.3)(y)

        y = Dense(1024, activation='sigmoid')(y)

        # y = Dropout(0.3)(y)

        y = Dense(r//q, activation='sigmoid')(y)

        # y = LayerNormalization()(y)

        out.append(y)

    return inp, out


#***********#
#   model   #
#***********#
def deepMUSICorig():
    q = 24

    inp = Input((3, m, m))

    x = BatchNormalization()(inp)
    x = Permute((2, 3, 1))(x)

    out = []
    for i in range(q):
        y = Conv2D(filters=256, strides=(1, 1), kernel_size=(5, 5))(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3))(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Flatten()(y)

        y = Dense(1024)(y)

        # y = Dropout(0.3)(y)

        y = Activation('softmax')(y)

        y = Dense(r//q)(y)

        out.append(y)

    return inp, out