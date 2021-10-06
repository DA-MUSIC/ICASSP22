####################################################################################################
#                                           trainModel.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 23/03/21                                                                                #
#                                                                                                  #
# Purpose: Training of the augmentation for the MUSIC algorithm. Neural network outputs noise      #
#          subspaces directly from the measurements.                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import h5py
import keras
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import time

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import Regularizer, L1, L1L2
from tensorflow.keras.utils import plot_model

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize, Normalizer, MinMaxScaler, StandardScaler

from scipy import linalg
from scipy import signal
from scipy.stats import laplace

from augMUSIC import augMUSIC
from beamformer import beamformer
from classicMUSIC import classicMUSIC
from errorMeasures import *
from losses import *
from models import *
from syntheticEx import *


#***********************************************#
#   eliminate randomness to reproduce results   #
#***********************************************#
np.random.seed(42)
tf.random.set_seed(42)

tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)


#****************************************#
#   read files (may need to alter path)  #
#****************************************#
hf = h5py.File('data/m8/d5_l200_snr10_10k_c.h5', 'r')

dataX = np.array(hf.get('X'))
dataY = np.array(hf.get('Y'))

# dataX, dataY = utils.shuffle(dataX, dataY)

trainX_real = np.real(dataX)
trainX_imag = np.imag(dataX)

# trainX = np.stack((trainX_real, trainX_imag), axis=3)
trainX = np.concatenate((trainX_real, trainX_imag), axis=1)


#********************#
#   initialization   #
#********************#
num_samples = dataX.shape[0]
n = m - dataY.shape[1]   # number of noise vectors
r = angles.shape[1]   # resolution (i.e. angle grid size)

batch_size = 16


# extract features: real(Kx), imag(Kx), angle(Kx), (where Kx is covariance of X)
trainKx = np.zeros((num_samples, 3, m, m))
for i in range(num_samples):
    Kx = np.cov(dataX[i])

    trainKx[i, 0] = np.real(Kx)
    trainKx[i, 1] = np.imag(Kx)
    trainKx[i, 2] = np.angle(Kx)


#****************************#
#   build perfect spectrum   #
#****************************#
y = ((dataY + np.pi/2) / np.pi * r).astype(int)
trainY = np.zeros((num_samples, r))
for i in tqdm(range(num_samples)):
    # dirac impulses
    # trainY[i] = (10 ** (snr / 10)) * signal.unit_impulse(r, y[i]) + 1
    # trainY[i] = signal.unit_impulse(r, y[i])

    # # laplace distributions
    # for j in range(d):
    #    trainY[i] +=  100 * laplace.pdf(np.linspace(0, r, r), loc=y[i, j], scale=1)

    # classic MUSIC spectrum
    trainY[i] = classicMUSIC(trainX[i, :m] + 1j * trainX[i, m:], array, angles, d)[1]

# trainY = MinMaxScaler().fit_transform(trainY)


#*******************************************#
#   transform DoA angles to spectrum locs   #
#*******************************************#
# trainY = ((dataY + np.pi/2) / np.pi * r).astype(int).astype('float64')


#**************************#
#   create EVD as labels   #
#**************************#
trainEVD = np.zeros((num_samples, m, n)) + 1j * np.zeros((num_samples, m, n))
# for i in range(num_samples):
#     X = trainX[i, :m] + 1j * trainX[i, m:]
#     covariance = np.cov(X)
#     eigenvalues, eigenvectors = linalg.eig(covariance)
#
#     # the noise matrix
#     trainEVD[i] = eigenvectors[:, d:]
#     # trainEVD[i] = trainEVD[i] / np.linalg.norm(trainEVD[i])
#
# trainEVD_real = np.real(trainEVD) / np.linalg.norm(np.real(trainEVD))
# trainEVD_imag = np.imag(trainEVD) / np.linalg.norm(np.imag(trainEVD))
#
# trainEVD = np.concatenate((trainEVD_real, trainEVD_imag), axis=1)


ENTIRE = False   # set to true when testing with entire data

if ENTIRE:
    # take entire set for testing
    testX, testY, trainDoA, testDoA, testKx = trainX, trainY, trainEVD, dataY, trainKx
else:
    # split train set for testing
    trainX, testX, trainY, testY, trainEVD, testEVD, trainDoA, testDoA, trainKx, testKx = \
    train_test_split(trainX, trainY, trainEVD, dataY, trainKx, test_size=0.1)


if __name__ == "__main__":

    TRAIN = True   # set to true when training a model
    E2E = True   # set to true when evaluating an end2end model
    SPEC = False   # set to true when training with Cov and Spec

    LOSS = inversePeaks

    x, y = create_model_alternative()
    model = Model(x, y)

    if TRAIN:
        if E2E: trainY, testY, LOSS = trainDoA, testDoA, perm_rmse
        if SPEC: trainX, LOSS = trainKx, 'mse'

        model.summary()
        model.compile(loss=LOSS, optimizer=Adam(lr=0.001))
        checkpoint = ModelCheckpoint(save_best_only=True, filepath='model/deepAugMUSIC_d5.h5',
                                     save_weights_only=True, verbose=1)

        if SPEC:
            q = 24

            trainY, testY = np.array_split(trainY, q, axis=1), np.array_split(testY, q, axis=1)
            for i in range(q): trainY[i] = StandardScaler().fit_transform(trainY[i])
            for i in range(q): testY[i] = StandardScaler().fit_transform(testY[i])

            history = model.fit(x=trainX, y=trainY, batch_size=batch_size, epochs=70,
                                validation_split=0.2, callbacks=[checkpoint], verbose=1)

            results = model.evaluate(testKx, testY, batch_size=batch_size)
            print("TEST LOSS:", results)

        else:
            history = model.fit(x=trainX, y=trainY, batch_size=batch_size, epochs=1,
                                validation_split=0.2, callbacks=[checkpoint], verbose=1)

            results = model.evaluate(testX, testY, batch_size=batch_size)
            print("TEST LOSS:", results)

    else: model.load_weights("model/deepAugMUSIC_d5.h5")


    #*********************#
    #   evaluate models   #
    #*********************#
    num_samples = testX.shape[0]

    E2eDoAall = np.zeros((num_samples, d))
    DeepDoAall = np.zeros((num_samples, d))
    AugDoA = np.zeros((num_samples, d))
    ClasDoA = np.zeros((num_samples, d))
    BeamDoA = np.zeros((num_samples, d))
    RanDoA = np.random.uniform(- np.pi / 2, np.pi / 2, size=(num_samples, d))
    ZeroDoA = np.zeros((num_samples, d))
    for i in tqdm(range(num_samples)):

        # end2end #
        #*********#
        if E2E:
            X = np.repeat(testX[i][np.newaxis, :, :], 1, axis=0)
            E2eDoAall[i] = model.predict(X)

        # deepMUSIC #
        #***********#
        elif SPEC:
            X = np.repeat(testKx[i][np.newaxis, :, :], 1, axis=0)
            spectrum = np.concatenate(model.predict(X), axis=None)
            DoA, _ = signal.find_peaks(spectrum, distance=10)

            # only keep d largest peaks
            DoA = DoA[np.argsort(spectrum[DoA])[-d:]]

            # transform to radians
            DoA = DoA * np.pi / r - np.pi / 2

            # ensure exact number of DoA are compared
            if len(DoA) < d:
                # add zero for all non-present angles
                DoA = np.append(DoA, [np.random.uniform(- np.pi / 2, np.pi / 2)
                                      for _ in range(d - len(DoA))])

            DeepDoAall[i] = DoA


        # aug MUSIC #
        #***********#
        else:
            X = np.repeat(testX[i][np.newaxis, :, :], 1, axis=0)
            DoA, spectrum = augMUSIC(model.predict(X), array, angles, d)

            # transform to radians
            DoA = DoA * np.pi / r - np.pi / 2

            # ensure exact number of DoA are compared
            if len(DoA) < d:
                # add zero for all non-present angles
                DoA = np.append(DoA, [np.random.uniform(- np.pi / 2, np.pi / 2)
                                      for _ in range(d - len(DoA))])

            AugDoA[i] = DoA

        # classic MUSIC #
        #***************#
        X = testX[i, :m] + 1j * testX[i, m:]
        DoAMUSIC, spectrum = classicMUSIC(X, array, angles, d)

        # transform to radians
        DoAMUSIC = DoAMUSIC * np.pi / r - np.pi / 2

        # ensure exact number of DoA are compared
        if len(DoAMUSIC) < d:
            # add zero for all non-present angles
            DoAMUSIC = np.append(DoAMUSIC, [np.random.uniform(- np.pi / 2, np.pi / 2)
                                            for _ in range(d - len(DoAMUSIC))])

        ClasDoA[i] = DoAMUSIC

        # Beamformer #
        #************#
        DoABF, spectrum = beamformer(X, array, angles, d)

        # transform to radians
        DoABF = DoABF * np.pi / r - np.pi / 2

        # ensure exact number of DoA are compared
        if len(DoABF) < d:
            # add zero for all non-present angles
            DoABF = np.append(DoABF, [np.random.uniform(- np.pi / 2, np.pi / 2)
                                            for _ in range(d - len(DoABF))])

        BeamDoA[i] = DoABF

    if E2E: print("END2END TEST ERROR:", mean_min_perm_rmse(E2eDoAall, testDoA))
    elif SPEC: print("DEEP MUSIC TEST ERROR:", mean_min_perm_rmse(DeepDoAall, testDoA))
    else: print("AUG MUSIC TEST ERROR:", mean_min_perm_rmse(AugDoA, testDoA))
    print("CLASSIC MUSIC TEST ERROR:", mean_min_perm_rmse(ClasDoA, testDoA))
    print("BEAMFORMER TEST ERROR:", mean_min_perm_rmse(BeamDoA, testDoA))
    print("RANDOM TEST ERROR:", mean_min_perm_rmse(RanDoA, testDoA))
    print("ZERO TEST ERROR:", mean_min_perm_rmse(ZeroDoA, testDoA))






