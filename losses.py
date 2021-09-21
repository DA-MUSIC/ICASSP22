####################################################################################################
#                                            losses.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 25/03/21                                                                                #
#                                                                                                  #
# Purpose: Definitions of custom losses used to train a neural augmentation of the MUSIC           #
#          algorithm.                                                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import Regularizer

from scipy import signal
from scipy.stats import laplace

from syntheticEx import *
from utils import *


#***********************************************#
#   eliminate randomness to reproduce results   #
#***********************************************#
np.random.seed(42)
tf.random.set_seed(42)


#************************#
#   MUSIC spectrum mse   #
#************************#
def mseSpectrum(y_true, y_pred):
    """
        Defines a custom loss to estimate performance of subspace estimator.

        @param y_true -- The "perfect" spectrum.
        @param y_pred -- The estimated noise space vectors.

        @returns -- The mse of the estimated spectrum and the spectrum given by y_true.
    """
    y = calculate_spectrum(y_pred)

    y = LayerNormalization()(y)
    y_true = LayerNormalization()(y_true)

    return K.mean(K.square(y - y_true), axis=-1)


#****************************************#
#   inverse spectrum at location peaks   #
#****************************************#
def inversePeaks(y_true, y_pred):
    """
        Defines a custom loss to estimate performance of subspace estimator.

        @param y_true -- The true DoA of the sources.
        @param y_pred -- The estimated noise space vectors.

        @returns -- The inverse of the sum of the estimated spectrum at the true DoA
                    (thereby favouring large values at these locations).
    """
    y = calculate_spectrum(y_pred)

    # normalize spectrum to [-1, 1] and shift +1 up to assure non-negative values
    # ! error occurred ! -> possibly float rounding error -> shift up > 1 to be sure
    shift_up = 2
    y = LayerNormalization()(y) + shift_up

    peaks = tf.gather(y, indices=tf.cast(y_true, dtype='int32'), axis=1, batch_dims=1)

    # sum up all inverse peaks and normalize
    inverse_peaks =  shift_up * tf.reduce_sum(1. / peaks, axis=-1) / y_true.shape[1]

    return inverse_peaks # + tf.reduce_sum(y, axis=-1) / num_samples


#***************************************************#
#   difference between peaks and rest of spectrum   #
#***************************************************#
def PeakSpektrumDiff(y_true, y_pred):
    """
        Defines a custom loss to estimate performance of subspace estimator.
        Caution! - This loss becomes negative instead of smaller!

        @param y_true -- The true DoA of the sources.
        @param y_pred -- The estimated noise space vectors.

        @returns -- The difference between the peaks and the mean of the rest
                    of the spectrum.
    """
    batch_size = y_pred.shape[0]
    num_samples = angles.shape[1]

    y = calculate_spectrum(y_pred)

    # normalize spectrum to [-1, 1] and shift +1 up to assure non-negative values
    # ! error occurred ! -> possibly float rounding error -> shift up > 1 to be sure
    shift_up = 2
    y = LayerNormalization()(y) + shift_up

    peaks = tf.gather(y, indices=tf.cast(y_true, dtype='int32'), axis=1, batch_dims=1)

    # create spectrum without peaks
    for a in range(y_true.shape[1]):
        y_remove = tf.sparse.SparseTensor(indices=[[j, y_true[j, a]] for j in range(batch_size)],
                                          values=peaks[:, a],
                                          dense_shape=[batch_size, num_samples])
        y = y - tf.sparse.to_dense(y_remove, default_value=0.)

    # return difference between the mean of the rest of the spectrum and the peaks
    return tf.reduce_sum(tf.math.subtract(tf.reduce_mean(y), peaks), axis=-1)


#***********************************************#
#   mse of the estimated DoA and the true DoA   #
#***********************************************#
def mseDoA(y_true, y_pred):
    """
        Defines a custom loss to estimate performance of subspace estimator.

        @param y_true -- The true DoA of the sources.
        @param y_pred -- The estimated noise space vectors.

        @returns -- The (angular) mse of the estimated DoA and the true DoA.
    """
    batch_size = y_pred.shape[0]
    num_samples = angles.shape[1]
    d = y_true.shape[1]

    y = calculate_spectrum(y_pred)
    y = LayerNormalization()(y) + 1   # normalize to [0, 2]

    beta = 1e3
    y_range = tf.range(y.shape.as_list()[-1], dtype=y.dtype)
    DoAs = []
    for _ in range(d):

        # soft-argmax
        DoA = tf.reduce_sum(tf.nn.softmax(y * beta) * y_range, axis=-1)

        DDoA = tf.expand_dims(tf.dtypes.cast(DoA, tf.int64), axis=1)
        DDoA = tf.dtypes.cast(DDoA, float)

        DoAs.append(DoA)

        if d > 1:
            # get distances to true DoA
            dist = tf.sort(abs(((y_true - DDoA) + num_samples/2) % num_samples - num_samples/2))

            # build range around estimated DoA using second smallest distance
            indices = []
            for b in range(batch_size):
                for idx in range(DDoA[b] - dist[b, 1]//2, DDoA[b] + dist[b, 1]//2 + 1):
                    indices.append([b, idx % num_samples])

            # remove maximal value and range around it
            values = tf.gather_nd(y, indices=tf.convert_to_tensor(indices, dtype='int32'))
            y_remove = tf.sparse.SparseTensor(indices=indices, values=values,
                                              dense_shape=[batch_size, num_samples])
            y = y - tf.sparse.to_dense(tf.sparse.reorder(y_remove), default_value =0.)

    DoAs = tf.transpose(tf.convert_to_tensor(DoAs, dtype=float))

    # account for angular overflow (pi / 2 = - pi /2)
    diff = ((tf.sort(DoAs) - tf.sort(y_true)) + num_samples/2) % num_samples - num_samples/2

    return K.mean(diff ** 2, axis=-1) / ((num_samples/4) ** 2)


#***************#
#  mse of evd   #
#***************#
def evd(y_true, y_pred):
    """
        Defines a custom loss to estimate performance of subspace estimator.

        @param y_true -- The evd of the measurements.
        @param y_pred -- The estimated noise space vectors.

        @returns -- The mse of the estimated evd and the true evd given by y_true.
    """

    return K.mean(K.mean(tf.math.subtract(y_pred, y_true) ** 2, axis=-1), axis=-1)


#******************#
#   angular rmse   #
#******************#
def angular_rmse(y_true, y_pred):
    """
        Defines a custom loss for a DoA estimator.

        @param y_true -- The evd of the measurements.
        @param y_pred -- The estimated noise space vectors.

        @returns -- The rmse with respect to the angles.
    """
    # angular difference
    diff = tf.math.floormod((tf.subtract(tf.sort(y_pred), tf.sort(y_true)) + np.pi / 2), np.pi) - np.pi / 2
    return tf.reduce_mean((diff ** 2), axis=-1) ** (1 / 2)


#***********************************#
#   mean minimal permutation rmse   #
#***********************************#
def perm_rmse(predDoA, trueDoA):
    # differentiable version of errorMeasures.py/mean_min_perm_rmse
    """
            Calculates the mean of all the samples of the minimal rmse of
            (all permutations of) the predicted DoA and the true DoA.

            @param predDoA -- The estimated DoA angles in radians (same length as true DoA).
            @param trueDoA -- The ground truth DoA angles in radians.

            @returns -- The mean of minimal rmse.
    """
    num_samples = trueDoA.shape[0]
    num_sources = trueDoA.shape[1]

    # get permutations of estimated DoA
    allPerms = np.zeros((num_samples, np.math.factorial(num_sources), num_sources))
    for i in range(num_samples):
        allPerms[i] = permutations(list(predDoA[i]))

    # angular difference
    diff = tf.math.floormod(((Subtract()([allPerms, trueDoA])) + np.pi / 2), np.pi) - np.pi / 2

    # rmse
    diff = tf.reduce_mean((diff ** 2), axis=-1) ** (1 / 2)

    # return minimal rmse as error
    return tf.math.reduce_min(diff, axis=-1)