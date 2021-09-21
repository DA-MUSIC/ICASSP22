####################################################################################################
#                                             utils.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/04/21                                                                                #
#                                                                                                  #
# Purpose: Definitions of helpful functions.                                                       #
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


#***********************************************#
#   eliminate randomness to reproduce results   #
#***********************************************#
np.random.seed(42)
tf.random.set_seed(42)


#**********************************#
#   calculate the MUSIC spectrum   #
#**********************************#
def calculate_spectrum(y_pred):
    """
        Calculates the MUSIC spectrum according to P = 1 / (a^H En En^H a).

        @param y_pred -- The estimated noise space vectors.

        @returns -- The estimated spectrum.
    """
    batch_size = y_pred.shape[0]
    num_samples = angles.shape[1]

    EnReal = Lambda(lambda y: y[:, :m])(y_pred)
    EnImag = Lambda(lambda y: y[:, m:])(y_pred)

    En = tf.cast(tf.dtypes.complex(EnReal, EnImag), dtype=tf.complex64)
    # calculate spatial spectrum
    spectrum = []
    for axis in angles:
        for i in range(num_samples):
            # establish array steering vector
            a = tf.cast(ULA_action_vector(array, axis[i]), dtype=tf.complex64)
            # a = np.repeat(a[np.newaxis, :], batch_size, axis=0)

            H = tf.linalg.matvec((En @ tf.transpose(En, conjugate=True, perm=[0, 2, 1])), a)

            H = tf.reduce_sum(tf.math.multiply(tf.math.conj(a), H), 1)
            spectrum.append(1. / H)

    return tf.transpose(tf.convert_to_tensor(spectrum, dtype=float))


#****************************#
#   calculate permutations   #
#****************************#
def permutations(predDoA):
    """
        Calculates all permutations of the given list.

        @param predDoA -- The estimated DoA angles to be permuted.

        @returns -- All permutations of the estimated DoA.
    """
    if len(predDoA) == 0:
        return []
    if len(predDoA) == 1:
        return [predDoA]

    perms = []
    for i in range(len(predDoA)):
       remaining = predDoA[:i] + predDoA[i + 1:]

       for perm in permutations(remaining):
           perms.append([predDoA[i]] + perm)

    return perms