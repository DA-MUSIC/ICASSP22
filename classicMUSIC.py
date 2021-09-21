####################################################################################################
#                                          classicMUSIC.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 06/03/21                                                                                #
#                                                                                                  #
# Purpose: Implementation of the purely model-based MUSIC algorithm.                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import warnings

from scipy import linalg
from scipy import signal

# shut up casting warnings
warnings.simplefilter("ignore")


#*********************************#
#   the classic MUSIC algorithm   #
#*********************************#
def classicMUSIC(incident, array, continuum, sources=None):
    """
        The classic MUSIC algorithm calculates the spatial spectrum, which is used to estimate
        the directions of arrival of the incident signals by finding its d peaks.

        @param incident -- The measured waveforms (= incident signals and noise).
        @param array -- Holds the positions of the array elements.
        @param continuum -- The continuum of all possible mode vectors
        @param sources -- The number of signal sources (optional).

        @returns -- The d locations of the spatial spectrum peaks.
    """
    # calculate EVD of covariance matrix
    covariance = np.cov(incident)
    eigenvalues, eigenvectors = linalg.eig(covariance)

    if sources:   # number of sources known
        d = sources
    else:
        n = cluster(eigenvalues).shape[0]   # estimate multiplicity of smallest eigenvalue...
        d = array.shape[0] - n   # and get number of signal sources

    # the noise matrix
    En = eigenvectors[:, d:]

    # calculate spatial spectrum
    numSamples = continuum.shape[1]
    spectrum = np.zeros(numSamples)
    for axis in continuum:
        for i in range(numSamples):
            # establish array steering vector
            a = ULA_action_vector(array, axis[i])
            spectrum[i] = 1./(a.conj().transpose() @ En @ En.conj().transpose() @ a)

    DoA, _ = signal.find_peaks(spectrum)

    # only keep d largest peaks
    DoA = DoA[np.argsort(spectrum[DoA])[-d:]]

    return DoA, spectrum


#*******************************#
#   cluster small eigenvalues   #
#*******************************#
def cluster(evs):
    """
        Estimates multiplicity of smallest eigenvalue.

        @param evs -- The eigenvalues in descending order.

        @returns -- The eigenvalues similar or equal to the smallest eigenvalue.
    """
    # simplest clustering method: with threshold
    threshold = 0.4
    return evs[np.where(abs(evs) < abs(evs[-1]) + threshold)]


#*******************************************#
#   uniform linear array steering vector    #
#*******************************************#
def ULA_action_vector(array, theta):
    """
        Establish the possible mode vectors (steering vectors) given the
        positions of a uniform linear array.

        @param array -- Holds the positions of the array elements.
        @param theta -- The value of the given axis to be evaluated.

        @returns -- The action vector.
    """
    return np.exp(- 1j * np.pi * array * np.sin(theta))