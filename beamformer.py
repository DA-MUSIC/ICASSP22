####################################################################################################
#                                           beamformer.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 22/04/21                                                                                #
#                                                                                                  #
# Purpose: Implementation of the purely model-based classical Beamformer algorithm.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import warnings

from scipy import linalg

from syntheticEx import *

# shut up casting warnings
warnings.simplefilter("ignore")


#******************************#
#   the Beamformer algorithm   #
#******************************#
def beamformer(incident, array, continuum, sources=2):
    """
        The classical Beamformer algorithm calculates a spatial spectrum, which is used to
        estimate the directions of arrival of the incident signals by finding its d peaks.

        @param incident -- The measured waveforms (= incident signals and noise).
        @param array -- Holds the positions of the array elements.
        @param continuum -- The continuum of all possible mode vectors
        @param sources -- The number of signal sources.

        @returns -- The d locations of the spatial spectrum peaks.
    """
    # calculate EVD of covariance matrix
    covariance = np.cov(incident)

    # calculate spatial spectrum
    numSamples = continuum.shape[1]
    spectrum = np.zeros(numSamples)
    for axis in continuum:
        for i in range(numSamples):
            # establish array steering vector
            a = ULA_action_vector(array, axis[i])
            spectrum[i] = (a.conj().transpose() @ covariance @ a) / linalg.norm(a)**2

    DoAsMUSIC, _ = signal.find_peaks(spectrum)

    # only keep d largest peaks
    DoAsMUSIC = DoAsMUSIC[np.argsort(spectrum[DoAsMUSIC])[-sources:]]

    return DoAsMUSIC, spectrum


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