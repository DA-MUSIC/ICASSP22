####################################################################################################
#                                          plotFigures.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/03/21                                                                                #
#                                                                                                  #
# Purpose: Plot synthetic examples to test correctness and performance of algorithms estimating    #
#          directions of arrival (DoA) of multiple signals.                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from tensorflow.keras.models import Model

from augMUSIC import augMUSIC
from beamformer import beamformer
from classicMUSIC import *
from errorMeasures import mean_min_perm_rmse
from models import *
from syntheticEx import *


#********************#
#   initialization   #
#********************#
np. set_printoptions(threshold=10)
np.set_printoptions(suppress=True)

# doa = [-0.1, 0.1]

x, s = construct_signal(doa)
# x, s = construct_coherent_signal(doa)


#*************************#
#   aug MUSIC algorithm   #
#*************************#
inX, outY = deepMUSICorig()
model = Model(inX, outY)
model.load_weights("model/deep_d5.h5")

# crate measurement for the augmentation
x_real = np.real(x)
x_imag = np.imag(x)
X = np.concatenate((x_real, x_imag), axis=0)
X = np.repeat(X[np.newaxis, :, :], 1, axis=0)

K = np.zeros((3, m, m))
Kx = np.cov(x)
K[0] = np.real(Kx)
K[1] = np.imag(Kx)
K[2] = np.angle(Kx)



# true if the augmentation outputs DoA and not subspace
DoAOut = False
spec = True

if DoAOut:
    inp = model.input
    outputs = [layer.output for layer in model.layers]
    # select output layer forming the spectrum and final layer (with DoA)
    functor = [K.function([inp], [outputs[-5]]), K.function([inp], [outputs[-1]])]

    pred_spec , DoA = [f([X]) for f in functor]
    pred_spec = pred_spec[0][0]

    # transform to spectrum indices
    DoA = ((np.array(DoA[0][0]) + np.pi / 2) / np.pi * r).astype(int)

elif spec:
    X = np.repeat(K[np.newaxis, :, :], 1, axis=0)
    pred_specs = model.predict(X)
    pred_spec = np.concatenate(pred_specs, axis=None)
    # DoA = []
    # for i, subspec in enumerate(pred_specs):
    #     angle, _ = signal.find_peaks(subspec[0], distance= r//36)
    #     DoA.append(angle + i * r//36)
    #
    # DoA = np.concatenate(DoA, axis=None)

    DoA, _ = signal.find_peaks(pred_spec, distance=10)

    # only keep d largest peaks
    DoA = DoA[np.argsort(pred_spec[DoA])[-d:]]

else:
    DoA, pred_spec = augMUSIC(model.predict(X), array, angles, d)


#*****************************#
#   classic MUSIC algorithm   #
#*****************************#
DoAMUSIC, spectrum = classicMUSIC(x, array, angles, d)
# DoA, pred_spec = testMUSIC(x, array, angles, d)


#****************#
#   beamformer   #
#****************#
DoABF, spectrumBF = beamformer(x, array, angles, d)


#***********************#
#   plot BF vs. MUSIC   #
#***********************#
def plotBFvMUSIC():
    plt.figure(figsize=(5, 4))
    plt.tight_layout()
    plt.plot(angles[0], spectrumBF, 'k--')
    # plt.plot(angles[0], spectrum)
    plt.plot(doa, np.mean(s, axis=1),'g*')
    plt.plot(angles[0, DoABF], spectrumBF[DoABF], color='grey', linestyle='', marker='x')
    # plt.plot(angles[0, DoAMUSIC], spectrum[DoAMUSIC], 'x')
    plt.xlabel('Azimuth angle (rad)')
    plt.ylabel('Spatial spectrum')
    plt.legend(['Beamformer', 'Actual DoA'], loc='center')


#************************#
#   plot classic MUSIC   #
#************************#
def plotMUSIC():
    plt.figure(figsize=(8, 4))
    plt.tight_layout()
    plt.plot(angles[0], spectrum)
    plt.plot(angles[0, DoAMUSIC], spectrum[DoAMUSIC], 'x')
    plt.plot(doa, np.mean(s, axis=1), '*')
    plt.xlabel('Azimuth angle (rad)')
    plt.ylabel('Spatial spectrum')
    plt.legend(['Classic MUSIC', 'Estimated DoA', 'Actual DoA'])


#************************#
#   plot est. spectrum   #
#************************#
def plotAugMUSIC():
    plt.figure(figsize=(8, 4))
    plt.plot(angles[0], pred_spec)
    plt.plot(angles[0, DoA], pred_spec[DoA],'x')
    plt.plot(doa, [0 for i in range(d)],'*')
    plt.xlabel('Azimuth angle (rad)')
    plt.ylabel('Spatial spectrum')
    plt.legend(['Aug MUSIC', 'Estimated DoA', 'Actual DoA'])


plt.style.use(['grid', 'science', 'no-latex'])

# plotBFvMUSIC()
plotMUSIC()
plotAugMUSIC()

plt.show()