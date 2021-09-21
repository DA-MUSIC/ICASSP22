# DEEP AUGMENTED MUSIC ALGORITHM FOR DATA-DRIVEN DOA ESTIMATION

## Abstract

Direction of arrival (DoA) estimation is a crucial task in sensor array signal processing, giving rise to various successful model-based (MB) algorithms as well as recently developed data-driven (DD) methods. This paper introduces a new hybrid MB/DD DoA estimation architecture, based on the classical multiple signal classification (MUSIC) algorithm. Our approach augments crucial aspects of the original MUSIC structure with specifically designed neural architectures, allowing it to overcome certain limitations of the purely MB method, such as its inability to successfully localize coherent sources. The deep augmented MUSIC algorithm is shown to outperform its unaltered version with a superior resolution.


## Overview

This repository consists of following Python scripts:
* The `augMUSIC.py` implements the augmented MUSIC algorithm.
* The `beamformer.py` implements the classic beamforming algorithm.
* The `classicMUSIC.py` implements the purely model-based MUSIC algorithm.
* The `errorMeasures.py` defines error measures used to evaluate the DoA estimation algorithms.
* The `losses.py` script  defines custom losses used to train neural augmentations for the MUSIC algorithm.
* The `models.py` defines neural augmentation architectures for the MUSIC algorithm.
* The `plotFigures.py` provides visualization of the performances of different DoA algortihms.
* The `regularizers.py` script  defines custom regularizers for the neural augmentations.
* The `syntheticEx.py` script implements synthetic examples for DoA and combines them to a datase.
* The `trainModel.py` implements the training of the neural augmentation.
* The `utils.py` defines some helpful functions.
