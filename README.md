# DEEP AUGMENTED MUSIC ALGORITHM FOR DATA-DRIVEN DOA ESTIMATION

[Deep Augmented MUSIC Algorithm for Data-Driven DoA Estimation](https://arxiv.org/abs/2109.10581)

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


## Requirements

| Module | Version |
| :--- | :---: |
| scipy  | 1.6.2  |
| h5py  | 2.10.0 |
| pandas  | 0.25.1 |
| matplotlib  | 3.1.1 |
| keras | 2.3.1 |
| numpy  | 1.19.3 |
| tensorflow  | 2.4.1 |
| tqdm  | 4.36.1 |
| scikit_learn | 0.24.2 |
