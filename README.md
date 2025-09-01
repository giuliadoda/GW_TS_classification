Final project of *Neural Networks & Deep Learning* course.

Based on [*Neural network time-series classifiers for gravitational-wave searches in single-detector periods* by Trovato *et al.*, 2023](https://arxiv.org/abs/2307.09268).

The complete dataset can be found [here](https://zenodo.org/records/11093596).

The aim of this project is to implement various time series classifiers and analyze their performances in order to establish the best model for signal - noise classification.

Three neural networks model have been implemented:
- Convolutional Neural Network,
- [Temporal Convolutional Neural Network](https://arxiv.org/abs/1803.01271),
- [Inception Time Neural Network](https://link.springer.com/article/10.1007/s10618-020-00710-y). 
  
They have been used to perform time series classification between data segments containing gravitational waves signal or noise. 