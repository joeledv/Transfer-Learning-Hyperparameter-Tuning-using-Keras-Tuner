# Transfer Learning-Based Classification with Hyperparameter Optimization

This project implements a production-grade dog/cat classifier using a strategic combination of transfer learning and hyperparameter optimization. The solution leverages MobileNetV2's pre-trained convolutional base as a feature extractor, initialized with ImageNet weights while keeping the bottom layers frozen during initial training. Through careful fine-tuning of the upper convolutional blocks combined with an optimized classification head, the model achieves exceptional discriminative performance.

The architecture employs Keras Tuner's Hyperband algorithm to automatically determine the optimal configuration of dense layers, activation functions, and regularization parameters. 

This implementation demonstrates an effective template for adapting pre-trained vision models to specialized classification tasks.
