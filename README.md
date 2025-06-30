# Transfer Learning-Based Classification with Hyperparameter Optimization

This project implements a production-grade dog/cat classifier using a strategic combination of transfer learning and hyperparameter optimization. The solution leverages MobileNetV2's pre-trained convolutional base as a feature extractor, initialized with ImageNet weights while keeping the bottom layers frozen during initial training. Through careful fine-tuning of the upper convolutional blocks combined with an optimized classification head, the model achieves exceptional discriminative performance.

The architecture employs Keras Tuner's Hyperband algorithm to automatically determine the optimal configuration of dense layers (depth 1-3, width 32-512 units), activation functions (ReLU/Swish), and regularization parameters (dropout 0-25%, L2 weight decay 1e-10 to 1e-3). Training incorporates early stopping monitored by validation AUC (patience=3), with the system automatically reverting to the best observed weights (epoch 16, 0.995 AUC) when terminating at epoch 19.

This implementation demonstrates an effective template for adapting pre-trained vision models to specialized binary classification tasks through:


