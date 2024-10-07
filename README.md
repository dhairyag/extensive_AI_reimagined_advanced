# Extensive AI Reimagined Advanced

This repository contains a collection of advanced AI and Machine Learning projects, showcasing various techniques and applications. Below is a summary of key folders and their contents:

## [Session 07: MNIST Digit Classification](./session_07)

Iterative optimization of a CNN model for MNIST digit classification

#### Technologies:
- Framework: PyTorch
- Key libraries: torchvision, matplotlib

#### Methodologies:
- Convolutional Neural Networks
- Progressive model refinement
- Regularization techniques (Batch Normalization, Dropout)
- Data augmentation
- Learning rate optimization
- Architecture modifications (Global Average Pooling)

Focus: Achieving high accuracy (>98\%) with a compact model (<8K parameters) through systematic iterations and performance analysis.

## [Session 08: Image Classification](./session_08)
To study and compare different normalization techniques (Group Normalization, Layer Normalization, Batch Normalization) applied to convolutional neural networks (CNNs) for image classification on the CIFAR-10 dataset.

#### Technologies:
- Framework: PyTorch
- Key libs: torchvision, matplotlib, numpy

#### Methodology:
- Implemented custom CNN architectures with various normalization layers
- Trained models using SGD optimizer and CrossEntropyLoss
- Evaluated performance using accuracy metrics and loss curves
- Visualized misclassified images for error analysis

#### Focus:
- Comparing the effectiveness of different normalization techniques
- Optimizing model architectures to achieve >70\% accuracy with <50,000 parameters
- Analyzing the impact of normalization on model convergence and generalization
- The project demonstrates a systematic approach to evaluating normalization techniques in deep learning, with a focus on practical implementation and performance analysis.

## [Session 11: ](./session_11)
Objective: Implement CIFAR10 image classification using a ResNet18 model from another [repo](https://github.com/dhairyag/main_ml_models_utils/tree/main).

Key Components:
- Model: ResNet18 architecture
- Dataset: CIFAR10
- Training: Custom training loop with 20 epochs
- Evaluation: Test set accuracy and loss tracking
- Visualization: Loss plots and misclassified images display
- Model Persistence: Saving and loading capabilities

#### Tech:
- PyTorch

#### Methodology:
- Utilizes a pre-defined ResNet18 architecture
- Trains the model on CIFAR10 data
- Implements custom training and evaluation loops

The project demonstrates a complete machine learning workflow for image classification, from model training to result visualization and model persistence.

