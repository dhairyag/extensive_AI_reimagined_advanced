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

## [Session 11: CIFAR10 Image Classification with ResNet18](./session_11)
Objective: Implement CIFAR10 image classification by importing ResNet18 model from another [repo](https://github.com/dhairyag/main_ml_models_utils/tree/main).

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

## [Session 13: Pytorch-Lightning on CIFAR10](./session_13)
The code demonstrates image classification on CIFAR10 dataset using [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)

#### Technology:
- PyTorch Lightning

#### Methodology:
- Interactive data visualization and model training
- Utilizes progress bars and other UI elements for real-time feedback
- Modular approach with extensive use of widget models and layouts

#### Focus:
- Creating an interactive and user-friendly interface for machine learning tasks
- Emphasis on real-time progress tracking and data presentation
- Involves training, evaluating models on image datasets

The code tries to create a rich, interactive environment for machine learning experiments or demonstrations.

## [Session 19: Building GPT: Text generator](./session_19)
Implement a small-scale GPT (Generative Pre-trained Transformer) model for text generation, trained on Shakespeare's works.

#### Technology:
- PyTorch

#### Methodology:
- Character-level tokenization
- Transformer architecture with self-attention mechanisms
- Training on a corpus of Shakespeare's text

#### Key Components:
- Data preprocessing and tokenization
- Custom PyTorch modules for the transformer architecture
- Training loop with gradient descent optimization
- Text generation functionality

#### Focus:
The project emphasizes understanding and implementing core concepts of transformer-based language models on a smaller scale. It serves as an educational tool for learning about GPT-like architectures and their application to text generation tasks.

