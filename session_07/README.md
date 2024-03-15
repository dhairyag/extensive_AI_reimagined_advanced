# Iterations for MNIST classification

# Iterations_01
#### Target
- Use modular code to generate first set of results
- Establish the code flow
- Use old model to begin with
#### Results:
- Total parameters: 592,660
- Best train accuracy: 99.37% at epoch 15
- Best test accuracy: 99.00% at epoch 10
#### Analysis
- As expected, the model has huge number of parameters
- The model is overfitted
- Next step should have very few parameters and proper conv blocks

# Iteration_02
#### Target
- Make model very lighter with less than 8k parameters
- Use batch normalization
- Use fully connected layer
- Use max pooling at appropriate places
#### Results
- Total parameters: 6,754
- Best train accuracy: 99.68% at epoch 15
- Best test accuracy: 99.06% at epoch 11
#### Analysis
- Decent accuracy with very few parameters
- The model is still overfitted
- Regularization can be useful to reduce overfitting
- Image processing with random nature can be useful to increase test accuracy

# Iteration_03
#### Target
- Use regularization with dropout (drop value as 0.15)
- Make model very lighter with less than 8k parameters
- Use batch normalization
- Use fully connected layer
- Use max pooling at appropriate places
#### Results
- Total parameters: 6,754
- Best train accuracy: 98.54% at epoch 15
- Best test accuracy: 98.97% at epoch 13
#### Analysis
- Regularization has removed overfitting by dropping 15% kernel values every step
- Image processing with random rotation/crop can be useful to increase test accuracy

# Iteration_04
#### Target
- Use GAP and 1x1 kernel for convolution at output and remove fully connected layer at output and
#### Results
- Total parameters: 5,464
- Best train accuracy: 98.17% at epoch 15
- Best test accuracy: 98.66% at epoch 13
#### Analysis
- Accuracy has decreased after adding GAP and 1x1 convolution at output layer
- Image processing with random rotation/crop can be useful to increase test accuracy

# Iteration_05
#### Target
- Image augmentation with random rotation by max 15 degrees
- Cropping images and resizing them back to original size
#### Results
- Total parameters: 5,464
- Best train accuracy: 97.56% at epoch 15
- Best test accuracy: 98.45% at epoch 15
#### Analysis
- Accuracy has decreased more for training dataset than test dataset as expected as images came with various modifications randomly
- Test accuracy is almost in same range as before
- Next, learning rate can be played with to check it's effect on accuracy


