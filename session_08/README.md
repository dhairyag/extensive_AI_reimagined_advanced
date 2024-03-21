# Normalization 

This study focuses on different types of normalizations carried on convolutional neural networks (CNN). Following types of normalizations have been studied here:
 
- Network with Group Normalization
- Network with Layer Normalization
- Network with Batch Normalization

All the networks follow `C1 C2 c3 P1 C4 C5 C6 c7 P2 C8 C9 C10 GAP c11` pattern where `Ci` is convolution step with 3x3 kernel and `ci` is with 1x1 kernel. `Pi` indicates the max pooling and `GAP` as global average pooling.

# Group Normalization
### Overview
- Group normalization has been used where channels are divided into groups and normalization is carried out within each group separately.
- A group is just simply independent set of channels
- Total number of parameters should be less than 50000 and and minimum accuracy of 70% 

### Results:
- Total parameters: 40,248
- Best train accuracy: 74.83% at epoch 20
- Best test accuracy: 74.03% at epoch 19

#### Loss and Accuracy Plots
![loss_group](plots/loss_group_norm.png)

#### Misclassified Images in Test Data
![mis_img_group](plots/misclass_group.png)


### Analysis
- Accuracy of more than 70% can be easily achieved with fewer than 50000 parameters
- Dropout was required for this model to get rid of overfitting

