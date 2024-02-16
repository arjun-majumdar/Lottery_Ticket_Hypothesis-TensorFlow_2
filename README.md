# The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks - Using TensorFlow 2

A GitHub repository implementing __The Lottery Ticket Hypothesis__ paper by _Jonathan Frankle & Michael Carbin_

"lottery ticket hypothesis:" dense, randomly-initialized, feed-forward and/or convolutional networks contain subnetworks ("winning tickets") that - when trained in isolation - reach test accuracy comparable to the original network in a similar number of iterations. The winning tickets we find have won the initialization lottery: their connections have initial weights that make training particularly effective. 

The paper can be downloaded from:
[The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)


# Comparing Rewinding and Fine-tuning in Neural Network Pruning - using PyTorch 2.X

Implementation for the paper __Comparing Rewinding and Fine-tuning in Neural Network Pruning__ by Alex Renda et al.


## LTH Codes:
1. MNIST dataset using 300-100-10 Dense Fully connected neural network winning ticket identification.
1. MNIST dataset using LeNet-5 Convolutional Neural Networks.
1. Validation of the winning ticket identified for MNIST and CIFAR-10 dataset using relevant neural networks.
1. Conv-2/4/6 Convolutional Neural Network (CNN) for CIFAR10 dataset; pruning network till 0.5% of original connections remain and observe training and testing accuracies and losses.
1. Pruning Algorithm implementation: numpy based unstructured, layer-wise, absolute magnitude pruning and _tensorflow_model_optimization_ toolkit based pruning (not the focus of most codes)




### Prerequisites for the code to run:
- Python 3.X
- numpy 1.17 and/or above
- TensorFlow 2.0
- PyTorch 2.X
- [tensorflow_model_optimization](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras) (not focused on)
