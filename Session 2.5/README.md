# Assignment 2.5

## Problem Statement

Write a neural network that can:
take 2 inputs:
    1. an image from the MNIST dataset (say 5), and
    2. a random number between 0 and 9, (say 7)
and gives two outputs:
    1. the "number" that was represented by the MNIST image (predict 5), and
    
    2. the "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)

## Data Generation

The MNIST dataset contains 70,000 images of handwritten digits from 0-9. There are a total of 60,000 images used for training and 10,000 images used for testing. 

We use the __getitem()__ which we use to return the MNIST image, its corresponding label, a random number and the sum of the random number with the corresponding label. 

To obtain the random numbers we use random.randint(). 

## Representation of Data

1. **MNIST DATA**: These are image matrices which we convert to tensors and apply normalisation technique based on mean and standard deviation.We use Totensor() to convert the data matrices to tensors. We have loaders to help us iterate through the datasets.

2. **Random Numbers**: We generate these using random.randint()

## Model

We define a class class Network(nn.Module). Then we use super().init() run the initialization for nn.Module. 

**nn.Conv2d**: Convolutes over the input image. 

**nn.MaxPool2d**: Does max pooling which helps in downsampling the detection of features in a feature map. 

**forward()**: The forward function computes output Tensors from input Tensors

**F.relu()**: We use the ReLU activation function because it is simple, easy to calculate, fast to compute and doesn't suffer from vanishing gradient function.

**x.view()** is similar to reshape in numpy. The view function is meant to reshape the tensor. 

**torch.cat((x,randomno),dim=1)**: This is used to concatenate the random number to the output from the x.view()

# Training Log and Loss

We take the loss of both MNIST predicted label and the loss from the addition of the MNIST label with the random number. Please view the logs below to infer the same. 

