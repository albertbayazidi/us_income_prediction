# US_income_prediction

## Overview
This is a notebook that uses neural network to predict a persons income (>50K or <=50K) based on 14 features (education, work sector, marital status, etc)

## Usage
This code uses census data from 1996 to predict income. The accuracy is 80.459% ± 5.809% with only 5 epoch worth of training.

## Functionality
This code was mainly built using the PyTorch framework. It has two hidden layers, first one with 8 neurons/nodes and second one with 4.  
The activation function used is leakyReLu and for the last layer the sigmoid function. The optimizer is stochastic gradient descent (SGD) and the loss function is binarycrossentropyloss (BCELoss).

## Source
Adult. (1996). UCI Machine Learning Repository.
