# The Idea of Recurrent Neural Network
A recurrent neural network (RNN) is a type of artificial neural network which uses sequential data or time series data. These deep learning algorithms are commonly used for ordinal or temporal problems, such as language translation, natural language processing (nlp), speech recognition, and image captioning; they are incorporated into popular applications such as Siri, voice search, and Google Translate. Like feedforward and convolutional neural networks (CNNs), recurrent neural networks utilize training data to learn. They are distinguished by their “memory” as they take information from prior inputs to influence the current input and output. While traditional deep neural networks assume that inputs and outputs are independent of each other, the output of recurrent neural networks depend on the prior elements within the sequence. While future events would also be helpful in determining the output of a given sequence, unidirectional recurrent neural networks cannot account for these events in their predictions.

![image](https://user-images.githubusercontent.com/86812576/171314293-b9d65e62-d600-4b16-80cf-b3a20cd2c695.png)

Another distinguishing characteristic of recurrent networks is that they share parameters across each layer of the network. While feedforward networks have different weights across each node, recurrent neural networks share the same weight parameter within each layer of the network. That said, these weights are still adjusted in the through the processes of backpropagation and gradient descent to facilitate reinforcement learning.

Recurrent neural networks leverage backpropagation through time (BPTT) algorithm to determine the gradients, which is slightly different from traditional backpropagation as it is specific to sequence data. The principles of BPTT are the same as traditional backpropagation, where the model trains itself by calculating errors from its output layer to its input layer. These calculations allow us to adjust and fit the parameters of the model appropriately. BPTT differs from the traditional approach in that BPTT sums errors at each time step whereas feedforward networks do not need to sum errors as they do not share parameters across each layer.

# RNN-with-PyTorch

# Import Packages
import common packages:

import **pandas as pd**

import **numpy as np**

import **matplotlib.pyplot as plt**

from **sklearn.mode_selection** import **train_test_split**

import PyTorch's common packages:

import **torch**

from **torch** import **nn, optim**

from **jcopdl.callback** import **Callback, set_config**

from **torchvision** import **datasets, transforms**

from **torch.utils.data** import **DataLoader**

**checking for GPU/CPU**

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import Data
The data used is the daily minimum temperature.
Because our data has not been split, we need to split it manually with scikit-learn. the index column is "Date", and don't forget to parse_dates because it's important to take the time in weeks, days, hours, or minutes.

# Added Features
Previously I've tried this model with a regular RNN with only one "Temp" feature, and the results weren't great.
If you look at the feature itself in the past it lacks meaning. So I tried to add more meaningful features. I will try to input the moon, because in certain months the temperature on earth is low. Extract "Date" to datetime and take it quarterly, dividing the year into 4 seasons. This means that it will add 4 new features so that the total becomes 5 features.

# Plot Data
![image](https://user-images.githubusercontent.com/86812576/171154314-ec8ca758-cb59-4766-a7b8-641c7c99fdd3.png)

We can see interesting things, at the beginning of each year the temperature looks warmer, while at the end of the year the temperature is colder.

# Dataset Splitting
![Screenshot 2022-05-31 175411](https://user-images.githubusercontent.com/86812576/171157781-6c115177-f73e-4a92-9c1b-af80b8c98702.png)

In time series data, there is a difference in splitting the data. The first thing is that time series data should not be shuffled, because if it is shuffled it will result in data leakage, as if to train future data. In fact, if we train the time series in the present, and test data for the future. Remember, test data only simulates what will happen (future data). So that the split does not use a shuffle, but a raw split.

![Screenshot 2022-05-31 175615](https://user-images.githubusercontent.com/86812576/171158100-00318cbb-cf05-4073-99ac-9f8ca60c5e12.png)

The way to do it in time series is not to use X and y, but what is split is the original data.

# Dataset & Dataloader
At this stage, how to convert the split data to NSF format? Why should I change to NSF? because RNN only accepts in the form of NSF. 

1 row, 2920 columns, but it has only one feature which is "Temp". So you have to be careful, the column is not a feature but 2920 sequences with 1 data. Whereas machine learning cannot learn with one data. So that in the RNN the data must be batched, because the time series data is only one.

The idea of batching is to divide the data sequence into several parts, for example, it is divided every 10 sequences. So that after batching, the form of the data is divided so that the data is not only one data.

![Screenshot 2022-05-31 194259](https://user-images.githubusercontent.com/86812576/171175971-4221e2f1-8d78-40b4-bd45-2b9cb6e1df6b.png)

I divide the data per 14 sequences with a batch size of 32. As a result, some data will be discarded to meet the number of 2920/14 = 208.57. So a total of 7 data were discarded in the train data, and 1 data was discarded in the test data.

# Architecture and Config

![Screenshot 2022-06-01 174708](https://user-images.githubusercontent.com/86812576/171387484-5336c31b-73a7-4068-a921-b7762758eb03.png)

The RNN architecture I'm using is the GRU (Gated Recurrent Unit) architecture. The parameters requested include "input_size", "hidden_layer", "num_layers", "dropout", and "batch_first"

### Config

Contains the parameters you want to keep when the model is reloaded. In this case I will save the "output_size", "input_size", "seq_len", "hidden_size", "num_layers", "dropout" which will be tuning later.

# Training Preparation 
### MCOC (Model, Criterion, Optimizer, Callback)
![mc](https://user-images.githubusercontent.com/86812576/171389937-76c8fddf-16e2-48dd-a54d-89d2316a1a96.png)

On the criterion using MSE Loss (Mean Square Error) for regression. But in RNN, the loss should be a number that you want to backprop, while in RNN the loss is more than one, because every output after the hidden size has a loss, then all losses must be combined. In this case I use reduction = "mean", meaning all losses will be averaged.

# Training and Result
![scr](https://user-images.githubusercontent.com/86812576/171390003-9cd34d34-6cb2-4d1b-81dc-c585f4a565e2.png)

You can see the results are not too overfit.

# Sanity Check
### Data for Pred
![image](https://user-images.githubusercontent.com/86812576/171196074-d9e098bc-1ce8-42b4-b15f-2b0a24180beb.png)

The results on test data that really never saw the future from 1989 to 1991.
Notice that in training it catches a trend of no more than 17 degrees, so we look at the test data if it's more than 17 degrees then the RNN seems to be stuck at 17, because if it's more than 17 then it's considered noise.

### Pred for Pred
If we want to be realistic when we meet data that has never been seen, and we use predictions for predictions, the reality will be like this.

![image](https://user-images.githubusercontent.com/86812576/171390579-0da0c062-606c-4af5-bf36-07709e3c5ee5.png)

The result looks bad, because of the domino effect. The longer the farther, the worse the prediction. So usually only one or two data is reliable in the future. Basically we can't predict the future, unless there are features that make sense. Therefore, our prediction looks bad because it does not find the pattern of the features.

