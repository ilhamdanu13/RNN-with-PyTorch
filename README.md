# The Idea of Recurrent Neural Network

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

# Plot dataset

![image](https://user-images.githubusercontent.com/86812576/171154314-ec8ca758-cb59-4766-a7b8-641c7c99fdd3.png)

We can see interesting things, at the beginning of each year the temperature looks warmer, while at the end of the year the temperature is colder.

# Dataset Splitting
![Screenshot 2022-05-31 175411](https://user-images.githubusercontent.com/86812576/171157781-6c115177-f73e-4a92-9c1b-af80b8c98702.png)

In time series data, there is a difference in splitting the data. The first thing is that time series data should not be shuffled, because if it is shuffled it will result in data leakage, as if to train future data. In fact, if we train the time series in the present, and test data for the future. Remember, test data only simulates what will happen (future data). So that the split does not use a shuffle, but a raw split.

![Screenshot 2022-05-31 175615](https://user-images.githubusercontent.com/86812576/171158100-00318cbb-cf05-4073-99ac-9f8ca60c5e12.png)

The way to do it in time series is not to use X and y, but what is split is the original data.

# Dataset & Dataloader
At this stage, how to convert the split data to NSF format? Why should I change to NSF? because RNN only accepts in the form of NSF. The interesting thing is that the data structure of the RNN is described (as shown below).

![Screenshot 2022-05-31 182338](https://user-images.githubusercontent.com/86812576/171162431-6c3b926a-4d1a-45f7-bd5c-14eb75fcfe10.png)


1 row, 2930 columns, but it has only one feature which is "Temp". So you have to be careful, the column is not a feature but 2930 sequences with 1 data. Whereas machine learning cannot learn with one data. So that in the RNN the data must be batched, because the time series data is only one.

The idea of batching is to divide the data sequence into several parts, for example, it is divided every 10 sequences. So that after batching, the form of the data is divided so that the data is not only one data.

![Screenshot 2022-05-31 194259](https://user-images.githubusercontent.com/86812576/171175971-4221e2f1-8d78-40b4-bd45-2b9cb6e1df6b.png)

I divide the data per 14 sequences with a batch size of 32. As a result, some data will be discarded to meet the number of 2930/14 = 208.57. So a total of 7 data were discarded in the train data, and 1 data was discarded in the test data.

# Architecture and Config

![Screenshot 2022-05-31 204111](https://user-images.githubusercontent.com/86812576/171187702-f6517c4f-1dcf-4046-9336-a6ce1fcb7390.png)

PyTorch has prepared an RNN architecture, namely the RNN Block. The parameters requested include "input_size", "hidden_layer", "num_layers", "dropout", and "batch_first"

### Config

Contains the parameters you want to keep when the model is reloaded. In this case I will save the "output_size", "input_size", "seq_len", "hidden_size", "num_layers", "dropout". 

# Training Preparation 
### MCOC (Model, Criterion, Optimizer, Callback)
![Screenshot 2022-05-31 204747](https://user-images.githubusercontent.com/86812576/171189043-4d45a63d-70bb-4b05-84f6-8c3c42fb04ce.png)

On the criterion using MSE Loss (Mean Square Error) for regression. But in RNN, the loss should be a number that you want to backprop, while in RNN the loss is more than one, because every output after the hidden size has a loss, then all losses must be combined. In this case I use reduction = "mean", meaning all losses will be averaged.

# Training and Result
![Screenshot 2022-05-31 210420](https://user-images.githubusercontent.com/86812576/171192589-97907b7d-a4f0-4551-a1a1-179b329c21b4.png)

You can see the results are not too overfit.

# Sanity Check
### Data for Pred
![image](https://user-images.githubusercontent.com/86812576/171196074-d9e098bc-1ce8-42b4-b15f-2b0a24180beb.png)

The results on test data that really never saw the future from 1989 to 1991.
Notice that in training it catches a trend of no more than 17 degrees, so we look at the test data if it's more than 17 degrees then the RNN seems to be stuck at 17, because if it's more than 17 then it's considered noise.

### Pred for Pred
If we want to be realistic when we meet data that has never been seen, and we use predictions for predictions, the reality will be like this.

![image](https://user-images.githubusercontent.com/86812576/171197762-2df535b9-f3c3-486e-a34f-68f816423161.png)

The result looks bad, because of the domino effect. The longer the farther, the worse the prediction. So usually only one or two data is reliable in the future. Basically we can't predict the future, unless there are features that make sense. Therefore, our prediction looks bad because it does not find the pattern of the features.

