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
