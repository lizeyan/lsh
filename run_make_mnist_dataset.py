#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import MNIST

from config import n_dims, n_test_samples, n_train_samples

# In[2]:
dataset_name = 'MNIST'
output_path = 'outputs'

# In[3]:


dataset_class = eval(dataset_name)
transform = Compose([ToTensor()])
original_train_data = dataset_class(
    f"{output_path}/{dataset_name}_data", download=True, transform=transform, train=True
)
train_dataset_length = len(original_train_data)

# In[4]:


loader = DataLoader(original_train_data, batch_size=4096 * 16, num_workers=2, shuffle=False)
mean = np.zeros((28, 28), dtype=np.float32)
for images, _ in loader:
    images = np.squeeze(images.numpy())
    mean += np.sum(images, axis=0, keepdims=False)
mean /= train_dataset_length
total_mean = np.mean(mean)
logger.debug(f"train dataset mean: {total_mean}")
var = np.zeros((28, 28))
original_train_dataset_arr = []
for images, _ in loader:
    images = np.squeeze(images.numpy())
    var += np.sum((images - mean) ** 2, axis=0)
    original_train_dataset_arr.append(images)
original_train_dataset_arr = np.concatenate(original_train_dataset_arr)
var /= train_dataset_length
total_std = np.sqrt(np.mean(var))
logger.debug(f"train dataset standard deviation: {total_std}")

# In[5]:


selected_positions = np.unravel_index(np.argsort(var, axis=None)[-1:-n_dims - 1:-1], np.shape(var))
logger.debug(f"selected positions: {selected_positions}")
del var

# In[6]:


train_arr = (original_train_dataset_arr[:, selected_positions[0], selected_positions[1]] - total_mean) / total_std
train_arr = train_arr[:n_train_samples]
train_arr = train_arr.astype(np.float32)
logger.debug(f"selected train dataset shape: {np.shape(train_arr)}")
fp = np.memmap(f"{output_path}/train_arr", dtype=np.float32, mode='w+', shape=np.shape(train_arr))
fp[:] = train_arr[:]
del fp
del original_train_dataset_arr

# In[7]:


original_test_data = dataset_class(
    f"{output_path}/{dataset_name}_data", download=True, transform=transform, train=False
)
loader = DataLoader(original_test_data, batch_size=4096 * 16, num_workers=2, shuffle=False)
original_test_dataset_arr = []
for images, _ in loader:
    images = np.squeeze(images.numpy())
    original_test_dataset_arr.append(images)
original_test_dataset_arr = np.concatenate(original_test_dataset_arr)

test_arr = (original_test_dataset_arr[:, selected_positions[0], selected_positions[1]] - total_mean) / total_std
test_arr = test_arr[:n_test_samples]
test_arr = test_arr.astype(np.float32)
logger.debug(f"selected test dataset shape: {np.shape(test_arr)}")
fp = np.memmap(f"{output_path}/test_arr", dtype=np.float32, mode='w+', shape=np.shape(test_arr))
fp[:] = test_arr[:]
del fp
del original_test_dataset_arr

# In[8]:


distances = np.sum(np.square(np.expand_dims(test_arr, 1) - np.expand_dims(train_arr, 0)), axis=-1)
ground_truth = np.argsort(distances, axis=-1)
ground_truth = ground_truth.astype(np.int32)
logger.debug(f'ground truth shape: {np.shape(ground_truth)}')
fp = np.memmap(f"{output_path}/ground_truth", dtype=np.int32, mode='w+', shape=np.shape(ground_truth))
fp[:] = ground_truth[:]
del fp

# In[ ]:
