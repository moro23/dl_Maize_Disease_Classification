#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


## libraries for os operations
import os

## libraries for data preprocessing
import numpy as np 
import pandas as pd 

## libraries for visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

## libraries for training dl models
import tensorflow as tf
from tensorflow import keras 

## libraries for pre-trained neural network
from tensorflow.keras.applications.xception import preprocess_input

## libraries for loading batch images
from tensorflow.keras.preprocessing.image import load_img

## ignoring warnings
import warnings 
warnings.filterwarnings('ignore')


# ## Loading The Model

# In[2]:


model = keras.models.load_model('xception_v1_15_0.812.h5')


# ## Getting The Predictions
# - Loading an image
# - Preprocessing Image

# In[11]:


## lets load an image sample
path = 'testImages/maize_fall_armyworm.jpg'
img = load_img(path, target_size=(299, 299))


# In[12]:


## convert and preprocess the image
x = np.array(img)
X = np.array([x])
X = preprocess_input(X)


# In[8]:


## lets get the predictions
pred = model.predict(X)

## lets get the element with the highest score
pred[0].argmax()


# In[9]:


## lets create our labels
labels = {
    0: 'maize ear rot',
    1: 'maize fall armyworm',
    2: 'maize stem borer'
}


# In[10]:


labels[pred[0].argmax()]


# In[ ]:




