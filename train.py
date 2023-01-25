#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[16]:


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
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions

## libraries for loading batch images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## ignoring warnings
import warnings 
warnings.filterwarnings('ignore')


# ## Loading Dataset
# - Building The Training And Validation Dataset
# - Data Argumentation

# ## 

# In[35]:


da_train_gen = ImageDataGenerator(
    shear_range=10.0,
    zoom_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input

)

da_train_ds = train_gen.flow_from_directory(
    "../input/d/moro23/maize-disease-dataset/maize_disease/training_data",
    target_size=(299,299),
    batch_size=32
)


# In[36]:


da_valid_gen = ImageDataGenerator(
    shear_range=10.0,
    zoom_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

da_valid_ds = valid_gen.flow_from_directory(
    "../input/d/moro23/maize-disease-dataset/maize_disease/validation_data",
    target_size=(299,299),
    batch_size=32

)


# ## Creating The Model Function
# 

# In[ ]:


def make_model(learning_rate, dropout_rate):
    """
    Args:
    learning_rate: float
    dropout_rate: float
    """
    base_model = Xception(
        weights='imagenet',
        input_shape=(299, 299, 3),
        include_top=False
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(299, 299, 3))
    
    base = base_model(inputs, training=False)
    
    vector = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(100, activation='relu')(vector)
    ## adding a dropout
    drop = keras.layers.Dropout(dropout_rate)(inner)
    
    outputs = keras.layers.Dense(3)(drop)
    
    model = keras.Model(inputs, outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate)
    
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )
    
    return model


# ## Training And Saving Final Model

# In[39]:


model = make_model(learning_rate=0.01, dropout_rate=0.2)

checkpoint = keras.callbacks.ModelCheckpoint(
    "xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5",
    save_best_only=True,
    monitor="val_accuracy",
    mode='max'
)

history = model.fit(da_train_ds, epochs=50, validation_data=da_valid_ds, callbacks=checkpoint)


# In[ ]:




