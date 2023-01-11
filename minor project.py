#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Input,Dropout, Flatten, Conv2D
from tensorflow.keras.layers import Activation, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from IPython.display import SVG, Image


# In[3]:


import tensorflow as tf
print("Tensorflow version:", tf.version)


# In[9]:


for dirname, _, filenames in os.walk(r"C:\Users\pmjv0\Downloads\test"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[12]:


img_size = 48
batch_size = 64

data_train = ImageDataGenerator(horizontal_flip=True)
train_generator = data_train.flow_from_directory(r"C:\Users\pmjv0\Downloads\train",
                                                   target_size=(img_size,img_size),
                                        
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=True)

data_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = data_train.flow_from_directory(r"C:\Users\pmjv0\Downloads\test",
                                                   target_size=(img_size,img_size),
                                                   
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=True)


# In[14]:


model = Sequential()

#conv1
model.add(Conv2D(64,(3,3), padding="same", input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#conv2
model.add(Conv2D(128,(5,5), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#conv3
model.add(Conv2D(512,(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#conv4
model.add(Conv2D(512,(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

opt=Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[16]:


print(model)


# In[19]:


model.inputs


# In[20]:


model.outputs


# In[21]:


model.get_weights


# In[22]:


model.get_config()


# In[27]:


labels=train_generator.class_indices
print(labels)


# In[ ]:




