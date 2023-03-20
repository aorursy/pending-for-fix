#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras

import os
import pandas as pd
import numpy as np
from skimage.io import imread_collection
import skimage.io
import skimage.color
import skimage.transform
from platform import python_version
import matplotlib.pyplot as plt

print(tf.__version__)
print(python_version())


# In[2]:


# extract filenames from the folder of images
filenames = []
for root, dirs, files in os.walk('../input/rsna-hemorrhage-jpg/train_jpg/train_jpg'):
    for file in files:
        if file.endswith('.jpg'):
            filenames.append(file)
            
# should be the same as the images imported
len(filenames)


# In[3]:


col_dir = '../input/rsna-hemorrhage-jpg/train_jpg/train_jpg/*.jpg'

# Create a collection with the available images
images = imread_collection(col_dir)
#we could also try what is below,
#this should load the images in the order that we expect, 
#but if automatically alphabetical this isn't necessary:
#images = imread_collection(col_dir, load_pattern = filenames)

#make sure this is equivalent with the number of filenames
len(images)


# In[4]:


# Plot the first image
plt.figure()
plt.imshow(images[0])
plt.colorbar()
plt.grid(False)
plt.show()

print(images[0])


# In[5]:


# Check shape
print(images[0].shape)
print(images[1].shape)
print(images[2].shape)


# In[6]:


# Select only the first 5000 images
images_trn = images[:20000]
print(len(images_trn))
images_val = images[20000:25000]
print(len(images_val))
images_tst = images[25000:30000]
print(len(images_tst))


# In[7]:


images_arr_trn = skimage.io.collection.concatenate_images(images_trn)
images_arr_val = skimage.io.collection.concatenate_images(images_val)
images_arr_tst = skimage.io.collection.concatenate_images(images_tst)


# In[8]:


# Import labels and selct only first 5000 labels without any additional columns
#labels = pd.read_feather('../input/rsna-hemorrhage-jpg/meta/meta/labels.fth')
#labels = labels.iloc[:5000, 1]
#print(labels)
#print(type(labels))
#print(labels.sum())


# In[9]:


labels = pd.read_feather('../input/rsna-hemorrhage-jpg/meta/meta/labels.fth')

#manipulate the filenames list, stripping the .jpg at the end
idstosearch = [item.rstrip(".jpg") for item in filenames]

#now search the "ID" column for ids that correspond to our filenames
#made the reduced dataframe "labels2" for now
labels2 = labels[labels['ID'].isin(idstosearch)]
labels2.shape


# In[10]:


labels = labels2.iloc[:, 1]
print(labels)


# In[11]:


labels_trn = labels[:20000]
print(len(labels_trn))
labels_val = labels[20000:25000]
print(len(labels_val))
labels_tst = labels[25000:30000]
print(len(labels_tst))


# In[12]:


print(type(labels_trn))
print(labels_trn.sum())


# In[13]:


# Transform labels into array
labels_trn = pd.Series.to_numpy(labels_trn)
len(labels_trn)


# In[14]:


labels_val = pd.Series.to_numpy(labels_val)
len(labels_val)


# In[15]:


labels_tst = pd.Series.to_numpy(labels_tst)
len(labels_tst)


# In[16]:


# Build the model
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(256, 256, 3)),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(2, activation='softmax')
#])


# In[17]:


# CNN -> train/test accuracy both at 50%
#model = keras.Sequential()
#model.add(keras.layers.Conv2D(20, kernel_size=(6, 6), strides=(1, 1),
                 activation='relu',
                 input_shape=(256, 256, 3)))
#model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(50, activation='relu'))
#model.add(keras.layers.Dense(2, activation='softmax'))


# In[18]:


# CNN -> train/test accuracy at 60%/50%
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(256, 256, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))


# In[19]:


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# data = train_images.reshape(2000,75,100,1)


# In[20]:


# Train model
model.fit(images_arr_trn, labels_trn, epochs=8)


# In[21]:


# Validate model
test_loss, test_acc = model.evaluate(images_arr_val, labels_val, verbose=2)

print('\nTest accuracy:', test_acc)


# In[ ]:




