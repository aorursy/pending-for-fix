#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import os.path
import zipfile
import cv2


# In[ ]:





# In[2]:


train_path_labels = '../input/aptos2019-blindness-detection/train.csv'
test_path_labels = '../input/aptos2019-blindness-detection/test.csv'
train_images = '../input/aptos2019-blindness-detection/train_images.zip'
test_images = '../input/aptos2019-blindness-detection/test_images.zip'



train_labels = pd.read_csv(train_path_labels)
test_labels = pd.read_csv(test_path_labels)


# In[3]:


train_labels.head()


# In[4]:


## Reading some images and compare 

fig=plt.figure(figsize=(10, 2))

with zipfile.ZipFile(train_images, 'r') as zfile:
    data1 = zfile.read('005b95c28852.png')
    data2 = zfile.read('001639a390f0.png')

img1 = cv2.imdecode(np.frombuffer(data1, np.uint8), 5)
img2 = cv2.imdecode(np.frombuffer(data2, np.uint8), 5)

img1 = cv2.resize(img1, (960, 540)) 
img2 = cv2.resize(img2, (960, 540))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imshow('Healthy',gray1)
cv2.imshow('Number4',gray2)
cv2.waitKey(0)


# In[5]:


hist = np.histogram(gray2.flatten(),256,[0,256])[0]


# In[6]:


hist


# In[7]:


from os import listdir,makedirs
from os.path import isfile,join

path_train = r'Train_Images' # Source Folder
path_train_gray = r'Gray_Images' # Destination Folder


files = [f for f in listdir(path_train) if isfile(join(path_train,f))] 

for image in files:
    img = cv2.imread(os.path.join(path_train,image))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dstPath = join(path_train_gray,image)
    cv2.imwrite(dstPath,gray)


# In[8]:


files = [f for f in listdir(path_train_gray) if isfile(join(path_train_gray,f))] 

for images in files:
    img = cv2.imread(os.path.join(path_train_gray,image))
    resized_image = cv2.resize(img, (128, 128))
    same_path = join(path_train_gray,images)
    cv2.imwrite(same_path, resized_image)
    


# In[9]:


def image_processing(original_path, new_path):
    files = [f for f in listdir(original_path) if isfile(join(original_path,f))] 
    for image in files:
        img = cv2.imread(os.path.join(original_path,image))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #Check image entrance do CNN model
        resized_image = cv2.resize(gray, (128, 128))
        different_path = join(new_path,image)
        cv2.imwrite(different_path,resized_image)


# In[10]:


original_path_train = r'Train_Images' # Source Folder
new_path_train = r'Gray_Images' # Destination Folder

image_processing(original_path_train, new_path_train)


# In[11]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

import warnings


# In[12]:


def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['id_code']:
        #load images into images of size 100x100x3
        img = image.load_img(dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train


# In[13]:


def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encode


# In[14]:


train_csv = pd.read_csv(train_path_labels)


X = prepareImages(train_labels, train_labels.shape[0], 'Train_Images')
X /= 255


# In[15]:


y, label_encoder = prepare_labels(train_labels['id_code'])


# In[16]:


model = Sequential()
model.add(Conv2D(32, (7,7), input_shape = X.shape[1:])
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(64, (3,3), input_shape = X.shape[1:])
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(64, (3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


model.fit(X,y, batch_size = 32,epochs = 3,  validation_split = 0.1)


# In[17]:


history = model.fit(X, y, epochs=100, batch_size=100, verbose=1)
gc.collect()

