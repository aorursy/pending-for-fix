#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




os.listdir('../input/aptos2019-blindness-detection/')




base_image_dir = os.path.join('..','input/aptos2019-blindness-detection/')
train_dir = os.path.join(base_image_dir,'train_images/')
train_df = pd.read_csv(os.path.join(base_image_dir,'train.csv'))




train_df.head()




train_df['image']= train_df['id_code'].map(lambda x: '{}.png'.format(x))




train_df['diagnosis']= train_df['diagnosis'].map(lambda x: str(x))




train_df.head()




train_df = train_df.drop(columns=['id_code'])




train_df.head()




train_df = train_df.sample(frac=1).reset_index(drop=True)




train_df.head()




print("Number of images: {}".format(len(train_df)))




train_df['diagnosis'].hist(figsize=(10,5))




from PIL import Image
from matplotlib import pyplot as plt
img = Image.open('../input/aptos2019-blindness-detection/train_images/'+train_df['image'][1])
w, h = img.size
print(w,h)




plt.imshow(np.asarray(img))




from sklearn.model_selection import train_test_split




train, valid = train_test_split(train_df,test_size=0.2, random_state=42, shuffle=True)




train.shape




valid.shape




from tensorflow import keras




from keras.applications.inception_v3 import InceptionV3




model = InceptionV3()




model.summary()




from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
import tensorflow as tf




flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(5, activation='softmax')(class1)




model = Model(inputs=model.inputs, outputs=output)




model.summary()




model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])




#history = model.fit(train_images, train_labels, epochs=10, 
 #                   validation_data=(test_images, test_labels))




from keras_preprocessing.image import ImageDataGenerator




datagen=ImageDataGenerator()




train_generator=datagen.flow_from_dataframe(
dataframe=train,
directory=train_dir,
x_col="image",
y_col="diagnosis",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(299,299))




valid_generator=datagen.flow_from_dataframe(
dataframe=valid,
directory=train_dir,
x_col="image",
y_col="diagnosis",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(299,299))




STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size




model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=2
)




test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="./test/",
x_col="id",
y_col=None,
batch_size=32,
seed=42,
shuffle=False,
class_mode=None,
target_size=(32,32))




model.evaluate_generator(generator=valid_generator,
steps=1)




mkdir model




ls




model.save('/kaggle/working/model')




ls model/




model_new = keras.models.load_model('/kaggle/working/model')




model_new.summary()




valid_generator.reset()




pred=model.predict_generator(valid_generator,
steps=STEP_SIZE_VALID,
verbose=1)




predicted_class_indices=np.argmax(pred,axis=1)




valid




x = train_dir+'0024cdab0c1e.png'




x




from PIL import Image
from matplotlib import pyplot as plt
img = Image.open(x)
w, h = img.size
print(w,h)




x= np.array(img)
x.shape




x = np.resize(x,(299,299,3))




x = np.expand_dims(x,0)




x.shape




model.summary()




pred = model.predict(x,batch_size=1, verbose=0, steps=None, callbacks=None)




pred




np.argmax(pred,axis=1)




pred_new = model_new.predict(x,batch_size=1, verbose=0, steps=None, callbacks=None)




pred_new




np.argmax(pred_new,axis=1)




import cv2
import numpy as np
img_list = []
def prepare(filename):
  frame = cv2.imread(filename)
  im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  model_image_size = (299, 299)
  resized_image = cv2.resize(im, model_image_size, interpolation = cv2.INTER_CUBIC)
  resized_image = resized_image.astype(np.float32)
  resized_image /= 255.
  image_data = np.expand_dims(resized_image, 0)
  return image_data
img = prepare(x)
prediction =  model_new.predict([img])
print(prediction)




np.argmax(prediction,axis=1)






