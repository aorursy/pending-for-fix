#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import random, re, math
import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from kaggle_datasets import KaggleDatasets
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as L
from tensorflow.keras.applications import ResNet152V2, InceptionResNetV2, InceptionV3, Xception, VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# In[2]:


get_ipython().system('pip install efficientnet')
import efficientnet.tfkeras as efn


# In[3]:


AUTO = tf.data.experimental.AUTOTUNE
# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)


# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path()


# In[4]:


img = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_0.jpg')
print(img.shape)
plt.imshow(img)


# In[5]:


path='../input/plant-pathology-2020-fgvc7/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
sub = pd.read_csv(path + 'sample_submission.csv')

train_paths = train.image_id.apply(lambda x: GCS_DS_PATH + '/images/' + x + '.jpg').values
test_paths = test.image_id.apply(lambda x: GCS_DS_PATH + '/images/' + x + '.jpg').values

train_labels = train.loc[:, 'healthy':].values


# In[6]:


nb_classes = 4
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
img_size = 768
EPOCHS = 5
SEED = 123


# In[7]:


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    # uint8（符号なし8ビット整数）形式のテンソルでデコードされた画像を出力
    # 出力テンソルのインデックス1つ目が高さ、2つ目が幅、3つ目がR/G/B各チャンネル
    image = tf.image.decode_jpeg(bits, channels=3)
    # tf.cast(変換したいもの, 変換後の型)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
        return image, label


# In[8]:


# データ拡張
def data_augment(image, label=None, seed=2020):
    # 画像を上下左右にランダムに flip する
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
           
    if label is None:
        return image
    else:
        return image, label


# In[9]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    )


# In[10]:


test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)


# In[11]:


# 学習率の減衰
# LR_START = 0.00001
# LR_MAX = 0.0001 * strategy.num_replicas_in_sync
# LR_MIN = 0.00001
# LR_RAMPUP_EPOCHS = 15
# LR_SUSTAIN_EPOCHS = 3
# LR_EXP_DECAY = .8

# def lrfn(epoch):
#     if epoch < LR_RAMPUP_EPOCHS:
#         lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
#     elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
#         lr = LR_MAX
#     else:
#         lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
#     return lr
    
# lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

# rng = [i for i in range(EPOCHS)]
# y = [lrfn(x) for x in rng]
# plt.plot(rng, y)
# print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


# In[12]:


def get_model1():
    base_model =  efn.EfficientNetB7(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    predictions = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)


# In[13]:


# def get_model2():
#     base_model =  efn.EfficientNetB6(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False, pooling='avg')
#     x = base_model.output
#     predictions = Dense(nb_classes, activation="softmax")(x)
#     return Model(inputs=base_model.input, outputs=predictions)


# In[14]:


# def get_model3():
#     model = tf.keras.Sequential([
#         ResNet152V2(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False),
#         L.GlobalAveragePooling2D(),
#         L.Dense(train_labels.shape[1], activation='softmax')
#     ])
#     return model


# In[15]:


# def get_model4():
#     model = tf.keras.Sequential([
#         InceptionResNetV2(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False),
#         L.GlobalAveragePooling2D(),
#         L.Dense(train_labels.shape[1], activation='softmax')
#     ])
#     return model


# In[16]:


# def get_model5():
#     model = tf.keras.Sequential([
#         InceptionV3(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False),
#         L.GlobalAveragePooling2D(),
#         L.Dense(train_labels.shape[1], activation='softmax')
#     ])
#     return model


# In[17]:


# def get_model6():
#     model = tf.keras.Sequential([
#         Xception(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False),
#         L.GlobalAveragePooling2D(),
#         L.Dense(train_labels.shape[1], activation='softmax')
#     ])
#     return model


# In[18]:


# def get_model7():
#     model = tf.keras.Sequential([
#         VGG19(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False),
#         L.GlobalAveragePooling2D(),
#         L.Dense(train_labels.shape[1], activation='softmax')
#     ])
#     return model


# In[19]:


strategy.scope()でコードのどの部分を分散して実行するかを示す
このスコープ内でモデルを作成すると、通常の変数の代わりにミラーリングされた変数を作成
with strategy.scope():
    model1 = get_model1()
    
model1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])


# In[20]:


get_ipython().run_cell_magic('time', '', 'model1.fit(\n    train_dataset, \n    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,\n#     callbacks=[lr_callback],\n    epochs=EPOCHS\n)')


# In[21]:


# with strategy.scope():
#     model2 = get_model2()
    
# model2.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])


# In[22]:


# %%time
# model2.fit(
#     train_dataset, 
#     steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
#     callbacks=[lr_callback],
#     epochs=EPOCHS
# )


# In[23]:


# with strategy.scope():
#     model3 = get_model3()
    
# model3.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])


# In[24]:


# %%time
# model3.fit(
#     train_dataset, 
#     steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
# #     callbacks=[lr_callback],
#     epochs=EPOCHS
# )


# In[25]:


# with strategy.scope():
#     model4 = get_model4()
    
# model4.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])


# In[26]:


# %%time
# model4.fit(
#     train_dataset, 
#     steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
#     callbacks=[lr_callback],
#     epochs=EPOCHS
# )


# In[27]:


# with strategy.scope():
#    model5 = get_model5()
    
# model5.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])


# In[28]:


# %%time
# model5.fit(
#    train_dataset, 
#    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
#    callbacks=[lr_callback],
#    epochs=EPOCHS
# )


# In[29]:


# with strategy.scope():
#     model6 = get_model6()
    
# model6.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])


# In[30]:


# %%time
# model6.fit(
#     train_dataset, 
#     steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
#     callbacks=[lr_callback],
#     epochs=EPOCHS
# )


# In[31]:


# with strategy.scope():
#     model7 = get_model7()
    
# model7.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])


# In[32]:


get_ipython().run_cell_magic('time', '', "probs1 = model1.predict(test_dataset, verbose=1)\n# probs2 = model2.predict(test_dataset, verbose=1)\n# probs3 = model3.predict(test_dataset, verbose=1)\n# probs4 = model4.predict(test_dataset, verbose=1)\n# probs5 = model5.predict(test_dataset, verbose=1)\n# probs6 = model6.predict(test_dataset, verbose=1)\n# probs7 = model7.predict(test_dataset, verbose=1)\n# probs_avg = (2*probs1 + probs2 + probs3 + probs4) / 5  # + probs5+ probs6 + probs7) / 8\n# probs_avg = (probs1 + probs3 + probs5) / 3\n# sub.loc[:, 'healthy':] = probs_avg\nsub.loc[:, 'healthy':] = probs1\nsub.to_csv('submission_effnets.csv', index=False)\nsub.head()")


# In[33]:


for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

