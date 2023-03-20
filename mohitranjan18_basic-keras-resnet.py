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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




train=pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test=pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')




train.shape




train.head()




train.isnull().sum()




(train.isnull().sum()/len(train.index))*100




# Dropping rows having null values

train=train.dropna(axis=0).reset_index(drop=True)




(train.isnull().sum()/len(train.index))*100




train['image_name']=train['image_name'].apply(lambda x:x+'.jpg')




test['image_name']=test['image_name'].apply(lambda x:x+'.jpg')




sample_df=pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')




sample_df.head()




train.target.value_counts()




train.target.unique()




import matplotlib.pyplot as plt
import seaborn as sns




sns.countplot(train.target)




# import seaborn as sns
# import matplotlib.pyplot as plt
# import tensorflow as tf

# from keras_preprocessing.image import ImageDataGenerator
# # from keras.applications.densenet import DenseNet121
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras.models import Model
# from keras import backend as K

# from keras.models import load_model




# importing few important libraries for pre-processing,generator and model building

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from tensorflow.keras.models import load_model





from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Add,Dense,Activation,ZeroPadding2D,BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense,Dropout,Flatten
# from keras.applications import resnet50
# from keras.applications.resnet50 import preprocess_input
K.set_image_data_format('channels_last')
K.set_learning_phase




import os




from tensorflow.keras.applications.resnet50  import ResNet50




def checkLeakage(df1,df2,patient):
    df1_unique = set(df1[patient].values)
    df2_unique = set(df2[patient].values)
    
    patients_in_train_test = list(df1_unique.intersection(df2_unique))
    if len(patients_in_train_test)>0:
        leakage = True# boolean (true if there is at least 1 patient in both groups)
        print(patients_in_train_test)
    else:
        leakage=False
    
    
    return leakage




checkLeakage(train,test,'patient_id')




def train_val_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 240, target_h = 240):
        print("getting train generator...") 
    # normalize images
        image_generator = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization= True,
            zoom_range=0.2,
            shear_range=0.2,
            validation_split=0.25)
    
    # flow from directory with specified batch size
    # and target image size
        train_generator = image_generator.flow_from_dataframe(
            dataframe=train,
            directory=image_dir,
            validation_split=0.25,
            x_col=x_col,
            y_col=y_cols,
            class_mode='raw',
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            validate_filenames=True,
            subset='training',
            target_size=(target_w,target_h))
        
        valid_generator=image_generator.flow_from_dataframe(
            dataframe=train,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            validate_filenames=False,
            batch_size=32,
            shuffle=False,
            class_mode='raw',
            seed=seed,
            subset='validation',
            target_size=(target_w,target_h))
        
        return train_generator,valid_generator
    
    




IMAGE_DIR = "../input/siim-isic-melanoma-classification/jpeg/train"




os.makedirs('saved_models_noFold')




def get_model():
    
#     image_input=Input(shape=(240, 240, 3))
    base_model= ResNet50(weights='imagenet',include_top=False,input_shape=(240,240,3))
    for layer in base_model.layers:
        layer.trainable=False
        
#     Get base model output

    base_model_output=base_model.output
    
#     Adding our own layers

    x=GlobalAveragePooling2D()(base_model_output)
#     adding fully connected layers

    x=Dense(512,activation='relu')(x)
    x=Dense(1,activation='sigmoid',name='fcnew')(x)
    
    model=Model(base_model.input,x)
    
    return model




get_model()




# Creating a function control gradient descent rate
# Initially we will keep the loss high 
# will decrease it with the training
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr




from sklearn.utils import class_weight




train_data,valid_data=train_val_generator(train, IMAGE_DIR, "image_name", 'target')




train_data.labels




class_weights = class_weight.compute_class_weight('balanced',np.unique(train_data.labels),train_data.labels)




class_weights_map={0:class_weights[0],1:class_weights[1]}




class_weight_map={}




import tensorflow




import math




# get model

model=get_model()

batch_size=32



# Compile the model

model.compile(loss='binary_crossentropy',optimizer ='sgd',metrics=[tf.keras.metrics.AUC()])

model.summary()






checkpoint = tensorflow.keras.callbacks.ModelCheckpoint('saved_models_noFold'+'.h5', verbose=1,save_best_only=True)

callbacks_list = [checkpoint]

# model.compile(loss=get_weighted_loss(pos_weights,neg_weights),optimizer ='sgd',metrics=['accuracy'])



history = model.fit_generator(train_data,
                              validation_data=valid_data,
                              steps_per_epoch=int(math.ceil(1. * len(train_data)// batch_size)),
                              validation_steps=int(math.ceil(1. * len(valid_data)// batch_size)),
                              callbacks=callbacks_list,
                              class_weight=class_weights_map,
#                                   workers=1,                        # maximum number of processes to spin up when using process-based threading
#                                   use_  multiprocessing=False,
                              epochs =10)




model.load_weights("saved_models_noFold.h5")




# Writing test generator
def test_generator(df,image_dir, x_col, shuffle=False, seed=1, target_w = 240, target_h = 240):
        print("getting test generator...") 
        print(image_dir)
    # normalize images
        image_generator = ImageDataGenerator(rescale=1/255)
        
        
        test_generator = image_generator.flow_from_dataframe(
            dataframe=test,
            directory=image_dir,
            x_col=x_col,
            y_col=None,
#             classes=['test'],
#             x_col=x_col,
#             y_col=y_cols,
            class_mode=None,
#             batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
        
        return test_generator




test_path='../input/siim-isic-melanoma-classification/jpeg/test'




test.head()




test_gen=test_generator(test,test_path, "image_name",target_w = 240, target_h = 240)




# type(test_gen)




predicted_vals = model.predict_generator(test_gen,verbose=1)




test.shape




len(predicted_vals)




l=test['image_name'].apply(lambda x:x.split('.jpg'))




img=[im[0] for im in l]




img.index




img




test_sub=test['image_name']









img




submission_df=pd.DataFrame({'image_name':img})




submission_df.head()




submission_df['target']=predicted_vals




submission_df.to_csv('submission.csv', index=False)




get_ipython().system('ls -ltr')




sample_df.head()




test_sub['target']=pd.DataFrame(data=predicted_vals)




test_sub.head()




predicted_class_indices = np.argmax(predicted_vals, axis=1)
# labels = train_gen.class_indices
predictions = [labels[k] for k in predicted_class_indices]

test_df['target'] = pd.DataFrame(data=predictions)
# submission_df.to_csv('submission.csv', index=False)




# Compile the model
    model.compile(loss='binary_crossentropy',optimizer ='sgd',metrics=[tf.keras.metrics.AUC()])
    # CREATE CALLBACKS
	
    checkpoint = keras.callbacks.ModelCheckpoint('model_'+str(fold_var)+'.h5', verbose=1,save_best_only=True)
    
    callbacks_list = [checkpoint]
    
    train_data_generator=train_val_generator(train, IMAGE_DIR, "image_name", 'target')
    
    history = model.fit_generator(train_data_generator,validation_data=valid_data_generator,
                                  steps_per_epoch=int(math.ceil(1. * X_train.shape[0] // batch_size)),
                                  validation_steps=int(math.ceil(1. * X_Val.shape[0] // batch_size)),
                                  callbacks=callbacks_list,
                                  class_weight=class_weights,
#                                   workers=1,                        # maximum number of processes to spin up when using process-based threading
                                  use_multiprocessing=False,
                                  epochs =10)
    
    # LOAD BEST MODEL to evaluate the performance of the model
   
	
    model.load_weights("model_"+str(fold_var)+".h5")
	




# Implementing stratified K-fold for best model

from sklearn.model_selection import StratifiedKFold




X =train.loc[:, ~train.columns.isin(['target'])].copy()
y = train['target']
skf = StratifiedKFold(n_splits=10,shuffle=False)
skf.get_n_splits(X, y)




X.index




y.index




print(skf)




X.head()




y.value_counts()




import os




os.makedirs('saved_models')




get_ipython().system('cd ../../outputs/saved_models')
# ../input/siim-isic-melanoma-classification




get_ipython().system('ls -ltr')




get_ipython().system('cd saved_models')




get_ipython().system('ls -ltr')
get_ipython().system('pwd')




# create the base pre-trained model
# base_model = DenseNet121(weights='./nih/densenet.hdf5', include_top=False)

# x = base_model.output

# # add a global spatial average pooling layer
# x = GlobalAveragePooling2D()(x)

# # and a logistic layer
# predictions = Dense(len(labels), activation="sigmoid")(x)

# model = Model(inputs=base_model.input, outputs=predictions)
# model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))

from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Add,Dense,Activation,ZeroPadding2D,BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense,Dropout,Flatten
# from keras.applications import resnet50
# from keras.applications.resnet50 import preprocess_input
K.set_image_data_format('channels_last')
K.set_learning_phase




from tensorflow.keras.applications.resnet50  import ResNet50
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions




from tensorflow.keras import backend




tf.keras.applications.ResNet50





def get_model():
    
#     image_input=Input(shape=(240, 240, 3))
    base_model= ResNet50(weights='imagenet',include_top=False,input_shape=(240,240,3))
    for layer in base_model.layers:
        layer.trainable=False
        
#     Get base model output

    base_model_output=base_model.output
    
#     Adding our own layers

    x=GlobalAveragePooling2D()(base_model_output)
#     adding fully connected layers

    x=Dense(512,activation='relu')(x)
    x=Dense(1,activation='sigmoid',name='fcnew')(x)
    
    model=Model(base_model.input,x)
    
    return model




model=get_model()




# Writing function so that we speed up training 
# We will keep the learning rate high when the loss is high 
# As the loss decreases we will reduce the learning rate




from sklearn.utils import class_weight




VALIDATION_ACCURACY = []
VALIDAITON_LOSS = []

save_dir = 'saved_models'
fold_var = 1









image_generator = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization= True,
            zoom_range=0.2,
            shear_range=0.2)




import keras




train.shape




import math




for train_index, val_index in skf.split(X, y):
#     print("TRAIN:", train_index, "val:", val_index)
    batch_size=32
    X_train, X_Val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    train_df=pd.concat([X_train, y_train],axis=1)
    val_df=pd.concat([X_Val, y_val],axis=1)
#     train_generator,valid_generator =train_val_generator(train_df, IMAGE_DIR, "image_name", 'target',val_df)
    train_data_generator = image_generator.flow_from_dataframe(train_df,
                                                               directory = IMAGE_DIR,
                                                               x_col = "image_name",
                                                               y_col = "target",
                                                               class_mode = "raw",
                                                               batch_size=batch_size, target_size=(240,240),
                                                               shuffle = True)
    
    valid_data_generator  = image_generator.flow_from_dataframe(val_df, 
                                                                directory = IMAGE_DIR,
                                                                x_col = "image_name",
                                                                y_col = "target",
                                                                class_mode = "raw", 
                                                                batch_size=batch_size, 
                                                                target_size=(240,240),
                                                                shuffle = True)
    
    class_weights = class_weight.compute_class_weight('balanced',np.unique(train_data_generator.labels),train_data_generator.labels)
# create model 
    model=get_model()

# Compile the model
    model.compile(loss='binary_crossentropy',optimizer ='sgd',metrics=[tf.keras.metrics.AUC()])
    # CREATE CALLBACKS
	
    checkpoint = keras.callbacks.ModelCheckpoint('model_'+str(fold_var)+'.h5', verbose=1,save_best_only=True)
    
    callbacks_list = [checkpoint]
    
    history = model.fit_generator(train_data_generator,validation_data=valid_data_generator,
                                  steps_per_epoch=int(math.ceil(1. * X_train.shape[0] // batch_size)),
                                  validation_steps=int(math.ceil(1. * X_Val.shape[0] // batch_size)),
                                  callbacks=callbacks_list,
                                  class_weight=class_weights,
#                                   workers=1,                        # maximum number of processes to spin up when using process-based threading
                                  use_multiprocessing=False,
                                  epochs =10)
    
    # LOAD BEST MODEL to evaluate the performance of the model
   
	
    model.load_weights("model_"+str(fold_var)+".h5")
	
    results = model.evaluate(valid_data_generator)
    
    results = dict(zip(model.metrics_names,results))
	
    VALIDATION_ACCURACY.append(history.history['val_loss'])
    VALIDATION_LOSS.append(history.history['val_acc'])
	
    fold_var += 1
    




train_generator,valid_generator =train_val_generator(train, IMAGE_DIR, "image_name", 'target')





x, y = train_generator.__getitem__(0)
plt.imshow(x[0]);




import glob




# print(glob.glob("../input/siim-isic-melanoma-classification/jpeg/train/*.jpg"))

def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    
    # total number of patients (rows)
    N = labels.shape
    
    
    positive_frequencies = np.sum(labels,axis=0)/N[0]
    negative_frequencies = (N[0]-np.sum(labels,axis=0))/N[0]

    return positive_frequencies, negative_frequencies




# Function to get weight loss which we will be used for training model

def get_weighted_loss(pos_weights,neg_weight,epsilon=1e-7):
    def weighted_loss(pred_pos,pred_neg):
        loss=0#initialising the loss to 0
        loss +=  K.mean(pos_weights * np.array(train['target'])*K.log(y_pred_1+epsilon) + neg_weights*(1- np.array(train['target'])*K.log(1-y_pred_2+epsilon)))
        return loss       
        
    return weighted_loss




# create the base pre-trained model
# base_model = DenseNet121(weights='./nih/densenet.hdf5', include_top=False)

# x = base_model.output

# # add a global spatial average pooling layer
# x = GlobalAveragePooling2D()(x)

# # and a logistic layer
# predictions = Dense(len(labels), activation="sigmoid")(x)

# model = Model(inputs=base_model.input, outputs=predictions)
# model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))

from keras import layers
from keras.layers import Input,Add,Dense,Activation,ZeroPadding2D,BatchNormalization
from keras import optimizers
from keras.layers import Dense,Dropout,Flatten
# from keras.applications import resnet50
# from keras.applications.resnet50 import preprocess_input
K.set_image_data_format('channels_last')
K.set_learning_phase




from keras.applications import ResNet50
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions




# ! wget https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5






def get_model():
    
    base_model= ResNet50(weights='imagenet',include_top=False)
    for layer in base_model.layers:
        layer.trainable=False
        
#     Get base model output

    base_model_output=base_model.output
    
#     Adding our own layers

    x=GlobalAveragePooling2D()(base_model_output)
#     adding fully connected layers

    x=Dense(512,activation='relu')(x)
    x=Dense(1,activation='sigmoid',name='fcnew')(x)
    
    model=Model(input=base_model.input,output=x)
    
    return model




set(train_generator.labels)




freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
freq_pos




freq_neg




pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights




pos_weights




from sklearn.utils import class_weight




class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_generator.labels),
                                                 train_generator.labels)




class_weights




class_weights={0:class_weights[0],1:class_weights[1]}




# y_pred_1  = K.constant(pos_weights*np.array(train['target']).reshape(33126,1))





# y_pred_2  = K.constant(neg_weights*np.array(train['target']).reshape(33126,1))




# data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
# data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
# plt.xticks(rotation=90)
# f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)




# get model

model=get_model()

# Compile the model

model.compile(loss='binary_crossentropy',optimizer ='sgd',metrics=['accuracy'])

# model.compile(loss=get_weighted_loss(pos_weights,neg_weights),optimizer ='sgd',metrics=['accuracy'])

model.summary()




history = model.fit_generator(train_generator, 
                              steps_per_epoch=100,
                              class_weight=class_weights,
                              epochs =10)




plt.plot(history.history['loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
plt.show()

