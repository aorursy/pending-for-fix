#!/usr/bin/env python
# coding: utf-8



import os
import time
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set()




print("Files in the folder:")
print(os.listdir("../input"))




train = pd.read_csv('../input/X_train.csv')
test = pd.read_csv('../input/X_test.csv')




train.head()




def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,5,figsize=(16,8))

    for feature in features:
        i += 1
        plt.subplot(2,5,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();

def plot_feature_class_distribution(classes,tt, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(5,2,figsize=(16,24))

    for feature in features:
        i += 1
        plt.subplot(5,2,i)
        for clas in classes:
            ttc = tt[tt['surface']==clas]
            sns.kdeplot(ttc[feature], bw=0.5,label=clas)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();
    




features = train.columns.values[3:]
plot_feature_distribution(train, test, 'train', 'test', features)




labels = pd.read_csv('../input/y_train.csv')




classes = (labels['surface'].value_counts()).index
aux = train.merge(labels, on='series_id', how='inner')
plot_feature_class_distribution(classes, aux, features)




# first drop columns such as measurement and row_id in training data
train_d = train.drop(['row_id', 'measurement_number'], axis=1)
test_d = test.drop(['row_id','measurement_number'], axis=1)
train_f = train_d.groupby('series_id').agg(['min', 'max', 'mean', 'median', 'var'])
test_f = test_d.groupby('series_id').agg(['min', 'max', 'mean', 'median', 'var'])




# lets see what we got
train_f.head()




test_f.head()




# I am going to write to temporary files and then use them for model building
# add surface to the training points
#aux = train_f.merge(labels, on='series_id', how='inner')
#aux.to_csv('training.csv', index=False, header=None)
#test_f.to_csv(input/testing.csv', index=False, header=None)
train_f['surface'] = labels['surface']




train_f.head()




train_f.to_csv('training.csv', index=False, header=None)
test_f.to_csv('testing.csv', index=False, header=None)




# load the traing data 
data = pd.read_csv('training.csv', header=None)




test_data = pd.read_csv('testing.csv', header=None)




from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# lets use GBM and AdaBoost, see how these work out
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier




from sklearn.tree import DecisionTreeClassifier




# First lets see AdaBoost  
model_ab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7), random_state=41)




train, valid = train_test_split(data, test_size=0.2)




model_ab.fit(train.loc[:, 0:49], train[50])




predict_train = model_ab.predict(train.loc[:, 0:49])
print(classification_report(train[50], predict_train))
confusion_matrix(train[50], predict_train)




predict_valid = model_ab.predict(valid.loc[:, 0:49])
print(classification_report(valid[50], predict_valid))
confusion_matrix(valid[50], predict_valid)




test_predict = model_ab.predict(test_data)




# read sample submission file
submit = pd.read_csv('../input/sample_submission.csv')




submit['surface'] = test_predict




submit.head()




submit.to_csv('naive_submission.csv', index=False)




more naive_submission.csv






