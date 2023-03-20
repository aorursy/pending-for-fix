#!/usr/bin/env python
# coding: utf-8



# This is a simple data exploration notebook. 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




print(os.listdir('../input'))




train= pd.read_csv('../input/train_sample.csv')
train.head()




train.info()




cols=train.columns
cols_time=['click_time', 'attributed_time']
cols_categorical=[col for col in cols if col not in cols_time]

for col in cols_categorical:
    train[col]=train[col].astype('category')
for col in cols_time:
    train[col]=pd.to_datetime(train[col])




train.describe()




train['conversion_time']=pd.to_timedelta(train['attributed_time']-train['click_time']).astype('timedelta64[s]')
print(train['conversion_time'].quantile(0.9)/3600)
train.describe()




ctimes=train['conversion_time'].dropna()




sns.distplot(ctimes)




sns.distplot(np.log10(ctimes))




# min_time=np.nanmin(train['click_time'])
# max_time=np.nanmax(train['attributed_time'])
# print(min_time)
# print(max_time)




# max_engagement_window_size=str(int(np.ceil(np.nanmax(ctimes)/3600)))+'H'
# max_engagement_window_size




sns.pairplot(train[train['is_attributed']==1])




col_list=[col for col in train.columns if col!='conversion_time']
# print(col_list)

sns.pairplot(train.loc[train['is_attributed']!=1,col_list])




from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier




train['is_attributed'].value_counts()/train.shape[0]




from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve




train['chour']=train['click_time'].dt.hour
train['cminute']=train['click_time'].dt.minute
train['cday']=train['click_time'].dt.day




del train['click_time']
del train['attributed_time']




Check how much of the data still has missing values




train.isnull().any()




del train['conversion_time']




X=train
Y=train['is_attributed']
del X['is_attributed']
X.head()




Xtrain, Xtest, Ytrain, Ytest=train_test_split(X,Y,test_size=0.2, random_state=32)




param_grid={'n_estimators':np.arange(10,100,20), 
           'max_depth':[None, 5, 10],
           'max_features':['auto','sqrt']}
rfgrid=GridSearchCV(RandomForestClassifier(),param_grid,cv=StratifiedKFold(5))




rfgrid.fit(Xtrain, Ytrain)




rfgrid.score(Xtest,Ytest)

