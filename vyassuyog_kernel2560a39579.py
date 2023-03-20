#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




my_cmp=pltc.LinearSegmentedColormap.from_list("",["red","green","blue"])




train_data=pd.read_csv("../input/2el1730-machinelearning/train.csv")
train_data.fillna(0)
train_data




train_desc=train_data.describe()
train_desc.columns




for i in train_desc.columns:
    plt.scatter(train_data[i],train_data["label"])
    plt.xlabel(i)
    plt.show()




train_data.loc[:,"website"]=train_data.loc[:,"org"]  + train_data.loc[:,"tld"]
train_data




org_freq=train_data.groupby("website").size()/len(train_data)
train_data.loc[:,"website_freq"]=train_data.loc[:,"website"].map(org_freq)
train_data




mail_freq=train_data.groupby("mail_type").size()/len(train_data)
train_data.loc[:,"mail_type_freq"]=train_data.loc[:,"mail_type"].map(mail_freq)
train_data




train_data.drop(["Id","date","org","tld","mail_type"],axis=1,inplace=True)
train_data
train_data.drop(["website"],axis=1,inplace=True)

train_data.fillna(0,inplace=True)




X=train_data.drop(["label"],axis=1)
X




Y=train_data["label"]
Y




X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=0)
print(X_train.shape,X_val.shape,Y_train.shape,Y_val.shape)




std=StandardScaler()
X_train_std=std.fit_transform(X_train)
X_val_std=std.transform(X_val)




X_train_std=pd.DataFrame(X_train_std,columns=X_train.columns)
X_train_std.fillna(0,inplace=True)
X_train_std




X_val_std=pd.DataFrame(X_val_std,columns=X_val.columns)
X_val_std.fillna(0,inplace=True)
X_val_std




for i in range(10):
    knn=KNeighborsClassifier(n_neighbors=i+1)

    knn.fit(X_train_std,Y_train)
    Y_pred=knn.predict(X_val_std)
    acc=accuracy_score(Y_val,Y_pred)
    print(acc)




for i in range(10):
    knn=KNeighborsClassifier(n_neighbors=i+1)

    knn.fit(X_train_std,Y_train)
    Y_pred=knn.predict(X_train_std)
    acc=accuracy_score(Y_train,Y_pred)
    print(acc)




X_train.fillna(0,inplace=True)
X_val.fillna(0,inplace=True)




log=LogisticRegression()

log.fit(X_train_std,Y_train)
Y_pred=log.predict(X_val_std)
acc=accuracy_score(Y_val,Y_pred)
print(acc)




log=LogisticRegression()

log.fit(X_train,Y_train)
Y_pred=log.predict(X_val)
acc=accuracy_score(Y_val,Y_pred)
print(acc)




nb=GaussianNB()
nb.fit(X_train_std,Y_train)
Y_pred=nb.predict(X_val_std)
acc=accuracy_score(Y_val,Y_pred)
print(acc)




trainl_data=pd.read_csv("../input/2el1730-machinelearning/train.csv")




trainl_data




trainl_data.loc[:,"website"]=trainl_data.loc[:,"org"]  + trainl_data.loc[:,"tld"]
trainl_data




label=preprocessing.LabelEncoder()
trainl_data["website"]=label.fit_transform(list(trainl_data["website"]))
trainl_data["website"]




label=preprocessing.LabelEncoder()
trainl_data["mail_type"]=label.fit_transform(list(trainl_data["mail_type"]))
trainl_data["mail_type"]




trainl_data.drop(["Id","date","org","tld"],axis=1,inplace=True)
trainl_data




X_l=trainl_data.drop(["label"],axis=1)
X_l




Y_l=trainl_data["label"]
Y_l




X_trainl,X_vall,Y_trainl,Y_vall=train_test_split(X_l,Y_l,test_size=0.2,stratify=Y,random_state=0)
print(X_trainl.shape,X_vall.shape,Y_trainl.shape,Y_vall.shape)




std=StandardScaler()
X_trainl_std=std.fit_transform(X_trainl)
X_vall_std=std.transform(X_vall)




X_trainl_std=pd.DataFrame(X_trainl_std,columns=X_trainl.columns)
X_trainl_std.fillna(0,inplace=True)
X_trainl_std




X_vall_std=pd.DataFrame(X_vall_std,columns=X_vall.columns)
X_vall_std.fillna(0,inplace=True)
X_vall_std




for i in range(10):
    knn=KNeighborsClassifier(n_neighbors=i+1)

    knn.fit(X_trainl_std,Y_trainl)
    Y_pred=knn.predict(X_vall_std)
    acc=accuracy_score(Y_vall,Y_pred)
    print(acc)




for i in range(10):
    knn=KNeighborsClassifier(n_neighbors=i+1)

    knn.fit(X_trainl_std,Y_trainl)
    Y_pred=knn.predict(X_trainl_std)
    acc=accuracy_score(Y_trainl,Y_pred)
    print(acc)




X_train.(kind="box",subplots=True,layout=(3,4))




corr=X_train.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(corr,vmax=1,vmin=-1)
fig.colorbar(cax)
ticks=np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticklabels(X_train.columns)
ax.set_yticklabels(X_train.columns)
plt.show()




test_data=pd.read_csv("../input/2el1730-machinelearning/test.csv")
test_data.fillna(0)
test_data




test_data.loc[:,"website"]=test_data.loc[:,"org"]  + test_data.loc[:,"tld"]
test_data




label=preprocessing.LabelEncoder()
test_data["website"]=label.fit_transform(list(test_data["website"]))
test_data["website"]




label=preprocessing.LabelEncoder()
test_data["mail_type"]=label.fit_transform(list(test_data["mail_type"]))
test_data["mail_type"]




test_data.drop(["Id","date","org","tld"],axis=1,inplace=True)
test_data




std=StandardScaler()
test_data_std=std.fit_transform(test_data)
test_data_std




knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_trainl_std,Y_trainl)
Y_pred=knn.predict(test_data_std)
Y_pred




Y_pred=pd.DataFrame(Y_pred)




Submit=pd.concat((test_data,Y_pred),axis=1)
Submit
Submit.rename(columns={0: 'Y_pred'},inplace=True)




Submit






