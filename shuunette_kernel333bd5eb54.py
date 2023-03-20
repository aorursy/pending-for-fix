#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)




df = pd.read_csv("/kaggle/input/predict-west-nile-virus/train.csv")




df.head()




pd.isna(df).sum(axis = 0)




df_test = pd.read_csv("/kaggle/input/predict-west-nile-virus/test.csv")




df_test.head()




pd.isna(df_test).sum(axis = 0)




df.nunique()




df['Species'].unique()




df.duplicated().sum()




df.shape




Species = pd.get_dummies(df['Species'])
Species




Species['CULEX PIPIENS'] += Species['CULEX PIPIENS/RESTUANS']
Species['CULEX RESTUANS'] += Species['CULEX PIPIENS/RESTUANS']
Species_dummies = Species.drop(columns=['CULEX PIPIENS/RESTUANS'])
Species_dummies.head()




Block41 = df[df['Block'] == 41]
Block41




X = pd.concat([df[['Block','NumMosquitos']], Species_dummies],axis = 1)




#X['Block'] = (X['Block'] - X['Block'].mean())/X['Block'].std()




X.head()




Y = df['WnvPresent']




from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.25, random_state=0)

model = LogisticRegression(solver = 'lbfgs')
model.fit(X_train, y_train)
predicted_classes = model.predict(X_test)




metrics.confusion_matrix(y_test, predicted_classes)




accuracy = metrics.accuracy_score(y_test,predicted_classes)
accuracy




y_train.sum()




y_test.sum()




# Construction du modèle SVM 
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',gamma = 'scale', random_state = 0)
classifier.fit(X_train, y_train)




# Faire de nouvelles prédictions
y_pred = classifier.predict(X_test)

# Matrice de confusion
from sklearn.metrics import confusion_matrix
cm_ker = confusion_matrix(y_test, y_pred)
cm_ker




accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy




from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
clf = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
# Faire de nouvelles prédictions
y_pred = clf.predict(X_test)
# Matrice de confusion
cm_RF = confusion_matrix(y_test, y_pred)
cm_RF




accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy




from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBClassifier
xgb_clf = GradientBoostingClassifier()
xgb_clf.fit(X_train, y_train)
# Faire de nouvelles prédictions
y_pred = xgb_clf.predict(X_test)
# Matrice de confusion
cm_XGB = confusion_matrix(y_test, y_pred)
cm_XGB




accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy




#matrice de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
#plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
 
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()




X = pd.concat([df[['Latitude','Longitude','NumMosquitos']], Species_dummies],axis = 1)




X.head()




X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.25, random_state=0)




X_train




# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[['Latitude','Longitude','NumMosquitos']] = sc.fit_transform(X_train[['Latitude','Longitude','NumMosquitos']])
X_test[['Latitude','Longitude','NumMosquitos']] = sc.transform(X_test[['Latitude','Longitude','NumMosquitos']])




X_train




X_test




from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
clf = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
# Faire de nouvelles prédictions
y_pred = clf.predict(X_test)




#matrice de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
#plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
 
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()




accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy




df['month'] = pd.to_datetime(df['Date']).dt.month




df.head()




X = pd.concat([df[['Latitude','Longitude','NumMosquitos','month']], Species_dummies],axis = 1)




X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.25, random_state=0)




# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[['Latitude','Longitude','NumMosquitos']] = sc.fit_transform(X_train[['Latitude','Longitude','NumMosquitos']])
X_test[['Latitude','Longitude','NumMosquitos']] = sc.transform(X_test[['Latitude','Longitude','NumMosquitos']])




from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
clf = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
# Faire de nouvelles prédictions
y_pred = clf.predict(X_test)




#matrice de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
#plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
 
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()




accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy




if 
