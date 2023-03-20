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




import numpy as np
import pandas as pd




train = pd.read_csv('/kaggle/input/flavours-of-physics/training.csv.zip')
train.head()




train.signal.nunique()




train.signal.value_counts()




import seaborn as sns
import matplotlib.pyplot as plt




sns.distplot(train['signal'])




train.info()




train = train.drop(columns=['min_ANNmuon'])




train.columns




test = pd.read_csv('/kaggle/input/flavours-of-physics/test.csv.zip')




test.head()




test.columns




train= train.drop(['production','mass'],axis=1)




train.head()




y = train['signal']




type(y)




train.skew(axis = 0, skipna = True) 




train['DOCAone'] = np.tanh(train.DOCAone)




train.skew(axis = 0, skipna = True) 




train.corr()




sns.heatmap(train.corr(),vmin=0, vmax=1)




for n in range(2):
  fig = plt.figure(figsize=(35,20))

  for i in range(1,26):
    ax = fig.add_subplot(5, 5, i)
    col = train.columns[i + 25*n]
    ax.set_title(col)

    plt.hist([train[train['signal'] == 1][col], train[train['signal'] == 0][col]], bins=50, histtype='stepfilled', color=['r', 'b'], alpha=0.4, label=['signal', 'background'])
    
    if (i == 5): ax.legend()
        
  fig.tight_layout(pad=1, w_pad=1, h_pad=1)
  fig.savefig('hist'+str(n+1)+'.png', dpi=150)




train.columns




X= train.drop(columns=['signal'])




X.head()




X= X.drop(columns=['id'])




X.head()




X.var()




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled= pd.DataFrame(sc.fit_transform(X), columns=X.columns)
X_scaled.var()




x=X_scaled.values




type(x)




X.head()




X_scaled.head()




X_scaled.shape




np.shape(x)




x




y.shape




y=y.values




y=y.reshape(-1,1)




np.shape(y)




y




from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split




X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.score(X_test,y_test)




pip install tpot




from tpot import TPOTClassifier

parameters = {'criterion': ['entropy', 'gini'],
               'max_depth': [2],
               'max_features': ['auto'],
               'min_samples_leaf': [4, 12, 16],
               'min_samples_split': [5, 10,15],
               'n_estimators': [10]}
               
tpot_classifier = TPOTClassifier(generations= 5, population_size= 32, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,mutation_rate=0.9,
                                 config_dict=
                                 {'sklearn.ensemble.RandomForestClassifier': parameters}, 
                                 cv = 4, scoring = 'accuracy')
tpot_classifier.fit(X_train,y_train) 




accuracy = tpot_classifier.score(X_test, y_test)
print(accuracy)




import xgboost as xgb
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,params=params)

xgb_model.fit(X_train,y_train)




xgb_model.score(X_test,y_test)




xgb_model.score(X_train,y_train)




from sklearn.metrics import confusion_matrix
y_pred = xgb_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

