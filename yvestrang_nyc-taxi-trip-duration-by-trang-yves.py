#!/usr/bin/env python
# coding: utf-8



import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

get_ipython().run_line_magic('matplotlib', 'inline')




train = pd.read_csv ('../input/train.csv')
test = pd.read_csv ('../input/test.csv')




train.head()




train.shape




train.dtypes




train.describe()




train.isna().sum()




#train = train[train['passenger_count']>0]
#train = train[train['passenger_count']<6]




#train.loc[train.trip_duration<4000
#          ,"trip_duration"].hist(bins=120
                                                        )




#train = train[(train['trip_duration'] > 60) & (train['trip_duration'] < 4000*2)]
train['trip_duration'] = np.log(train['trip_duration'].values)




sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)
sns.despine(left=True)
sns.distplot(train['pickup_latitude'].values, label = 'pickup_latitude',color="m",bins = 100, ax=axes[0,0])
sns.distplot(train['pickup_longitude'].values, label = 'pickup_longitude',color="m",bins =100, ax=axes[0,1])
sns.distplot(train['dropoff_latitude'].values, label = 'dropoff_latitude',color="m",bins =100, ax=axes[1, 0])
sns.distplot(train['dropoff_longitude'].values, label = 'dropoff_longitude',color="m",bins =100, ax=axes[1, 1])
plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()




#train = train.loc[train['pickup_longitude']> -80]
#train = train.loc[train['pickup_latitude']< 44]
#train = train.loc[train['dropoff_longitude']> -90]
#train = train.loc[train['dropoff_latitude']> 34]




def haversine(lat1, lon1, lat2, lon2):
   R = 6372800  # Earth radius in meters
   phi1, phi2 = math.radians(lat1), math.radians(lat2)
   dphi       = math.radians(lat2 - lat1)
   dlambda    = math.radians(lon2 - lon1)

   a = math.sin(dphi/2)**2 +        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

   return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))




train["distance"] = train.apply(lambda row: haversine(row["pickup_latitude"], row["pickup_longitude"], row["dropoff_latitude"], row["dropoff_longitude"]), axis=1)
test["distance"]  = test.apply(lambda row: haversine(row["pickup_latitude"], row["pickup_longitude"], row["dropoff_latitude"], row["dropoff_longitude"]), axis=1)




train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

train['hour'] = train.pickup_datetime.dt.hour
train['day'] = train.pickup_datetime.dt.dayofweek
train['month'] = train.pickup_datetime.dt.month
test['hour'] = test.pickup_datetime.dt.hour
test['day'] = test.pickup_datetime.dt.dayofweek
test['month'] = test.pickup_datetime.dt.month
                                                    




y_train = train["trip_duration"] # <-- target
X_train = train[["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","month","hour","day","distance"]] # <-- features

X_testdata = test[["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","month","hour","day","distance"]]









from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import ShuffleSplit
import xgboost as xgb




X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42,test_size= 0.1)




X_train.shape, y_train.shape, X_test.shape, y_test.shape




rfr = RandomForestRegressor(n_estimators=100,min_samples_leaf=5, min_samples_split=50, max_depth=80,verbose=0,max_features="auto",n_jobs=-1)
rfr.fit(X_train, y_train)














train_pred = rfr.predict(X_testdata)




train_pred




len(train_pred)




sample = pd.read_csv('../input/sample_submission.csv')




#my_submission = pd.DataFrame({'id': test.id, 'trip_duration': train_pred})
my_submission = pd.DataFrame({'id': test.id, 'trip_duration': np.exp(train_pred)})




my_submission.to_csv('sub.csv', index=False)

