#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




# below 2 statements are taking 2 much time
#train_df = pd.read_csv('../input/train.csv',nrows=10000000,parse_dates=['key','pickup_datetime'])
#test_df = pd.read_csv('../input/test.csv',parse_dates=['key','pickup_datetime'])
train_df = pd.read_csv('../input/train.csv',nrows=10000000)
test_df = pd.read_csv('../input/test.csv')




train_df.dtypes




test_df.tail(3)




train_df.shape




# This dataset is having large number of rows
train_df.keys()




# as we see that null values are very less as compared to number of rows imported, so we can delete these 69 rows
train_df.isnull().sum()




test_df.isnull().sum().sort_values(ascending=False)




train_df = train_df.dropna(how='any',axis =0)




train_df.isnull().sum()




train_df.shape




test_df.isnull().sum()





train_df['fare_amount'].describe()




# there are 420 rows with fare_amount < 0  which is not possible . so we can delete these rows
train_df[train_df['fare_amount']<0]
         




from collections import Counter

Counter(train_df['fare_amount']<0)




#counter1 = for i in train_df['fare_amount'] 




#so the fare amount minimum is negative which cannot be possible . So this lines may be having wrong data.
# So we need to delete these lines.
train_df = train_df.drop(train_df[train_df['fare_amount'] < 0].index,axis = 0)




train_df.describe()




train_df.shape




# if we look at the passenger_count field then we see that it is coming as 208 which is an outlier . There are 12 rows where passenger count is greater than 10
len(train_df[train_df['passenger_count']>10].index)




len(test_df[test_df['passenger_count']>10].index)




train_df = train_df.drop(train_df[train_df['passenger_count'] > 10].index,axis = 0)




train_df.shape




#now coming to date time columns
train_df['key'] = pd.to_datetime(train_df['key'])
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])




test_df.info()




test_df['key'] = pd.to_datetime(test_df['key'])
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])




train_df.head(2)




#getting the values from date time column
train_df['month'] = train_df.pickup_datetime.dt.month
train_df['year'] = train_df.pickup_datetime.dt.year
train_df['day'] = train_df.pickup_datetime.dt.day
train_df['hour'] = train_df.pickup_datetime.dt.hour
train_df['dayweek'] = train_df.pickup_datetime.dt.dayofweek




train_df.sort_values(by=['year'],ascending=False)




test_df['month'] = test_df.pickup_datetime.dt.month
test_df['year'] = test_df.pickup_datetime.dt.year
test_df['day'] = test_df.pickup_datetime.dt.day
test_df['hour'] = test_df.pickup_datetime.dt.hour
test_df['dayweek'] = test_df.pickup_datetime.dt.dayofweek




#data = [train_df,test_df]
test_df.head(10)




train_df.describe()




# now checking latitude and longitude columns
# latitude range is -90 to 90
# longitude range is -180 to 180
len(train_df[train_df['pickup_latitude'] <-90])




len(test_df[test_df['pickup_latitude'] <-90])




len(train_df[train_df['pickup_latitude'] >90])




len(test_df[test_df['pickup_latitude'] >90])




train_df = train_df.drop(((train_df[train_df['dropoff_latitude']<-90])|(train_df[train_df['dropoff_latitude']>90])).index, axis=0)




train_df = train_df.drop(((train_df[train_df['pickup_latitude']<-90])|(train_df[train_df['pickup_latitude']>90])).index, axis=0)




train_df = train_df.drop(((train_df[train_df['dropoff_longitude']<-180])|(train_df[train_df['dropoff_longitude']>180])).index, axis=0)




train_df = train_df.drop(((train_df[train_df['pickup_longitude']<-180])|(train_df[train_df['pickup_longitude']>180])).index, axis=0)




test_df = test_df.drop(((test_df[test_df['dropoff_latitude']<-90])|(test_df[test_df['dropoff_latitude']>90])).index, axis=0)
test_df = test_df.drop(((test_df[test_df['pickup_latitude']<-90])|(test_df[test_df['pickup_latitude']>90])).index, axis=0)
test_df = test_df.drop(((test_df[test_df['dropoff_longitude']<-180])|(test_df[test_df['dropoff_longitude']>180])).index, axis=0)
test_df = test_df.drop(((test_df[test_df['pickup_longitude']<-180])|(test_df[test_df['pickup_longitude']>180])).index, axis=0)




train_df.describe()




test_df.shape




# using the haversine formula
import math
def haversine(pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude):
    for i in [train_df]:
        delta_latitude = np.radians(np.abs(i['dropoff_latitude'] - i['pickup_latitude']))
        delta_longitude = np.radians(np.abs(i['dropoff_longitude'] - i['pickup_longitude']))
        lat1 = np.radians(i['pickup_latitude'])
        lat2 = np.radians(i['dropoff_latitude'])
        a = np.sin(delta_latitude / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_longitude / 2.0) ** 2
        #c = 2 * a * np.arctan2((np.sqrt(a), np.sqrt(1-a) ))
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = 6371  * c  # distance in kilometres
        i['distance'] = d
    




# using the haversine formula
import math
def haversine_test(pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude):
    for i in [test_df]:
        delta_latitude = np.radians(np.abs(i['dropoff_latitude'] - i['pickup_latitude']))
        delta_longitude = np.radians(np.abs(i['dropoff_longitude'] - i['pickup_longitude']))
        lat1 = np.radians(i['pickup_latitude'])
        lat2 = np.radians(i['dropoff_latitude'])
        a = np.sin(delta_latitude / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_longitude / 2.0) ** 2
        #c = 2 * a * np.arctan2((np.sqrt(a), np.sqrt(1-a) ))
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = 6371  * c  # distance in kilometres
        i['distance'] = d




haversine('pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude')




haversine_test('pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude')




train_df.head(10)




test_df.head(2)




train_df['distance'].describe()






plt.figure(figsize=(15,7))
plt.scatter(x=train_df['passenger_count'], y=train_df['fare_amount'], s=1.5)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')




plt.figure(figsize=(15,8))
plt.hist(train_df['passenger_count'],bins=20)




#maximum frequency is by passengers travelling alone and highest fare also has come from a single passenger
train_df['fare_amount'].describe()
train_df[train_df['fare_amount']>1000]









# if we see the fare_amount then it is 1276 with distance as 0
# there seems to be a problem on the part as all latitude's and longitude's are 0
# so we need to delete the same as it is not appropriate





train_df.head(1)




plt.figure(figsize=(14,6))
plt.scatter(x=train_df['year'],y=train_df['fare_amount'],s=1.5)
plt.xlabel('year')
plt.ylabel('fare')




plt.figure(figsize=(14,6))
plt.hist(train_df['year'],bins=20)
plt.xlabel('year')
plt.ylabel('fare')




# year 2015 has lowest rides but fare amount has increased alot maybe due to inflation




plt.figure(figsize=(15,8))
plt.scatter(x=train_df['month'],y=train_df['fare_amount'],s=1.5)
plt.xlabel('month')
plt.ylabel('fare')




plt.figure(figsize=(15,6))
plt.hist(train_df['month'],bins=12)




# nothing to derive based on month




plt.figure(figsize=(16,6))
plt.scatter(x=train_df['day'],y=train_df['fare_amount'],s=1.5)
plt.xlabel('day')
plt.ylabel('fare_amount')




plt.figure(figsize=(15,6))
plt.hist(train_df['day'],bins=31)




#again there is not much of a difference as 31 is not there in many months so its value is less




plt.figure(figsize=(16,6))
plt.scatter(x=train_df['hour'],y=train_df['fare_amount'],s=1.5)
plt.xlabel('hour')
plt.ylabel('fare_amount')




plt.figure(figsize=(20,10))
plt.hist(train_df['hour'],bins=24)




# hours is giving us some insights as we see from the graph that from 1am to 6 am , rides are very less and from 6 pm to 12 am rides frequency is more.
#fare_Amount is highest at 3 pm




plt.figure(figsize=(16,6))
plt.scatter(x=train_df['dayweek'],y=train_df['fare_amount'],s=1.5)
plt.xlabel('dayweek')
plt.ylabel('fare_amount')




train_df.head(1)




train_df.shape




#  NY is a state which is not located at either 0 longitude or 0 latitude or 180 longitude or 180 latitude
train_df.describe()




#train_df[((train_df['dropoff_longitude'] == 0)&(train_df['pickup_longitude'] == 0)&(train_df['dropoff_latitude'] == 0)&(train_df['pickup_latitude'] == 0)) & (train_df['fare_amount']!=0) ]




#train_df[((train_df['dropoff_longitude'] ==train_df['pickup_longitude'])&(train_df['dropoff_latitude'] == train_df['pickup_latitude'])) & (train_df['fare_amount']!=0) ]






len(train_df[(train_df['distance']==0)&(train_df['fare_amount']==0)])




train_df.shape




train_df=train_df.drop(train_df[(train_df['distance']==0)&(train_df['fare_amount']==0)].index,axis=0)









train_df.shape




train_Df.shape









correlations = train_df.corr()
corrmat = correlations['fare_amount'].sort_values(ascending=False)
corrmat




x = train_df.drop(['key','pickup_datetime', 'fare_amount'], axis=1)
y = train_df['fare_amount']





x.head(1)




y.head(1)






from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=0, test_size=0.25)





from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV




def rmse(model,x,y):
    return(np.sqrt(np.abs(cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = 5))))




from sklearn.ensemble import RandomForestRegressor
forestreg = RandomForestRegressor(max_depth=20, min_sa
                                  mples_split=5, n_estimators=250, random_state=0).fit(x_train, y_train)
forestreg_ypredict = forestreg.predict(x_validate)
forestreg_cvscores = rmse(forestreg, x, y)
print('Random forest regression  and cross validation score is {0}'.format(forestreg_cvscores.mean()*10))




train_df.head(5)




x_test = test_df.drop(['key','pickup_datetime'], axis=1)
y_test['fare_amount'] = forestreg.predict(x_test)
data1  = pd.Dataframe({'Id':test_df['id'],'Fare':y_test['fare_amount']})
data1.to_csv('submission.csv', index=False)
#x_test = x_test[feat[feat['FeatureImportances'] > 0.0001].index]
#x_test.shape
#x_test = scaler.transform(x_test)




#linreg = LinearRegression()
#parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}
#grid_linreg = GridSearchCV(linreg, parameters_lin, verbose=1 , scoring = "r2")
#grid_linreg.fit(x_train, y_train)

#print("Best LinReg Model: " + str(grid_linreg.best_estimator_))
#print("Best Score: " + str(grid_linreg.best_score_))

