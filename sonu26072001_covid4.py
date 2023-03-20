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




df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")




df.head()




len(df['Province_State'].unique())




def create_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df




def categoricalToInteger(df):
    from sklearn.preprocessing import OrdinalEncoder
    #convert NaN Province State values to a string
    df.Province_State.fillna('NaN', inplace=True)
    #Define Ordinal Encoder Model
    oe = OrdinalEncoder()
    df[['Province_State','Country_Region']] = oe.fit_transform(df.loc[:,['Province_State','Country_Region']])
    return df









df.columns




df['Date'] = pd.to_datetime(df['Date'], errors='coerce')




df = create_features(df)




df.head()




df['Country_Region'].value_counts(sort=False)




df = categoricalToInteger(df)




df.head()




df.head()









df




df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')




len(df)




df_test['Date'] = pd.to_datetime(df_test['Date'], errors='coerce')




df_test=create_features(df_test)




df_test




df_test = categoricalToInteger(df_test)




df_test.head()




from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
column_names = ['ForecastId','ConfirmedCases','Fatalities']
preds = []
for country in df.Country_Region.unique():
    df_train_1 = df[df['Country_Region'] == country]
    for province in df_train_1.Province_State.unique():
        df_train2 = df_train_1[df_train_1['Province_State']==province]
        columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','ConfirmedCases','Fatalities']
        df_train_2 = df_train2[columns]
        train = df_train_2.values
        X = train[:,:-2]
        y = train[:,-2:]
        xg_reg_1 = xgb.XGBRegressor(n_estimators=1000)
        xg_reg_1.fit(X,y[:,0])
        xg_reg_2 = xgb.XGBRegressor(n_estimators=1000)
        xg_reg_2.fit(X,y[:,1])
        df_test_1 = df_test[(df_test['Country_Region']==country) & (df_test['Province_State']==province)]
        s = df_test_1['ForecastId']
        X_test = df_test_1[columns[:-2]].values
        pred_1 = xg_reg_1.predict(X_test)
        pred_2 = xg_reg_2.predict(X_test)
        preds.append([s,pred_1,pred_2])




len(preds)




preds[0]




preds_ = np.array(preds)




preds_.shape




preds_1 = preds_.reshape(3,313*43)




preds_1[0] = preds_1[0].astype(np.int64)
preds_1[0][0:10]




preds_1[0].dtype




for i in range(313):
    if preds_[i][0]




preds[0][0]




preds_1[0] = preds_1[0].astype(int)




len(np.unique(preds_1[0].astype(int)))




submission = []
df_train = df
import xgboost as xgb
#Loop through all the unique countries
for country in df_train.Country_Region.unique():
    #Filter on the basis of country
    df_train1 = df_train[df_train["Country_Region"]==country]
    #Loop through all the States of the selected country
    for state in df_train1.Province_State.unique():
        columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','ConfirmedCases','Fatalities']
        #Filter on the basis of state
        df_train2 = df_train1[df_train1["Province_State"]==state]
        df_train2 = df_train2[columns]
        #Convert to numpy array for training
        train = df_train2.values
        #Separate the features and labels
        X_train, y_train = train[:,:-2], train[:,-2:]
        #model1 for predicting Confirmed Cases
        model1 = xgb.XGBRegressor(n_estimators=1000)
        model1.fit(X_train, y_train[:,0])
        #model2 for predicting Fatalities
        model2 = xgb.XGBRegressor(n_estimators=1000)
        model2.fit(X_train, y_train[:,1])
        #Get the test data for that particular country and state
        df_test1 = df_test[(df_test["Country_Region"]==country) & (df_test["Province_State"] == state)]
        #Store the ForecastId separately
        ForecastId = df_test1.ForecastId.values
        #Remove the unwanted columns
        df_test2 = df_test1[columns[:-2]]
        #Get the predictions
        y_pred1 = model1.predict(df_test2.values)
        y_pred2 = model2.predict(df_test2.values)
        #Append the predicted values to submission list
        for i in range(len(y_pred1)):
            d = {'ForecastId':ForecastId[i], 'ConfirmedCases':y_pred1[i], 'Fatalities':y_pred2[i]}
            submission.append(d)




submission




df_submit = pd.DataFrame(submission)
df_submit.to_csv(r'submission.csv', index=False)






