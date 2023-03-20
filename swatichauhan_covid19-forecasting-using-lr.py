#!/usr/bin/env python
# coding: utf-8



pip install pycountry_convert




#Libraries to import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import requests
import sys
from itertools import chain
import pycountry
import seaborn as sns
import pycountry_convert as pc
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV




df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv') 
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')




display(df_train)
display(df_train.describe())
display(df_train.info())




df_train.isnull().sum()




df_train.drop(['Province_State'],axis=1,inplace=True)




df_train.groupby(['Country_Region']).sum()




df_train['Date'] = pd.to_datetime(df_train['Date'], format = '%Y-%m-%d')
df_test['Date'] = pd.to_datetime(df_test['Date'], format = '%Y-%m-%d')




train_date_min = df_train['Date'].min()
train_date_max = df_train['Date'].max()
print('Minimum date from training set: {}'.format(train_date_min))
print('Maximum date from training set: {}'.format(train_date_max))




test_date_min = df_test['Date'].min()
test_date_max = df_test['Date'].max()
print('Minimum date from test set: {}'.format(test_date_min))
print('Maximum date from test set: {}'.format(test_date_max))




#Reading the cumulative cases dataset
covid_cases = pd.read_csv('/kaggle/input/covid19cases/times_series_covid19_cases_country.csv')
covid_cases




covid_cases.drop(['Lat','Long_','Incident_Rate','People_Tested','People_Hospitalized','UID','ISO3'],axis=1,inplace=True)




covid_cases




sns.set_style('darkgrid')
plt.figure(figsize=(15,9))
sns.barplot(x=df_train['Date'],y=df_train['Fatalities'],palette = "YlOrRd")
plt.xticks(rotation=90)
plt.show()




df_train['Death']=(df_train['Fatalities']/df_train['ConfirmedCases'])*100




plt.figure(figsize=(12,6))
plt.plot(df_train['Death'], label = 'Death')
plt.show()




covid_country=df_train.groupby(['Country_Region']).sum()




covid_country.loc['India'].transpose().plot(title='Time Series of confirmend cases of india')

Mortality Rate 


df_train['Mortality Rate']=round((df_train['Death']/df_train['ConfirmedCases'])*100,2)
df_train




plt.figure(figsize=(15,8))
plt.plot(df_train['Mortality Rate'],label='Mortality Rate')
plt.show()




plt.figure(figsize=(15,15))
stats = [df_train.loc[:,['Country_Region','Death']]]
label = ["Death"]
threshold = [10000]
for i, stat in enumerate(stats):
    df_train = stat.groupby(["Country_Region"]).sum()
    df_train = df_countries.sort_values(df_countries.columns[-1],ascending= False)
    others = df_countries[df_countries[df_countries.columns[-1]] < threshold[i] ].sum()[-1]
    df_countries = df_countries[df_countries[df_countries.columns[-1]] > threshold[i]]
    df_countries = df_countries[df_countries.columns[-1]]
    df_countries["others"] = others
    labels = [df_countries.index[i] for i in range(df_countries.shape[0])]
    plt.pie(df_countries, labels=labels, autopct='%1.1f%%')
    plt.title('Pie Chart', fontsize = 18)
    plt.legend()
    plt.show()




df_train




df_train['Days']=df_train.index-df_train.index[0]
df_train




from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression(normalize=True)




x=np.array(df_train['Days']).reshape(-1,1)
x




y=np.array(df_train['Fatalities']).reshape(-1,1)
y




lin_reg.fit(x,y)




df_test['Days']=df_test.index-df_test.index[0]
df_test




predicted_value=lin_reg.predict(np.array(df_test['Days']).reshape(-1,1))
predicted_value




df_test['Fatalities']=predicted_value




df_test




l=np.array(df_train['Days']).reshape(-1,1)
l




m=np.array(df_train['ConfirmedCases']).reshape(-1,1)
m




lin_reg.fit(l,m)




predicted_confirm=lin_reg.predict(np.array(df_test['Days']).reshape(-1,1))
predicted_confirm




submission_dataset = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")




#Adding results to the dataset
submission_dataset['ConfirmedCases'] = predicted_confirm
submission_dataset['Fatalities'] = predicted_value

submission_dataset.head()




#Submitting the dataset
submission_dataset.to_csv("submission.csv" , index = False)






