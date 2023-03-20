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




from datetime import datetime
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost
import math

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import tree, linear_model
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt




train= pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
test= pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")




train.head()




test.head()




#print(train.groupby(['Date','Lat']).count().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'])




display(train.head(5))
display(train.describe())
print("Number of Country_Region: ", train['Country_Region'].nunique())
print("Number of Country_Region: ", train['Country_Region'].unique())
print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")
print("Countries with Province/State informed: ", train[train['Province_State'].isna()==False]['Country_Region'].unique())




print("Countries with Province/State informed: ", train[train['Province_State'].isna()==False]['Province_State'].unique())




print(train.groupby(['Date']).mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'][[1,2]])
print(train.groupby('Date').mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'][[4,5,6]])




print(train.groupby(['Date','Country_Region']).mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'])




training_data=train.groupby('Date')['ConfirmedCases','Fatalities'].sum().reset_index()




training_data=train.groupby('Date')['ConfirmedCases','Fatalities'].sum().reset_index()




training_data




y_train = train[["ConfirmedCases", "Fatalities"]]
train = train[["Province_State","Country_Region","Date"]]
X_test_Id = test.loc[:, 'ForecastId']
test = test[["Province_State","Country_Region","Date"]]




#print(train_encoded.count())
print(y_train)




#xyz = train.groupby(['Country_Region']).count().sort_values(by='ConfirmedCases', ascending=False)()
#train.groupby(['Country_Region']).sort_values(by='Date', ascending=False)[:100]




#train.groupby(['Country_Region','col2']).count()

y_train.head()


#train.groupby(['Date','Country_Region'])['count','ConfirmedCases']




#train = train.sort_values(by=['Date','Country_Region'], ascending=True)




#train[train.groupby(['Date','Country_Region'])'count','ConfirmedCases')]




#train['count']=1




#train['count'] = train.groupby(['Date','Country_Region'])['ConfirmedCases'].apply()#




EMPTY_VAL = "EMPTY_VAL"

def fillState(state, country):
    if state == EMPTY_VAL: return country
    return state




from sklearn.preprocessing import LabelEncoder

print("fill blanks and add region for counting")
#train.fillna(' ',inplace=True)
#train['Lat']=train['Province_State']+train['Country_Region']
#train.drop('Province_State',axis=1,inplace=True)
#train.drop('Country_Region',axis=1,inplace=True)


cols = ['ConfirmedCases', 'Fatalities']
index_split = train.shape[0]

full_df = pd.concat([train,test],sort=False)
full_df.fillna(' ',inplace=True)
#full_df['Lat']=full_df['Province_State']+full_df['Country_Region']
#X_xTrain['State'].fillna(EMPTY_VAL, inplace=True)
full_df['Province_State'] = full_df.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)



#full_df.drop('Province_State',axis=1,inplace=True)
#full_df.drop('Country_Region',axis=1,inplace=True)
display(full_df.head())
#full_df = pd.concat([train.drop(cols, axis=1))
#full_df.Date = full_df.Date.astype('int64')
full_df['Mon'] = full_df['Date'].apply(lambda x: int(x.split('-')[1]))
full_df['Day'] = full_df['Date'].apply(lambda x: int(x.split('-')[2]))
full_df['serial'] = full_df['Mon'] * 30 + full_df['Day']
full_df['serial'] = full_df['serial'] - full_df['serial'].min()
full_df.Date = pd.to_datetime(full_df.Date)
full_df['Date'] = full_df['Date'].apply(pd.to_datetime)

full_df['day_of_week'] = full_df['Date'].apply(lambda ts: ts.weekday()).astype('int')
full_df['month'] = full_df['Date'].apply(lambda ts: ts.month)
full_df['day'] = full_df['Date'].apply(lambda ts: ts.day)
full_df.loc[:, 'Date'] = full_df.Date.dt.strftime("%m%d")
full_df['Date']  = full_df['Date'].astype(int)

#full_df.drop(['Province_State','Country_Region'],axis=1, inplace= True )
full_df.drop(['Mon','Day'],axis=1, inplace= True )
#full_df.drop(['Date', 'Province_State','Country_Region','Mon','Day'],axis=1, inplace= True )
display(full_df.dtypes)
display(full_df.head())




le = LabelEncoder()
def CustomLabelEncoder(df):
    for c in df.columns:
        if df.dtypes[c] == object:
            le.fit(df[c].astype(str))
            df[c] = le.transform(df[c].astype(str))
    return df





full_df_encoded = CustomLabelEncoder(full_df)

train_encoded = full_df[:index_split]
test_encoded= full_df[index_split:]




train_encoded.count()




test_encoded.iloc[250:350,:]




train_encoded.tail()




test_encoded.head()




train.head()




full_df_encoded.head(10)




train_encoded.info()




test_encoded




from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(train_encoded, y_train[['ConfirmedCases']],test_size=0.2, random_state=48)

X_train2, X_test2, y_train2, y_test2 = train_test_split(train_encoded, y_train[['Fatalities']] ,test_size=0.2, random_state=48)




y_train1,y_test1,y_train2,y_test2




# features that will be used in the model
y1 = y_train[['ConfirmedCases']]
y2 = y_train[['Fatalities']]




from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor(random_state = 48) 
from sklearn.metrics import mean_squared_error, mean_absolute_error




regressor.fit(X_train1,y_train1)
#predict_dt1 = regressor.predict(X_test1)
#predict_dt1 = pd.DataFrame(predict_dt1)
#predict_dt1.columns = ["ConfirmedCases"]
#print(predict_dt1)
display(regressor.score(X_test1,y_test1))

predict_dt1 = regressor.predict(X_test1)
#predict_dt1 = pd.DataFrame(predict_dt1)
#predict_dt1.columns = ["ConfirmedCases"]
predict_dt1 = predict_dt1.astype(int)
predict_dt1




regressor.fit(train_encoded, y_train[['ConfirmedCases']])
#predict_dt1 = regressor.predict(X_test1)
#predict_dt1 = pd.DataFrame(predict_dt1)
#predict_dt1.columns = ["ConfirmedCases"]
#print(predict_dt1)
display(regressor.score(X_test1,y_test1))

predict_dt1 = regressor.predict(test_encoded)
#predict_dt1 = pd.DataFrame(predict_dt1)
#predict_dt1.columns = ["ConfirmedCases"]
predict_dt1 = predict_dt1.astype(int)
predict_dt1




regressor.fit(train_encoded, y_train[['Fatalities']])
#predict_dt2 = regressor.predict(X_test2)
#predict_dt2 =predict_dt1.astype(int)
#predict_dt2 = pd.DataFrame(predict_dt2)
#predict_dt2.columns = ["Fatalities"]
predict_dt2 = regressor.predict(test_encoded)
#predict_dt1 = pd.DataFrame(predict_dt1)
#predict_dt1.columns = ["ConfirmedCases"]
predict_dt2 = predict_dt2.astype(int)
predict_dt2




print(X_test_Id,predict_dt1,predict_dt2)




sub = pd.DataFrame({'ForecastId':X_test_Id,'ConfirmedCases': predict_dt1, 'Fatalities': predict_dt2})

sub.ForecastId = sub.ForecastId.astype('int') 
sub.head()

sub.to_csv('submission.csv', index=False)




import lightgbm as lgb




SEED = 42
params = {'num_leaves': 8,
          'min_data_in_leaf': 5,  # 42,
          'objective': 'regression',
          'max_depth': 8,
          'learning_rate': 0.02,
          'boosting': 'gbdt',
          'bagging_freq': 5,  # 5
          'bagging_fraction': 0.8,  # 0.5,
          'feature_fraction': 0.8201,
          'bagging_seed': SEED,
          'reg_alpha': 1,  # 1.728910519108444,
          'reg_lambda': 4.9847051755586085,
          'random_state': SEED,
          'metric': 'mse',
          'verbosity': 100,
          'min_gain_to_split': 0.02,  # 0.01077313523861969,
          'min_child_weight': 5,  # 19.428902804238373,
          'num_threads': 6,
          }




y_test1









sub.head()









#subb.ForecastId = subb.ForecastId.astype('int')
#sub.tail()
#sub.to_csv('submission.csv', index=False)




xgb = xgboost.XGBRegressor()




xgb1=xgb.fit(X_train1,y_train1)
predictions = xgb1.predict(X_test1)
print(explained_variance_score(predictions,y_test1))

predict_xgb1 = xgb1.predict(test_encoded)
#predict_xg2 = pd.DataFrame(predict_dt2)
#predict_xg2.columns = ["Fatalities"]
print(xgb1.score(X_test1,y_test1))
print(explained_variance_score(predictions,y_test1))
print(mean_squared_error(y_test1,predictions))
print(r2_score(y_test1,predictions))




xgb1.score(X_test1,y_test1)




xgb2=xgb.fit(X_train2,y_train2)
predictions = xgb2.predict(X_test2)
print(explained_variance_score(predictions,y_test2))

predict_xgb2 = xgb2.predict(test_encoded)
#predict_xg2 = pd.DataFrame(predict_dt2)
#predict_xg2.columns = ["Fatalities"]
print(xgb2.score(X_test2,y_test2))
print(explained_variance_score(predictions,y_test2))
print(mean_squared_error(y_test2,predictions))
print(r2_score(y_test2,predictions))




rf = RandomForestRegressor()
rf1 = rf.fit(X_train1,y_train1)

predictions = rf1.predict(X_test1)
print(explained_variance_score(predictions,y_test1))

predict_rf1 = rf1.predict(test_encoded)
#predict_xg2 = pd.DataFrame(predict_dt2)
#predict_xg2.columns = ["Fatalities"]
print(rf1.score(X_test1,y_test1))
print(explained_variance_score(predictions,y_test1))
print(mean_squared_error(y_test1,predictions))
print(r2_score(y_test1,predictions))




from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

regr = linear_model.LinearRegression()
regr1 = regr.fit(X_train1, y_train1)
print(regr.predict(X_test1))

predictions = regr1.predict(X_test1)
print(explained_variance_score(predictions,y_test1))

predict_rf1 = regr1.predict(test_encoded)
#predict_xg2 = pd.DataFrame(predict_dt2)
#predict_xg2.columns = ["Fatalities"]
print(regr1.score(X_test1,y_test1))
print(explained_variance_score(predictions,y_test1))
print(mean_squared_error(y_test1,predictions))
print(r2_score(y_test1,predictions))




print("RMSE: %.2f"
      % math.sqrt(np.mean((regr.predict(X_test1) - y_test1) ** 2)))




from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
# Fit the classifier to the data
knn1 = knn.fit(X_train1,y_train1.values.ravel())

predictions = knn1.predict(X_test1)
print(explained_variance_score(y_test1,predictions))
predict_knn1 = knn1.predict(test_encoded)
#predict_xg2 = pd.DataFrame(predict_dt2)
#predict_xg2.columns = ["Fatalities"]
print(knn1.score(X_test1,y_test1))
print(explained_variance_score(y_test1,predictions))
print(mean_squared_error(y_test1,predictions))
print(r2_score(predictions,y_test1))




from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
# Fit the classifier to the data
knn1 = knn.fit(X_train2,y_train2.values.ravel())

predictions = knn1.predict(X_test2)
print(explained_variance_score(y_test2,predictions))
predict_knn1 = knn1.predict(test_encoded)
#predict_xg2 = pd.DataFrame(predict_dt2)
#predict_xg2.columns = ["Fatalities"]
print(knn1.score(X_test2,y_test2))
print(explained_variance_score(y_test2,predictions))
print(mean_squared_error(y_test2,predictions))
print(r2_score(predictions,y_test2))




knn.score(X_test2, y_test2)




#xgb1=xgb.fit(X_train1,y_train1)
predictions = knn1.predict(test_encoded)
predictions1 = knn1.predict(X_test1)
print(explained_variance_score(predictions1,y_test1))
print(predictions1)




predictions




knn.fit(X_train1,y_train1)
#predict_dt1 = regressor.predict(X_test1)
#predict_dt1 = pd.DataFrame(predict_dt1)
#predict_dt1.columns = ["ConfirmedCases"]
#print(predict_dt1)
knn.score(X_test1,y_test1)

predict_dt1 = knn.predict(X_test1)
#predict_dt1 = pd.DataFrame(predict_dt1)
#predict_dt1.columns = ["ConfirmedCases"]
predict_dt1 = predict_dt1.astype(int)
predict_dt1




knn.fit(train_encoded, y_train[['ConfirmedCases']])
#predict_dt1 = regressor.predict(X_test1)
#predict_dt1 = pd.DataFrame(predict_dt1)
#predict_dt1.columns = ["ConfirmedCases"]
#print(predict_dt1)
knn.score(X_test1,y_test1)

predict_dt1 = knn.predict(test_encoded)
#predict_dt1 = pd.DataFrame(predict_dt1)
#predict_dt1.columns = ["ConfirmedCases"]
predict_dt1 = predict_dt1.astype(int)
predict_dt1




knn.fit(train_encoded, y_train[['Fatalities']])
#predict_dt2 = regressor.predict(X_test2)
#predict_dt2 =predict_dt1.astype(int)
#predict_dt2 = pd.DataFrame(predict_dt2)
#predict_dt2.columns = ["Fatalities"]
predict_dt2 = knn.predict(test_encoded)
#predict_dt1 = pd.DataFrame(predict_dt1)
#predict_dt1.columns = ["ConfirmedCases"]
predict_dt2 = predict_dt2.astype(int)
predict_dt2




print(X_test_Id,predict_dt1,predict_dt2)




#sub = pd.DataFrame({'ForecastId':X_test_Id,'ConfirmedCases': predict_dt1, 'Fatalities': predict_dt2})

#sub.ForecastId = sub.ForecastId.astype('int')
#sub.head()
#sub.to_csv('submission.csv', index=False)




pd.DataFrame(predictions)




from sklearn.model_selection import GridSearchCV
#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=10,n_jobs=-1, verbose=3)
#fit model to data
knn_gscv.fit(X_train1, y_train1.values.ravel())









#check top performing n_neighbors value
knn_gscv.best_params_




#check mean score for the top performing value of n_neighbors
knn_gscv.best_score_




#xgb1=xgb.fit(X_train1,y_train1)
predictions = knn_gscv.predict(test_encoded)
predictions1 = knn_gscv.predict(X_test2)
print(explained_variance_score(predictions1,y_test2))
print(predictions)




from sklearn.neighbors import KNeighborsRegressor
# Create KNN classifier
knn = KNeighborsRegressor()
# Fit the classifier to the data
knn1 = knn.fit(X_train1,y_train1) 




#knn.fit(train_encoded, y_train[['ConfirmedCases']])
#predict_dt1 = regressor.predict(X_test1)
#predict_dt1 = pd.DataFrame(predict_dt1)
#predict_dt1.columns = ["ConfirmedCases"]
#print(predict_dt1)
knn1.score(X_test1,y_test1)

predict_dt1 = knn1.predict(X_test1)
#predict_dt1 = pd.DataFrame(predict_dt1)
#predict_dt1.columns = ["ConfirmedCases"]
predict_dt1 = predict_dt1.astype(int)
#predict_dt1




knn1.score(X_test1,y_test1)




y_test1




from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression
from numpy import testing




ensemble = BaggingRegressor(base_estimator=SVR(),
                                max_features=1,
                                bootstrap_features=False,
                                random_state=0).fit(X_train1, y_train1.values.ravel())




svr = SVR()
svr.fit(X_train1, y_train1.values.ravel())




#svr.score(X_test1,y_test1)
svr.predict(X_test1)




y_test1




dt = DecisionTreeRegressor()
dt.fit(X_train1, y_train1.values.ravel())
dt.score(X_test1,y_test1)




ensemble.score(X_test1, y_test1)




def test_bootstrap_features():
    # Test that bootstraping features may generate dupplicate features.
    rng = 48
    X_train1, X_test1, y_train1, y_test1 = train_test_split(train_encoded,
                                                        y_train[['ConfirmedCases']],
                                                        random_state=rng)

    ensemble = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                                max_features=1.0,
                                bootstrap_features=False,
                                random_state=rng).fit(X_train1, y_train1.values.ravel())

    return ensemble




for features in ensemble.estimators_features_:
        assert_equal(boston.data.shape[1], np.unique(features).shape[0])

    ensemble = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                                max_features=1.0,
                                bootstrap_features=True,
                                random_state=rng).fit(X_train1, y_train1)

    for features in ensemble.estimators_features_:
        assert_greater(boston.data.shape[1], np.unique(features).shape[0]) 




np.testing.assert_array_equal




model=test_bootstrap_features()




model.get_params




Ridge = linear_model.Ridge()
Ridge.fit(X_train1,y_train1)




Ridge.score(X_train1,y_train1)

