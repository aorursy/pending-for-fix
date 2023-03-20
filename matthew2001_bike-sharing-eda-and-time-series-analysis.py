#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 




plt.rcParams["figure.figsize"] = (10, 6) # (w, h)














ls ../input/




data = pd.read_csv('../input/train.csv', parse_dates = [0]) # convert to datetime object 




data.shape




data.columns




# renaming columns to non pandas "reserved" object attributes (to avoid mistakes / confusion)
data.rename(columns={'datetime':'logged',
                          'count':'cnt'}, 
                 inplace=True)









# ordinal value which represents different levels of precipitation. rename to more appropriate name 
data.rename(columns=
            {'weather':'precipitation'          
            },
            inplace=True)




data.head(20)




data.describe()









#check for any missing values
data.isnull().values.any()




# nice!




#however, windspeed has many zeros. will check that out later 









# making sure a holiday is not a working day
len(data[((data.holiday & data.workingday) == 1)])




# todo: still there might be correlation between holiday and workingday. will check this out 









data.info()




'''
logged : datetime 
holiday : categorical
workingday : categorical
precipitation : ordinal . represents the "level" of precipitation
season : ordinal . one season is "greater" than another
temp : numerical
atemp : numerical
humidity : numerical
windspeed : numerical
casual : numerical
registered :numerical
cnt : numerical
'''









data.info()









dataOrig = data.copy() # deep copy
#data = dataOrig.copy()









colsToDrop = []









# unravel date time data and add them as as features 
data['hour'] = data.logged.dt.hour # hour in day
data['day'] = data.logged.dt.day # day in month
data['dayofweek'] = data.logged.dt.dayofweek # day of week 
data['month'] = data.logged.dt.month # month in year
data['year'] = data.logged.dt.year # year within the two year timespan 









# drop unraveled datetime object
colsToDrop.append('logged')









data.head(20)









data.hist(figsize = (25,22), bins=25);




for col in data.columns:
    print(data[col].value_counts().head(20))
    print()









sns.boxplot(data['cnt']).set_title('Cnt distribution')




# cnt has many outlier datapoints beyond the outer quartile 









sns.boxplot(data['season'],data['cnt']).set_title('Cnt distribution across season')




# season 1 (spring) has a significant drop in count 









sns.boxplot(data['precipitation'],data['cnt']).set_title('Cnt distribution across precipitation')




# amount of rentals drop as it gets more rainy . 









sns.boxplot(data['hour'],data['cnt']).set_title('Cnt distribution across hour')




# peak hours are during commuting hours: between 7-8 in the morning, and 17-18 in evening









dateCols = ['hour','day','dayofweek','month','year']
nonNumCols = ['season','holiday', 'workingday', 'precipitation']
continuousCols = ['temp','atemp','humidity','windspeed']
targetCols = ['casual','registered','cnt']














for col in continuousCols:
    sns.boxplot(col, data=data,orient='v') 
    plt.show()




'''
temp :  distributed normal
atemp : distributed normal
windspeed : has many outliers 
humidity : few outliers 
'''









# windspeed count plot 




sns.countplot(data['windspeed']) 




'''
there are many zero values

logically, if I were renting a bike, the wind levels won't matter so much (as long there are not tornado / hurricane winds)
and it wont really wouldn't be part of my decision to rent a boke 

will need to check for correlation to decide if this feature is kept or not 

'''














# bar plot with sum estimator 
for col in dateCols + nonNumCols:
    sns.barplot(x=col, y='cnt', data=data, estimator = sum)
    plt.show()




# bar plot with mean estimator
# with mean can relatively compare between classes despite high imbalances

for col in dateCols + nonNumCols:
    sns.barplot(x=col, y='cnt', data=data) #estimator = mean (default param)
    plt.show()




'''
1. summer months had the highest amount of rentals  
2. cnt varies well with: hour, month, year, season, precipitation levels 


todo: check if day in month is correlated to day in week 
'''









fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.set_title('Weekday - Time of Day')

days = {0: 'sunday', 1:'monday', 2:'tuesday',3:'wednesday',4:'thursday',5:'friday',6:'saturday'}

for day in range(7): 
    datetimeGroup = data[data['dayofweek'] == day].groupby(['hour']).mean().reset_index() # reset index turns index back into col after groupby turned it into an index 
    ax.plot(datetimeGroup.hour,datetimeGroup.cnt, label = days[day])

ax.legend(loc=2)
fig.show()




'''
we now see that the high usage during commuting hours is not during friday+saturday

friday and saturday had high usages during the morning-afternoon (10-16 oclock)
'''














fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.set_title('Season - Time of Day')

seasons = {0: 'spring', 1:'summer', 2:'fall',3:'winter'}


for season in range(4): 
    datetimeGroup = data[data['season'] == season].groupby(['hour']).mean().reset_index() # reset index turns index back into col after groupby turned it into an index 
    ax.plot(datetimeGroup.hour,datetimeGroup.cnt, label = seasons[season])

ax.legend(loc=2)
fig.show()




'''
we see the same  amounts of rental peaks during those certain hours of the day, just a less amount of rentals in the summer season
'''









fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.set_title('Customer Type - Time of Day')

customers = {0: 'casual', 1:'registered'}

for customer in range(len(customers)): 
    datetimeGroup = data.groupby(['hour']).mean().reset_index() # reset index turns index back into col after groupby turned it into an index 
    ax.plot(datetimeGroup.hour,datetimeGroup[customers[customer]], label = customers[customer])

ax.legend(loc=2)
fig.show()




# registered users alone are the ones who ride during the peak commuting hours, not casual users
# specifally, the casual users ride on the off hours of the peak users.
# could be that the registered users are registered since they use this only to commute, as the more ideal tiem to ride
# seems to be the middle of the day

#also, most riding is done during the second peak (evening)














# 24 hour window 




data['cnt'].rolling(24).mean().plot()
plt.show()




# can see the spikes which are the changes during the different days of the week as discussed . 




#zooming out 




# one month window




daysPerMonth = len(data.day.unique())




data['cnt'].rolling(24*daysPerMonth).mean().plot()
plt.show()









#each season (each 3 months)
# https://www.google.com/search?q=how+many+months+a+season&oq=how+many+months+a+season




#there are 19 days a month




data['cnt'].rolling(24*daysPerMonth*3).mean().plot()
plt.show()




# as seen from the daily, monthly, and seasonal rolling windows, the count grows over each season of time




# since season already correlates to the unraveled datetime features, no need to keep as a feature, it will already be incorporated 









corrMatrix = data.corr(method='spearman')

fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(corrMatrix,square = False,annot =True,cmap='Spectral', ax=ax)

plt.show()




# pearson
corrMatrix = data.corr(method='pearson')
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(corrMatrix,square = False,annot =True,cmap='Spectral', ax=ax)

plt.show()




# windspeed has a very low correlation with cnt
colsToDrop.append('windspeed')









#logically there is a high correlation between month and season 




colsToDrop.append('season')









# hifh corr between cnt, registered and casual. makes sense since they are leakage variables! 




(data['cnt'] - data['registered']).equals(data['casual'])




#bingo!
#  drop registered and casual
colsToDrop.append('registered')
colsToDrop.append('casual')









# temperature (both temp and atemp) has a corr with cnt. not very high  but still significant enough




# humidiy has a neg corr with count. not very high  but still significant enough









# temp is also hightly correlated with atemp




tempDiff = (data['atemp'] - data['temp'])




print('mean: ' + str(tempDiff.mean()))
print('median: ' + str(tempDiff.median()))
print('mode: ' + str(tempDiff.mode()[0]))




tempDiff.hist(bins=50)




tempDiff.plot()




tempDiff.plot(kind='box')




data['atemp'].max()




data['temp'].max()




# temp diff does have a few outliers due to atemp having a few spikes, yet temp and atemp are highly correleated and very close in value. so will simply just drop one of them
# decided to drop atemp due to the outliers  




colsToDrop.append('atemp')









#highly neg correlated with day of week and workingday
# working day: whether the day is neither a weekend nor holiday. that is already derived from day of week and holiday
colsToDrop.append('workingday')









colsToDrop









data.columns









#todo: if have time, explore methods to remove corrlelated dimensions such as pca, knn+pca, lasso, tsne...









# setting the target variable: given an order, predict the amount of rentals there will be that day




data.head()




years = data.logged.dt.year.unique()
months = data.logged.dt.month.unique()
days = data.logged.dt.day.unique()




years




months




days









data.head(10)




#dataOrig = data.copy()









# alrady accomplished this with the datagrouping / aggregation 

'''
data["cntDay"] = np.nan

for y in years:
    for m in months:
        for d in days:
            selection = (data['year'] == y) & (data['month'] == m) & (data['day'] == d)
            data.loc[selection, 'cntDay'] = (data[ selection ])['cnt'].sum()
  '''          




#making sure data is sorted properly 
data.sort_index(ascending=True).equals(data)




dataset = data.copy()




dataset.head(2)




dataset.columns




colsToDrop




dataset = dataset.drop(colsToDrop,axis=1)




dataset.columns




dataset = dataset.groupby(['year','month','day']).agg({'precipitation':'mean','temp':'mean','holiday':'mean','humidity':'mean','cnt':'sum'}).reset_index()




dataset.head()




dataset['precipitation'] = round(dataset['precipitation'])
dataset['humidity'] = round(dataset['humidity'])
dataset['temp'] = round(dataset['temp'],2)




dataset.head(10)




dataset['tomorrowCnt'] = dataset['cnt'].shift(-1)




dataset.head(10)




#making sure the only thing na is the last col due to the upshift 




dataset.isna().sum() #




dataset = dataset.dropna()




dataset.isna().sum() #









dataset.head(20)




target = dataset.tomorrowCnt




tomorrowCntMean = target.mean()




tomorrowCntMean




dataset = dataset.drop(['cnt','tomorrowCnt'],axis=1)




dataset.columns














dataset.head()




from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()




dataBeforeNorm = dataset.copy()




#dataset = dataBeforeNorm.copy()




#commenting out since got better results not scaling 

#dataset = pd.DataFrame(min_max_scaler.fit_transform(dataset), columns=dataset.columns, index=dataset.index)
#dataset.head()  




for col in dataset.columns:
    print(dataset[col].value_counts().head())
    print()














from sklearn.model_selection import train_test_split




import xgboost as xgb









# we should split this into train, validation, test but i understood from the problem to only split it into train and test 




#For training, use 90% of the data and 10% for test
X_train, X_test, y_train, y_test  = train_test_split(dataset, target, test_size=0.1, random_state=26)
dm_train = xgb.DMatrix(X_train, label=y_train)
dm_test = xgb.DMatrix(X_test, label=y_test)




'''
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    assert len(preds) == len(labels)
    #labels = labels.tolist()
    #preds = preds.tolist()

    return 'error', mean_squared_error(preds)/tomorrowCntMean > 1
'''




clParams = {}
clParams['objective'] = 'reg:linear'
#clParams['eval_metric'] = 'mae' # 'error'

clParams['eta'] = 0.1
clParams['max_depth'] = 8

watchlist = [(dm_train, 'train'), (dm_test, 'test')]

#cl = xgb.train(clParams, dm_train, 10000, watchlist, feval = evalerror, early_stopping_rounds=500, maximize=False, verbose_eval=10)
cl = xgb.train(clParams, dm_train, 10000,  watchlist, early_stopping_rounds=500, maximize=False, verbose_eval=10)









xgb.plot_importance(cl)














# Prediction on test 




def MeasurePerf(preds,y_test):
    
    predsMse = mean_squared_error(y_test,preds)
    meanPreds = np.full(len(preds),tomorrowCntMean)
    meanMse = mean_squared_error(y_test,meanPreds)
    
    res = predsMse / meanMse
    print('predMse/meanMse: '+ str(res))
    
    if (res>=1): # if it equals 1, why train a whole model when u can use average 
        print('\tbetter to use average')
    else:
        print('\tgood prediction')
        
    
    




def CalculateCompanyGain(ordered,actual): 
    
    extraOrders =  ordered.sum() - actual.sum()
    print("extra orders: " + str(extraOrders))
    
    gain = actual.sum()*15 - ordered.sum()*10  + (extraOrders<0)*extraOrders*100 # supplier charges 10 per vehicle and registered pays 15

    gain = round(gain)
    print('gain: ' + str(gain))    
    return gain









from sklearn.metrics import mean_squared_error




preds = cl.predict(dm_test)









MeasurePerf(preds,y_test)









# company gains with forcasted 
gainsWithForcasted = CalculateCompanyGain(preds,y_test)




# company gains with mean
gainsWithAverage = CalculateCompanyGain(np.full(len(preds),tomorrowCntMean),y_test)




# model doesn't seem to put enough of a penalty on ordering less than required. this is bad since it's more expensive to lose
# a customer then to order extra. the loss function is root mean squared error which substracts the difference between them and squares it
# this gives it equal penalties to under and over ordering. 














# did we make more money with forcased or average?

gainsWithForcasted - gainsWithAverage > 0









# sample test 




sampleX = X_train.head(1)
sampleY = y_train.head(1)




dm_sample = xgb.DMatrix(sampleX, label=sampleY)




samplePreds = cl.predict(dm_sample)




MeasurePerf(samplePreds,sampleY)














'''

I would have some threshold defined which marks when we are running low on bikes.

Once that threshold is hit, if a registered customer orders, i will supply him. 

However if a casual user orders, I would need to check the time in the day.
If it's past the second commuting peak, I will grant the casual customer his order.
If it's before the second commuting peak, I will deny the casual customer's order, since I am expecting a large amount of orders
from the registered customers during the next peak. 
(Although the casual users pay more money and their peak time is between the two commuting peaks,
based of the graphs it seems the high volumes of the registered users during their peak hours can generate a higher profit.)

Optionally, more tresholds can be set defining the policy of before the first commuting peak or in between peaks
'''










'''

I would use a custom loss function instead of mse. This was trained on rmse which 
gives an loss of an equal balance to more bikes predicted or less bikes predicted

Instead, we need to give a higher penalty loss when less than the needed amount of bikes are perdicted, 
(since being short of bikes causes the company to suffer an expensive fee of losing customers),
and will penalize less for ordering too many bikes

I'd try a loss of: (actual.sum()*15 - forcasted.sum()*10  + (extraOrders<0)*extraOrders*100 )^2
it is squared so that it will be convex function and have a min 


'''

