#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')
store= pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
test= pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')


# In[4]:


print(data.shape)
print(store.shape)


# In[5]:


data.head()


# In[6]:


store.head()


# In[7]:


data.info()
# data.dtypes


# In[8]:


data.describe(include='object')


# In[9]:


data.describe()[['Sales','Customers']]


# In[10]:


data.describe()[['Sales','Customers']].loc['mean']


# In[11]:


data.describe()[['Sales','Customers']].loc['min']


# In[12]:


data.describe()[['Sales','Customers']].loc['max']


# In[13]:


data.Store.nunique()


# In[14]:


data.head()
data.Store.value_counts().head(50).plot.bar()


# In[15]:


data.Store.value_counts().tail(50).plot.bar()


# In[16]:


data.Store.value_counts()


# In[17]:


data.DayOfWeek.value_counts()


# In[18]:


data.Open.value_counts()


# In[19]:


data.Promo.value_counts()


# In[20]:


data['Date']=pd.to_datetime(data['Date'],format='%Y-%m-%d')
store_id= data.Store.unique()[0]
print(store_id)
store_rows=data[data['Store']==store_id]
print(store_rows.shape)
# store_rows.resample('1D',on='Date')['Sales'].sum().plot.line(figsize=(14,4))


# In[21]:


# store_rows[store_rows['Sales']==0]


# In[22]:


test['Date']=pd.to_datetime(test['Date'],format='%Y-%m-%d')
store_test_rows = test[test['Store']==store_id]
store_test_rows['Date'].min(),store_test_rows['Date'].max()

store_test_rows
# In[23]:


store_rows['Sales'].plot.hist()
# it is slightly skewed.


# In[24]:


data['Sales'].plot.hist()
# it is slightly skewed.


# In[25]:


store.head()


# In[26]:


# store.isna.sum()


# In[27]:


store_id=store[store['Store']==1].T


# In[28]:


store[~store['Promo2SinceYear'].isna()].iloc[0]


# In[29]:


# Method1
store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
store['Promo2SinceWeek']= store['Promo2SinceWeek'].fillna(0)
store['Promo2SinceYear']= store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])
store['PromoInterval']= store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0]) 

store['CompetitionDistance']=store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())
store['CompetitionOpenSinceMonth']= store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])
store['CompetitionOpenSinceYear']= store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])
store.isna().sum()
      


# In[30]:


data_merged = data.merge(store, on='Store',how='left')
print(data.shape)
print(data_merged.shape)
print(data_merged.isna().sum().sum()) #to cross check if there are any missing values


# In[31]:


# encoding
# 3 categorical column,1 date column, rest are numerical
# data_merged.dtypes
data_merged['day']=data_merged['Date'].dt.day
data_merged['month']=data_merged['Date'].dt.month
data_merged['year']=data_merged['Date'].dt.year
#data_merged['dayofweek']=data_merged['Date'].dt.strftime('%a')


# In[32]:


# Decision tress - label encoding should be used.
# regression - one hot encoding must be used.


# In[33]:


# data_merged.dtypes
# StateHoliday,StoreType,Assortment,PromoInterval
data_merged['StateHoliday'].unique()
# for creating dummy variables - label encoding is used
data_merged['StateHoliday']=data_merged['StateHoliday'].map({'0':0,0:0,'a':1,'b':2,'c':3})
data_merged['StateHoliday']=data_merged['StateHoliday'].astype(int)
data_merged


# In[34]:


# encoding assorted
data_merged['Assortment']
# for creating dummy variables - label encoding is used
data_merged['Assortment']=data_merged['Assortment'].map({'a':1,'b':2,'c':3})
data_merged['Assortment']=data_merged['Assortment'].astype(int)
data_merged


# In[35]:


data_merged['StoreType'].unique()
data_merged['StoreType']=data_merged['StoreType'].map({'a':1,'b':2,'c':3,'d':4})
data_merged['StoreType']=data_merged['StoreType'].astype(int)
data_merged


# In[36]:


data_merged['PromoInterval'].unique()
map_promo = {'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3}
data_merged['PromoInterval']=data_merged['PromoInterval'].map(map_promo)
data_merged


# In[37]:


# Train and validate Split
features= data_merged.columns.drop(['Sales','Date'])
from sklearn.model_selection import train_test_split
train_x,validate_x,train_y,validate_y = train_test_split(data_merged[features],np.log(data_merged['Sales']+1),test_size=0.2,random_state=1)
train_x.shape,validate_x.shape,train_y.shape,validate_y.shape


# In[38]:


# from sklearn.tree import DecisionTreeRegressor

# model_dt = DecisionTreeRegressor(max_depth=20,random_state=1).fit(train_x,train_y)
# validate_y_pred = model_dt.predict(validate_x)

from sklearn.tree import DecisionTreeRegressor
model_dt=DecisionTreeRegressor(max_depth=10,random_state=1).fit(train_x,train_y)
validate_y_pred=model_dt.predict(validate_x)


# In[39]:


get_ipython().system('pip install pydotplus')


# In[40]:


def draw_tree(model, columns):
    import pydotplus
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    import os
    from sklearn import tree
    
    graphviz_path = 'C:\Program Files (x86)\Graphviz2.38/bin/'
    os.environ["PATH"] += os.pathsep + graphviz_path

    dot_data = StringIO()
    tree.export_graphviz(model,
                         out_file=dot_data,
                         feature_names=columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())


# In[41]:


draw_tree(model_dt,features)


# In[42]:


validate_y_pred = model_dt.predict(validate_x)
from sklearn.metrics import mean_squared_error
validate_y_inv = np.exp(validate_y) - 1
validate_y_pred_inv = np.exp(validate_y_pred) - 1
np.sqrt(mean_squared_error(validate_y_inv , validate_y_pred_inv))


# In[43]:


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(y, yhat):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe
validate_y_inv = np.exp(validate_y) - 1
validate_y_pred_inv = np.exp(validate_y_pred) - 1
np.sqrt(mean_squared_error(validate_y_inv , validate_y_pred_inv))
rmspe(validate_y_inv,validate_y_pred_inv)


# In[44]:


# submitting the train on test data set


# In[45]:


model_dt.feature_importances_


# In[46]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.barh(features,model_dt.feature_importances_)
pd.Series(model_dt.feature_importances_,index=features)


# In[47]:


from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':list(range(5,20))}
base_model = DecisionTreeRegressor()
cv_model = GridSearchCV(base_model,param_grid =parameters,cv=5,return_train_score =True).fit(train_x,train_y)


# In[48]:


cv_model.best_params_


# In[49]:


df_cv_results=pd.DataFrame(cv_model.cv_results_).sort_values(by='mean_test_score',ascending=False)[['param_max_depth','mean_test_score']]
df_cv_results.set_index('param_max_depth')['mean_test_score'].plot.line()


# In[50]:


df_cv_results[df_cv_results['param_max_depth']==11].T


# In[51]:


stores_avg_cust = data.groupby(['Store'])[['Customers']].mean().reset_index().astype(int)
test_1 = test.merge(stores_avg_cust,on='Store',how='left')
test.shape,test_1.shape
test_merged = test_1.merge(store,on='Store',how='inner')
test_merged['Open']=test_merged['Open'].fillna(1)
test_merged['Date']=pd.to_datetime(test_merged["Date"],format='%Y-%m-%d')
test_merged['day']=test_merged['Date'].dt.day
test_merged['month']=test_merged['Date'].dt.month
test_merged['year']=test_merged['Date'].dt.year
test_merged['StateHoliday']=test_merged['StateHoliday'].map({'0':0,'a':1})
test_merged['StateHoliday']=test_merged['StateHoliday'].astype(int)
test_merged['Assortment']=test_merged['Assortment'].map({'a':1,'b':2,'c':3})
test_merged['Assortment']=test_merged['Assortment'].astype(int)
test_merged['StoreType']=test_merged['StoreType'].map({'a':1,'b':2,'c':3,'d':4})
test_merged['StoreType']=test_merged['StoreType'].astype(int)
map_promo = {'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3}
test_merged['PromoInterval']=test_merged['PromoInterval'].map(map_promo)


# In[52]:


test_merged


# In[53]:


test_pred = model_dt.predict(test_merged[features])
test_pred_inv = np.exp(test_pred) - 1


# In[54]:


submission = pd.read_csv('/kaggle/input/rossmann-store-sales/sample_submission.csv')
submission_predicted = pd.DataFrame({'Id':test['Id'],'Sales':test_pred_inv})
submission_predicted.to_csv('submission.csv',index=False)
submission_predicted.head()


# In[55]:


import xgboost as xgb


# In[56]:


dtrain = xgb.Dmatrix(train_x[features],train_y)
dvalidate =xgb.Dmatrix(validate_x[features],validate_y)

params ={'max_depth':2,'eta':0.1,'objective':'reg:linear'}
model_xg = xgb.train(param,dtrain,5)
predict_y = 

