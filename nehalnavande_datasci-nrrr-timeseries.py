#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import gc
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import featuretools as ft
import lightgbm as lgb
from lightgbm import plot_tree
from graphviz import Digraph
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,GroupKFold
from sklearn.metrics import roc_auc_score,mean_squared_error
import time
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def check_fline(fpath):
    """check total number of lines of file for large files
    
    Args:
    fpath: string. file path
    
    Returns:
    None
    
    """
    lines = subprocess.run(['wc', '-l', fpath], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(lines, end='', flush=True)


# In[3]:


fs=['../input/ashrae-energy-prediction/train.csv', 
    '../input/ashrae-energy-prediction/test.csv', 
    '../input/ashrae-energy-prediction/weather_test.csv',
    '../input/ashrae-energy-prediction/weather_train.csv',
    '../input/ashrae-energy-prediction/building_metadata.csv']
[check_fline(s) for s in fs]


# In[4]:


# Load sample training data
df_train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
df_train_weather = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')
df_test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')
df_test_weather = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')
df_building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')


# In[5]:


# Show data shape
[print(item.shape) for item in [df_train,df_train_weather,df_test,df_test_weather,df_building]]


# In[6]:


df_train.head()


# In[7]:


df_test.head()


# In[8]:


df_train_weather.head()


# In[9]:


df_building.head()


# In[10]:


df_train_total = pd.merge(df_train,df_building,how='left',on='building_id')
df_train_total = pd.merge(df_train_total,df_train_weather,how='left',on=["site_id", "timestamp"])


# In[11]:


df_train_total.head()


# In[12]:


df_test_total = pd.merge(df_test,df_building,how='left',on='building_id')
df_test_total = pd.merge(df_test_total,df_test_weather,how='left',on=["site_id", "timestamp"])


# In[13]:


df_test_total.head()


# In[14]:


def feat_value_count(df,colname):
    """value count of each feature
    
    Args
    df: data frame.
    colname: string. Name of to be valued column
    
    Returns
    df_count: data frame.
    """
    df_count = df[colname].value_counts().to_frame().reset_index()
    df_count = df_count.rename(columns={'index':colname+'_values',colname:'counts'})
    return df_count


# In[15]:


feat_value_count(df_train,'building_id')


# In[16]:


feat_value_count(df_test,'building_id')


# In[17]:


len(set(df_train.building_id) & set(df_test.building_id))


# In[18]:


feat_value_count(df_train,'meter')


# In[19]:


feat_value_count(df_train_weather,'site_id')


# In[20]:


feat_value_count(df_building,'primary_use')


# In[21]:


feat_value_count(df_building,'site_id')


# In[22]:


df_train_total.dtypes


# In[23]:


df_train_total["timestamp"] = pd.to_datetime(df_train_total["timestamp"], format='%Y-%m-%d %H:%M:%S')


# In[24]:


df_test_total["timestamp"] = pd.to_datetime(df_test_total["timestamp"], format='%Y-%m-%d %H:%M:%S')


# In[25]:


def check_missing(df,cols=None,axis=0):
    """check data frame column missing situation
    Args
    df: data frame.
    cols: list. List of column names
    axis: int. 0 means column and 1 means row
    
    Returns
    missing_info: data frame. 
    """
    if cols != None:
        df = df[cols]
    missing_num = df.isnull().sum(axis).to_frame().rename(columns={0:'missing_num'})
    missing_num['missing_percent'] = df.isnull().mean(axis)*100
    return missing_num.sort_values(by='missing_percent',ascending = False) 


# In[26]:


df_colmissing = check_missing(df_train_total,cols=None,axis=0)
df_colmissing


# In[27]:


del df_colmissing
gc.collect()


# In[28]:


print(max(df_train_total.timestamp),min(df_train_total.timestamp))


# In[29]:


print(max(df_test_total.timestamp),min(df_test_total.timestamp))


# In[30]:


df_one_building = df_train_total[df_train_total.building_id == 1258]


# In[31]:


df_one_building.head()


# In[32]:


# electricity
sns.lineplot(x='timestamp',y='meter_reading',data=df_one_building[df_train_total.meter == 0])


# In[33]:


del df_one_building
gc.collect()


# In[34]:


# chilledwater
sns.lineplot(x='timestamp',y='meter_reading',data=df_one_building[df_train_total.meter == 1])


# In[35]:


# steam
sns.lineplot(x='timestamp',y='meter_reading',data=df_one_building[df_train_total.meter == 2])


# In[36]:


# hotwater
sns.lineplot(x='timestamp',y='meter_reading',data=df_one_building[df_train_total.meter == 3])


# In[37]:


df_lots_building = df_train_total[df_train_total['building_id'].isin([1258,1298,1249])]


# In[38]:


for i in range(0,4):
    f, ax = plt.subplots(figsize=(15, 6))
    sns.lineplot(x='timestamp',y='meter_reading', hue = 'building_id',legend=False,
             data=df_lots_building[df_lots_building.meter == i])


# In[39]:


del df_lots_building
gc.collect()


# In[40]:


for i in range(0,4):
    corr = df_train_total[df_train_total.meter == i][['timestamp','meter_reading','square_feet','year_built','floor_count',
             'air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',
             'sea_level_pressure','wind_direction','wind_speed']].corr()
    f, ax = plt.subplots(figsize=(15, 6))
    sns.heatmap(corr, vmin=-1, vmax=1, annot=True)


# In[41]:


del corr
gc.collect()


# In[42]:


del df_train
del df_train_weather
del df_test
del df_test_weather
del df_building
gc.collect()


# In[43]:


def label_encoder(df, categorical_columns=None):
    """Encode categorical values as integers (0,1,2,3...) with pandas.factorize. """
    # if categorical_colunms are not given than treat object as categorical features
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])
    return df, categorical_columns


# In[44]:


df_train_total,colname = label_encoder(df_train_total, categorical_columns=['primary_use'])
df_test_total,colname = label_encoder(df_test_total, categorical_columns=['primary_use'])


# In[45]:


params = {'objective':'regression',
          'boosting_type':'gbdt',
          'metric':'rmse',
          'learning_rate':0.1,
          'num_leaves': 2**8,
          'max_depth':-1,
          'colsample_bytree':0.5,# feature_fraction 0.7
          'subsample_freq':1,
          'subsample':0.7,
          'verbose':-1,
          #'num_threads':8,
          'seed': 47,#42
                } 


# In[46]:


category_cols = ['building_id', 'site_id', 'primary_use']


# In[47]:


def fold_train_model(splits_num,features_train,labels_train,features_test,categorical):
    splits = splits_num
    folds = KFold(n_splits = splits,random_state=50)
    predictions = np.zeros(len(features_test))
    ave_score = 0
    
    for fold_num, (trn_idx, val_idx) in enumerate(folds.split(features_train.values, labels_train.values)):
        print("Fold {}".format(fold_num))
        train_df, y_train_df = features_train.iloc[trn_idx], labels_train.iloc[trn_idx]
        valid_df, y_valid_df = features_train.iloc[val_idx], labels_train.iloc[val_idx]

        trn_data = lgb.Dataset(train_df, label=y_train_df,categorical_feature=categorical)
        val_data = lgb.Dataset(valid_df, label=y_valid_df,categorical_feature=categorical)

        valid_results = {}
        clf = lgb.train(params,
                        trn_data,
                        10000,
                        valid_sets = [trn_data, val_data],
                        verbose_eval=500,
                        early_stopping_rounds=500,
                        evals_result=valid_results)

        pred = clf.predict(valid_df)
        score = np.sqrt(mean_squared_error(y_valid_df, pred))
        ave_score += score / splits
        predictions += clf.predict(features_test) / splits
    return ave_score,predictions


# In[48]:


def train_meter_type(meter_type,df_train_total,df_test_total,category_cols):
    # prepare data
    df_type_train = df_train_total[df_train_total.meter == meter_type]
    # transfer label with log
    df_type_label = np.log1p(df_type_train['meter_reading'])
    df_type_train.drop(columns = ['meter','meter_reading'],inplace=True)
    df_type_train['timestamp'] = df_type_train['timestamp'].astype('int64') // 10**9

    df_type_test = df_test_total[df_test_total.meter == meter_type]
    df_type_row_id = df_type_test['row_id']
    df_type_test.drop(columns = ['row_id','meter'],inplace=True)
    df_type_test['timestamp'] = df_type_test['timestamp'].astype('int64') // 10**9
    
    # train model
    print('train model')
    ave_score,predictions_type = fold_train_model(3,df_type_train,df_type_label,df_type_test,category_cols)
    print('ave socre is %s'%(ave_score))
    
    # get prediction
    print('get prediction')
    sub_type = pd.DataFrame({'row_id': df_type_row_id, 'meter_reading': np.expm1(predictions_type)})
    return sub_type,ave_score


# In[49]:


#sub_ele_f,ave_score = train_meter_type(0,df_train_total,df_test_total,category_cols)


# In[50]:


#sub_cw_f,ave_score_cw = train_meter_type(1,df_train_total,df_test_total,category_cols)


# In[51]:


#sub_stm_f,ave_score_stm = train_meter_type(2,df_train_total,df_test_total,category_cols)


# In[52]:


#sub_hw_f,ave_score_hw = train_meter_type(3,df_train_total,df_test_total,category_cols)


# In[53]:


#sub_all = pd.concat([sub_ele_f,sub_cw_f,sub_stm_f,sub_hw_f])
#sub_all.sort_values(by='row_id')


# In[54]:


sub_all.to_csv(['../output/baseline_log.csv', index = False)

