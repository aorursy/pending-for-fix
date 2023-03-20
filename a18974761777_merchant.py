#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')
np.random.seed(4590)


# In[2]:


def get_new_columns(name,aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]


# In[3]:


df_train = pd.read_csv('../input/elo-merchant-category-recommendation/train.csv')
df_test = pd.read_csv('../input/elo-merchant-category-recommendation/test.csv')
df_hist_trans = pd.read_csv('../input/elo-merchant-category-recommendation/historical_transactions.csv')
df_new_merchant_trans = pd.read_csv('../input/elo-merchant-category-recommendation/new_merchant_transactions.csv')
df_merchants = pd.read_csv('../input/elo-merchant-category-recommendation/merchants.csv')


# In[4]:


gdf = df_hist_trans.groupby("card_id")
print(type(gdf))
gdf = gdf.agg({'merchant_category_id':['min']}).reset_index()
print(type(gdf))
gdf.columns = ["card_id", "merchant_category_id"]
df_train = pd.merge(df_train, gdf, on="card_id", how="left")
df_test = pd.merge(df_test, gdf, on="card_id", how="left")


# In[5]:


gdf = df_hist_trans.groupby("card_id")
print(type(gdf))
gdf = gdf.agg({'merchant_category_id':['max']}).reset_index()
print(type(gdf))
gdf.columns = ["card_id", "max_merchant_category_id"]
df_train = pd.merge(df_train, gdf, on="card_id", how="left")
df_test = pd.merge(df_test, gdf, on="card_id", how="left")


# In[6]:


for df in [df_hist_trans,df_new_merchant_trans]:
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)


# In[7]:


for df in [df_hist_trans,df_new_merchant_trans]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 
    #https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']


# In[8]:


aggs = {}
for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']
aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean']
aggs['weekend'] = ['sum', 'mean']
aggs['category_1'] = ['sum', 'mean']
aggs['card_id'] = ['size']

for col in ['category_2','category_3']:
    df_new_merchant_trans[col+'_mean'] = df_new_merchant_trans.groupby([col])['purchase_amount'].transform('mean')
    aggs[col+'_mean'] = ['mean']
    
new_columns = get_new_columns('new_hist',aggs)
df_hist_trans_group = df_new_merchant_trans.groupby('card_id').agg(aggs)
df_hist_trans_group.columns = new_columns
df_hist_trans_group.reset_index(drop=False,inplace=True)
df_hist_trans_group['new_hist_purchase_date_diff'] = (df_hist_trans_group['new_hist_purchase_date_max'] - df_hist_trans_group['new_hist_purchase_date_min']).dt.days
df_hist_trans_group['new_hist_purchase_date_average'] = df_hist_trans_group['new_hist_purchase_date_diff']/df_hist_trans_group['new_hist_card_id_size']
df_hist_trans_group['new_hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['new_hist_purchase_date_max']).dt.days
df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')
df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')
del df_hist_trans_group;gc.collect()
del df_new_merchant_trans;gc.collect()


# In[9]:


df_train.columns


# In[10]:


aggs = {}
for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']

aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean']
aggs['authorized_flag'] = ['sum', 'mean']
aggs['weekend'] = ['sum', 'mean']
aggs['category_1'] = ['sum', 'mean']
aggs['card_id'] = ['size']

for col in ['category_2','category_3']:
    df_hist_trans[col+'_mean'] = df_hist_trans.groupby([col])['purchase_amount'].transform('mean')
    aggs[col+'_mean'] = ['mean']    

new_columns = get_new_columns('hist',aggs)
df_hist_trans_group = df_hist_trans.groupby('card_id').agg(aggs)
df_hist_trans_group.columns = new_columns
df_hist_trans_group.reset_index(drop=False,inplace=True)
df_hist_trans_group['hist_purchase_date_diff'] = (df_hist_trans_group['hist_purchase_date_max'] - df_hist_trans_group['hist_purchase_date_min']).dt.days
df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff']/df_hist_trans_group['hist_card_id_size']
df_hist_trans_group['hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['hist_purchase_date_max']).dt.days
df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')
df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')
del df_hist_trans_group;gc.collect()


# In[11]:


del df_hist_trans;gc.collect()
#del df_new_merchant_trans;gc.collect()#清理内存-
df_train.head(5)


# In[12]:


from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
df_merchants['avg_sales_lag3']=imp.fit(pd.DataFrame(df_merchants['avg_sales_lag3'].values.reshape(-1,1))).transform(pd.DataFrame(df_merchants['avg_sales_lag3'].values.reshape(-1,1)))
df_merchants['avg_sales_lag6']=imp.fit(pd.DataFrame(df_merchants['avg_sales_lag6'].values.reshape(-1,1))).transform(pd.DataFrame(df_merchants['avg_sales_lag6'].values.reshape(-1,1)))
df_merchants['avg_sales_lag12']=imp.fit(pd.DataFrame(df_merchants['avg_sales_lag12'].values.reshape(-1,1))).transform(pd.DataFrame(df_merchants['avg_sales_lag12'].values.reshape(-1,1)))
df_merchants['category_2']=imp.fit(pd.DataFrame(df_merchants['category_2'].values.reshape(-1,1))).transform(pd.DataFrame(df_merchants['category_2'].values.reshape(-1,1)))
df_merchants.head()


# In[13]:


aggs={}
for col in ['avg_sales_lag3','avg_purchases_lag3','active_months_lag3','avg_sales_lag6','avg_purchases_lag6','active_months_lag6','avg_sales_lag12','avg_purchases_lag12','active_months_lag12','numerical_1','numerical_2']:
    aggs[col]= ['mean']
    
new_columns= get_new_columns('merchants',aggs)
df_merchants_group = df_merchants.groupby('merchant_category_id').agg(aggs)
df_merchants_group.columns = new_columns
df_merchants_group.reset_index(drop=False,inplace=True)
df_train=df_train.merge(df_merchants_group,on='merchant_category_id',how='left')
df_test=df_test.merge(df_merchants_group.reset_index(),on='merchant_category_id',how='left')
df_train.head()


# In[14]:


df_merchants['max_merchant_category_id']=df_merchants['merchant_category_id']
#df_test['max_merchant_category_id']=df_test['merchant_category_id']

aggs={}
for col in ['avg_sales_lag3','avg_purchases_lag3','active_months_lag3','avg_sales_lag6','avg_purchases_lag6','active_months_lag6','avg_sales_lag12','avg_purchases_lag12','active_months_lag12','numerical_1','numerical_2']:
    aggs[col]= ['mean']
    
new_columns= get_new_columns('max_merchants',aggs)
df_merchants_group = df_merchants.groupby('max_merchant_category_id').agg(aggs)
df_merchants_group.columns = new_columns
df_merchants_group.reset_index(drop=False,inplace=True)
df_train=df_train.merge(df_merchants_group,on='max_merchant_category_id',how='left')
df_test=df_test.merge(df_merchants_group.reset_index(),on='max_merchant_category_id',how='left')
df_train.head()


# In[15]:


df_train['outliers'] = 0
df_train.loc[df_train['target'] < -30, 'outliers'] = 1
df_train['outliers'].value_counts()


# In[16]:


for df in [df_train,df_test]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['dayofweek'] = df['first_active_month'].dt.dayofweek
    df['weekofyear'] = df['first_active_month'].dt.weekofyear
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days
    for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',                     'new_hist_purchase_date_min']:
        df[f] = df[f].astype(np.int64) * 1e-9
    df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']
    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']

for f in ['feature_1','feature_2','feature_3']:
    order_label = df_train.groupby([f])['outliers'].mean()
    df_train[f] = df_train[f].map(order_label)
    df_test[f] = df_test[f].map(order_label)


# In[17]:


df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target','outliers']]
target = df_train['target']
del df_train['target']


# In[18]:


param = {'num_leaves': 31,
         'min_data_in_leaf': 27, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.015,
         "min_child_samples": 50,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 4590}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['outliers'].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)
    val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = df_train_columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits

np.sqrt(mean_squared_error(oof, target))


# In[19]:


model_without_outliers = pd.DataFrame({"card_id":df_test["card_id"].values})
model_without_outliers["target"] = predictions


# In[20]:


#del df_train['outliers']
#del df_train['target']
#target = df_train['outliers']


# In[21]:


features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = [c for c in features if 'feature_' in c]


# In[22]:


param = {'num_leaves': 31,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 6,
         'learning_rate': 0.01,
         "boosting": "rf",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'binary_logloss',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 2333}


# In[23]:


from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
%%time
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

start = time.time()


for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(log_loss(target, oof)))


# In[24]:


### 'target' is the probability of whether an observation is an outlier
df_outlier_prob = pd.DataFrame({"card_id":df_test["card_id"].values})
df_outlier_prob["target"] = predictions
df_outlier_prob.head()


# In[25]:


outlier_id = pd.DataFrame(df_outlier_prob.sort_values(by='target',ascending = False).head(25000)['card_id'])


# In[26]:


best_submission = pd.read_csv('../input/finaldata/submission_ashish.csv')
most_likely_liers = best_submission.merge(outlier_id,how='right')
most_likely_liers.head()


# In[27]:


get_ipython().run_cell_magic('time', '', "for card_id in most_likely_liers['card_id']:\n    model_without_outliers.loc[model_without_outliers['card_id']==card_id,'target']\\\n    = most_likely_liers.loc[most_likely_liers['card_id']==card_id,'target'].values")


# In[28]:


model_without_outliers.to_csv("combining_submission.csv", index=False)


# In[29]:


#sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})
#sub_df["target"] = predictions
#sub_df.to_csv("submission.csv", index=False)


# In[30]:


#cols = (feature_importance_df[["Feature", "importance"]]
        #.groupby("Feature")
        #.mean()
        #.sort_values(by="importance", ascending=False)[:1000].index)

#best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

#plt.figure(figsize=(14,25))
#sns.barplot(x="importance",
            #y="Feature",
            #data=best_features.sort_values(by="importance",
                                          # ascending=False))
#plt.title('LightGBM Features (avg over folds)')
#plt.tight_layout()
#plt.savefig('lgbm_importances.png')

