#!/usr/bin/env python
# coding: utf-8
We are trying to explore the datasets. 
Our objective to improve zestimate residual error.

Using the following libraries. 


import numpy as np #  algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt #Provides MATLAB-like plotting framework
import seaborn as sns #python visualization library 
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')




train = pd.read_csv('../input/train_2016_v2.csv' , parse_dates=["transactiondate"])
prop = pd.read_csv('../input/properties_2016.csv')
sample = pd.read_csv('../input/sample_submission.csv')




train.shape




train.head(3).transpose()




prop.shape




prop.head(3).transpose()




plt.figure(figsize=(10,7.5))
plt.scatter(range(train.shape[0]), np.sort(train.logerror.values))
plt.xlabel('index', fontsize=15)
plt.ylabel('logerror', fontsize=15)
plt.show()




ulimit = np.percentile(train.logerror.values, 99)
llimit = np.percentile(train.logerror.values, 1)
train['logerror'].ix[train['logerror']>ulimit] = ulimit
train['logerror'].ix[train['logerror']<llimit] = llimit

plt.figure(figsize=(10,7.5))
sns.distplot(train.logerror.values, bins=100, kde=True)
plt.xlabel('logerror', fontsize=20)
plt.show()




merged = pd.merge(train,prop,on="parcelid",how="left")




merged.shape




merged.head(3).transpose()




pd.options.display.max_rows = 65

dtype_df = merged.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df




dataTypeDf = pd.DataFrame(merged.dtypes.value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})
fig,ax = plt.subplots()
fig.set_size_inches(10,7.5)
sns.barplot(data=dataTypeDf,x="variableType",y="count",ax=ax,color="#C28D82")
ax.set(xlabel='Variable Type', ylabel='Count',title="Variables Count Across Datatype")




import missingno as msno

missingValueColumns = merged.columns[merged.isnull().any()].tolist()
msno.bar(merged[missingValueColumns],            figsize=(30,8),color="#34495e",fontsize=15,labels=True,)




msno.matrix(merged[missingValueColumns],width_ratios=(10,1),            figsize=(40,10),fontsize=12,sparkline=True,labels=True)




msno.heatmap(merged[missingValueColumns],figsize=(19,19))




msno.dendrogram(merged)




missing_df = merged.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['missing_ratio'] = missing_df['missing_count'] / merged.shape[0]
missing_df.ix[missing_df['missing_ratio']>0.999]




train['transaction_month'] = train['transactiondate'].dt.month

cnt_srs = train['transaction_month'].value_counts()
plt.figure(figsize=(10,7.5))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[5])
plt.xticks(rotation='vertical')
plt.xlabel('Month of transaction', fontsize=15)
plt.ylabel('Number of Occurrences', fontsize=15)
plt.show()




for col in merged.columns:
    print(col, len(merged[col].unique()))




corr_zero_cols = ['assessmentyear', 'storytypeid', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'poolcnt', 'decktypeid', 'buildingclasstypeid']
for col in corr_zero_cols:
    print(col, len(merged[col].unique()))

