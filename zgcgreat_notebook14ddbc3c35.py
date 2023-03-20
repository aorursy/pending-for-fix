#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc # clean memory a lot
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

p = sns.color_palette()

print('#File size')
for f in os.listdir('../input'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')


# In[2]:


# Clicks 
df_train = pd.read_csv('../input/clicks_train.csv')
df_test = pd.read_csv('../input/clicks_test.csv')


# In[3]:


sizes_train = df_train.groupby('display_id')['ad_id'].count().value_counts()
sizes_test = df_test.groupby('display_id')['ad_id'].count().value_counts()
sizes_train = sizes_train / np.sum(sizes_train)
sizes_test = sizes_test / np.sum(sizes_test)

plt.figure(figsize=(12, 4))
sns.barplot(sizes_train.index, sizes_train.values, alpha=0.8, color=p[0], label='train')
sns.barplot(sizes_test.index, sizes_test.values, alpha=0.8, color=p[1], label='test')
plt.legend()
plt.xlabel('Number of Ads in display', fontsize=12)
plt.ylabel('Proportion of set', fontsize=12)


# In[4]:


sizes_test


# In[5]:


ad_usage_train = df_train.groupby('ad_id')['ad_id'].count()

for i in [2, 10, 50, 100, 1000]:
    print('Ads that appear less than {} times: {}%'.format(i, round((ad_usage_train < i).mean() * 100, 2)))
    
plt.figure(figsize=(12, 6))
plt.hist(ad_usage_train.values, bins=50, log=True)
plt.xlabel('Number of times ad appeared', fontsize=12)
plt.ylabel('log(Count of displays with ad)', fontsize=12)
plt.show()


# In[6]:


# Check how many ads in the test set are not in the training set
ad_prop = len(set(df_test.ad_id.unique()).intersection(df_train.ad_id.unique())) / len(df_test.ad_id.unique())
print('Proportion of test ads in test that are in training: {}%'.format(round(ad_prop * 100, 2)))


# In[7]:


# Events
try:del df_train, df_test 
except:pass
gc.collect()
    
events = pd.read_csv('../input/events.csv')

events.head()


# In[8]:


print('Shape:', events.shape)
print('Columns:', events.columns.tolist())


# In[9]:


# 分析Platform
plat = events.platform.value_counts()
print(plat)
print('\nUnique values of platform:', events.platform.unique())


# In[10]:


events.platform = events.platform.astype(str) # 将str型的'1','2','3'和整型的1,2,3看作一样
plat = events.platform.value_counts()

plt.figure(figsize=(12, 4))
sns.barplot(plat.index, plat.values, alpha=0.8, color=p[2])
plt.xlabel('Platform', fontsize=12)
plt.ylabel('Occurence count', fontsize=12)


# In[11]:


# 分析uuid
uuid_counts = events.groupby('uuid')['uuid'].count().sort_values()

print(uuid_counts.tail())

for i in [2, 5, 10]:
    print('Users that appear less than {} times: {}%'.format(i, round((uuid_counts < i).mean() * 100, 2)))
    
plt.figure(figsize=(12, 4))
plt.hist(uuid_counts.values, bins=50, log=True)
plt.xlabel('Number of times user appeared in set', fontsize=12)
plt.ylabel('log(Count of users)', fontsize=12)
plt.show()


# In[12]:


# Categorices
try del:events
except:pass
gc.collect() # 垃圾回收，释放内存

topics = pd.read_csv('../input/documents_topics.csv')
print('Columns:', topics.columns.tolist())
primt('Number of unique topics:', len(topics.topics_id.unique()))

topics.head()

