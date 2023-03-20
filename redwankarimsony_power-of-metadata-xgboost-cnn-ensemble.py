#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install xgboost')

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
import xgboost as xgb
from sklearn.metrics import accuracy_score


# In[2]:


train= pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test= pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
sub   = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
train.head()

train.target.value_counts()


# In[3]:


train['sex'] = train['sex'].fillna('na')
train['age_approx'] = train['age_approx'].fillna(0)
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')

test['sex'] = test['sex'].fillna('na')
test['age_approx'] = test['age_approx'].fillna(0)
test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')


# In[4]:


train['sex'] = train['sex'].astype("category").cat.codes +1
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype("category").cat.codes +1
train.head()


# In[5]:


test['sex'] = test['sex'].astype("category").cat.codes +1
test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype("category").cat.codes +1
test.head()


# In[ ]:





# In[6]:


x_train = train[['sex', 'age_approx','anatom_site_general_challenge']]
y_train = train['target']


x_test = test[['sex', 'age_approx','anatom_site_general_challenge']]
# y_train = test['target']


train_DMatrix = xgb.DMatrix(x_train, label= y_train)
test_DMatrix = xgb.DMatrix(x_test)


# In[7]:


param = {
    'booster':'gbtree', 
    'eta': 0.3,
    'num_class': 2,
    'max_depth': 
}

epochs = 100


# In[8]:


# model = xgb.train(param, 
#                   train_DMatrix, 
#                   num_boost_round=epochs)

clf = xgb.XGBClassifier(n_estimators=2000, 
                        max_depth=8, 
                        objective='multi:softprob',
                        seed=0,  
                        nthread=-1, 
                        learning_rate=0.15, 
                        num_class = 2, 
                        scale_pos_weight = (32542/584))


# In[9]:


clf.fit(x_train, y_train)


# In[10]:


# predictions = model.predict(test_DMatrix)
# proba = model.predict_proba(test_DMatrix) 
clf.predict_proba(x_test)[:,1]
# clf.predict(x_test)
sub.target = clf.predict_proba(x_test)[:,1]
sub_tabular = sub.copy()


# In[11]:


sub_public_merge = pd.read_csv('/kaggle/input/submission-9/submission_935.csv')
sub_mean = pd.read_csv('/kaggle/input/siim-isic-multiple-model-training-stacking-923/submission_mean.csv')


# In[12]:


sub.target = sub_mean.target *0.1 + sub_public_merge.target *0.7 + sub_tabular.target *0.2


# In[ ]:





# In[ ]:





# In[13]:


sub.head()
sub.to_csv('submission.csv', index = False)


# In[14]:


# train_df.diagnosis.value_counts()


# In[ ]:




