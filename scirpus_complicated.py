#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ls ../input


# In[3]:


def OrdinalConverter(d):
    a1 = ord(d[:1])-65
    if(a1>26):
        a1-=6
    #if(len(d)==1):
    return a1
#     a2 = ord(d[1:2])-65
#     if(a2>26):
#         a2-=6
#     return a1*52+a2


# In[4]:


df_train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')
df_test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')


# In[5]:


allcolumns = df_train.columns
binaries = [c for c in df_train.columns if 'bin' in c]
nominals = [c for c in df_train.columns if 'nom' in c]
ordinals = [c for c in df_train.columns if 'ord' in c]


# In[6]:


df_train['nans'] = df_train.isnull().sum(axis=1)
df_test['nans'] = df_test.isnull().sum(axis=1)


# In[7]:


df_train = df_train.set_index('id')
df_test = df_test.set_index('id')
y_train = df_train.target
del df_train['target']


# In[8]:


ord_0_mapping = {1 : 0, 2 : 1, 3 : 2}
ord_1_mapping = {'Novice' : 0, 'Contributor' : 1, 'Expert' : 2, 'Master': 3, 'Grandmaster': 4}
ord_2_mapping = { 'Freezing': 0, 'Cold': 1, 'Warm' : 2, 'Hot': 3, 'Boiling Hot' : 4, 'Lava Hot' : 5}
df_train['real_ord_0'] = df_train.loc[df_train.ord_0.notnull(), 'ord_0'].map(ord_0_mapping)
df_train['real_ord_1'] = df_train.loc[df_train.ord_1.notnull(), 'ord_1'].map(ord_1_mapping)
df_train['real_ord_2'] = df_train.loc[df_train.ord_2.notnull(), 'ord_2'].map(ord_2_mapping)
df_test['real_ord_0'] = df_test.loc[df_test.ord_0.notnull(), 'ord_0'].map(ord_0_mapping)
df_test['real_ord_1'] = df_test.loc[df_test.ord_1.notnull(), 'ord_1'].map(ord_1_mapping)
df_test['real_ord_2'] = df_test.loc[df_test.ord_2.notnull(), 'ord_2'].map(ord_2_mapping)
otherordinals = ['ord_3','ord_4','ord_5']

for c in otherordinals:
    print(c)
    df_train['real_'+c] = df_train[[c]].apply(lambda a: OrdinalConverter(a[c]) if not pd.isnull(a[c]) else np.nan,axis=1)
    df_test['real_'+c] = df_test[[c]].apply(lambda a: OrdinalConverter(a[c]) if not pd.isnull(a[c]) else np.nan,axis=1)


# In[9]:


df_train.drop(ordinals,inplace=True,axis=1)
df_test.drop(ordinals,inplace=True,axis=1)


# In[10]:


df = pd.concat([df_train,df_test])


# In[11]:


df = pd.get_dummies(df, dummy_na=False, columns=binaries)


# In[12]:


df.head()


# In[13]:


df.columns


# In[14]:


for c in nominals:
    le = LabelEncoder()
    df.loc[~df[c].isnull(),c] = le.fit_transform(df.loc[~df[c].isnull(),c])


# In[15]:


df.shape


# In[16]:


df_train = df[:600000].copy()
df_test = df[600000:].copy()


# In[17]:


df_train.head()


# In[18]:


import category_encoders as ce
cat_feat_to_encode = list(set(df_train.columns).difference(set(['real_ord_0', 'real_ord_1',
       'real_ord_2', 'real_ord_3', 'real_ord_4', 'real_ord_5'])))
smoothing=100

folds = 20
oof = np.zeros(df_train[cat_feat_to_encode].shape)
test_oof = np.zeros(df_test[cat_feat_to_encode].shape)
 


# In[19]:


cat_feat_to_encode


# In[20]:


df_train.columns


# In[21]:


df_train[['real_ord_0', 'real_ord_1',
       'real_ord_2', 'real_ord_3', 'real_ord_4', 'real_ord_5']].head()


# In[22]:


from sklearn.model_selection import StratifiedKFold
cat_feat_to_encode = df_train.columns[:].tolist()

oof = np.zeros(df_train.shape)
test_oof = np.zeros(df_test.shape)
from sklearn.model_selection import StratifiedKFold
for tr_idx, oof_idx in StratifiedKFold(n_splits=folds, random_state= 1032, shuffle=True).split(df_train, y_train):
    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
    ce_target_encoder.fit(df_train.iloc[tr_idx, :], y_train.iloc[tr_idx])
    oof[oof_idx,:] = ce_target_encoder.transform(df_train.iloc[oof_idx, :]).values
    test_oof[:,:] += ce_target_encoder.transform(df_test).values
test_oof /= folds  


# In[23]:


new_train = pd.DataFrame(data=oof,columns=['te_'+ c for c in cat_feat_to_encode],index=df_train.index.values)
new_test = pd.DataFrame(data=test_oof,columns=['te_'+ c for c in cat_feat_to_encode],index=df_test.index.values)  


# In[24]:


df_train = pd.concat([df_train,new_train],axis=1)
df_test = pd.concat([df_test,new_test],axis=1)


# In[25]:


alldata = pd.concat([df_train,df_test])


# In[26]:


alldata.head()


# In[27]:


alldata.columns


# In[28]:


ordinals


# In[29]:


for c in alldata.columns:
    print(c)
    mn = alldata.loc[~alldata[c].isnull(),c].mean()
    sd = alldata.loc[~alldata[c].isnull(),c].std()
    alldata.loc[~alldata[c].isnull(),c] -= mn
    alldata.loc[~alldata[c].isnull(),c] /= sd


# In[30]:


alldata.drop(nominals,inplace=True,axis=1)


# In[31]:


df_train = alldata[:600000].copy()
df_test = alldata[600000:].copy()


# In[32]:


df_train.head()


# In[33]:


df_test.head()


# In[34]:


df_train['target'] = y_train


# In[35]:


from sklearn.linear_model import LogisticRegression
glm =LogisticRegression(C=1, random_state=2, solver='lbfgs', max_iter=20600, fit_intercept=True, penalty='l2', verbose=0)
glm.fit(df_train[df_train.columns[:-1]].fillna(df_train[df_train.columns[:-1]].mean()), df_train.target)


# In[36]:


roc_auc_score(y_train,glm.predict_proba(df_train[df_train.columns[:-1]].fillna(df_train[df_train.columns[:-1]].mean()))[:,1])


# In[37]:


def Output(p):
    return 1.0/(1.0+np.exp(-p))

def GP(data):
    return Output(  -1.468275 +
                    0.047290*np.tanh(((((np.tanh((((np.tanh((((((((((data["te_real_ord_3"]) + ((((-1.0)) / 2.0)))) + (((data["te_nom_8"]) + (((data["te_nom_7"]) + (((((data["te_real_ord_5"]) + (((data["te_bin_2_0.0"]) / 2.0)))) + (data["te_real_ord_2"]))))))))) + (data["te_real_ord_0"]))) + (data["te_real_ord_3"]))))) * 2.0)))) * 2.0)) * 2.0)) +
                    0.041999*np.tanh(((((((data["real_ord_5"]) + (((((data["te_nom_7"]) + (((data["te_real_ord_3"]) + (((data["te_real_ord_0"]) + (((data["te_real_ord_3"]) + (((data["te_nom_9"]) + (np.where(data["te_real_ord_3"] > -998, data["te_nom_8"], data["te_nom_9"] )))))))))))) + (np.where(data["real_ord_2"] > -998, data["real_ord_2"], np.tanh((data["real_ord_5"])) )))))) * 2.0)) * 2.0)) +
                    0.048760*np.tanh((((((((((data["te_bin_0_1.0"]) + ((((((data["te_real_ord_2"]) + (((data["te_nom_8"]) + (((data["te_nom_1"]) + (data["te_real_ord_0"]))))))/2.0)) * 2.0)))) + (((((((data["te_real_ord_3"]) + (data["te_nom_9"]))) + (data["te_nom_7"]))) + (((((data["te_real_ord_5"]) + (data["te_real_ord_3"]))) + (data["te_real_ord_4"]))))))/2.0)) * 2.0)) * 2.0)) +
                    0.049284*np.tanh(((((data["te_real_ord_0"]) + (((data["te_real_ord_1"]) + (((((data["te_month"]) - ((-((((((((((((((data["te_real_ord_2"]) + (data["te_real_ord_3"]))/2.0)) * 2.0)) + (np.minimum(((data["te_real_ord_5"])), (((((data["te_bin_0_1.0"]) + (data["te_real_ord_5"]))/2.0))))))) * 2.0)) + (((data["te_nom_9"]) + (data["te_real_ord_4"]))))/2.0))))))) * 2.0)))))) * 2.0)) +
                    0.048754*np.tanh(((((((data["bin_2_1.0"]) + (data["te_nom_1"]))) + (((np.where(data["te_real_ord_2"] < -998, data["te_nom_9"], ((((data["te_real_ord_3"]) + (data["te_month"]))) + (data["te_real_ord_2"])) )) * 2.0)))) + (((data["te_real_ord_5"]) + (((data["te_real_ord_0"]) + (((data["te_nom_9"]) + (((data["te_nom_7"]) + (data["te_real_ord_4"]))))))))))) +
                    0.049983*np.tanh(((((data["te_bin_0_0.0"]) + (((((data["te_month"]) + (((data["te_real_ord_0"]) + (data["te_real_ord_4"]))))) + (((((((data["te_nom_7"]) + (data["te_real_ord_2"]))) + (((data["te_real_ord_1"]) + (data["te_nom_1"]))))) + (((data["te_real_ord_3"]) * 2.0)))))))) + (((data["te_real_ord_5"]) + (data["te_nom_8"]))))) +
                    0.049995*np.tanh(((data["te_month"]) + (((((np.minimum(((data["te_bin_0_1.0"])), ((data["real_ord_3"])))) - ((-((((data["te_real_ord_2"]) + (((((data["te_real_ord_5"]) - ((-((((data["te_nom_1"]) + (((((data["te_real_ord_0"]) + (data["te_nom_8"]))) + (data["te_real_ord_4"])))))))))) + (data["te_nom_3"])))))))))) + (data["real_ord_3"]))))) +
                    0.049850*np.tanh(((data["te_real_ord_2"]) + (((((data["te_real_ord_0"]) + (((data["te_nom_1"]) + (data["te_real_ord_1"]))))) + (((((((((((data["te_nom_8"]) + (((data["te_real_ord_3"]) + (data["te_month"]))))) + (data["te_nom_7"]))) + (data["te_real_ord_5"]))) + (np.minimum(((data["te_nom_8"])), ((data["te_real_ord_3"])))))) + (data["te_bin_2_1.0"]))))))) +
                    0.041341*np.tanh(((((((((np.minimum(((np.sin((data["real_ord_0"])))), ((data["te_nom_7"])))) + (((data["te_real_ord_2"]) + (((data["te_real_ord_3"]) + (data["te_nom_9"]))))))) - ((-((((data["te_month"]) + (data["te_nom_8"])))))))) * 2.0)) + (((data["te_bin_0_1.0"]) + (((((data["te_real_ord_1"]) + (data["te_real_ord_5"]))) + (data["te_real_ord_3"]))))))) +
                    0.049940*np.tanh(((((((((data["te_real_ord_3"]) + (data["te_real_ord_2"]))) + (((((((data["te_real_ord_0"]) + (data["te_bin_2_0.0"]))) + (((((data["te_nom_8"]) + (data["te_real_ord_4"]))) + (np.tanh((data["te_nom_5"]))))))) + (((((data["te_month"]) + (data["te_nom_7"]))) + ((-1.0)))))))) + (data["te_real_ord_5"]))) + (data["te_nom_9"]))) +
                    0.049852*np.tanh(((data["te_nom_2"]) + (((data["te_real_ord_0"]) + (((((((((((((((data["te_real_ord_3"]) * 2.0)) + (data["te_month"]))) + (np.minimum(((data["te_bin_0_1.0"])), ((data["te_nom_9"])))))) + (((((data["real_ord_5"]) + (((data["te_nom_7"]) + (data["te_real_ord_4"]))))) + (data["te_nom_8"]))))) * 2.0)) / 2.0)) + (data["te_nom_3"]))))))) +
                    0.049998*np.tanh(((data["te_real_ord_3"]) + (((((((((data["te_real_ord_2"]) + (((data["te_real_ord_4"]) + (data["te_nom_9"]))))) + (((((np.tanh((data["te_bin_2_1.0"]))) + (((data["te_real_ord_1"]) + (data["te_nom_3"]))))) + (((data["te_day"]) + (data["te_month"]))))))) + (np.minimum(((data["te_real_ord_3"])), ((data["te_real_ord_5"])))))) + (data["te_real_ord_5"]))))) +
                    0.049904*np.tanh(((data["te_real_ord_0"]) + (((((((np.minimum(((data["te_nom_8"])), ((((data["te_month"]) + (((data["te_nom_8"]) + (data["te_nom_5"])))))))) + (data["te_real_ord_2"]))) + (((data["te_real_ord_3"]) + (((((((data["te_real_ord_5"]) + (data["te_day"]))) + (data["te_nom_7"]))) + (np.tanh((data["te_real_ord_1"]))))))))) * 2.0)))) +
                    0.049975*np.tanh(((((data["te_nom_8"]) + (((((data["te_real_ord_2"]) + (data["te_real_ord_3"]))) + (((((data["te_real_ord_0"]) + (data["te_month"]))) + (((data["te_day"]) + (((((data["te_real_ord_4"]) + (data["te_nom_4"]))) + (((np.tanh((data["te_nom_2"]))) + (((data["te_nom_7"]) + (np.tanh((data["te_bin_2_0.0"]))))))))))))))))) * 2.0)) +
                    0.044503*np.tanh(((((((data["te_real_ord_3"]) * 2.0)) + (((((((((data["te_real_ord_0"]) + (data["te_nom_2"]))) + (data["te_real_ord_2"]))) + (((((((((((data["te_real_ord_5"]) + (data["te_real_ord_1"]))) + (data["te_real_ord_4"]))) + (data["te_month"]))) + (data["te_nom_9"]))) - ((0.318310)))))) - ((0.318310)))))) + (data["te_nom_1"]))) +
                    0.041910*np.tanh((((((((((((((((data["te_nom_7"]) + (data["te_real_ord_2"]))) + (((data["te_real_ord_3"]) + ((((data["te_bin_2_1.0"]) + (data["te_real_ord_1"]))/2.0)))))) + (np.minimum(((((data["te_nom_1"]) + (((data["te_nom_8"]) + (data["te_nom_5"])))))), ((((data["te_nom_5"]) + (data["te_month"])))))))) + (data["te_real_ord_5"]))/2.0)) * 2.0)) * 2.0)) * 2.0)) +
                    0.049845*np.tanh(((data["te_nom_3"]) + (((data["te_month"]) + (((((data["te_real_ord_0"]) + (data["te_nom_7"]))) + (((((((((((data["te_real_ord_2"]) + (data["te_real_ord_3"]))) + (data["te_nom_1"]))) + (np.minimum(((data["te_day"])), ((data["te_real_ord_5"])))))) + ((((data["te_nom_8"]) + (data["te_nom_2"]))/2.0)))) + (np.tanh((data["te_nom_9"]))))))))))) +
                    0.028460*np.tanh(((((((((data["real_ord_3"]) + ((((data["te_nom_5"]) + (((((data["te_real_ord_0"]) + (((data["te_nom_3"]) + (((data["te_real_ord_2"]) + (data["te_nom_7"]))))))) - (((data["bin_2_0.0"]) - (data["te_nom_8"]))))))/2.0)))) * 2.0)) + (np.minimum(((data["bin_0_0.0"])), ((data["te_real_ord_2"])))))) + (np.minimum(((data["te_month"])), ((data["te_nom_7"])))))) +
                    0.049901*np.tanh(((((((data["te_real_ord_5"]) + (((((((((((data["te_nom_8"]) + (data["te_real_ord_1"]))) + (data["real_ord_3"]))) + (data["te_bin_0_1.0"]))) - ((-((data["te_nom_9"])))))) + ((((((np.minimum(((data["te_month"])), ((((data["te_real_ord_4"]) + (data["te_nom_3"])))))) * 2.0)) + (data["te_real_ord_2"]))/2.0)))))) + (data["te_day"]))) * 2.0)) +
                    0.049735*np.tanh(((((((data["te_real_ord_0"]) + (data["te_real_ord_3"]))) + (data["te_real_ord_2"]))) + (((((((data["te_nom_2"]) + (((data["te_day"]) + (np.minimum(((data["te_month"])), ((np.tanh((data["real_ord_4"])))))))))) + (((((data["te_nom_8"]) - (data["bin_2_0.0"]))) + (data["te_real_ord_5"]))))) + (np.minimum(((data["te_nom_3"])), ((data["te_nom_7"])))))))) +
                    0.049679*np.tanh(((data["te_nom_4"]) + (((((((data["te_month"]) + (((data["te_real_ord_4"]) + (((data["te_real_ord_3"]) + ((((((data["te_real_ord_5"]) + (np.minimum(((data["te_nom_9"])), ((data["te_bin_0_1.0"])))))/2.0)) * 2.0)))))))) + (((data["te_nom_1"]) + ((((((data["bin_2_1.0"]) + (np.minimum(((data["te_nom_8"])), ((data["te_nom_7"])))))/2.0)) * 2.0)))))) * 2.0)))) +
                    0.048315*np.tanh(((data["te_real_ord_1"]) + (((((((np.minimum(((data["te_real_ord_4"])), ((data["te_nom_1"])))) + (((((data["te_nom_8"]) + (((data["te_nom_3"]) + (data["bin_0_0.0"]))))) + (np.sin((data["real_ord_3"]))))))) + (((((data["te_real_ord_2"]) + (data["te_nom_7"]))) + (((data["te_day"]) + (np.sin((data["real_ord_0"]))))))))) * 2.0)))) +
                    0.049833*np.tanh(((((((data["te_real_ord_0"]) + (data["te_month"]))) + (data["te_real_ord_5"]))) + (((((data["te_nom_5"]) + (((((data["te_real_ord_1"]) + (data["te_real_ord_4"]))) + (data["te_nom_2"]))))) + (np.minimum(((((data["te_real_ord_3"]) * 2.0))), ((((np.minimum(((((data["te_real_ord_2"]) * 2.0))), ((((data["te_real_ord_3"]) * 2.0))))) + (data["te_nom_9"])))))))))) +
                    0.049972*np.tanh((((((((((((data["te_day"]) - ((-((data["te_real_ord_4"])))))) + (((data["te_bin_0_1.0"]) * 2.0)))) + (((((data["te_nom_9"]) + (((((data["te_nom_4"]) - ((-((((data["te_bin_2_1.0"]) + (data["te_nom_8"])))))))) + (((data["te_real_ord_3"]) + (data["te_real_ord_0"]))))))) * 2.0)))/2.0)) + (data["te_nom_7"]))) * 2.0)) +
                    0.049905*np.tanh(((data["real_ord_2"]) + (((data["te_nom_7"]) + (((((((np.minimum(((data["te_real_ord_3"])), ((data["te_bin_1_0.0"])))) + (data["te_nom_4"]))) + (data["te_nom_5"]))) + ((((((data["te_real_ord_5"]) + (((((data["te_nom_1"]) + (data["te_nom_3"]))) + (((data["te_real_ord_1"]) + (((data["te_real_ord_0"]) + (data["te_real_ord_3"]))))))))/2.0)) * 2.0)))))))) +
                    0.049980*np.tanh(((data["te_month"]) + (((((((data["te_nom_8"]) + (((data["te_real_ord_3"]) + (((data["bin_2_1.0"]) + (data["te_nom_2"]))))))) + (((((data["te_real_ord_1"]) + (data["te_nom_4"]))) + (np.minimum(((data["te_day"])), ((data["te_nom_1"])))))))) + (((data["te_real_ord_4"]) + (((data["te_real_ord_5"]) + (np.sin((data["te_nom_5"]))))))))))) +
                    0.049438*np.tanh(((data["te_bin_4_N"]) + (((data["te_month"]) + (((((data["te_bin_0_1.0"]) + (((data["te_nom_9"]) + (((data["te_nom_3"]) + (data["te_real_ord_5"]))))))) + (((data["te_nom_5"]) + (((data["te_real_ord_3"]) + (((np.minimum(((((data["te_real_ord_2"]) * 2.0))), ((data["te_real_ord_4"])))) + (((data["te_nom_4"]) + (data["te_nom_1"]))))))))))))))) +
                    0.047113*np.tanh(((data["te_nom_3"]) + ((((2.94935059547424316)) * ((((((((data["te_nom_8"]) + (np.minimum(((data["te_real_ord_3"])), ((data["te_real_ord_4"])))))) + (data["te_nom_7"]))) + (((data["te_nom_9"]) + ((((((((data["te_nom_1"]) - (np.tanh((data["bin_2_0.0"]))))) + (np.minimum(((data["te_real_ord_5"])), ((data["te_real_ord_0"])))))/2.0)) * 2.0)))))/2.0)))))) +
                    0.049168*np.tanh((((((((((data["te_nom_7"]) + (((data["te_bin_2_0.0"]) + (data["te_real_ord_5"]))))/2.0)) + (data["te_nom_0"]))) + (np.tanh((data["te_day"]))))) + (((data["te_real_ord_2"]) + (((((data["te_real_ord_4"]) + (np.minimum(((data["te_real_ord_0"])), ((data["te_real_ord_3"])))))) + (np.minimum(((data["te_nom_9"])), ((((data["te_nom_3"]) + (data["te_month"])))))))))))) +
                    0.049541*np.tanh(((np.minimum(((((data["te_bin_1_0.0"]) + (data["te_nom_1"])))), ((((data["te_bin_0_1.0"]) + (data["te_real_ord_0"])))))) + (((((data["te_real_ord_2"]) + (((data["te_month"]) + (data["te_real_ord_4"]))))) + (((((data["te_real_ord_1"]) + ((((((((data["te_nom_7"]) * 2.0)) + (data["te_real_ord_3"]))/2.0)) + (data["te_nom_8"]))))) + (data["te_bin_4_Y"]))))))) +
                    0.049973*np.tanh(((((data["te_real_ord_1"]) + ((((((((data["te_nom_7"]) + (data["bin_2_1.0"]))/2.0)) + (data["te_bin_1_1.0"]))) + (data["te_nom_8"]))))) - ((((-((((((np.minimum(((data["te_real_ord_5"])), ((data["te_real_ord_0"])))) + ((((data["te_nom_2"]) + (data["te_day"]))/2.0)))) + ((((data["te_nom_5"]) + (data["te_real_ord_2"]))/2.0))))))) * 2.0)))) +
                    0.048645*np.tanh(np.where(data["real_ord_3"] < -998, data["nans"], (((((((data["te_month"]) + (data["te_nom_3"]))) + (data["te_nom_2"]))/2.0)) + (((np.minimum(((data["te_real_ord_0"])), ((data["te_real_ord_5"])))) + ((((data["te_nom_9"]) + (((((data["te_day"]) + (((data["te_bin_0_1.0"]) + (data["bin_2_1.0"]))))) + (((data["te_bin_4_Y"]) + (data["real_ord_3"]))))))/2.0))))) )) +
                    0.046152*np.tanh((((((((data["te_day"]) + (((data["te_month"]) + (data["te_real_ord_2"]))))) + (data["te_bin_1_1.0"]))/2.0)) + (((((data["te_nom_6"]) + (((data["te_bin_0_1.0"]) + (((data["te_nom_5"]) + (data["te_nom_1"]))))))) + (np.minimum(((data["te_nom_7"])), ((((np.where(data["te_nom_5"] < -998, data["real_ord_4"], data["te_real_ord_3"] )) * 2.0))))))))) +
                    0.049996*np.tanh((((((data["te_bin_2_1.0"]) + (data["te_month"]))/2.0)) + (((data["te_nom_4"]) + (((((((((((((data["te_nom_7"]) + (data["te_nom_2"]))/2.0)) + ((((((data["te_real_ord_2"]) + (data["te_nom_9"]))) + (((np.minimum(((data["bin_0_0.0"])), ((data["te_real_ord_1"])))) + (np.tanh((data["te_real_ord_3"]))))))/2.0)))/2.0)) * 2.0)) * 2.0)) + (data["te_nom_3"]))))))) +
                    0.047794*np.tanh((((((data["te_real_ord_0"]) + ((((((((np.minimum(((data["te_nom_1"])), ((data["te_real_ord_2"])))) + (np.minimum(((data["te_real_ord_5"])), ((((data["te_real_ord_3"]) * 2.0))))))) + (((data["bin_2_1.0"]) - ((-(((((data["te_month"]) + ((((((data["te_nom_0"]) + (((data["te_nom_8"]) * 2.0)))/2.0)) * 2.0)))/2.0))))))))/2.0)) * 2.0)))/2.0)) * 2.0)) +
                    0.049944*np.tanh((((((((data["te_month"]) + (((data["te_real_ord_5"]) + (((data["te_real_ord_1"]) + (data["te_real_ord_0"]))))))) + (((data["te_nom_4"]) + (((((data["te_bin_0_1.0"]) + (data["te_nom_9"]))) + (((((data["te_nom_5"]) + (((data["te_real_ord_4"]) + (data["bin_1_0.0"]))))) + (data["te_nom_4"]))))))))/2.0)) + (data["te_nom_8"]))) +
                    0.037920*np.tanh(((((data["te_nom_3"]) + (np.minimum(((data["te_real_ord_2"])), ((((data["te_bin_4_N"]) + (data["te_day"])))))))) + (((np.minimum(((data["te_real_ord_1"])), ((data["te_real_ord_3"])))) + (((((((((((np.minimum(((data["te_nom_7"])), ((data["te_real_ord_3"])))) + (data["te_nom_1"]))/2.0)) * 2.0)) + (np.maximum(((data["te_real_ord_4"])), ((data["te_nom_0"])))))/2.0)) * 2.0)))))) +
                    0.046360*np.tanh((((((((data["te_real_ord_4"]) + (((data["te_real_ord_2"]) + (np.sin((data["te_real_ord_3"]))))))) + (np.minimum(((data["te_nom_4"])), ((data["te_day"])))))) + (((((data["te_nom_2"]) + (((data["te_nom_5"]) - (data["real_ord_3"]))))) + (((((data["te_real_ord_3"]) + (data["te_nom_9"]))) + (np.minimum(((data["te_nom_7"])), ((data["te_real_ord_3"])))))))))/2.0)) +
                    0.049740*np.tanh((((data["te_bin_0_1.0"]) + ((((data["te_nom_6"]) + (((((((np.minimum(((data["te_real_ord_5"])), (((((((data["te_real_ord_1"]) + (((data["te_real_ord_0"]) + (data["te_nom_1"]))))/2.0)) + (data["te_nom_8"])))))) - (np.cos(((((((data["month"]) + (data["te_nom_7"]))/2.0)) + ((14.39997959136962891)))))))) * 2.0)) * 2.0)))/2.0)))/2.0)) +
                    0.048550*np.tanh((((((data["te_day"]) + (((np.minimum(((data["bin_4_Y"])), ((((data["te_bin_1_1.0"]) + (((data["te_month"]) + (data["te_nom_6"])))))))) + ((((data["te_real_ord_0"]) + (data["te_real_ord_2"]))/2.0)))))) + (((np.tanh((data["bin_2_1.0"]))) + (((data["te_nom_4"]) + ((((data["te_nom_3"]) + (data["te_real_ord_3"]))/2.0)))))))/2.0)) +
                    0.047951*np.tanh((((((np.minimum(((((((((np.minimum(((data["te_bin_0_1.0"])), ((np.sin((data["real_ord_5"])))))) + (data["te_nom_7"]))) / 2.0)) + (data["te_real_ord_4"])))), ((((data["te_nom_2"]) + (((data["te_bin_0_1.0"]) + (data["te_nom_8"])))))))) + (((data["te_nom_0"]) + (data["te_real_ord_1"]))))/2.0)) + ((((data["te_real_ord_3"]) + (np.sin((data["te_nom_5"]))))/2.0)))) +
                    0.031950*np.tanh((((data["te_real_ord_5"]) + (((np.where(data["te_nom_4"] < -998, data["te_month"], ((data["te_nom_4"]) + (data["te_nom_1"])) )) + (((np.where((0.0) > -998, np.maximum(((data["te_month"])), ((np.minimum(((data["te_nom_8"])), ((data["te_nom_9"])))))), data["te_real_ord_5"] )) + (np.minimum(((data["te_real_ord_2"])), ((data["te_nom_9"])))))))))/2.0)) +
                    0.049920*np.tanh(np.sin((np.minimum((((((((((((np.minimum(((data["te_nom_7"])), ((data["te_nom_8"])))) + (data["bin_2_1.0"]))/2.0)) + (((data["real_ord_2"]) / 2.0)))/2.0)) + ((((data["te_nom_8"]) + (data["real_ord_5"]))/2.0)))/2.0))), (((((data["te_nom_9"]) + ((((data["bin_2_1.0"]) + (np.minimum(((data["real_ord_2"])), ((data["te_real_ord_0"])))))/2.0)))/2.0))))))) +
                    0.044820*np.tanh(np.sin((((((((data["real_ord_3"]) + (((data["te_nom_3"]) + (data["te_nom_7"]))))/2.0)) + ((((np.minimum(((((np.maximum(((data["te_bin_0_1.0"])), ((((((data["te_month"]) + (data["te_nom_3"]))) * 2.0))))) - (data["real_ord_3"])))), ((np.where(data["nans"] > -998, data["te_nom_5"], data["real_ord_3"] ))))) + (data["te_real_ord_0"]))/2.0)))/2.0)))) +
                    0.049998*np.tanh((((((np.maximum((((-((np.sin((np.sin((data["te_bin_0_1.0"]))))))))), ((data["real_ord_2"])))) * (data["te_day"]))) + (((((((data["nans"]) + (((data["te_real_ord_4"]) + (data["te_bin_0_1.0"]))))) / 2.0)) + (np.sin((np.sin(((((data["te_nom_2"]) + (((((data["real_ord_5"]) + (data["te_bin_4_N"]))) / 2.0)))/2.0)))))))))/2.0)) +
                    0.026570*np.tanh(np.maximum(((np.minimum(((np.sin((np.minimum(((((data["te_bin_1_1.0"]) / 2.0))), ((data["te_nom_6"]))))))), ((np.sin((((((((data["te_bin_1_1.0"]) + (data["te_month"]))/2.0)) + (np.where(np.minimum(((data["real_ord_0"])), ((data["te_nom_6"]))) > -998, ((data["te_real_ord_3"]) * 2.0), data["te_month"] )))/2.0)))))))), ((np.minimum(((data["real_ord_0"])), ((((data["real_ord_2"]) * (data["te_real_ord_3"]))))))))) +
                    0.049720*np.tanh((((-((((data["te_real_ord_3"]) * (((((data["te_bin_4_Y"]) - (((data["te_real_ord_2"]) + (((data["te_real_ord_0"]) / 2.0)))))) - ((((((((data["te_month"]) + (((data["te_nom_9"]) + (data["te_bin_4_Y"]))))) + (data["te_nom_8"]))) + (data["te_bin_4_Y"]))/2.0))))))))) * (data["te_real_ord_5"]))) +
                    0.049180*np.tanh((((((data["te_real_ord_2"]) + (np.minimum(((((data["te_nom_1"]) + (data["te_nom_2"])))), ((((data["te_month"]) + (((np.where(np.cos((data["te_month"])) > -998, data["te_month"], data["te_nom_1"] )) + (np.cos((data["te_bin_1_1.0"])))))))))))/2.0)) / 2.0)) +
                    0.039470*np.tanh(np.minimum(((np.maximum(((data["real_ord_5"])), ((((data["te_real_ord_3"]) * 2.0)))))), ((((((((((np.tanh((data["real_ord_1"]))) + ((((((data["te_bin_2_1.0"]) + (data["real_ord_5"]))/2.0)) * (data["te_real_ord_3"]))))/2.0)) + ((((((data["te_bin_2_1.0"]) + (data["real_ord_5"]))/2.0)) * (data["real_ord_5"]))))/2.0)) + (np.sin((((data["te_nom_5"]) * 2.0))))))))) +
                    0.045492*np.tanh(((((((data["te_nom_9"]) * ((((data["te_real_ord_4"]) + ((((data["te_nom_8"]) + (data["te_bin_0_1.0"]))/2.0)))/2.0)))) * 2.0)) * ((((data["te_bin_0_1.0"]) + (((((((((data["te_nom_8"]) + (np.sin((data["real_ord_5"]))))/2.0)) + (((data["bin_4_Y"]) + (np.tanh((np.sin((data["te_nom_9"]))))))))) + (np.sin((data["real_ord_5"]))))/2.0)))/2.0)))) +
                    0.034819*np.tanh(((np.cos((data["real_ord_5"]))) * (((((((data["te_nom_7"]) + ((-((((data["real_ord_3"]) / 2.0))))))/2.0)) + ((((((np.cos((data["bin_0_1.0"]))) + ((((data["real_ord_5"]) + (np.where(data["te_real_ord_3"] < -998, data["bin_0_1.0"], ((((data["te_real_ord_3"]) * (data["bin_0_1.0"]))) * (data["bin_0_1.0"])) )))/2.0)))/2.0)) * (data["bin_0_1.0"]))))/2.0)))) +
                    0.049001*np.tanh(((data["bin_0_1.0"]) * (np.minimum(((((((((data["bin_0_1.0"]) * (data["real_ord_4"]))) * (np.maximum(((data["te_nom_4"])), ((np.where(data["te_nom_7"] > -998, data["te_nom_7"], data["te_nom_4"] ))))))) / 2.0))), ((np.sin((((data["real_ord_4"]) - ((-((np.minimum(((data["real_ord_3"])), ((np.where(data["real_ord_4"] > -998, data["te_nom_7"], data["real_ord_3"] ))))))))))))))))) +
                    0.035700*np.tanh((((((((data["te_real_ord_5"]) + (((data["te_nom_9"]) + (np.minimum(((data["te_nom_7"])), ((data["te_real_ord_5"])))))))/2.0)) + (((data["te_nom_7"]) + (data["te_real_ord_0"]))))) * (((((data["te_nom_8"]) + (data["te_real_ord_2"]))) * (data["te_real_ord_3"]))))) +
                    0.048870*np.tanh((-((((np.minimum(((((np.minimum(((((np.minimum((((0.0))), ((data["te_bin_0_0.0"])))) * (data["te_nom_8"])))), ((np.minimum(((data["te_bin_0_0.0"])), ((data["te_bin_0_0.0"]))))))) * (data["te_nom_8"])))), ((np.minimum(((data["te_real_ord_5"])), ((data["te_bin_0_1.0"]))))))) * (np.where(data["te_bin_0_1.0"] < -998, data["te_nom_8"], ((((data["te_nom_8"]) / 2.0)) / 2.0) ))))))) +
                    0.0*np.tanh(((data["te_real_ord_3"]) * ((((((np.sin((((((((((((np.cos((data["day"]))) + (data["bin_0_1.0"]))/2.0)) * 2.0)) * 2.0)) + (data["bin_0_1.0"]))/2.0)))) * 2.0)) + (data["bin_0_1.0"]))/2.0)))) +
                    0.038040*np.tanh(((((((-((((data["nans"]) * (data["te_real_ord_2"])))))) + (((((-((((data["te_bin_0_1.0"]) * ((((2.88482022285461426)) * (data["te_real_ord_2"])))))))) + (((data["nans"]) * (((np.where(data["real_ord_5"] > -998, data["te_real_ord_3"], data["te_real_ord_2"] )) * ((-(((2.88482022285461426))))))))))/2.0)))/2.0)) / 2.0)) +
                    0.023170*np.tanh(((np.minimum(((np.sin((data["te_nom_6"])))), ((np.maximum(((data["te_bin_2_1.0"])), (((-((((np.minimum(((data["te_nom_0"])), ((data["te_bin_0_1.0"])))) * (((((((data["nans"]) - (data["te_nom_0"]))) / 2.0)) + (((data["nans"]) + (data["te_bin_1_1.0"])))))))))))))))) / 2.0)) +
                    0.045005*np.tanh((((np.maximum(((np.minimum(((data["te_real_ord_0"])), (((((np.sin((data["te_nom_3"]))) + (((data["te_nom_7"]) * (data["te_real_ord_2"]))))/2.0)))))), ((np.minimum(((((np.sin((data["bin_2_1.0"]))) * (data["te_real_ord_2"])))), ((data["real_ord_5"]))))))) + (((np.tanh((data["bin_2_1.0"]))) * (((data["real_ord_5"]) / 2.0)))))/2.0)) +
                    0.048440*np.tanh(((np.where(data["real_ord_5"] < -998, ((np.where(((((np.where(data["te_bin_0_0.0"] > -998, data["real_ord_5"], data["real_ord_5"] )) - (data["te_real_ord_5"]))) - (data["te_real_ord_4"])) < -998, data["te_bin_3_T"], data["real_ord_5"] )) / 2.0), ((((data["real_ord_5"]) - (data["te_real_ord_5"]))) * 2.0) )) * 2.0)) +
                    0.043283*np.tanh(np.minimum(((np.sin((((((data["real_ord_4"]) / 2.0)) * (((np.where(data["real_ord_3"] > -998, data["te_nom_6"], data["real_ord_4"] )) / 2.0))))))), ((((np.maximum(((data["te_day"])), ((((np.maximum(((data["te_real_ord_5"])), ((np.where(data["real_ord_3"] < -998, data["real_ord_4"], data["real_ord_3"] ))))) / 2.0))))) * 2.0))))) +
                    0.047520*np.tanh(((np.minimum(((np.minimum(((np.sin((np.cos((np.minimum(((data["te_nom_7"])), ((((np.sin((((data["te_nom_9"]) * 2.0)))) * 2.0)))))))))), ((((np.cos((np.minimum(((data["te_nom_8"])), ((np.sin((((((data["te_nom_7"]) * 2.0)) * 2.0))))))))) / 2.0)))))), ((np.cos((np.minimum(((data["te_nom_7"])), ((np.sin((data["te_nom_7"]))))))))))) / 2.0)) +
                    0.044310*np.tanh(np.where(((data["real_ord_3"]) - (np.sin((data["real_ord_5"])))) < -998, data["bin_0_1.0"], ((np.tanh((np.sin((np.where(((data["real_ord_5"]) * 2.0) < -998, data["real_ord_3"], np.sin((data["real_ord_5"])) )))))) - (np.sin((((data["real_ord_5"]) * 2.0))))) )) +
                    0.049185*np.tanh(((data["bin_0_1.0"]) * (np.maximum(((data["te_bin_0_0.0"])), ((np.minimum(((data["bin_0_1.0"])), ((np.cos((((((np.tanh((np.minimum(((data["bin_0_1.0"])), ((np.tanh((np.minimum(((np.minimum(((data["te_nom_9"])), ((data["month"]))))), ((np.cos((((np.minimum(((data["te_real_ord_3"])), ((np.cos((((data["te_nom_9"]) * 2.0))))))) * 2.0)))))))))))))) * 2.0)) * 2.0)))))))))))) +
                    0.014602*np.tanh(((((np.sin(((((np.where(data["real_ord_5"] < -998, np.cos(((3.141593))), data["te_real_ord_2"] )) + (np.sin(((((np.sin(((((np.cos(((3.141593)))) + (data["real_ord_5"]))/2.0)))) + (data["bin_2_1.0"]))/2.0)))))/2.0)))) * (data["te_month"]))) * (data["te_real_ord_3"]))) +
                    0.028560*np.tanh((((((data["te_nom_7"]) + (np.minimum(((data["te_bin_2_1.0"])), ((((((((data["nans"]) + (((data["nans"]) + ((-(((-1.0))))))))) + ((-((((data["te_bin_1_0.0"]) / 2.0))))))) + (data["nans"])))))))/2.0)) * ((-((((data["te_bin_1_1.0"]) / 2.0))))))) +
                    0.026585*np.tanh(np.where(np.minimum(((data["day"])), (((-((data["real_ord_3"])))))) < -998, ((data["real_ord_3"]) * 2.0), (-((np.tanh((((data["te_day"]) * (np.sin((((np.sin((((np.where(np.minimum(((data["day"])), ((((data["real_ord_3"]) * 2.0)))) < -998, data["te_nom_8"], data["real_ord_3"] )) / 2.0)))) / 2.0)))))))))) )) +
                    0.034420*np.tanh(((data["bin_0_1.0"]) * (np.sin((((data["te_real_ord_3"]) + (((data["te_nom_8"]) - ((-(((((data["te_nom_1"]) + (((np.sin((((data["bin_0_1.0"]) + (((data["real_ord_0"]) - (np.tanh(((-(((((((data["bin_0_1.0"]) + (data["real_ord_1"]))/2.0)) * 2.0))))))))))))) * 2.0)))/2.0))))))))))))) +
                    0.017700*np.tanh(((((data["te_real_ord_3"]) * ((((((data["te_bin_2_0.0"]) * (data["te_month"]))) + (np.maximum((((-((np.where(data["te_real_ord_3"] < -998, data["te_month"], data["te_month"] )))))), ((((data["te_real_ord_2"]) * (data["te_real_ord_4"])))))))/2.0)))) / 2.0)) +
                    0.030000*np.tanh((((-((np.where(data["day"] < -998, data["te_bin_1_1.0"], ((((np.minimum(((data["te_bin_2_1.0"])), ((((((data["te_real_ord_2"]) * (data["te_bin_2_1.0"]))) - (data["day"])))))) - (data["te_real_ord_0"]))) * (((((data["te_nom_6"]) * (data["te_bin_2_1.0"]))) / 2.0))) ))))) / 2.0)) +
                    0.014400*np.tanh(((((data["te_nom_2"]) * (np.where(data["real_ord_4"] < -998, data["te_real_ord_3"], np.sin((np.sin(((((((-((data["te_real_ord_3"])))) * 2.0)) - (np.where(data["real_ord_3"] < -998, data["bin_2_1.0"], ((((data["bin_2_1.0"]) * (data["te_nom_2"]))) / 2.0) ))))))) )))) / 2.0)) +
                    0.003110*np.tanh(np.tanh(((((((((data["te_nom_0"]) * (((((data["bin_0_0.0"]) - ((((data["nans"]) + (((data["te_real_ord_1"]) / 2.0)))/2.0)))) * 2.0)))) + ((((((((data["bin_1_0.0"]) * 2.0)) + (data["te_nom_8"]))/2.0)) * (((np.cos((data["te_bin_0_1.0"]))) * (((data["te_nom_8"]) * (data["te_bin_2_1.0"]))))))))/2.0)) / 2.0)))) +
                    0.0*np.tanh(np.where(data["day"] < -998, ((data["real_ord_5"]) + (np.cos((((np.cos((((data["te_month"]) + (data["real_ord_1"]))))) + (data["te_real_ord_2"])))))), np.sin((((data["te_real_ord_5"]) * ((-((np.cos((((((data["te_month"]) + (data["real_ord_1"]))) + (data["te_real_ord_2"]))))))))))) )) +
                    0.042930*np.tanh(((data["bin_0_0.0"]) * (np.sin((((data["real_ord_0"]) * (np.minimum(((np.minimum(((data["real_ord_5"])), ((((((((((data["te_nom_8"]) + (np.tanh((data["te_real_ord_4"]))))/2.0)) + ((((((data["te_nom_8"]) + (data["te_real_ord_4"]))/2.0)) * 2.0)))/2.0)) * 2.0)))))), (((((data["te_nom_8"]) + ((((data["te_nom_8"]) + (data["te_real_ord_4"]))/2.0)))/2.0))))))))))) +
                    0.043940*np.tanh(((data["bin_0_1.0"]) * (np.minimum(((((data["te_real_ord_5"]) * (((data["te_real_ord_1"]) * (data["bin_0_1.0"])))))), (((((2.0)) * (np.minimum(((((data["te_real_ord_5"]) * (((data["te_real_ord_5"]) * (np.where(data["bin_0_1.0"] < -998, data["te_bin_1_0.0"], ((data["te_real_ord_1"]) * (data["bin_0_1.0"])) ))))))), ((np.maximum(((data["bin_0_1.0"])), ((data["real_ord_3"])))))))))))))) +
                    0.044472*np.tanh((((-((data["bin_0_0.0"])))) * (np.sin((((np.maximum(((data["te_month"])), ((((((((data["te_real_ord_1"]) + (data["te_nom_3"]))/2.0)) + (np.where(data["real_ord_1"] < -998, data["real_ord_3"], data["te_bin_0_1.0"] )))/2.0))))) / 2.0)))))) +
                    0.000004*np.tanh(((data["te_real_ord_2"]) * (((((np.sin((data["real_ord_3"]))) - ((-((((np.sin((data["te_nom_9"]))) - ((-((((data["te_nom_3"]) - ((-((data["real_ord_5"])))))))))))))))) * (((((np.sin((data["month"]))) - ((-((np.sin((data["real_ord_3"])))))))) / 2.0)))))) +
                    0.025296*np.tanh((((1.0)) - (np.sin((((((np.cos((data["real_ord_2"]))) + (np.cos((np.maximum(((data["real_ord_5"])), ((np.cos((data["real_ord_2"])))))))))) + (np.cos((np.maximum(((data["te_real_ord_3"])), ((np.minimum(((np.cos((((data["te_real_ord_3"]) * 2.0))))), ((data["real_ord_0"]))))))))))))))) +
                    0.025824*np.tanh(np.sin((((((np.cos((np.maximum(((data["te_bin_2_1.0"])), ((data["real_ord_5"])))))) - (((data["te_real_ord_0"]) + (((data["te_nom_8"]) + (data["real_ord_2"]))))))) - (((data["real_ord_3"]) + (data["te_nom_7"]))))))) +
                    0.046440*np.tanh(np.minimum((((0.0))), ((np.maximum(((data["te_nom_5"])), (((((((np.minimum(((np.cos((data["real_ord_1"])))), ((np.maximum(((np.maximum(((data["te_nom_7"])), ((((((((np.cos((data["te_nom_7"]))) + (((data["te_nom_9"]) * (np.maximum(((data["bin_4_Y"])), ((data["te_nom_9"])))))))/2.0)) + (data["te_nom_7"]))/2.0)))))), ((data["te_nom_5"]))))))) * 2.0)) + (data["te_nom_7"]))/2.0)))))))) +
                    0.041090*np.tanh(((((((data["te_real_ord_2"]) * (data["te_real_ord_3"]))) / 2.0)) * (((((data["te_month"]) / 2.0)) + (((((data["real_ord_4"]) / 2.0)) + ((((data["te_bin_3_F"]) + (((((data["te_real_ord_3"]) / 2.0)) + (((data["te_bin_2_1.0"]) + (np.where(data["te_real_ord_2"] > -998, data["te_nom_1"], ((data["real_ord_4"]) / 2.0) )))))))/2.0)))))))) +
                    0.023230*np.tanh(np.where(np.where(data["real_ord_3"] > -998, data["te_real_ord_3"], data["real_ord_3"] ) < -998, ((data["nans"]) * (data["te_real_ord_1"])), np.where(((data["real_ord_4"]) - (data["te_nom_0"])) < -998, ((data["te_nom_3"]) * 2.0), np.sin((((((data["te_real_ord_3"]) * (((data["te_real_ord_0"]) * (((data["te_real_ord_5"]) * (((data["te_bin_0_1.0"]) * 2.0)))))))) / 2.0))) ) )) +
                    0.034410*np.tanh(((((data["te_nom_0"]) * ((-((np.where(data["real_ord_0"] < -998, data["nans"], np.sin((np.where(data["month"] < -998, data["real_ord_0"], np.sin((np.sin((((data["nans"]) + (np.where(((data["real_ord_0"]) + (data["real_ord_0"])) < -998, data["real_ord_0"], ((data["month"]) + (data["real_ord_0"])) ))))))) ))) ))))))) / 2.0)) +
                    0.002430*np.tanh(((np.where(data["real_ord_4"] < -998, data["day"], ((data["te_nom_1"]) * (np.sin(((-((np.where(data["day"] < -998, data["real_ord_2"], np.where((0.0) < -998, data["day"], ((((data["bin_1_0.0"]) * 2.0)) * ((-((np.tanh((np.cos((data["day"]))))))))) ) )))))))) )) / 2.0)) +
                    0.043500*np.tanh(np.minimum((((0.0))), ((np.maximum(((np.minimum(((np.maximum(((data["te_month"])), ((np.where(data["te_month"] > -998, data["real_ord_0"], data["te_real_ord_3"] )))))), ((np.minimum(((np.maximum(((data["te_month"])), ((data["te_nom_8"]))))), ((np.maximum(((data["te_real_ord_3"])), ((np.maximum(((data["te_real_ord_2"])), ((np.where(data["te_real_ord_3"] < -998, data["te_nom_8"], data["te_nom_8"] ))))))))))))))), ((data["te_real_ord_3"]))))))) +
                    0.049399*np.tanh(((data["bin_0_1.0"]) * (((data["bin_0_1.0"]) * ((((data["te_nom_3"]) + ((((((data["te_real_ord_3"]) + (np.maximum(((data["te_real_ord_3"])), ((data["te_real_ord_0"])))))) + (np.minimum(((((np.tanh((data["te_nom_9"]))) + (((data["te_real_ord_3"]) + (np.maximum(((data["te_real_ord_3"])), ((data["te_real_ord_0"]))))))))), ((((data["bin_0_1.0"]) * (data["te_nom_2"])))))))/2.0)))/2.0)))))) +
                    0.049815*np.tanh(((((data["te_nans"]) * (np.cos((((data["te_nom_6"]) + (np.where(data["real_ord_5"] < -998, ((data["bin_3_F"]) + (data["real_ord_2"])), ((data["real_ord_5"]) + (np.where(((data["real_ord_2"]) * 2.0) < -998, ((data["real_ord_5"]) + (np.where(data["real_ord_2"] < -998, (6.49070024490356445), data["te_nans"] ))), data["bin_3_F"] ))) )))))))) / 2.0)) +
                    0.009500*np.tanh(((data["te_bin_2_0.0"]) * (np.sin(((((-((np.maximum(((data["te_real_ord_1"])), ((((np.maximum(((np.minimum(((data["real_ord_3"])), ((np.cos((data["bin_2_1.0"]))))))), ((np.minimum(((np.maximum(((np.minimum(((data["te_real_ord_0"])), ((np.tanh((data["te_real_ord_5"]))))))), ((data["real_ord_3"]))))), ((np.tanh((np.maximum(((data["te_real_ord_5"])), ((data["te_real_ord_5"])))))))))))) * 2.0)))))))) * 2.0)))))) +
                    0.0*np.tanh(((((((np.sin((((data["real_ord_4"]) - (data["te_real_ord_4"]))))) * 2.0)) * 2.0)) * 2.0)) +
                    0.040760*np.tanh(((((np.sin((data["te_bin_1_1.0"]))) * (np.maximum(((data["real_ord_5"])), ((((data["day"]) + (((data["day"]) + ((-((((((np.maximum(((data["te_nom_0"])), ((((np.maximum(((data["real_ord_5"])), ((((np.sin((data["te_bin_2_1.0"]))) + (data["te_real_ord_2"])))))) * 2.0))))) * 2.0)) * 2.0)))))))))))))) / 2.0)) +
                    0.0*np.tanh(((((data["te_month"]) * (np.sin((((((((data["real_ord_3"]) + (((data["te_month"]) * (np.tanh(((((data["real_ord_5"]) + (((((data["real_ord_3"]) * (data["te_real_ord_0"]))) * (data["te_nom_9"]))))/2.0)))))))/2.0)) + (((data["real_ord_5"]) + ((((data["te_real_ord_0"]) + (data["te_nom_9"]))/2.0)))))/2.0)))))) * (data["te_real_ord_2"]))) +
                    0.020989*np.tanh(np.where(data["real_ord_4"] < -998, np.sin((data["real_ord_4"])), ((((data["te_nom_8"]) * (np.sin((np.sin((data["te_nom_5"]))))))) * (np.minimum((((-((data["bin_1_1.0"]))))), ((np.minimum(((np.maximum(((data["real_ord_4"])), ((data["te_nom_5"]))))), ((np.maximum(((data["bin_2_1.0"])), (((-((data["te_bin_0_0.0"])))))))))))))) )) +
                    0.017192*np.tanh(((((data["real_ord_0"]) * (((data["real_ord_3"]) * (((data["te_nom_7"]) + ((((((1.570796)) * (((data["te_nom_9"]) + (((np.tanh((data["bin_2_1.0"]))) + (((((data["bin_2_1.0"]) * (data["te_month"]))) + (data["te_real_ord_2"]))))))))) + ((((1.570796)) * (data["te_month"]))))))))))) / 2.0)) +
                    0.0*np.tanh(((data["te_real_ord_0"]) * (((np.maximum((((-((np.cos((np.maximum(((data["real_ord_3"])), ((((data["te_month"]) + (data["te_real_ord_2"])))))))))))), (((((((((((-((np.maximum(((data["real_ord_3"])), ((((data["te_month"]) + (data["te_real_ord_2"]))))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0))))) / 2.0)))) +
                    0.008157*np.tanh(((data["te_bin_0_0.0"]) * (((((((data["te_nom_0"]) + (data["real_ord_5"]))/2.0)) + ((-((((data["real_ord_5"]) + (np.minimum(((((data["bin_0_1.0"]) * (np.minimum(((data["te_bin_1_0.0"])), ((data["te_nom_9"]))))))), ((((data["bin_2_1.0"]) * (np.minimum(((data["month"])), (((((data["te_nom_0"]) + (data["te_bin_1_0.0"]))/2.0))))))))))))))))/2.0)))) +
                    0.014110*np.tanh(np.tanh((np.tanh((np.tanh((((data["te_real_ord_2"]) * (((((data["te_nom_4"]) * 2.0)) * (((data["real_ord_3"]) + (np.sin((((np.maximum(((np.maximum(((data["bin_4_N"])), ((np.maximum(((((data["te_nom_4"]) * 2.0))), ((((data["te_nom_1"]) * 2.0))))))))), ((((data["te_nom_0"]) * (data["te_nom_4"])))))) * (data["te_bin_2_1.0"]))))))))))))))))) +
                    0.0*np.tanh(((np.maximum(((np.where(data["te_real_ord_4"] > -998, np.maximum(((((data["bin_0_1.0"]) / 2.0))), ((((data["bin_2_1.0"]) * ((((((data["te_nom_7"]) + (((((((data["te_real_ord_4"]) + (data["te_month"]))/2.0)) + (data["te_month"]))/2.0)))/2.0)) / 2.0)))))), data["te_real_ord_4"] ))), ((((data["te_real_ord_3"]) * (data["te_real_ord_4"])))))) * ((((data["real_ord_0"]) + (data["te_month"]))/2.0)))) +
                    0.025100*np.tanh(((np.cos((np.minimum(((data["real_ord_3"])), ((data["bin_4_Y"])))))) - (np.cos((np.minimum(((((data["bin_4_Y"]) * (np.where(data["real_ord_3"] > -998, data["real_ord_3"], ((data["bin_4_Y"]) * (np.where(data["real_ord_3"] > -998, data["real_ord_3"], data["te_real_ord_2"] ))) ))))), ((((data["bin_3_T"]) * (data["te_nom_0"])))))))))) +
                    0.033340*np.tanh(((data["te_real_ord_2"]) * (((np.maximum(((np.sin((((data["te_real_ord_1"]) * (((np.maximum(((data["te_nom_1"])), ((((data["te_real_ord_0"]) * (np.maximum(((data["te_nom_9"])), ((data["te_real_ord_3"]))))))))) / 2.0))))))), ((np.maximum(((((data["te_bin_0_1.0"]) * (data["real_ord_0"])))), ((((data["te_real_ord_3"]) * (data["te_nom_9"]))))))))) / 2.0)))) +
                    0.0*np.tanh(np.sin((((data["te_nom_1"]) * ((((((data["real_ord_0"]) + (np.where(data["real_ord_0"] < -998, data["real_ord_0"], np.where(data["real_ord_0"] < -998, data["te_nom_8"], data["te_nom_8"] ) )))/2.0)) * ((((((data["real_ord_0"]) + (np.where(data["real_ord_0"] < -998, data["te_nom_8"], data["te_bin_2_1.0"] )))/2.0)) / 2.0)))))))) +
                    0.028169*np.tanh((-((((np.where(data["te_real_ord_3"] < -998, data["te_real_ord_2"], np.sin((((((data["te_real_ord_2"]) + (((data["te_real_ord_3"]) + (data["te_real_ord_4"]))))) + (data["real_ord_5"])))) )) + (np.sin((((((((data["te_real_ord_3"]) + (data["te_nom_7"]))) + (data["real_ord_5"]))) + (data["te_nom_7"])))))))))))


# In[38]:


roc_auc_score(y_train,GP(df_train[df_train.columns[:-1]].fillna(-999)))


# In[39]:


def GPComplex(data):
    return Output(  0.009000*np.tanh(np.real(((data["te_nom_8"]) + (((((data["te_nom_9"]) + (((((((data["real_ord_3"]) * 2.0)) + (((((data["te_real_ord_2"]) + (((((data["te_real_ord_5"]) + (((data["real_ord_0"]) + (((data["te_real_ord_4"]) + (data["te_nom_8"]))))))) + (data["te_nom_7"]))))) + (data["te_month"]))))) * 2.0)))) * 2.0))))) +
                    0.010475*np.tanh(np.real(((data["real_ord_4"]) + (((((((((data["te_real_ord_3"]) * 2.0)) + (data["te_month"]))) * 2.0)) + (((data["real_ord_5"]) + (((((((data["te_nom_8"]) + ((((data["te_nom_1"]) + (data["te_real_ord_0"]))/2.0)))) * 2.0)) + (((data["te_real_ord_2"]) + (((data["te_nom_3"]) + (((data["te_real_ord_2"]) + (data["real_ord_5"])))))))))))))))) +
                    0.003510*np.tanh(np.real(((((((((data["real_ord_0"]) - (complex(0.636620)))) + (data["te_real_ord_2"]))) + (((np.sinh((data["real_ord_3"]))) + (data["te_nom_8"]))))) + (np.sinh((((data["real_ord_3"]) + (((data["te_nom_7"]) + ((((((((((data["te_nom_1"]) - (data["te_month"]))) + (data["te_real_ord_5"]))/2.0)) + (data["te_month"]))) * 2.0))))))))))) +
                    0.012129*np.tanh(np.real(((((data["te_month"]) + (((data["te_nom_9"]) + ((((((((((((((((np.sinh((data["te_real_ord_0"]))) + (data["te_nom_8"]))) + (data["te_month"]))) + (((data["te_real_ord_3"]) * 2.0)))) + (data["te_real_ord_2"]))) + (np.conjugate(((data["te_nom_7"]) + (data["te_real_ord_5"])))))/2.0)) * 2.0)) * 2.0)))))) + (data["te_nom_1"])))) +
                    0.014315*np.tanh(np.real(((((((data["real_ord_5"]) + (((((data["real_ord_3"]) + (data["te_real_ord_2"]))) + (data["te_nom_8"]))))) * 2.0)) + (((np.conjugate(((((data["te_real_ord_0"]) + (((((data["te_day"]) + (data["real_ord_3"]))) + (data["te_real_ord_1"]))))) + (data["te_real_ord_4"])))) + (((data["te_nom_7"]) - (np.cosh((data["real_ord_3"])))))))))) +
                    0.0*np.tanh(np.real(((((((data["te_real_ord_2"]) + (((((((data["real_ord_0"]) + (data["real_ord_3"]))) + (((data["te_real_ord_4"]) + (((((((((data["te_nom_8"]) + (data["te_real_ord_5"]))) + (((data["te_real_ord_2"]) + (data["real_ord_3"]))))) + (((data["te_month"]) + (np.conjugate(data["te_nom_1"])))))) * 2.0)))))) * 2.0)))) * 2.0)) * 2.0))) +
                    0.006388*np.tanh(np.real(((((((((data["te_real_ord_5"]) + (((data["real_ord_3"]) + ((((data["real_ord_0"]) + (data["te_nom_3"]))/2.0)))))) + (((((np.tanh((data["real_ord_0"]))) + (((data["te_month"]) + (((np.tanh((data["te_bin_2_0.0"]))) + (((data["te_real_ord_3"]) + (data["te_real_ord_4"]))))))))) + (data["te_real_ord_2"]))))) * 2.0)) * 2.0))) +
                    0.038000*np.tanh(np.real(((((((((((((data["te_real_ord_3"]) + (((((data["te_nom_7"]) + (data["te_real_ord_4"]))) + (data["te_real_ord_3"]))))) + (data["te_month"]))) + (data["te_real_ord_0"]))) + (((((data["real_ord_5"]) + (data["te_nom_8"]))) + (((data["real_ord_1"]) + (data["te_real_ord_2"]))))))) * 2.0)) + (((data["te_real_ord_2"]) + (data["te_nom_9"])))))) +
                    0.031940*np.tanh(np.real(((((((data["te_real_ord_3"]) + (data["te_nom_3"]))) + (data["real_ord_1"]))) + (((((((data["real_ord_5"]) + (((data["te_bin_2_0.0"]) + (((((data["real_ord_0"]) + (data["te_nom_8"]))) + (data["te_real_ord_3"]))))))) + (((((data["real_ord_2"]) + (data["month"]))) + (data["te_nom_7"]))))) * 2.0))))) +
                    0.027526*np.tanh(np.real(((((((data["te_real_ord_3"]) + (((((data["te_nom_9"]) + ((((((data["real_ord_0"]) + (data["te_real_ord_3"]))/2.0)) + (data["te_month"]))))) + (((data["te_real_ord_2"]) + (data["te_real_ord_5"]))))))) + ((((((np.sinh((((data["te_nom_8"]) + (data["te_real_ord_4"]))))) + (data["real_ord_1"]))) + (data["bin_2_1.0"]))/2.0)))) * 2.0))) +
                    0.031490*np.tanh(np.real(((((((data["te_nom_5"]) + (((data["te_real_ord_3"]) + (np.conjugate(((((((data["te_real_ord_3"]) + (((((((data["te_nom_9"]) + (((data["real_ord_2"]) + (((data["real_ord_5"]) + (data["te_real_ord_0"]))))))) + (data["te_nom_7"]))) + (data["month"]))))) + (data["te_day"]))) * 2.0))))))) + (data["real_ord_1"]))) * 2.0))) +
                    0.018550*np.tanh(np.real(((((((data["te_nom_7"]) + (((((((data["te_real_ord_5"]) + (((((((data["te_nom_8"]) + (data["te_real_ord_2"]))) + (data["real_ord_0"]))) + (data["real_ord_3"]))))) + (((data["real_ord_4"]) + (data["real_ord_3"]))))) + (np.conjugate(np.conjugate(((data["te_day"]) + (np.sinh((data["te_month"]))))))))))) * 2.0)) * 2.0))) +
                    0.013530*np.tanh(np.real(((((data["real_ord_1"]) + (data["real_ord_0"]))) + ((((((((((((data["te_day"]) + (data["real_ord_4"]))) + (((np.sin((data["real_ord_3"]))) + (((((data["te_month"]) + (((((data["te_real_ord_3"]) + (data["real_ord_5"]))) + (data["real_ord_2"]))))) * 2.0)))))) + (data["te_bin_2_1.0"]))/2.0)) + (data["te_nom_8"]))) * 2.0))))) +
                    0.049868*np.tanh(np.real(((((((data["te_month"]) + (data["te_nom_9"]))) + (data["te_real_ord_3"]))) + (((data["te_real_ord_4"]) + (((data["te_real_ord_3"]) + (((((data["te_real_ord_2"]) + (((((data["te_nom_1"]) + (((data["te_nom_5"]) + (data["te_bin_2_0.0"]))))) + (((data["real_ord_5"]) + (data["bin_0_0.0"]))))))) + (data["te_nom_7"])))))))))) +
                    0.048000*np.tanh(np.real(((((((data["real_ord_0"]) + (((((((data["te_nom_8"]) + (complex(-1.0)))) + (data["te_nom_7"]))) + (((data["real_ord_4"]) + (((data["te_nom_1"]) + (data["te_nom_9"]))))))))) + (data["real_ord_3"]))) + (((data["te_month"]) + (((((data["te_real_ord_3"]) + (data["te_real_ord_5"]))) + (data["te_real_ord_2"])))))))) +
                    0.048681*np.tanh(np.real(((((((data["te_real_ord_3"]) + (data["te_real_ord_5"]))) + (data["te_real_ord_4"]))) + (((((data["te_real_ord_3"]) + (data["te_month"]))) - ((-((((((((data["te_nom_7"]) + (data["real_ord_0"]))) + (((data["te_nom_8"]) - ((-((((data["real_ord_1"]) + (data["te_real_ord_2"])))))))))) + (data["te_nom_3"]))))))))))) +
                    0.009795*np.tanh(np.real(((((data["real_ord_2"]) + (((((data["real_ord_4"]) + (((data["te_nom_9"]) + (data["te_nom_7"]))))) + (((((((data["te_real_ord_3"]) * 2.0)) + (data["te_nom_8"]))) - (np.cos((np.sqrt((((data["te_bin_0_0.0"]) + (data["te_day"]))))))))))))) + (((data["real_ord_5"]) + (data["te_real_ord_0"])))))) +
                    0.005996*np.tanh(np.real(((((np.sinh((data["real_ord_2"]))) + (((data["te_real_ord_3"]) + (data["te_real_ord_1"]))))) + (((((((((data["te_nom_9"]) + (data["te_month"]))) + (((((((data["te_day"]) + (data["real_ord_4"]))) + (data["te_nom_7"]))) + (((data["te_real_ord_3"]) + (complex(-1.0)))))))) + (data["real_ord_0"]))) + (data["te_real_ord_5"])))))) +
                    0.044472*np.tanh(np.real(((((((data["te_real_ord_5"]) + (((data["te_bin_2_1.0"]) + (((complex(-1.0)) + (((data["te_nom_3"]) + (((data["te_nom_9"]) + (data["te_real_ord_0"]))))))))))) + (((((data["real_ord_3"]) + (((((data["te_nom_8"]) + (data["te_month"]))) + (data["te_bin_0_1.0"]))))) + (data["te_real_ord_3"]))))) + (data["te_real_ord_2"])))) +
                    0.036947*np.tanh(np.real(((data["te_month"]) + (((data["te_nom_8"]) + (((((((data["real_ord_2"]) + (((data["real_ord_3"]) + (np.conjugate(data["te_nom_7"])))))) * 2.0)) + (((((((data["te_bin_2_0.0"]) + (complex(-1.0)))) + (((data["te_real_ord_4"]) + (data["te_real_ord_5"]))))) + (((data["te_real_ord_0"]) + (data["te_nom_1"])))))))))))) +
                    0.048511*np.tanh(np.real(((np.sinh((complex(-1.0)))) + (((data["real_ord_0"]) + (((data["te_nom_7"]) + (((((data["te_real_ord_2"]) + (((data["te_real_ord_3"]) + (((data["te_real_ord_1"]) + (((data["te_month"]) + (((data["te_nom_1"]) + (data["real_ord_5"]))))))))))) + (((data["te_day"]) + (((data["te_nom_9"]) + (data["real_ord_4"])))))))))))))) +
                    0.034525*np.tanh(np.real(((((((((((data["te_real_ord_2"]) + (data["te_bin_0_1.0"]))) + (data["te_real_ord_1"]))) + (((data["te_real_ord_4"]) + (data["te_month"]))))) + (data["te_real_ord_0"]))) + (((((data["bin_2_1.0"]) + (((((data["te_nom_7"]) + (data["te_day"]))) + (data["te_real_ord_5"]))))) + (np.sinh((((data["real_ord_3"]) - (complex(1.0))))))))))) +
                    0.040610*np.tanh(np.real(((data["real_ord_5"]) + (((((((data["real_ord_3"]) + (((data["te_nom_8"]) + (data["te_real_ord_2"]))))) + (data["real_ord_4"]))) + (((((data["te_real_ord_0"]) + (((((data["te_nom_1"]) - (np.cosh((data["real_ord_3"]))))) + (((data["te_nom_9"]) + (((data["bin_2_1.0"]) + (data["real_ord_1"]))))))))) + (data["te_real_ord_3"])))))))) +
                    0.049501*np.tanh(np.real(((data["te_nom_8"]) + (((((data["te_bin_0_0.0"]) + (((((data["te_real_ord_1"]) + (data["real_ord_4"]))) + (((((data["te_real_ord_0"]) + (((((data["te_real_ord_2"]) + (((data["te_nom_3"]) + (((complex(-1.0)) + (data["te_month"]))))))) + (data["te_nom_7"]))))) + (data["real_ord_5"]))))))) + (data["real_ord_3"])))))) +
                    0.049955*np.tanh(np.real(((data["te_nom_8"]) + (((((((((data["te_bin_0_1.0"]) + (data["te_nom_9"]))) + (((data["te_real_ord_2"]) + (((data["te_month"]) + (data["te_nom_2"]))))))) + (((data["real_ord_0"]) + (np.sin((data["te_bin_2_0.0"]))))))) + (((data["te_nom_7"]) + (((((data["real_ord_5"]) + (data["te_nom_4"]))) + (data["real_ord_3"])))))))))) +
                    0.035810*np.tanh(np.real(((((data["te_nom_7"]) + (((data["te_real_ord_2"]) + (data["te_real_ord_3"]))))) + (((((((data["te_real_ord_5"]) + (data["real_ord_0"]))) - (((data["bin_0_1.0"]) + ((-((data["te_nom_5"])))))))) + (((((data["te_real_ord_3"]) - (((complex(1.0)) - (data["te_day"]))))) + (data["te_nom_8"])))))))) +
                    0.049505*np.tanh(np.real(((data["te_nom_2"]) + (((((data["real_ord_3"]) + (((((data["te_nom_7"]) + (((((data["te_real_ord_5"]) + (((data["te_nom_1"]) + (data["te_nom_5"]))))) + (complex(-1.0)))))) + ((((((np.sinh((data["te_month"]))) + (data["real_ord_0"]))/2.0)) + ((((((data["real_ord_2"]) + (data["te_nom_8"]))/2.0)) * 2.0)))))))) * 2.0))))) +
                    0.049705*np.tanh(np.real(((data["te_real_ord_2"]) + (((((((((((((data["real_ord_3"]) + (data["te_month"]))) + (np.sin((data["te_nom_8"]))))) * 2.0)) + (((((complex(-1.0)) + (data["te_nom_2"]))) + (((data["te_real_ord_1"]) + ((((((data["te_nom_9"]) + (data["te_nom_5"]))/2.0)) * 2.0)))))))) + (data["te_real_ord_4"]))) + (data["te_nom_1"])))))) +
                    0.049980*np.tanh(np.real(((data["bin_2_1.0"]) + (((data["te_nom_1"]) + (((((((data["te_nom_3"]) + (data["real_ord_4"]))) + (data["te_nom_8"]))) + (((data["real_ord_5"]) + (((complex(-1.0)) + (((((data["te_month"]) + ((((((data["te_nom_9"]) + (data["real_ord_0"]))/2.0)) + (data["te_real_ord_2"]))))) + (data["real_ord_3"])))))))))))))) +
                    0.049945*np.tanh(np.real(((((((((((((data["real_ord_0"]) - (np.cos((np.sqrt((data["te_real_ord_5"]))))))) + (data["te_nom_3"]))) + (data["te_real_ord_2"]))) + (((data["te_day"]) + (data["te_nom_7"]))))) + (data["te_real_ord_1"]))) + ((((((data["te_nom_1"]) + (data["te_nom_8"]))/2.0)) + (((data["te_real_ord_3"]) + (data["te_nom_4"])))))))) +
                    0.039107*np.tanh(np.real(((np.sinh((data["te_bin_0_1.0"]))) + (((((((((((data["te_day"]) + (data["real_ord_4"]))) + (((data["te_real_ord_5"]) + (((((((data["bin_2_1.0"]) + (data["real_ord_2"]))) + (data["real_ord_0"]))) + (data["te_month"]))))))) + (((data["te_nom_8"]) + (data["real_ord_3"]))))) * 2.0)) + (np.sinh((data["te_nom_7"])))))))) +
                    0.049750*np.tanh(np.real(((data["te_nom_7"]) + (((data["te_nom_9"]) + (((((((((data["te_real_ord_5"]) - (((complex(0.636620)) - (((data["te_nom_2"]) + (((data["te_nom_1"]) + ((((data["te_month"]) + (data["real_ord_2"]))/2.0)))))))))) + (((data["real_ord_3"]) + (data["te_day"]))))) + (data["real_ord_0"]))) + (data["te_nom_3"])))))))) +
                    0.030000*np.tanh(np.real(((data["te_real_ord_4"]) + (((data["te_nom_5"]) + (((np.sinh((((((data["real_ord_2"]) + (data["te_real_ord_5"]))) + (((data["real_ord_3"]) + (((data["te_nom_8"]) - (np.cos((np.sqrt((((((data["te_nom_7"]) + (data["te_nom_9"]))) * 2.0)))))))))))))) + (((data["te_nom_3"]) - (data["bin_2_0.0"])))))))))) +
                    0.049915*np.tanh(np.real(((data["te_nom_5"]) + (((((((data["real_ord_5"]) + (((((((data["te_bin_0_0.0"]) + (((data["te_nom_4"]) + (((data["te_real_ord_4"]) + (((data["te_nom_3"]) + (((data["real_ord_1"]) + (data["real_ord_0"]))))))))))) + (data["te_bin_2_1.0"]))) + (((data["te_nom_8"]) + (data["te_real_ord_3"]))))))) - (complex(1.570796)))) * 2.0))))) +
                    0.049963*np.tanh(np.real(((data["te_month"]) - ((-((((((data["te_nom_8"]) + (((((((data["te_nom_5"]) + (data["te_nom_7"]))) + (((data["te_nom_4"]) + (data["te_nom_9"]))))) + (((data["real_ord_1"]) - (complex(0.318310)))))))) + (((data["real_ord_3"]) + (((data["te_nom_1"]) + (((data["real_ord_5"]) + (data["te_bin_0_1.0"]))))))))))))))) +
                    0.049498*np.tanh(np.real(((data["te_real_ord_3"]) + (((((((data["te_nom_7"]) + (((data["te_bin_2_1.0"]) + (((data["te_nom_2"]) + (((data["real_ord_1"]) + (((data["te_nom_9"]) + (data["te_month"]))))))))))) + (((data["te_nom_4"]) + (((data["te_real_ord_2"]) + (((data["real_ord_0"]) + (data["te_real_ord_4"]))))))))) - (np.sqrt((complex(0.636620))))))))) +
                    0.049984*np.tanh(np.real(((((((data["te_nom_8"]) + (((data["te_bin_4_N"]) + (data["te_nom_1"]))))) + (((data["te_month"]) + (data["real_ord_2"]))))) + (((((((data["te_nom_9"]) + (((((data["te_nom_3"]) + (data["te_real_ord_3"]))) + (data["te_real_ord_5"]))))) + (data["te_nom_2"]))) + (((data["te_day"]) - (np.cosh((data["bin_0_1.0"])))))))))) +
                    0.045810*np.tanh(np.real(((data["real_ord_4"]) + (((((((data["te_day"]) + (((data["te_nom_8"]) + (((((data["te_nom_3"]) + (data["te_nom_2"]))) + (data["real_ord_0"]))))))) + (((((((data["real_ord_2"]) + (data["te_nom_7"]))) + (data["bin_2_1.0"]))) + (data["te_real_ord_3"]))))) - (((complex(1.0)) + (data["bin_0_1.0"])))))))) +
                    0.047890*np.tanh(np.real(((((((((np.conjugate(data["te_nom_5"])) + (data["te_real_ord_1"]))) + (((((data["te_nom_2"]) + (data["te_real_ord_2"]))) + (data["te_bin_1_1.0"]))))) - (np.sqrt((np.cosh((data["te_bin_0_1.0"]))))))) + (np.conjugate(((((((data["te_real_ord_0"]) + (data["te_real_ord_5"]))) + (data["te_month"]))) + (data["real_ord_3"]))))))) +
                    0.049928*np.tanh(np.real(((((data["te_nom_5"]) + (((data["te_real_ord_4"]) + (((data["te_nom_0"]) + (((data["te_nom_8"]) + (((data["real_ord_5"]) + (((((data["te_nom_1"]) + (((data["te_nom_9"]) - (np.cos((data["bin_2_1.0"]))))))) + (((data["te_bin_0_0.0"]) + (((data["real_ord_3"]) + (data["te_month"]))))))))))))))))) + (data["te_nom_4"])))) +
                    0.040868*np.tanh(np.real(((((((((data["te_day"]) + (data["te_nom_8"]))) - (((data["bin_2_0.0"]) - (((data["te_bin_4_N"]) + (((data["real_ord_2"]) - (((np.cosh((data["te_bin_1_1.0"]))) - (np.sinh(((((((data["te_nom_1"]) + (((data["te_real_ord_0"]) + (data["te_nom_7"]))))/2.0)) + (data["real_ord_3"]))))))))))))))) * 2.0)) + (data["te_day"])))) +
                    0.045998*np.tanh(np.real(((data["te_nom_7"]) + (((((data["te_bin_1_1.0"]) + (((((data["real_ord_0"]) + (((data["te_day"]) - (np.cos((np.sqrt((((data["te_month"]) + (np.sinh((data["te_nom_1"]))))))))))))) + (data["real_ord_3"]))))) + (((data["te_nom_4"]) + (((data["te_bin_0_1.0"]) + (((data["real_ord_4"]) + (data["te_nom_3"])))))))))))) +
                    0.049620*np.tanh(np.real(((((((((data["te_nom_9"]) + (data["real_ord_5"]))) + (np.conjugate((((data["te_nom_7"]) + (((((((((((data["te_nom_6"]) + (data["te_nom_8"]))) + (data["real_ord_1"]))) - ((-((data["te_nom_5"])))))) + (data["te_nom_4"]))) * 2.0)))/2.0))))) + (data["real_ord_2"]))) - (np.sqrt((data["bin_0_1.0"])))))) +
                    0.049767*np.tanh(np.real(((((data["te_real_ord_3"]) + (data["te_nom_7"]))) + (((((data["bin_2_1.0"]) + (((data["te_bin_4_Y"]) + (((((data["te_nom_0"]) + (((((data["real_ord_0"]) + (data["real_ord_5"]))) - (complex(1.0)))))) + (((data["te_nom_3"]) + (data["te_nom_1"]))))))))) + (((data["te_nom_9"]) + (data["te_nom_2"])))))))) +
                    0.043221*np.tanh(np.real(((((((((data["real_ord_1"]) + (data["te_day"]))) - (complex(1.0)))) + (((((data["te_nom_5"]) + (data["real_ord_3"]))) + (data["te_nom_3"]))))) + (((data["real_ord_2"]) + (((((((data["te_nom_2"]) / 2.0)) + (data["te_nom_9"]))) + (((data["real_ord_4"]) + (((data["te_month"]) + (data["te_nom_2"])))))))))))) +
                    0.035580*np.tanh(np.real(((data["te_bin_0_1.0"]) + (((data["te_nom_1"]) + (((data["te_nom_4"]) + ((((((data["real_ord_4"]) + (((data["te_day"]) + (((data["te_bin_1_1.0"]) + (((data["bin_2_1.0"]) + (((((data["real_ord_3"]) + (data["te_real_ord_1"]))) + (np.sinh((((data["te_month"]) - (complex(0.636620)))))))))))))))/2.0)) + (data["te_nom_8"])))))))))) +
                    0.047749*np.tanh(np.real((((((data["te_bin_0_1.0"]) + (((data["real_ord_5"]) + (data["bin_4_Y"]))))) + (((data["real_ord_4"]) - ((-((((((((((((data["te_real_ord_0"]) + (data["real_ord_5"]))) + (((data["te_nom_6"]) + (data["te_month"]))))) - (np.cosh((data["real_ord_5"]))))) + (data["real_ord_2"]))) - (np.sin((data["bin_2_0.0"])))))))))))/2.0))) +
                    0.048250*np.tanh(np.real(((np.sin((data["te_nom_5"]))) + (((data["te_nom_8"]) + (((((np.sin((data["te_nom_9"]))) + (np.conjugate((((((((data["real_ord_0"]) + (((data["te_nom_3"]) + (data["te_nom_2"]))))) + (data["te_nom_7"]))) + (((data["te_bin_0_1.0"]) + (((data["real_ord_3"]) + (data["te_day"]))))))/2.0))))) - (complex(0.318310))))))))) +
                    0.048170*np.tanh(np.real(((((data["real_ord_2"]) / 2.0)) + (np.conjugate(((((((((((data["real_ord_5"]) + (((data["te_month"]) + (data["real_ord_3"]))))) + (data["bin_1_0.0"]))/2.0)) + (data["te_nom_0"]))) + ((((((((data["te_day"]) + (data["te_nom_7"]))) + (((data["te_nom_1"]) + (data["real_ord_1"]))))/2.0)) - (complex(1.0)))))/2.0)))))) +
                    0.044730*np.tanh(np.real((((((((((((data["te_real_ord_3"]) + (data["te_nom_4"]))) + ((((data["nans"]) + (((np.sinh((data["month"]))) + (data["te_nom_7"]))))/2.0)))) + (data["real_ord_4"]))) - ((((np.cosh((((data["real_ord_3"]) - (data["te_nom_7"]))))) + (data["bin_1_1.0"]))/2.0)))) + ((((data["real_ord_3"]) + (data["te_nom_7"]))/2.0)))/2.0))) +
                    0.034540*np.tanh(np.real(((((data["real_ord_5"]) + ((((data["te_nom_1"]) + (np.sinh((((((data["real_ord_0"]) - (np.sqrt((data["real_ord_5"]))))) - (np.cos((((np.cos((data["te_month"]))) + (np.cos((data["real_ord_5"]))))))))))))/2.0)))) + ((((data["te_nom_6"]) + (((np.cos((data["te_bin_0_1.0"]))) + (data["te_nom_8"]))))/2.0))))) +
                    0.042491*np.tanh(np.real(((np.tanh((data["te_nom_9"]))) + ((((((data["te_bin_2_1.0"]) + ((((data["bin_1_0.0"]) + (data["bin_4_Y"]))/2.0)))) + (((((((data["te_month"]) + (((((data["te_nom_3"]) + (data["te_real_ord_2"]))) + (data["te_nom_4"]))))) - (data["bin_0_1.0"]))) + ((((((data["te_nom_3"]) + (data["te_day"]))) + (data["te_real_ord_1"]))/2.0)))))/2.0))))) +
                    0.016890*np.tanh(np.real(((((np.sin((((data["te_nom_5"]) + (np.sqrt((data["te_nom_5"]))))))) - (((data["real_ord_3"]) * (np.sin((np.tanh((np.sin((data["real_ord_3"]))))))))))) + ((((((np.tanh((data["real_ord_4"]))) + (data["real_ord_5"]))) + ((((((data["real_ord_3"]) + (data["real_ord_3"]))) + (data["te_day"]))/2.0)))/2.0))))) +
                    0.026100*np.tanh(np.real(np.tanh((complex(0,1)*np.conjugate((((((data["real_ord_2"]) + (np.sinh((data["te_real_ord_3"]))))) + ((((((((((((complex(0,1)*np.conjugate(data["real_ord_5"])) + (data["te_real_ord_4"]))/2.0)) + ((((((complex(0,1)*np.conjugate(data["te_bin_2_1.0"])) + (data["te_real_ord_3"]))/2.0)) + (data["te_bin_0_1.0"]))))/2.0)) + (data["real_ord_5"]))/2.0)) - (np.sqrt((data["te_real_ord_3"]))))))/2.0)))))) +
                    0.036756*np.tanh(np.real((((np.conjugate(np.conjugate(((((data["real_ord_4"]) + ((((data["te_nom_2"]) + (((complex(1.0)) * (data["real_ord_1"]))))/2.0)))) + (np.sin((data["te_nom_8"]))))))) + ((((data["te_nom_7"]) + (((((data["te_bin_4_Y"]) - (data["bin_0_1.0"]))) + (data["te_nom_1"]))))/2.0)))/2.0))) +
                    0.046000*np.tanh(np.real((((-((((np.cos((((((np.sqrt((np.cos((np.sinh((np.cos((data["real_ord_3"]))))))))) / (np.sqrt((np.cos((np.sin((((data["te_nom_8"]) * (np.sin((data["te_nom_8"]))))))))))))) * 2.0)))) * (np.sinh((((data["te_real_ord_2"]) - (np.cos((np.sqrt((data["real_ord_3"])))))))))))))) / 2.0))) +
                    0.024677*np.tanh(np.real(((((((data["te_month"]) + (((data["real_ord_5"]) + (data["te_bin_1_0.0"]))))/2.0)) + (((np.sin((np.sin((data["te_nom_9"]))))) + (((np.sin((data["te_nom_9"]))) + (((np.sin((data["te_nom_8"]))) + (((((complex(0.636620)) / (data["te_nom_5"]))) + ((((data["real_ord_1"]) + (data["te_nom_2"]))/2.0)))))))))))/2.0))) +
                    0.040479*np.tanh(np.real(((((((data["te_month"]) + ((((((np.cosh((complex(0,1)*np.conjugate(data["month"])))) * 2.0)) + (((data["te_real_ord_0"]) * 2.0)))/2.0)))/2.0)) + ((-((((np.sqrt(((-((((np.sinh((((((data["te_nom_7"]) * 2.0)) + (np.cosh((complex(0,1)*np.conjugate(data["te_month"])))))))) + (np.cosh((data["bin_3_F"])))))))))) * 2.0))))))/2.0))) +
                    0.049240*np.tanh(np.real(((data["te_bin_0_1.0"]) - (np.sinh((((data["bin_0_0.0"]) / (np.cosh(((((complex(0,1)*np.conjugate((-((data["te_nom_8"]))))) + (((np.sqrt((data["te_bin_0_1.0"]))) - (((((np.sin(((((data["bin_0_0.0"]) + (((data["te_bin_0_1.0"]) - ((-((data["te_bin_0_1.0"])))))))/2.0)))) / 2.0)) - (data["te_bin_2_1.0"]))))))/2.0))))))))))) +
                    0.033428*np.tanh(np.real((((((((((((data["te_nom_8"]) * (((((data["te_nom_9"]) * 2.0)) * 2.0)))) + (data["te_nom_8"]))/2.0)) + (np.sin((((data["te_nom_9"]) * 2.0)))))/2.0)) + (np.sinh((((np.sin((data["real_ord_3"]))) * (np.sqrt((((((data["te_nom_8"]) * (((data["te_nom_9"]) * 2.0)))) + ((-((data["real_ord_5"])))))))))))))/2.0))) +
                    0.023450*np.tanh(np.real(complex(0,1)*np.conjugate((-((complex(0,1)*np.conjugate((((((((data["real_ord_0"]) * 2.0)) + (((((data["te_real_ord_5"]) + (((data["real_ord_3"]) + ((((data["real_ord_4"]) + (data["real_ord_2"]))/2.0)))))) * 2.0)))/2.0)) * (((data["te_bin_0_1.0"]) + (((data["te_month"]) + ((((data["real_ord_2"]) + (data["real_ord_3"]))/2.0)))))))))))))) +
                    0.047993*np.tanh(np.real((((data["te_bin_0_1.0"]) + (((((data["real_ord_3"]) + (((data["real_ord_3"]) + (data["real_ord_3"]))))) * ((((((((((((data["te_month"]) + (data["real_ord_5"]))/2.0)) + (((data["real_ord_5"]) + (data["te_nom_7"]))))/2.0)) + (data["te_real_ord_2"]))/2.0)) * (((((data["real_ord_5"]) + (data["te_nom_8"]))) + (data["te_real_ord_0"]))))))))/2.0))) +
                    0.048155*np.tanh(np.real(np.sinh((((((((data["real_ord_5"]) + (np.cos((((data["te_nom_8"]) + (data["month"]))))))/2.0)) + ((((np.cos((((data["te_nom_8"]) + (data["month"]))))) + (((np.sinh((((data["real_ord_2"]) / (np.sqrt((data["te_day"]))))))) - (np.cosh((data["te_nom_7"]))))))/2.0)))/2.0))))) +
                    0.024506*np.tanh(np.real(((((((data["te_nom_0"]) + (((((data["real_ord_0"]) + (((np.cos((data["te_month"]))) + (data["te_nom_6"]))))) + (np.cos((((data["te_nom_6"]) - (data["te_real_ord_4"]))))))))/2.0)) + ((((np.sin((data["te_nom_5"]))) + (np.sinh(((((((data["te_nom_4"]) + (data["te_month"]))) + (data["real_ord_1"]))/2.0)))))/2.0)))/2.0))) +
                    0.034070*np.tanh(np.real((((((((complex(1.570796)) / (((data["te_nom_8"]) + (np.sinh((np.conjugate(np.tanh((((np.sinh((complex(1.0)))) + (np.cos((data["bin_2_1.0"])))))))))))))) / 2.0)) + ((-((np.cos((data["bin_2_1.0"])))))))/2.0))) +
                    0.041630*np.tanh(np.real(np.conjugate(np.sin((np.conjugate(np.sin((complex(0,1)*np.conjugate(((data["te_real_ord_3"]) - (np.tanh((((np.cos(((((np.sqrt(((((data["te_real_ord_0"]) + (data["te_real_ord_3"]))/2.0)))) + (data["te_day"]))/2.0)))) + (np.cos(((((np.sqrt((data["te_nom_7"]))) + ((((data["te_real_ord_0"]) + (data["te_nom_7"]))/2.0)))/2.0)))))))))))))))))) +
                    0.025860*np.tanh(np.real(((data["te_month"]) / (np.sqrt((np.sin((((((np.sqrt((((complex(9.94689178466796875)) - (((((data["te_bin_0_0.0"]) + (((data["te_nom_9"]) + (np.cosh((np.cosh((complex(0,1)*np.conjugate((((((data["te_month"]) * (data["te_nom_9"]))) + (np.sqrt((data["te_day"]))))/2.0))))))))))) / 2.0)))))) * 2.0)) * 2.0))))))))) +
                    0.015830*np.tanh(np.real((((((data["te_nom_7"]) + (((((data["real_ord_4"]) + (((((data["real_ord_0"]) + (data["real_ord_5"]))) * (np.conjugate(np.cosh((data["real_ord_5"])))))))) * (np.cosh((data["real_ord_5"]))))))/2.0)) * (((data["bin_0_1.0"]) + (((data["real_ord_3"]) * (np.tanh((data["te_real_ord_2"])))))))))) +
                    0.036150*np.tanh(np.real((-((((((((-((((((((-(((((((data["real_ord_0"]) + (data["te_nom_9"]))/2.0)) * (data["te_bin_0_0.0"])))))) * (data["te_bin_0_0.0"]))) + (np.sqrt((data["real_ord_0"]))))/2.0))))) * (data["te_bin_0_0.0"]))) + (((data["te_nom_9"]) * ((((data["real_ord_0"]) + (data["te_nom_9"]))/2.0)))))/2.0)))))) +
                    0.033190*np.tanh(np.real((((np.sin((np.sin(((-((((((((data["real_ord_5"]) / 2.0)) + (np.conjugate(data["real_ord_3"])))) * 2.0))))))))) + (((((data["te_nom_1"]) + (((data["te_real_ord_4"]) + (((data["te_real_ord_0"]) * ((((-((data["te_real_ord_4"])))) + ((((-((data["real_ord_3"])))) + (data["te_real_ord_5"]))))))))))) / 2.0)))/2.0))) +
                    0.048563*np.tanh(np.real(((data["te_real_ord_3"]) * (((data["bin_0_1.0"]) - (((((data["te_nom_1"]) - (np.sqrt((complex(0,1)*np.conjugate(((np.sinh(((((-((data["te_nom_1"])))) - (((np.conjugate(data["te_nom_9"])) * 2.0)))))) - (np.conjugate(((data["te_nom_1"]) - (np.conjugate(((np.conjugate(data["te_nom_9"])) * 2.0))))))))))))) / 2.0))))))) +
                    0.045370*np.tanh(np.real(((((np.sin((np.cos((((data["bin_0_1.0"]) - ((((data["te_nom_5"]) + (np.cosh((((data["bin_4_Y"]) * ((((-((data["real_ord_3"])))) - (np.sin((data["te_nom_5"]))))))))))/2.0)))))))) + (np.sin((data["te_nom_5"]))))) / (complex(0,1)*np.conjugate(np.sqrt((data["real_ord_2"]))))))) +
                    0.0*np.tanh(np.real(((((data["real_ord_0"]) + (((data["bin_1_0.0"]) + (np.sqrt(((((((data["real_ord_0"]) + ((((((data["real_ord_0"]) + ((((np.conjugate(np.tanh((np.sqrt(((((data["real_ord_0"]) + (data["te_month"]))/2.0))))))) + (data["real_ord_5"]))/2.0)))) + (data["real_ord_5"]))/2.0)))) + (data["real_ord_5"]))/2.0)))))))) * (complex(0,1)*np.conjugate(data["real_ord_2"]))))) +
                    0.031305*np.tanh(np.real((((-((data["te_nom_4"])))) * (((data["bin_0_0.0"]) / ((((((data["real_ord_3"]) - (data["bin_0_0.0"]))) + (complex(0,1)*np.conjugate(np.sinh((np.sqrt((complex(0,1)*np.conjugate(np.sinh((complex(0,1)*np.conjugate(np.sinh((np.sqrt((((data["te_real_ord_1"]) * (np.conjugate(np.conjugate(data["te_nom_1"])))))))))))))))))))/2.0))))))) +
                    0.042774*np.tanh(np.real(((data["bin_0_1.0"]) / (complex(0,1)*np.conjugate((-((((np.conjugate(data["real_ord_3"])) + ((((((data["te_real_ord_2"]) + (((np.sqrt(((-((data["real_ord_3"])))))) * (data["te_real_ord_4"]))))/2.0)) + (np.sqrt(((((data["real_ord_2"]) + (data["real_ord_0"]))/2.0))))))))))))))) +
                    0.0*np.tanh(np.real(((((((np.conjugate(np.sinh((np.tanh((data["real_ord_3"])))))) + (data["te_bin_0_0.0"]))) + (((np.sin((data["te_nom_8"]))) + (data["te_real_ord_4"]))))) * ((-(((((((data["te_real_ord_5"]) + (((data["real_ord_3"]) + (((data["real_ord_2"]) + (data["te_nom_3"]))))))/2.0)) / 2.0)))))))) +
                    0.023340*np.tanh(np.real(((np.sinh((data["month"]))) * ((((-(((((data["te_bin_0_1.0"]) + (((((((((np.sinh((data["month"]))) + (((((data["month"]) / 2.0)) + (data["te_nom_8"]))))/2.0)) + (((data["te_bin_3_F"]) - (np.sinh((data["bin_1_1.0"]))))))/2.0)) - ((-(((((data["te_bin_0_1.0"]) + (data["real_ord_2"]))/2.0))))))))/2.0))))) / 2.0))))) +
                    0.018510*np.tanh(np.real(((((((((data["te_nom_1"]) + (((data["te_nom_2"]) * (((data["te_nom_5"]) / (data["real_ord_1"]))))))/2.0)) / 2.0)) + (((((((((data["te_nom_2"]) * (((complex(0.636620)) / ((-((data["te_nom_1"])))))))) + (data["te_real_ord_2"]))/2.0)) + (((data["te_nom_2"]) * ((-(((((data["real_ord_3"]) + (data["te_nom_1"]))/2.0))))))))/2.0)))/2.0))) +
                    0.043253*np.tanh(np.real(((data["bin_0_1.0"]) * ((((((((data["real_ord_3"]) / 2.0)) + ((((complex(0,1)*np.conjugate(data["te_nom_2"])) + (complex(0,1)*np.conjugate(data["real_ord_1"])))/2.0)))/2.0)) / (((np.sqrt((complex(0,1)*np.conjugate((((data["te_nom_1"]) + (((((((data["te_nom_9"]) + (complex(0,1)*np.conjugate(data["te_nom_2"])))/2.0)) + (data["bin_0_1.0"]))/2.0)))/2.0))))) / 2.0))))))) +
                    0.009565*np.tanh(np.real(np.sinh(((-((np.sqrt(((((((data["te_bin_1_1.0"]) * (np.conjugate(((((np.sin((data["te_month"]))) / 2.0)) - (data["real_ord_5"])))))) + (np.conjugate(complex(0,1)*np.conjugate((((np.tanh(((-((np.sqrt((data["bin_1_1.0"])))))))) + (np.conjugate(((data["te_month"]) - (np.sin((data["te_month"])))))))/2.0)))))/2.0)))))))))) +
                    0.000417*np.tanh(np.real(np.sin((complex(0,1)*np.conjugate(((np.cos((np.sin((((((complex(0,1)*np.conjugate(data["real_ord_1"])) / 2.0)) - (np.sqrt((data["te_real_ord_3"]))))))))) - (np.tanh((np.conjugate(np.cos((((data["te_real_ord_3"]) - (np.tanh((np.conjugate(np.cos((((data["te_real_ord_2"]) - (np.sinh((np.sqrt((data["te_real_ord_0"]))))))))))))))))))))))))) +
                    0.0*np.tanh(np.real((((((data["real_ord_2"]) + (((((((data["real_ord_0"]) + (data["real_ord_1"]))/2.0)) + (data["te_real_ord_3"]))/2.0)))/2.0)) / (((data["bin_2_1.0"]) / (((((((data["te_real_ord_3"]) + ((((data["nans"]) + (data["te_real_ord_4"]))/2.0)))/2.0)) + ((((data["month"]) + (data["te_real_ord_5"]))/2.0)))/2.0))))))) +
                    0.045411*np.tanh(np.real(((np.sinh((data["te_nom_7"]))) * (complex(0,1)*np.conjugate(np.sinh((np.sinh((np.sqrt((((np.sinh((np.sinh((np.sin((((data["bin_1_0.0"]) + (((data["te_real_ord_3"]) / 2.0)))))))))) + (((((data["te_real_ord_4"]) / 2.0)) + (((((data["te_nom_7"]) / 2.0)) + (data["te_real_ord_5"]))))))))))))))))) +
                    0.0*np.tanh(np.real((((((((data["te_real_ord_2"]) + ((((((data["te_real_ord_2"]) + (data["te_month"]))/2.0)) * (np.sqrt((((data["te_month"]) * (data["real_ord_3"]))))))))/2.0)) * (np.sqrt(((((((data["real_ord_5"]) + (data["te_month"]))/2.0)) * (((data["real_ord_3"]) * 2.0)))))))) * (np.sqrt((np.sqrt((((data["bin_2_1.0"]) * (data["te_month"])))))))))) +
                    0.041440*np.tanh(np.real(complex(0,1)*np.conjugate(((complex(0,1)*np.conjugate((-(((((((-((np.sqrt(((-((data["real_ord_0"]))))))))) * 2.0)) * ((((data["te_nom_6"]) + ((((data["real_ord_3"]) + (((data["real_ord_2"]) + (((((data["te_nom_4"]) + (((data["real_ord_2"]) + (data["te_nom_1"]))))) + (data["real_ord_0"]))))))/2.0)))/2.0)))))))) / 2.0)))) +
                    0.008370*np.tanh(np.real(complex(0,1)*np.conjugate((((((data["te_nom_0"]) * (complex(0,1)*np.conjugate(np.sin((((data["te_nans"]) + (data["bin_0_0.0"])))))))) + (np.sin((np.sin((np.sqrt((np.sin((np.sin((((data["te_nans"]) + (data["bin_0_0.0"]))))))))))))))/2.0)))) +
                    0.025270*np.tanh(np.real(((((((data["bin_3_T"]) - (data["te_bin_3_T"]))) / (np.cosh((complex(0,1)*np.conjugate(((data["real_ord_4"]) * (((data["real_ord_4"]) / 2.0))))))))) / (np.cosh((complex(0,1)*np.conjugate(((data["real_ord_4"]) * ((((((data["te_nom_7"]) + (((complex(0.636620)) / (np.cos((data["bin_3_T"]))))))/2.0)) / 2.0)))))))))) +
                    0.006620*np.tanh(np.real(((data["real_ord_0"]) * ((((((((((-((data["te_real_ord_2"])))) - (data["te_real_ord_5"]))) * ((((((-((data["real_ord_2"])))) / 2.0)) - (data["real_ord_3"]))))) / 2.0)) + (np.tanh((((data["real_ord_3"]) / (data["te_bin_2_1.0"])))))))))) +
                    0.001180*np.tanh(np.real((-((((data["real_ord_3"]) * (((complex(0,1)*np.conjugate(((np.cos((data["te_real_ord_0"]))) - (np.sqrt(((((data["te_nom_7"]) + (((((((data["te_real_ord_0"]) + (((np.conjugate((((data["te_real_ord_0"]) + (((np.conjugate(data["te_month"])) * 2.0)))/2.0))) * 2.0)))/2.0)) + (((np.cos((data["te_month"]))) * 2.0)))/2.0)))/2.0))))))) * 2.0)))))))) +
                    0.027510*np.tanh(np.real(np.sin((((((data["te_nans"]) * 2.0)) * (((((complex(3.141593)) - (np.tanh((np.tanh((((np.tanh((np.tanh(((((data["te_nom_0"]) + (data["real_ord_2"]))/2.0)))))) * 2.0)))))))) - (np.tanh((np.tanh(((((np.tanh((data["te_nans"]))) + (data["te_real_ord_3"]))/2.0))))))))))))) +
                    0.025006*np.tanh(np.real((((((((((((data["real_ord_2"]) * (data["te_bin_2_0.0"]))) * (data["te_bin_0_1.0"]))) * (data["te_bin_2_0.0"]))) * (((data["bin_2_0.0"]) * (((data["te_nom_6"]) / (((data["te_bin_2_0.0"]) / 2.0)))))))) + (((data["te_month"]) * (np.cos((((data["te_nom_6"]) / (((data["te_month"]) / 2.0)))))))))/2.0))) +
                    0.043289*np.tanh(np.real(((((data["te_month"]) * (((np.conjugate(((data["real_ord_3"]) - (np.sinh((((np.cos((np.sqrt((data["real_ord_0"]))))) / 2.0))))))) * 2.0)))) * ((((((((((data["te_real_ord_2"]) + (data["te_bin_2_1.0"]))/2.0)) + (data["te_real_ord_2"]))/2.0)) + (((((((data["te_nom_9"]) + (data["te_real_ord_5"]))/2.0)) + (data["te_real_ord_5"]))/2.0)))/2.0))))) +
                    0.032970*np.tanh(np.real(((((np.conjugate(((np.sqrt((np.cos(((((((((data["te_nom_9"]) * 2.0)) * 2.0)) + (np.sqrt((data["te_nom_9"]))))/2.0)))))) * (complex(3.141593))))) * (np.cos((((complex(3.141593)) * (np.tanh((((data["te_nom_9"]) * 2.0)))))))))) / 2.0))) +
                    0.000001*np.tanh(np.real(((((np.sqrt((data["real_ord_2"]))) + (data["real_ord_3"]))) * (((data["te_real_ord_5"]) * ((((((((((data["te_bin_2_1.0"]) + (data["real_ord_2"]))) + (((((np.sinh((data["te_nom_7"]))) + (((((data["te_bin_2_1.0"]) + (data["real_ord_3"]))) - (complex(2.0)))))) + (data["real_ord_2"]))))/2.0)) + (data["te_real_ord_0"]))) / 2.0))))))) +
                    0.040591*np.tanh(np.real(((((((data["te_real_ord_5"]) * (((data["te_real_ord_5"]) * ((((data["te_real_ord_5"]) + (data["te_month"]))/2.0)))))) * 2.0)) * (((((complex(0,1)*np.conjugate((((complex(0,1)*np.conjugate(data["bin_0_1.0"])) + (((data["te_nom_9"]) * (complex(0,1)*np.conjugate(((((((data["te_month"]) + (((data["bin_0_1.0"]) / 2.0)))/2.0)) + (data["te_nom_9"]))/2.0))))))/2.0))) * 2.0)) / 2.0))))) +
                    0.047978*np.tanh(np.real(((((data["bin_0_1.0"]) * (((np.sin((data["real_ord_2"]))) - (((((complex(1.0)) - (np.sinh((np.sin((((data["te_nom_9"]) + (((data["real_ord_2"]) + (data["te_day"]))))))))))) - (np.sinh((np.sin((((data["te_month"]) + (((data["real_ord_2"]) + (data["te_nom_9"]))))))))))))))) / 2.0))) +
                    0.043971*np.tanh(np.real(((data["te_real_ord_5"]) * (((data["nans"]) * (((data["te_nom_4"]) / ((((complex(6.68887281417846680)) + (((data["nans"]) * (((data["te_real_ord_5"]) * (((data["te_real_ord_5"]) / (np.cos((((data["te_nom_8"]) * (((data["te_nom_4"]) / (np.cos((np.cos((np.conjugate(np.cos((data["te_nom_4"])))))))))))))))))))))/2.0))))))))) +
                    0.038160*np.tanh(np.real(((data["bin_0_1.0"]) * ((((((data["bin_0_1.0"]) * (((((((data["te_nom_2"]) + ((((((data["te_real_ord_3"]) + (data["te_nom_2"]))/2.0)) + (data["bin_1_0.0"]))))/2.0)) + ((((data["te_real_ord_3"]) + ((((((data["real_ord_0"]) + (data["te_nom_3"]))/2.0)) + (data["te_nom_3"]))))/2.0)))/2.0)))) + (data["te_nom_7"]))/2.0))))) +
                    0.038251*np.tanh(np.real(complex(0,1)*np.conjugate(np.tanh((((data["real_ord_1"]) + (np.cosh((((data["real_ord_4"]) * ((((np.cosh((np.sqrt(((((np.cosh((data["real_ord_1"]))) + (np.conjugate(np.sqrt((data["te_day"])))))/2.0)))))) + (np.conjugate(np.sqrt((data["te_day"])))))/2.0)))))))))))) +
                    0.043721*np.tanh(np.real(complex(0,1)*np.conjugate(((data["te_bin_0_1.0"]) / (np.tanh((((np.tanh((((((((data["te_bin_0_1.0"]) * (((np.tanh((((complex(0,1)*np.conjugate(data["te_nom_1"])) / 2.0)))) / 2.0)))) / 2.0)) + (data["month"]))))) + (np.tanh((np.sinh((((complex(0,1)*np.conjugate(data["te_nom_1"])) * 2.0)))))))))))))))


# In[40]:


roc_auc_score(y_train,GPComplex(df_train[df_train.columns[:-1]].astype(complex).fillna(complex(0,1))))


# In[41]:


pd.DataFrame({'id': df_test.index.values, 'target': glm.predict_proba(df_test[df_test.columns].fillna(df_test[df_test.columns].mean()))[:,1]}).to_csv('glmsubmission.csv', index=False)


# In[42]:


pd.DataFrame({'id': df_test.index.values, 'target': GP(df_test[df_test.columns].fillna(-999))}).to_csv('gpsubmission.csv', index=False)


# In[43]:


pd.DataFrame({'id': df_test.index.values, 'target': GPComplex(df_test[df_test.columns].astype(complex).fillna(complex(0,1)))}).to_csv('gpcomplexsubmission.csv', index=False)


# In[44]:


predictions = (.5*glm.predict_proba(df_test[df_test.columns].fillna(df_test[df_test.columns].mean()))[:,1] +
               .25*GP(df_test[df_test.columns].fillna(-999)) +
               .25*GPComplex(df_test[df_test.columns].astype(complex).fillna(complex(0,1))))


# In[45]:


pd.DataFrame({'id': df_test.index.values, 'target': predictions}).to_csv('allmodelssubmission.csv', index=False)

