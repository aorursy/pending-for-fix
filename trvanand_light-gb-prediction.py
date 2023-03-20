#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import gc

# Gradient Boosting
import lightgbm as lgb
import xgboost as xgb

# Scikit-learn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Skopt functions
from skopt import BayesSearchCV
from skopt import gp_minimize # Bayesian optimization using Gaussian Processes
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args # decorator to convert a list of parameters to named arguments
from skopt.callbacks import DeadlineStopper # Stop the optimization before running out of a fixed budget of time.
from skopt.callbacks import VerboseCallback # Callback to control the verbosity
from skopt.callbacks import DeltaXStopper # Stop the optimization If the last two positions at which the objective has been evaluated are less than delta

# Hyperparameters distributions
from scipy.stats import randint
from scipy.stats import uniform

# Metrics
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error

import os
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn




data = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')




# Taking the labels (price)
label_df = data['target']




''''data.drop(['ID_code','target'], axis=1, inplace=True)
data_test.drop('ID_code', axis=1, inplace=True)
data.head(5)









data.describe()




data[data.isnull().any(axis=1)]




data.select_dtypes(exclude=np.number).columns




len_train = len(data)
len_train




#Merge test and train
merged = pd.concat([data, data_test])
#Saving the list of original features in a new list `original_features`.
original_features = merged.columns
merged.shape




len(data.drop(['ID_code', 'target'], axis=1).columns)




get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
# Plot first 100 features.
data.iloc[:, 2:100].plot(kind='box', figsize=[16,8])




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = data
data_scaled.iloc[:, 2:] = scaler.fit_transform(data.iloc[:, 2:])




# Separate out the features.
x = data_scaled.iloc[:, 2:].values
# Separate out the target.
y = data_scaled.iloc[:, 1].values




from sklearn.decomposition import PCA
pca = PCA(2)
projected = pca.fit_transform(x)




print(projected)




plt.scatter(projected[:, 0], projected[:, 1],
           c=y, edgecolor='none', alpha=0.5,
           cmap=plt.cm.get_cmap('copper', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();




import lightgbm as lgb
from sklearn.model_selection import train_test_split




# Create stratified validation split.
# Stratifying makes the splits have the same class distribution (purchase/no-purchase).
train_x, validation_x, train_y, validation_y = train_test_split(x, y, stratify=y)




train_data = lgb.Dataset(train_x, label=train_y)




validation_data = lgb.Dataset(validation_x, label=validation_y, reference=train_data)




bst = lgb.train({
    'boosting': 'gbdt', #'dart', # Dropouts meet Multiple Additive Regression Trees, default='gbdt'
    'learning_rate': 0.005, # smaller increases accuracy, default=0.1
    'max_bin': 511, # larger increases accuracy, default=255
    'metric': 'auc',
    'num_leaves': 63, # larger increases accuracy, default=31
    'num_trees': 90,
    'num_iteration': 720, # default=100
    'objective': 'binary',
    },
    train_data,
    num_boost_round=800, # may be redundant with params#num_iteration
    valid_sets=[validation_data],
    early_stopping_rounds=100,
    verbose_eval=90, # logs every 90 trees
)




bst.save_model('model.txt', num_iteration=bst.best_iteration)




# Generate submission
test = pd.read_csv('../input/test.csv')
test_x = test.iloc[:, 1:].values # Drop the ID_code
ypred = bst.predict(test_x)
test_code = test.iloc[:, 0]
submission = pd.concat([test_code, pd.Series(ypred, name='target')], axis=1)
submission.to_csv('submissions.csv', index=False)
submission.head()




nunique  = data.nunique()




get_ipython().system('head submissions.csv')






