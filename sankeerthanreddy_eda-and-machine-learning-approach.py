#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




pip install kaggle




train = pd.read_csv('../input/mercedes-benz-greener-manufacturing/train.csv.zip')
test=pd.read_csv("../input/mercedes-benz-greener-manufacturing/test.csv.zip")




print("Train shape :", train.shape)
print("Test shape :" ,test.shape)




train.head()




import matplotlib.pyplot as plt




plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train.y.values))
plt.xlabel('value')
plt.ylabel('y')
plt.grid()
plt.show()




train["y"].max()




train["y"].min()




train["y"].mean()




print(train.columns)




train.describe()




import seaborn as sns




train['X0'].unique()




train.groupby('X0')['ID'].nunique()




col_sort_order = np.sort(train['X0'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X0', y='y', data=train,order=col_sort_order)
plt.xlabel('X0', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X0")
plt.grid()
plt.show()




col_sort_order = np.sort(train['X0'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X0', y='y', data=train, order=col_sort_order)
plt.xlabel('X0', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X0")
plt.show()




train['X1'].unique()




train.groupby('X1')['ID'].nunique()




col_sort_order = np.sort(train['X1'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X1', y='y', data=train,order=col_sort_order)
plt.xlabel('X1', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X1")
plt.grid()
plt.show()




col_sort_order = np.sort(train['X1'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X1', y='y', data=train, order=col_sort_order)
plt.xlabel('X1', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X1")
plt.show()




train['X2'].unique()




train.groupby('X2')['ID'].nunique()




col_sort_order = np.sort(train['X2'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X2', y='y', data=train,order=col_sort_order)
plt.xlabel('X2', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X2")
plt.grid()
plt.show()




col_sort_order = np.sort(train['X2'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X2', y='y', data=train, order=col_sort_order)
plt.xlabel('X2', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X2")
plt.show()




train['X3'].unique()




train.groupby('X3')['ID'].nunique()




col_sort_order = np.sort(train['X3'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X3', y='y', data=train,order=col_sort_order)
plt.xlabel('X3', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X3")
plt.grid()
plt.show()




col_sort_order = np.sort(train['X3'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X3', y='y', data=train, order=col_sort_order)
plt.xlabel('X3', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X3")
plt.show()




train['X4'].unique()




train.groupby('X4')['ID'].nunique()




col_sort_order = np.sort(train['X4'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X4', y='y', data=train,order=col_sort_order)
plt.xlabel('X4', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X4")
plt.grid()
plt.show()




col_sort_order = np.sort(train['X4'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X4', y='y', data=train, order=col_sort_order)
plt.xlabel('X4', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X4")
plt.show()




train['X5'].unique()




train.groupby('X5')['ID'].nunique()




col_sort_order = np.sort(train['X5'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X5', y='y', data=train,order=col_sort_order)
plt.xlabel('X5', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X5")
plt.grid()
plt.show()




col_sort_order = np.sort(train['X5'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X5', y='y', data=train, order=col_sort_order)
plt.xlabel('X5', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X5")
plt.show()




train['X6'].unique()




train.groupby('X6')['ID'].nunique()




col_sort_order = np.sort(train['X6'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X6', y='y', data=train,order=col_sort_order)
plt.xlabel('X6', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X6")
plt.grid()
plt.show()




col_sort_order = np.sort(train['X6'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X6', y='y', data=train, order=col_sort_order)
plt.xlabel('X6', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X6")
plt.show()




train['X8'].unique()




train.groupby('X8')['ID'].nunique()




col_sort_order = np.sort(train['X8'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X8', y='y', data=train,order=col_sort_order)
plt.xlabel('X8', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X8")
plt.grid()
plt.show()




col_sort_order = np.sort(train['X8'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X8', y='y', data=train, order=col_sort_order)
plt.xlabel('X8', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X8")
plt.show()




get_ipython().system(' pip install pycaret')




from pycaret.regression import *




train.info()




reg = setup(data = train, 
             target = 'y',
             categorical_features = ['X0','X1','X2','X3','X4','X5','X6','X8'],
              normalize = True,
             numeric_imputation = 'mean',
             pca=True,
             silent = True)




compare_models()




br=create_model('br')




br




tuned_dt = tune_model(br)




plot_model(br)




tuned_br = tune_model(br, optimize = 'R2')




tuned_br




final_br= finalize_model(tuned_br)




save_model(final_br, 'br_saved1126082020')









#sample= pd.read_csv('../input/sample/sample.csv')




#predictions = predict_model(tuned_br, data = test)
#sample['y'] = predictions['Label']




#sample.head()






