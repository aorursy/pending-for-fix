#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
print(__version__) # requires version >= 1.9.0
# For Notebooks
init_notebook_mode(connected=True)
# For offline use




# Handy code which I prefer to enlarge the plots
plt.rcParams["figure.figsize"] = (18,9)




train = pd.read_csv('../input/train_2016_v2.csv')
properties = pd.read_csv('../input/properties_2016.csv', low_memory=False)
dictionary = pd.read_excel('../input/zillow_data_dictionary.xlsx')




train.head()




properties.head()




null = pd.DataFrame(data= properties.isnull().sum()/len(properties)*100, 
                    columns=['Percentage of Values Missing'],
                    index=properties.columns
                   ).reset_index()




null['Percentage of Values Missing'].mean()




plt.rcParams["figure.figsize"] = (13,10)
sns.barplot(x= 'Percentage of Values Missing', 
            y='index', 
            data= null.sort_values(by='Percentage of Values Missing', ascending=False),
            color = '#ff004f') 




## Caution - Only 50% percentile missing values are taken. There are 29 MORE!!!
Notorious_null = null[null['Percentage of Values Missing'] > null['Percentage of Values Missing'].mean()]




Notorious_null.sort_values(by='Percentage of Values Missing', ascending=False).head(10)




plt.rcParams["figure.figsize"] = (13,10)
sns.barplot(x= 'Percentage of Values Missing', 
            y='index', 
            data= Notorious_null,
            color = '#ff004f') 




len(null) - len(Notorious_null)




alldata = pd.merge(train, properties, how='inner', on='parcelid')




alldata.head()




# sns.heatmap(alldata.corr(), cmap='viridis', vmax=0.8, vmin=0)




alldata.head(10)




null_drop = null[null['Percentage of Values Missing'] > 85]




col_to_drop = []
for cols in list(null_drop['index'].values):
    col_to_drop.append(cols)




alldata.drop(col_to_drop, axis=1, inplace=True)




alldata.head()




nullv2 = pd.DataFrame(data= alldata.isnull().sum()/len(alldata)*100, 
                    columns=['Percentage of Values Missing'],
                    index=alldata.columns
                   ).reset_index()




nullv2.sort_values(by='Percentage of Values Missing', ascending=False)




alldata.fillna(value=0, inplace=True)




alldata.head(8)




sns.heatmap(alldata.corr().head(500), cmap='viridis', vmax=0.8, vmin=0)




alldata.describe()




from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=20)




X = alldata.drop(['parcelid','logerror', axis=1)






