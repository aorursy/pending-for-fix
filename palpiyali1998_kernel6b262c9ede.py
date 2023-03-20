#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.









import pandas as pd
a = pd.read_csv("../input/train.csv")




a.head()




a['genres']




df=list()
for i in range(10):
  st=a['genres'][i].split()[3]
  st = st[1:-3]
  df.append(st)
print(df)




a.isnull().sum()





a.head()




ax=["belongs_to_collection","homepage","production_countries","tagline","Keywords"]




a




a.drop(ax,axis=1,inplace = True) 
  




a




a["genres"]




df=list()
for i in range(471):
  st=a['genres'][i].split()[3]
  st = st[1:-3]
  df.append(st)
print(df)




a.iloc[470]




a.iloc[100]




a = a.fillna(a['spoken_languages'].value_counts(),inplace=True)




a.iloc[470]




df=list()
for i in range(471):
  st=a['genres'][i].split()[3]
  st = st[1:-3]
  df.append(st)
print(df)





imp = SimpleImputer(strategy="most_frequent")
print(imp.fit_transform(a['spoken_languages'].reshape(-1,1)))




for each

