#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




from glob import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf

labels = pd.read_csv("../input/labels.csv")
submi = pd.read_csv("../input/sample_submission.csv")
#getting the all the files
data = pd.DataFrame([{'path': filepath} for filepath in glob("../input/train/*.jpg")])




df = pd.DataFrame()
#reading all the files and converting into matrix form
df['id'] = pd.DataFrame(data['path'].apply(lambda x : x.split('/')[-1].split('.')[0]))
df['value'] = pd.DataFrame(data['path'].apply(lambda x :plt.imread(str(x))))




df.head()




#Getting Breed against Dog id
df['Breed'] = None
for i in range(len(df['id'])):
    df['Breed'].iloc[i] =  labels[labels['id']== df['id'].iloc[i]]['breed'].any()




df.head()




df['value'][0].shape




(df['value'][0].flatten().reshape(194,237,3)).shape




df['value'] = df['value'].apply(lambda x : x.flatten())




df.head()




# Normalising the data diving it with 255

pixels = df['value'].values/255.0




plt.figure(figsize=(14,10))
a,b = 3,3
for i in range(0,(a*b)):
    #getting random index
    rand_ind = np.random.randint(0,len(df), size= a*b)
    plt.subplot(a,b, i+1)
    plt.imshow(df['value'].iloc[rand_ind[i]])
    plt.title(df['Breed'].iloc[rand_ind[i]])




t=[]
for i in range(len(df['value'])):
    t.append(df['value'].iloc[i]/255.0)u




#df['value'].values/255.0





























data.head(7)









plt.imshow(df['path'][0])




data['id'] = data['path'].apply(lambda x : x.split('/')[-1].split('.')[0]) 




data.head()






