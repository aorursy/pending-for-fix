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




df = pd.read_csv('../input/train.csv')




df.info()




df.head()




df['channelGrouping'].unique()




def DeviceSplitter(inputString):
    theList = inputString.split(',')
    browser = theList[0].split("\"")[-2]
    OS = theList[3].split("\"")[-2]
    isMobile = theList[5].split(":")[-1].strip()
    deviceCat = theList[-1].split("\"")[-2]
    return ({"browser": browser, "OS": OS, "isMobile": isMobile, "device": deviceCat})
    #return ([browser, OS, isMobile,deviceCat])
    #print(browser, OS, isMobile, deviceCat)
    #print(isMobile)




DeviceSplitter(df['device'].iloc[0])




get_ipython().run_cell_magic('time', '', "random_row = df['device'].apply(DeviceSplitter)")




from sklearn.feature_extraction import FeatureHasher
random_row




h = FeatureHasher(n_features=5)




f1 = h.fit_transform(random_row)




pd.concat([pd.DataFrame(f1.toarray()), random_row], axis = 1)




def 

