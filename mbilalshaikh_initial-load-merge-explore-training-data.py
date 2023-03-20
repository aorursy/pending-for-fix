#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns




# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train =pd.read_csv("../input/train_2016.csv")
train.head()
train.info()

props = pd.read_csv("../input/properties_2016.csv")
#props.columns

df = pd.merge(train,props,how="left",on="parcelid")

# Now we have both files merged in the pd object




small_df = df[:500]




pids = small_df['parcelid']
(pids.is_unique, pids.hasnans)
df.shape
small_df.shape




plt.figure(figsize=(8,6))
plt.scatter(range(df.shape[0]), np.sort(df.logerror.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.show()




df.logerror.values




***More to come. Stay tuned.!
Please upvote if you find it useful :)***

