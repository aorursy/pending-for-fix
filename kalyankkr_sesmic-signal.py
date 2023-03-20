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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action="ignore",category=FutureWarning)




train= pd.read_csv("../input/train.csv",dtype={"acoustic_data":np.int16, "time_ro_failure": np.float64},nrows=1500000)
train.head()




train.isna().sum()




train.describe()




plt.figure(figsize=(8,6))
plt.title("Distribution of Acoustic data")
ax= sns.distplot(train.acoustic_data,label="acustic_data")




This shows most of the signal data centred around mean value of the signals




plt.figure(figsize=(12,8))
plt.title("Sesmic signal time_to_failure")
plt.plot(train.time_to_failure,train.acoustic_data)

