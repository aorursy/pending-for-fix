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

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

 y = le.fit_transform(np.array([-1,-2,-3]))

 y = le.fit_transform(np.array([-1,-2,-3]))

a = np.random.rand(3,2)

index = np.array([[0,1],[1,0],[0,1]])

a[0,:][index[0,:]]

index

a


