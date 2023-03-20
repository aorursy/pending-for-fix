# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from ggplot import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
help(ggplot)

#如何做出age的分布图，以gender来区分。
gatrain = pd.read_csv('../input/gender_age_train.csv')

sns.distplot(gatrain.age)

app_events=pd.read_csv("../input/app_events.csv")

events=pd.read_csv("")