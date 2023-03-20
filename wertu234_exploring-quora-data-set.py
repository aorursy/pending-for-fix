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
print(check_output(["ls", ".."]).decode("utf8"))
# Any results you write to the current directory are saved as output.




train_raw = pd.read_csv("../input/train.csv")




sum(train_raw["is_duplicate"])/len(train_raw.index)
train_raw.head


len(train_raw.index)




sum(train_raw["is_duplicate"])/len(train_raw.index)




train_raw.columns.values




sum(train_raw["is_duplicate"])




test_raw = pd.read_csv("../input/test.csv")




test_raw.head




len(test_raw.index)




basepreds = np.repeat(sum(train_raw["is_duplicate"])/len(train_raw.index),len(test_raw.index))




sub1 = pd.DataFrame({"test_id" : test_raw["test_id"], "is_duplicate" : basepreds})




sub1.head




submission1 = sub1.to_csv("sub1.csv", index = False)




from sklearn.feature_extraction.text import CountVectorizer




count_vec = CountVectorizer()




train_q1_counts = count_vec.fit_transform(train_raw["question1"])




train_q1_counts.shape




print(train[1]"question1"])




import re

clean1 = train_raw.loc[1,"question1"].lower()
clean2 = re.sub(r'[?.,\/#!$%\^&\*;:{}=\-_`~()]','',clean1)

jacc1 = set(clean2.split())
jacc1




clean1 = train_raw.loc[1,"question2"].lower()
clean2 = re.sub(r'[?.,\/#!$%\^&\*;:{}=\-_`~()]','',clean1)

jacc2 = set(clean2.split())
jacc2




union = jacc1.union(jacc2)
union




intersect = jacc1 & jacc2
intersect




jacc = len(intersect)/len(union)
jacc




len(intersect)




len(union)
train_raw['question']




train_qs = pd.Series(train_raw['question1'].tolist() + train_raw['question2'].tolist()).astype(str)
train_qs.dtypes




words = (" ".join(train_qs)).lower().split()
words






