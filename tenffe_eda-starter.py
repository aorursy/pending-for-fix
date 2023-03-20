#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




import os
import os.path as osp
import sys
from tqdm import tqdm_notebook as tqdm
from IPython.display import display, clear_output




get_ipython().run_cell_magic('time', '', "path = '/kaggle/input/data-science-bowl-2019/'\ntrain_df = pd.read_csv(osp.join(path, 'train.csv'))\ntest_df = pd.read_csv(osp.join(path, 'test.csv'))\ntrain_labels_df = pd.read_csv(osp.join(path, 'train_labels.csv'))\nspecs_df = pd.read_csv(osp.join(path, 'specs.csv'))\nsub_df = pd.read_csv(osp.join(path, 'sample_submission.csv'))")




def show_df_info(df):
    display(df.head(2), df.columns, df.shape)




show_df_info(train_df)




show_df_info(train_labels_df)




def get_shared_columns(df_1, df_2):
    return [x for x in df_1.columns if x in df_1.columns and x in df_2.columns]
    
shares_column_names = get_shared_columns(train_labels_df, train_df)
display(shares_column_names)




show_df_info(test_df)




get_shared_columns(train_labels_df, test_df)




get_shared_columns(train_df, test_df)




show_df_info(specs_df)




display(get_shared_columns(specs_df, train_df),
        get_shared_columns(specs_df, train_labels_df),
        get_shared_columns(specs_df, test_df))




show_df_info(sub_df)




accuracy_group = np.array(train_labels_df['accuracy_group'])
display(set(accuracy_group))




get_ipython().run_cell_magic('time', '', "train = pd.merge(train_df, train_labels_df, on = ['game_session', 'installation_id', 'title'])\nshow_df_info(train)")




get_ipython().run_cell_magic('time', '', "train = pd.merge(train, specs_df, on = ['event_id'])\nshow_df_info(train)")




get_ipython().run_cell_magic('time', '', "test = pd.merge(test_df, sub_df, on=['installation_id'])\nshow_df_info(test)")




get_ipython().run_cell_magic('time', '', "test = pd.merge(test, specs_df, on=['event_id'])\nshow_df_info(test)")




columns = get_shared_columns(train, test)
id_str = 'installation_id'
target_str = 'accuracy_group'
features = [column for column in columns if column not in [id_str, target_str]]

display(columns, len(columns), features, len(features))




get_ipython().run_cell_magic('time', '', 'features_numbers = [len(set(train[feature])) for feature in features]\ndisplay(features, features_numbers)')




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader




class zx_data









# Get the random results
accuracy_group_list = np.random.randint(2, 4, size=(sub_df.shape[0], 1))
accuracy_group_list




sub_df['accuracy_group'] = accuracy_group_list
sub_df.head()




sub_df.to_csv('submission.csv', index=False)

