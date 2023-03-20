#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.style.use('fivethirtyeight')
palette = 'tab10'
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




train = pd.read_csv("../input/train.csv")
train.head()




train = train.sample(frac=.1)




structures = pd.read_csv("../input/structures.csv")
structures.head()




mulliken = pd.read_csv("../input/mulliken_charges.csv")
mulliken.head()




potential_energy = pd.read_csv("../input/potential_energy.csv")
potential_energy.head()




magnetic_shielding = pd.read_csv("../input/magnetic_shielding_tensors.csv")
magnetic_shielding.head()




test = pd.read_csv("../input/test.csv")
test.head()




train = pd.merge(train, structures, 'inner', left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'] )
train = pd.merge(train, structures, 'inner', left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])

train = pd.merge(train, mulliken, 'inner', left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])
train = pd.merge(train, mulliken, 'inner', left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])
train = pd.merge(train, potential_energy, 'inner', 'molecule_name')
train = pd.merge(train, potential_energy, 'inner', 'molecule_name')
train = pd.merge(train, magnetic_shielding, 'inner', left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])
train = pd.merge(train, magnetic_shielding, 'inner', left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])


test = pd.merge(test, structures, 'inner', left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])
test = pd.merge(test, structures, 'inner', left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])

train = train.drop(['atom_index_x', 'atom_index_y'], 1)
test = test.drop(['atom_index_x', 'atom_index_y'], 1)




train.head()




test.count()




train[train.columns[:-1]] .nunique()




train.isna().sum()




fig, ax = plt.subplots(1,2 , figsize=(15,6))
size_molecule = pd.pivot_table(train, index=['molecule_name'], aggfunc='size')
sns.distplot(size_molecule, ax=ax[0])
ax[0].set_title('Train Distribution of molecule duplicates')

size_molecule = pd.pivot_table(test, index=['molecule_name'], aggfunc='size')
sns.distplot(size_molecule, ax=ax[1])
ax[1].set_title('Test Distribution of molecule duplicates')
plt.tight_layout()
plt.show()




liste_mol_train = train['molecule_name'].unique()
len(test[test['molecule_name'].isin(liste_mol_train)])




table = pd.pivot_table(train, 'scalar_coupling_constant', ['atom_1', 'atom_2', 'type'] ).reset_index()

submission = pd.merge(test, table, 'left', ['atom_1', 'atom_2', 'type'])
table




submission.head()




submission['scalar_coupling_constant'].dropna().count()




fig, ax = plt.subplots(1,2 , figsize=(15,6))

sns.countplot(y='type', data=train, order=train['type'].value_counts().index, ax=ax[0], palette=palette)
sns.barplot('scalar_coupling_constant', 'type', data=train, order=train['type'].value_counts().index, palette=palette, ax=ax[1])
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(1,2 , figsize=(15,6))

sns.barplot('scalar_coupling_constant', 'atom_x', data=train, ax=ax[0])
sns.barplot('scalar_coupling_constant', 'atom_y', data=train, ax=ax[1])
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(1,2 , figsize=(15,6))

sns.barplot('scalar_coupling_constant', 'atom_x', data=train, ax=ax[0])
sns.barplot('scalar_coupling_constant', 'atom_y', data=train, ax=ax[1])
plt.tight_layout()
plt.show()




train.columns




fig, ax = plt.subplots(1,3 , figsize=(15,6))
sns.distplot(train['x_x'], ax=ax[0])
sns.distplot(train['y_x'], ax=ax[1])
sns.distplot(train['z_x'], ax=ax[2])

plt.tight_layout()
plt.show()




fig, ax = plt.subplots(1,3 , figsize=(15,6))
sns.distplot(train['x_y'], ax=ax[0])
sns.distplot(train['y_y'], ax=ax[1])
sns.distplot(train['z_y'], ax=ax[2])
#ax[0].set_title('Train Distribution of molecule duplicates')

plt.tight_layout()
plt.show()




g=sns.FacetGrid(train, col="atom_y", height=4, aspect=1)
g.map(sns.distplot, "x_y", hist=False)
plt.show()

g=sns.FacetGrid(train, col="atom_y", height=4 aspect=1)
g.map(sns.distplot, "y_y", hist=False)
plt.show()


g=sns.FacetGrid(train, col="atom_y", height=4, aspect=1)
g.map(sns.distplot, "z_y", hist=False)
plt.show()




train.head()

