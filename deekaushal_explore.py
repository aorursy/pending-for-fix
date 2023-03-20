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

print (2)


clients = pd.read_csv("../input/cliente_tabla.csv")


products = pd.read_csv("../input/producto_tabla.csv")

clients.shape

products.shape

clients

towns = pd.read_csv("../input/town_state.csv")

towns.State.value_counts()

train = pd.read_csv("../input/train.csv", nrows=10000000)
train

train.Demanda_uni_equil.value_counts()

print (@)
