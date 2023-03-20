#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from pathlib import Path




from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.cluster import hierarchy as hc
from fastai.tabular import *




import seaborn as sns




PATH = Path('../input')




train_df = pd.read_csv(PATH/'train.csv')
test_df = pd.read_csv(PATH/'test.csv')
train_df.head(), test_df.head()




dep_var = 'target'
cont_names = [column for column in train_df.columns][1:-1]




train_df[cont_names].head()




size = len(cont_names)
cols = int(math.sqrt(size))
rows = cols

fig = plt.figure(figsize=(60, 60))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(size):
    sp = fig.add_subplot(cols, cols, i+1)
    sp.set_title('%3d'%i)
    sp.hist(train_df[cont_names[i]])




train_df[cont_names[146]].hist()




X = train_df[cont_names].values
y = train_df[dep_var].values




X.shape, y.shape




X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=31)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape




m = RandomForestClassifier(n_estimators=40, max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')




m.




def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_})




fi = rf_feat_importance(m, train_df[cont_names]); fi[:10]




def plot_fi(fi, fs=(12,15)): return fi.plot('cols', 'imp', 'barh', figsize=fs, legend=False)




plot_fi(fi, fs=(12,50));




corr = np.round(scipy.stats.spearmanr(train_df[cont_names]).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,30))
dendrogram = hc.dendrogram(z, labels=cont_names, orientation='left', leaf_font_size=12)
plt.show()




procs = [FillMissing, Normalize]




test = TabularList.from_df(test_df, path='.', cont_names=cont_names)




data = (TabularList.from_df(train_df, path='.', cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(800,1000)))
                           .label_from_df(cols=dep_var)
                           .add_test(test)
                           .databunch())




data.show_batch(rows=10)




learn = tabular_learner(data, layers=[200,100], metrics=accuracy)




learn.fit(1, 1e-2)




learn.unfreeze()




learn.fit_one_cycle(10)






