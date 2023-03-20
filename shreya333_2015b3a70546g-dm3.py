#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np




df = read_csv("../input/opcode_frequency_benign.csv")




# df.head()




df_mal = read_csv("../input/opcode_frequency_malware.csv")




# df_mal.head()




# df.drop('FileName', axis=1)
# df_mal.drop('FileName', axis=1)




# df




# import pandas as pd
dff = pd.concat([df, df_mal])




# df.shape[0]
# # df_mal.shape[0]
# # dff.shape[0]




# dff['2'][0]




# df_final = dff




dff = dff.reset_index()




Y = [0]*1400
Y = Y + [1]*1999




len(Y)




dff = dff.drop('FileName', axis=1)




dff = dff.drop('index', axis=1)




# dff
X = dff.values




model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)




arr = np.array(model.feature_importances_)
inds = np.argsort(arr)
ar = arr
np.sort(ar)




# arr[0]




from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier




# knn_cv = KNeighborsClassifier(n_neighbors=3)
# cv_scores = cross_val_score(knn_cv, X, Y, cv=5)




# arr[926]




# inds[-404:]




# dff = df_final
inds_int = inds[-404:]
# inds_int




# dff_copy = dff




# dff




# dff = dff_copy
for x in range(1,1809):
    if x not in inds_int:
        dff = dff.drop(str(x), axis=1)




# dff




X = dff.values
# len(X[0])




# knn_cv = KNeighborsClassifier(n_neighbors=3)
# cv_scores = cross_val_score(knn_cv, X, Y, cv=5)




# print(cv_scores)
# print(np.mean(cv_scores))




# from sklearn.model_selection import GridSearchCV

# knn2 = KNeighborsClassifier()

# param_grid = {'n_neighbors': np.arange(1, 25)}

# knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

# knn_gscv.fit(X, Y)




# knn_gscv.best_params_




test_df = read_csv('../input/Test_data.csv')
# test_df
# file = df['FileName']




test_df = test_df.drop('Unnamed: 1809', axis=1)




test_df = test_df.drop('FileName', axis=1)




# test_df




# %matplotlib inline
# from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA
# pca = PCA().fit(dff)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');




from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier()




# len(X[0])
ranfor = RandomForestClassifier()
ranfor.fit(X, Y)




for x in range(1,1809):
    if x not in inds_int:
        test_df = test_df.drop(str(x), axis=1)




# test_df




X_val = test_df.values
predicted = ranfor.predict(X_val)




# predicted[0:18]




sample = read_csv('../input/sample_submission.csv')
sample['Class'] = predicted
# sample




sample = sample.set_index('FileName')




sample.to_csv('submission.csv')









# from sklearn.linear_model import LogisticRegression




# logreg = LogisticRegression()
# cv_scores = cross_val_score(logreg, X, Y, cv=5)




# print(cv_scores)
# print(np.mean(cv_scores))




# X = dff_404.values




# from sklearn.ensemble import RandomForestClassifier
# ranfor = RandomForestClassifier()
# cv_scores = cross_val_score(ranfor, X, Y, cv=5)




from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode())
payload = b64.decode()
html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
html = html.format(payload=payload,title=title,filename=filename)
return HTML(html)
create_download_link(sample)

