#!/usr/bin/env python
# coding: utf-8





import numpy as np
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")
loading_features = list(loading_df.columns[1:])
labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")
labels_df["is_train"] = True
loading_df=loading_df.merge(labels_df, on="Id", how="left")




test_df=loading_df[loading_df["is_train"] != True].copy()
train_df = loading_df[loading_df["is_train"] == True].copy()




print(test_df.shape,train_df.shape)




train_df['domain1_var1'].fillna(train_df['domain1_var1'].mean(),inplace=True)
train_df['domain1_var2'].fillna(train_df['domain1_var2'].mean(),inplace=True)
train_df['domain2_var1'].fillna(train_df['domain2_var1'].mean(),inplace=True)
train_df['domain2_var2'].fillna(train_df['domain2_var2'].mean(),inplace=True)




train_df.head()




from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_squared_error,mean_absolute_error




X=train_df.drop(columns=labels_df)
Y=labels_df.drop(columns=['Id','is_train'])




Y['domain1_var1'].fillna(Y['domain1_var1'].mean(),inplace=True)
Y['domain1_var2'].fillna(Y['domain1_var2'].mean(),inplace=True)
Y['domain2_var1'].fillna(Y['domain2_var1'].mean(),inplace=True)
Y['domain2_var2'].fillna(Y['domain2_var2'].mean(),inplace=True)




models=[LinearRegression(),Ridge(),Lasso(),ElasticNet()]
names=['linearregression','ridgeregression','lassoregression','elasticnetregression']
for model, name in zip(models, names):
    print(name)
    print(cross_val_score(model,X,Y.domain1_var2,scoring='neg_mean_absolute_error', cv=5).mean())




y.drop(columns=['Id','is_train']).columns




test_df.drop(columns=labels_df.columns,inplace=True)




test_df




lis=[]
for i in labels_df.drop(columns=['Id','is_train']).columns:
    elastic=ElasticNet()
    elastic.fit(X,Y[i])
    pred=elastic.predict(test_df)
    lis.append(pred)




lis=np.array(lis)




sub=pd.read_csv('/kaggle/input/trends-assessment-prediction/sample_submission.csv')




for i in 




for i in range(lis.shape[1]):
    sub.iloc[(5*i)+0,1]=lis[0,i]
    sub.iloc[(5*i)+1,1]=lis[1,i]
    sub.iloc[(5*i)+2,1]=lis[2,i]
    sub.iloc[(5*i)+3,1]=lis[3,i]
    sub.iloc[(5*i)+4,1]=lis[4,i]




sub.to_csv('submit.csv',index=False)






