#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import svm
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#import xgboost as xgb

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import math as ma

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 
Trainset = Trainset.dropna()

Trainset  = Trainset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])
Trainset  = Trainset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])
#Trainset  = Trainset.replace(to_replace=[True,False], value=[1,2])

sampleset = pd.read_csv("/kaggle/input/hackstat2k19/sample_submisison.csv", header = 0) 
sampleset = sampleset.dropna()

#Trainset1 = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 
#Trainset1 = Trainset1.dropna()

#dummies1 = pd.get_dummies(Trainset1.Month)
#merged1 = pd.concat([Trainset1,dummies1],axis = 'columns')

#dummies2 = pd.get_dummies(merged1.VisitorType)
#merged2 = pd.concat([merged1,dummies2],axis = 'columns')

#dummies3 = pd.get_dummies(merged2.OperatingSystems)
#dummies3 = dummies3.drop([1], axis = 'columns')
#merged3 = pd.concat([merged2,dummies3],axis = 'columns')

#dummies4 = pd.get_dummies(merged3.Browser)
#dummies4 = dummies4.drop([9], axis = 'columns')
#merged4 = pd.concat([merged3,dummies4],axis = 'columns')

#dummies5 = pd.get_dummies(merged4.Province)
#dummies5 = dummies5.drop([9], axis = 'columns')
#merged5 = pd.concat([merged4,dummies5],axis = 'columns')
y = Trainset['Revenue']
#Trainset = merged5.drop(['Month','VisitorType','Weekend','OperatingSystems','Browser','Province','Revenue'], axis = 'columns')

xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")
xset = xset.dropna()

xset  = xset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])
xset  = xset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])
#xset  = xset.replace(to_replace=[True,False], value=[1,2])

#xset1 = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")
#xset1 = xset1.dropna()

#xdummies1 = pd.get_dummies(xset1.Month)
#xmerged1 = pd.concat([xset1,xdummies1],axis = 'columns')

#xdummies2 = pd.get_dummies(xmerged1.VisitorType)
#xmerged2 = pd.concat([xmerged1,xdummies2],axis = 'columns')

#xdummies3 = pd.get_dummies(xmerged2.OperatingSystems)
#xdummies3 = xdummies3.drop([1], axis = 'columns')
#xmerged3 = pd.concat([xmerged2,xdummies3],axis = 'columns')

#xdummies4 = pd.get_dummies(xmerged3.Browser)
#xdummies4 = xdummies4.drop([9], axis = 'columns')
#xmerged4 = pd.concat([xmerged3,xdummies4],axis = 'columns')

#xdummies5 = pd.get_dummies(xmerged4.Province)
#xdummies5 = dummies5.drop([9], axis = 'columns')
#xmerged5 = pd.concat([xmerged4,xdummies5],axis = 'columns')

#xset = xmerged5.drop(['ID','Month','VisitorType','Weekend','OperatingSystems','Browser','Province'], axis = 'columns')
xset = xset.drop(['ID'], axis = 'columns')
#Trainset['is_train'] = np.random.uniform(0,1,len(Trainset)) <= 0.75
print(Trainset)

Train, Test = Trainset,xset

print('Number of Training examples', len(Train))
print('Number of Testing examples', len(Test))

#features = Test.columns[15:17]
features = Test.columns[:54]
print(features)
y

id_ = sampleset['ID']
#new_ = id_.DataFarame(id_)
clf = RandomForestClassifier(n_jobs = 2, random_state = 0)
clf.fit(Train[features],y)

clf.predict(Test[features])

clf.predict_proba(Test[features])

preds = clf.predict(Test[features])

print("Random forest = ", preds)
#print(tpe(new_))
#print(type(preds))
YArray= id_.as_matrix(columns=None)
#print (YArray)

df = pd.DataFrame({"ID" : YArray, "Revenue" : preds})
#df.to_csv("submission_withfive.csv", index=False)

#########################################################################################################
#from sklearn import datasets
#iris = datasets.load_iris()
#nb = MultinomialNB()
#nbclassifier = nb.fit(Train[features], y)
#y_pred_nb = nbclassifier.predict(Test[features])
#>>> print("Number of mislabeled points out of a total %d points : %d"
#...       % (iris.data.shape[0],(iris.target != y_pred).sum()))
#Number of mislabeled points out of a total 150 points : 6
#print("Gaussian Naive Bayes = ", y_pred_gnb)
#########################################################################################################

#ncclf = NearestCentroid()
#ncclf.fit(Train[features], y)
#y_pred_nc = ncclf.predict(Test[features])
#print("nb = ", y_pred_nb)

from sklearn.linear_model import SGDClassifier
sgdclf = MLPClassifier()
sgdclf.fit(Train[features], y) 
y_pred_sgd = sgdclf.predict(Test[features])  
print("sgd = ", y_pred_sgd)
#########################################################################################################

from sklearn import tree
#iris = load_iris()
treeclf = tree.DecisionTreeClassifier()
treeclf = treeclf.fit(Train[features], y)
y_pred_tree = treeclf.predict(Test[features])
print("Decision Tree classifier = ",y_pred_tree)

#########################################################################################################

#svmclf = svm.SVC()
#svmclf.fit(Train[features], y)  
#y_pred_svm = svmclf.predict(Test[features])
#print("svm classifier = ",y_pred_svm)

cclassifier = LinearDiscriminantAnalysis()
cclassifier.fit(Train[features], y) 
y_pred_svm = cclassifier.predict(Test[features])
print("c classifier = ",y_pred_svm)
########################################################################################################

knnclassifier = KNeighborsClassifier()
knnclassifier.fit(Train[features], y) 
y_pred_knn = knnclassifier.predict(Test[features])
print("KNN classifier = ",y_pred_knn)

#########################################################################################################
df = pd.DataFrame({"ID" : YArray, "Random" : preds, "sgd" : y_pred_sgd,"tree" : y_pred_tree, "c" : y_pred_svm, "knn" : y_pred_knn})
df.to_csv("submission_neeeew.csv", index=False)
df
#svclassifier = SVC(kernel='linear')
#svclassifier.fit(Train[features], y)
#y_pred = svclassifier.predict(Test)
#print("SVC classifier = ",y_pred)
count = 0
pred_list = []
for i in range(len(y_pred_knn)):
    count = 0
    c_list = [preds[i],y_pred_sgd[i],y_pred_tree[i],y_pred_svm[i],y_pred_knn[i]]
    for j in range(5):
        if c_list[j] == 1:
            count += 1
    if count >= 3:
        pred_list += [1]
    else:
        pred_list += [0]
print(len(pred_list))
o = pd.DataFrame(pred_list) 

pred_= o.as_matrix(columns=None)
df = pd.DataFrame({"ID" : YArray, "Revenue" : pred_list})
df.to_csv("submission_v2.csv", index=False)


# In[2]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn import tree

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math as ma

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Trainset1 = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 
Trainset1 = Trainset1.dropna()

#Trainset  = Trainset1.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])
#Trainset  = Trainset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])
#Trainset  = Trainset.replace(to_replace=[True,False], value=[1,2])
dummies1 = pd.get_dummies(Trainset1.Month)
merged1 = pd.concat([Trainset1,dummies1],axis = 'columns')

dummies2 = pd.get_dummies(merged1.VisitorType)
merged2 = pd.concat([merged1,dummies2],axis = 'columns')

dummies3 = pd.get_dummies(merged2.OperatingSystems)
#dummies3 = dummies3.drop([1], axis = 'columns')
merged3 = pd.concat([merged2,dummies3],axis = 'columns')

dummies4 = pd.get_dummies(merged3.Browser)
#dummies4 = dummies4.drop([1], axis = 'columns')
merged4 = pd.concat([merged3,dummies4],axis = 'columns')

dummies5 = pd.get_dummies(merged4.Province)
#dummies5 = dummies5.drop([9], axis = 'columns')
merged5 = pd.concat([merged4,dummies5],axis = 'columns')

#dummies3 = pd.get_dummies(merged2.OperatingSystems)
#dummies3 = merged3.drop(['Month','VisitorType','Weekend','Feb','Other','OperatingSystems',1], axis = 'columns')
#merged3 = pd.concat([merged2,dummies3],axis = 'columns')

Trainset = merged5.drop(['Month','VisitorType','Weekend','OperatingSystems','Browser','Province'], axis = 'columns')

print(Trainset)
xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")
xset = xset.dropna()

Trainset['is_train'] = np.random.uniform(0,1,len(Trainset)) <= 0.75
print(Trainset['is_train'])

Train, Test = Trainset[Trainset['is_train'] == True], Trainset[Trainset['is_train'] == False]
y = Train['Revenue']

Train = Train.drop(['Revenue'], axis = 'columns')
o = Test['Revenue']
Test = Test.drop(['Revenue'], axis = 'columns')
print('Number of Training examples', len(Train))
print('Number of Testing examples', len(Test))

features = Test.columns[:54]
print(features)

y

clf = tree.DecisionTreeClassifier()
clf.fit(Train[features],y)

#clf.predict(Test[features])

#clf.predict_proba(Test[features])

preds = clf.predict(Test[features])
print(pd.crosstab(o, preds, rownames = ['Actual Revenue'], colnames = ['Predicted Revenue']))

i = pd.crosstab(o, preds, rownames = ['Actual Revenue'], colnames = ['Predicted Revenue'])
print((i[0][0] + i[1][1])/len(Test[features]))
#clf = RandomForestClassifier(n_estimators=17, max_depth=None,min_samples_split=2, random_state=0)
#scores = cross_val_score(clf, X, y, cv=5)
#scores.mean() 
print("Number of Training examples", 7850)
print("Number of Testing examples", 2616)
print('Predicted Revenue   ', 0,"  ", 1)
print('Actual Revenue')
print(0, "                ", 2131 ," ",92)
print(1, "                 ", 149,"",244)
print("accuracy" , (21)


# In[3]:


import pandas as pd

df = pd.DataFrame({'A': [5, 2], 'B': [4, 8]})

print("Pearson correlation",df.corr(method='pearson'))

print("kendall correlation",df.corr(method='kendall'))

print("spearman correlation",df.corr(method='spearman'))


# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
#import math as ma

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Trainset1 = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 

dummies1 = pd.get_dummies(Trainset1.Month)
merged1 = pd.concat([Trainset1,dummies1],axis = 'columns')

dummies2 = pd.get_dummies(merged1.VisitorType)
merged2 = pd.concat([merged1,dummies2],axis = 'columns')

dummies3 = pd.get_dummies(merged2.OperatingSystems)
#dummies3 = dummies3.drop([1], axis = 'columns')
merged3 = pd.concat([merged2,dummies3],axis = 'columns')

dummies4 = pd.get_dummies(merged3.Browser)
#dummies4 = dummies4.drop([9], axis = 'columns')
merged4 = pd.concat([merged3,dummies4],axis = 'columns')

dummies5 = pd.get_dummies(merged4.Province)
#dummies5 = dummies5.drop([9], axis = 'columns')
merged5 = pd.concat([merged4,dummies5],axis = 'columns')

dummies6 = pd.get_dummies(merged5.Weekend)
merged6 = pd.concat([merged5,dummies6],axis = 'columns')

y = merged6['Revenue']

Trainset = merged6.drop(['Month','VisitorType','Weekend','OperatingSystems','Browser','Province','Revenue'], axis = 'columns')
Trainset = pd.concat([Trainset,y],axis = 'columns')

print(Trainset)
#df = pd.DataFrame(Trainset)
#df.to_csv("NewtTrainset.csv", index=False)

# calculate the correlation matrix
corr = Trainset.corr()


plt.figure(figsize=(40,40)) 
# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,vmin = -1,vmax =1 ,center = 0,cmap = sns.diverging_palette(-400,220,n=200),square = True)


# In[5]:


df


# In[6]:


import numpy as np # linear algebra
import pandas as pd

xset1 = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")
xset1 = xset1.dropna()

xdummies1 = pd.get_dummies(xset1.Month)
xmerged1 = pd.concat([xset1,xdummies1],axis = 'columns')

xdummies2 = pd.get_dummies(xmerged1.VisitorType)
xmerged2 = pd.concat([xmerged1,xdummies2],axis = 'columns')

xdummies3 = pd.get_dummies(xmerged2.OperatingSystems)
xmerged3 = pd.concat([xmerged2,xdummies3],axis = 'columns')

xdummies4 = pd.get_dummies(xmerged3.Browser)
xmerged4 = pd.concat([xmerged3,xdummies4],axis = 'columns')

xdummies5 = pd.get_dummies(xmerged4.Province)
xmerged5 = pd.concat([xmerged4,xdummies5],axis = 'columns')

xdummies6 = pd.get_dummies(xmerged5.Weekend)
xmerged6 = pd.concat([xmerged5,xdummies6],axis = 'columns')

xset = xmerged6.drop(['Month','VisitorType','Weekend','OperatingSystems','Browser','Province'], axis = 'columns')
print(xset)

dfx = pd.DataFrame(xset)
dfx.to_csv("xtest.csv", index=False)


# In[7]:


dfx


# In[ ]:




