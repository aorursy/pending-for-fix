#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2




data_o=pd.read_csv("../input/dataset.csv")
data=data_o




data.head()




data.tail()




data




data.dtypes




# from sklearn.ensemble import ExtraTreesClassifier
# X=data_o.drop(['id','Class'],axis=1)[:175]
# y=data_o['Class'][:175]
# model = ExtraTreesClassifier()
# model.fit(X,y)
# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
# #plot graph of feature importances for better visualization
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()




data['Account1'].unique()




x=float('nan')
data.replace('?',x,inplace=True)




datar['Account1'].unique()




for col in data.columns:
    data[col].fillna(data[col].mode()[0],inplace=True)




data['Account1'].unique()









data = pd.get_dummies(data, columns=['Account1'], prefix = ['Account1'])
data.head()




data['History'].unique()




data = pd.get_dummies(data, columns=['History'], prefix = ['History'])
data.head()




# data = pd.get_dummies(data, columns=['Motive'], prefix = ['Motive'])
# data.head()




data=data.drop(['Motive'],axis=1)




data['Account2'].unique()




data['Account2'].replace({'Sacc4':'sacc4'},inplace=True);




data = pd.get_dummies(data, columns=['Account2'], prefix = ['Account2'])
data.head()




data['Employment Period'].unique()




data = pd.get_dummies(data, columns=['Employment Period'], prefix = ['Emp Per'])
data.head()




data['Gender&Type'].unique()




data = pd.get_dummies(data, columns=['Gender&Type'], prefix = ['G&T'])
data.head()




data['Sponsors'].unique()




data['Sponsors'].replace({'g1':'G1'},inplace=True);




data['Sponsors'].unique()




data = pd.get_dummies(data, columns=['Sponsors'], prefix = ['Sponsors'])
data.head()




data['Plotsize'].unique()




data['Plotsize'].replace({
    'sm':0,
    'SM':0,
    'me':1,
    'ME':1,
    'M.E.':1,
    'la':2,
    'LA':2,
    'XL':3
},inplace=True);




data.head()




data['Housing'].unique()




data = pd.get_dummies(data, columns=['Housing'], prefix = ['Housing'])
data.head()




data['Plan'].unique()




data = pd.get_dummies(data, columns=['Plan'], prefix = ['Plan'])
data.head()




data['Post'].unique()




data = pd.get_dummies(data, columns=['Post'], prefix = ['Post'])
data.head()




# data = pd.get_dummies(data, columns=['Expatriate'], prefix = ['Expatriate'])
# data.head()




# data = pd.get_dummies(data, columns=['Phone'], prefix = ['Phone'])
# data.head()









data['InstallmentCredit']=pd.to_numeric(data['InstallmentCredit'],errors='coerce')




data['Yearly Period']=pd.to_numeric(data['Yearly Period'],errors='coerce')




data.dtypes




data['Monthly Period']=pd.to_numeric(data['Monthly Period'],errors='coerce')
data['Credit1']=pd.to_numeric(data['Credit1'],errors='coerce')
data['InstallmentRate']=pd.to_numeric(data['InstallmentRate'],errors='coerce')
data['Tenancy Period']=pd.to_numeric(data['Tenancy Period'],errors='coerce')
data['Age']=pd.to_numeric(data['Age'],errors='coerce')




data.dtypes




d1=data.drop(['#Credits','#Authorities','Expatriate','Phone'],axis=1)




d1=d1.drop(['id','Class'],axis=1)
len(d1)




# from sklearn.ensemble import ExtraTreesClassifier
# X=d1[:175]
# y=data_o['Class'][:175]
# model = ExtraTreesClassifier()
# model.fit(X,y)
# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
# #plot graph of feature importances for better visualization
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(26).plot(kind='barh')
# plt.show()




d1.dtypes




# bestfeatures = SelectKBest(score_func=chi2, k=10)
# fit = bestfeatures.fit(X,y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(10,'Score'))  #print 10 best features




# dx = pd.DataFrame()
# for x in featureScores.nlargest(10, 'Score')['Specs']:
#     dx[x] = data[x]
# dx




# feat_importances.nlargest(24)




do = pd.DataFrame()
for x in ['Post_Jb2','InstallmentCredit', 'Tenancy Period', 'Sponsors_G3', 'Sponsors_G1', 'Account2_sacc5', 'G&T_M1', 'Emp Per_time5', 'Plan_PL3','InstallmentRate','History_c4','Housing_H3', 'Age', 'Plotsize', 'History_c3', 'Yearly Period', 'Housing_H2']:
    do[x] = d1[x]
do.head()




import seaborn as sns
f, ax = plt.subplots(figsize=(30, 30))
corr = d1.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);
plt.show()




d1=d1.drop(['Monthly Period'],axis=1)




d1=d1.drop(['Credit1'],axis=1)




d1=d1.drop(['InstallmentCredit'],axis=1)









from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler




scaler=MinMaxScaler()
scaled_data=scaler.fit(d1).transform(d1)
d1=pd.DataFrame(scaled_data,columns=d1.columns)
d1.tail()




from sklearn.cluster import KMeans
wcss = []
for i in range(2, 50):
    kmean = KMeans(n_clusters = i, random_state = 42)
    kmean.fit(d1)
    wcss.append(kmean.inertia_)
plt.plot(range(2,50),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()




from sklearn.decomposition import PCA




pca = PCA().fit(d1)


plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()





pca1 = PCA(n_components=41)
pca1.fit(d1)
T1 = pca1.transform(d1)




kmean = KMeans(n_clusters = 3, random_state = 42)
kmean.fit(d1)
pred = kmean.predict(d1)





plt.scatter(T1[:, 0], T1[:, 1], c=pred)




centroids = kmean.cluster_centers_
centroids = pca1.transform(centroids)
plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)




plt.show()




pred




pred[:200]




data_o['Class'][:200]




dat = [x for x in data_o['Class'] if x==2.0]
len(dat)




from sklearn.cluster import AgglomerativeClustering as AC
aggclus = AC(n_clusters = 3,affinity='euclidean',linkage='ward',compute_full_tree='auto')
y_aggclus= aggclus.fit_predict(d1)
plt.scatter(T1[:, 0], T1[:, 1], c=y_aggclus)
plt.show()




y_aggclus[:200]




final1=pd.DataFrame()




final2=pd.DataFrame()




final2['id']=data['id']
final2['Class']=pred




final1['id']=data['id']
final1['Class']=y_aggclus




final1[:200]




final1[175:].to_csv('20150546g.csv',index=False)




len(final1[175:])




for i 




final2[175:].to_csv('20150546.csv',index=False)




for x in range(0, 175):
    if fina['Class'][x]==0:
        fina['Class'][x]=3
    if fina['Class'][x]==1:
        fina['Class'][x]=4
for x in range(0, 175):
    if fina['Class'][x]==3:
        fina['Class'][x]=1
    if fina['Class'][x]==4:
        fina['Class'][x]=0




count = 0
for x in range(0, 175):
    if y_aggclus[x] == data_o['Class'][x]:
        count+=1
print(count/175)




fina2[175:].to_csv('20150546G.csv',index=False)




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

create_download_link(final1)

