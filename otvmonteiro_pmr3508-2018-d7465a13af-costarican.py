#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import sklearn




xTrain = pd.read_csv("../input/train.csv",
           sep=r'\s*,\s*',engine='python'
           )
print(xTrain.head)




xTrain = xTrain.drop(columns=["v2a1", "v18q1", "rez_esc"], index =1)




xTrain = xTrain.dropna()
xTrain.shape




yTrain = xTrain.Target




xTrain = xTrain.drop(columns=["Id","Target"])




from sklearn import preprocessing




xTrain = xTrain.apply(preprocessing.LabelEncoder().fit_transform)




from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt




k_scores = []
k_values = []
k=80
while(k <= 130):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, xTrain, yTrain, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
    k_values.append(k)
    k = k + 5




plt.plot(k_values,k_scores)
plt.xlabel("Numero de vizinhos(k)")
plt.ylabel("Precisão Média por Validação Cruzada")




k = 116
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(xTrain, yTrain)




xTest = pd.read_csv("../input/test.csv",
           sep=r'\s*,\s*',engine='python')
Pred = pd.DataFrame(columns=["Id","Target"])
Pred["Id"]= xTest.Id




xTest = xTest.fillna(xTest.mean())
xTest = xTest.drop(columns=["Id", "v2a1", "v18q1", "rez_esc"])
xTest = xTest.apply(preprocessing.LabelEncoder().fit_transform)




yPred = knn.predict(xTest)
Pred["Target"] = yPred




Pred.to_csv("sample_submission.csv", index=False)
Pred




kaggle competitions submit -c costa-rican-household-poverty-prediction -f submission.csv -m "Message"

