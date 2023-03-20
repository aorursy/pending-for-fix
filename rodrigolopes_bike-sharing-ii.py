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




# Importando os Arquivos
df = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

df.shape, test.shape




#Verificando o datafreime de treino

df.info()




test.info()




#Realizando a transformação dos Dados

#Aplicar log na variavel de resposta
df['count']=np.log(df['count'])




#unindo os dataframe

df = df.append(test)




#Convertendo a coluna datetime

df['datetime']=pd.to_datetime(df['datetime'])




#Criando colunas derivadas da coluna datetime

df['year']=df['datetime'].dt.year
df['month']=df['datetime'].dt.month
df['day']=df['datetime'].dt.day
df['hour']=df['datetime'].dt.hour
df['dayofweek']=df['datetime'].dt.dayofweek




# Separando os dataframes de teste, treino e validação

#Primeiro os dados de teste
test=df[df['count'].isnull()]




#Agora os dados de treino

df=df[~df['count'].isnull()] # Comando ""~"" significa a negativa booleana




#Verificando os tamanhos
df.shape, test.shape




#Dividindo o dataframe de treino

#importando o metodo do scikitlearn para divisão da base

from sklearn.model_selection import train_test_split




#Dividir a base de treino

train, valid = train_test_split(df, random_state=42)




#Verificando os tamanhos 
train.shape,valid.shape




#selecionar as colunas que usaremos como entrada

#Lista das colunas não usadas
removed_cols = ['casual','registered','count','datetime']

#Lista das colunas de entrada
feats = [c for c in train.columns if c not in removed_cols]




#importando o mopdelo Random Foreste

from sklearn.ensemble import RandomForestRegressor




#instanciar o modelo rando state = 42 galhos 

rf=RandomForestRegressor(random_state=42, n_jobs=-1)




#Treinando o modelo - informar as entradas e saídas
rf.fit(train[feats], train['count'])




#Fazendo previsões em cima dos dados de valição
preds = rf.predict(valid[feats])




#Verificando as previsoes
preds




#Verificando o real
valid['count'].head(3)




#Verificando o modelo com relação a métrica

#Importando a métrica
from sklearn.metrics import mean_squared_error




#Aplicando a metrica
mean_squared_error(valid['count'],preds)**(1/2)




#Previsão com base nos dados de treino
#Como o modelo se comporta prevendo atraves de dados conhecidos
train_preds = rf.predict(train[feats])

mean_squared_error(train['count'],train_preds)**(1/2)




#Gerando as previsões para envio ao kaggle

test['count']=np.exp(rf.predict(test[feats]))




#Visualizando o arquivo para envio
test[test['datetime',]




#Gerando o arquivo
test[['datetime','count']].to_csv('rf.csv', index=False)






