#!/usr/bin/env python
# coding: utf-8



# Importar bibliotecas 
import os
import time
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model 
from keras.layers import Dense,Conv2D,Dropout,MaxPooling2D,Flatten
from keras.callbacks import EarlyStopping
from keras import optimizers




print(os.listdir("../input"))




train_path='../input/train/train'
test_path='../input/test/test'




label_train=pd.read_csv("../input/train.csv")
print(label_train.head(10))
# ordenar os ID
label_train=label_train.sort_values(by=['id'])
print(label_train.head(10))




# criar 2 matrizes distintas, uma para ids e outra para labels
ids=label_train['id'].values
labels=label_train['has_cactus'].values




print(ids)




train=[]
X=[]
Y=[]
for index,imagem in enumerate(sorted(os.listdir(train_path))):
    path=os.path.join(train_path,imagem)
    frame=cv2.imread(path,cv2.IMREAD_COLOR)
    X.append(frame)
    train.append([np.array(frame),labels[index]])




train=np.array(train)
print(train[0].shape)
print(len(train))
Y=train[:,1]

print(len(Y))
train=train[:,0]
X=np.array(X)

X.shape

X=X/255
train=train/255




plt.figure(figsize = (8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if Y[i]==1:
        label="Tem Cactus"
    elif Y[i]==0:
        label="Não tem Cactus"
    plt.xlabel(label,fontsize=8)
    plt.imshow(train[i])
plt.suptitle("Primeiras imagems ",fontsize=8)
plt.show()




Preparando os dados de Teste




#Preparando os dados de Teste
test_viz=[]
X_test=[]

# devolve o index do registro eo path da imagem 
for index, imagem in enumerate(os.listdir(test_path)):
    # concatena o caminho da imagem 
    path = os.path.join(test_path,imagem)
    # ler com opencv e gera uma matrix 
    frame = cv2.imread(path,cv2.IMREAD_COLOR)
    # indexa os frames ou imagems em array de imagens 
    X_test.append(frame)
    # indexa um array de na primeira posicao do index 
    test_viz.append([np.array(frame),index])




X_test=np.array(X_test)
print(X_test.shape)
#print(X_test[1])

# plt.imshow(X_test[1])
# plt.show()
test_viz=np.array(test_viz)
print(test_viz.shape)

id_test=test_viz[:,1]
print(id_test.shape)

test_viz=test_viz[:,0]

test_viz.shape

X_test=X_test/255
test_viz=test_viz/255
#print(test_viz[0])
# plt.imshow(test_viz[1])
# plt.show()




# Plotar as primeiras 25 imagens no Conjunto de teste
plt.figure(figsize = (8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_viz[i])
plt.suptitle("Primeiras imagems ",fontsize=8)
plt.show()




def convulacional():
  
  modelo = Sequential()
  
  modelo.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu",input_shape=(32,32,3)))
  
  modelo.add(MaxPooling2D(pool_size=2,strides=1))
  
  modelo.add(Dropout(0.2))
  
  modelo.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
  
  modelo.add(MaxPooling2D(pool_size=2,strides=1))
  
  modelo.add(Dropout(0.2))
  
  modelo.add(Conv2D(filters=128,kernel_size=2,padding="same",activation="relu"))
  
  modelo.add(MaxPooling2D(pool_size=2,strides=1))
  
  modelo.add(Dropout(0.2))
  
  modelo.add(Flatten())
  
  modelo.add(Dense(32,activation="relu"))
  
  modelo.add(Dense(64,activation="relu"))
  
  modelo.add(Dropout(0.5))
  
  modelo.add(Dense(1,activation="sigmoid"))
   
  return modelo 




# treinamento do modelo
# chama a funcao adam que possui a estrutura do rede 
modelo = convulacional()

#seta os parametros de compilação da rede o tipo de perda e tipo de otimizador 
# exemplos de performaces e acuracia de otimizadores 
modelo.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
modelo.summary()




# inicio do processo 
inicio = time.time()


epocas = 40
batch = 32

# modelo fit recebe treinamento X e teste Y do trainemnto
historico = modelo.fit(X,Y,batch_size=batch,validation_split=0.2,epochs=epocas)

# fim do processo 
fim = time.time()

fim_processo = fim-inicio

print("Treinamento concluído em% d minutos e% d segundos" %(fim_processo/60,fim_processo*60))




acc=historico.history['acc']
val_acc=historico.history['val_acc']
loss=historico.history['loss']
val_loss=historico.history['val_loss']




plt.plot(acc)
plt.plot(val_acc)
plt.title('Cactus_identifier_net1 Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.show()




plt.plot(loss)
plt.plot(val_loss)
plt.title('Cactus_identifier_net1 Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.show()




plt.figure()
plt.plot(historico.history['acc'],'b*',)
plt.plot(historico.history['val_acc'],'g')
plt.title('model accuracy - batch size = '+str(batch))
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')  
plt.show()




result = modelo.evaluate(X, Y, batch_size=32)
print('\ntrein. acc:', result[1])
print('trein. loss:', result[0])

