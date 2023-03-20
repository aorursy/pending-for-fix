#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
 

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np # linear algebra
import pandas as pd
Image_width,Image_height = 400,400
batch_size=16
val_split=0.01
n_train=25000*(1-val_split)
n_val=25000*val_split

train = pd.DataFrame({"name" :os.listdir("../input/dogs-vs-cats-redux-kernels-edition/train"), 
                      "path" :pd.Series(os.listdir("../input/dogs-vs-cats-redux-kernels-edition/train")).apply(lambda x : "../input/dogs-vs-cats-redux-kernels-edition/train/" + str(x)), 
                      "label" :pd.Series(os.listdir("../input/dogs-vs-cats-redux-kernels-edition/train")).apply(lambda x : str(x).split(".")[0])})

test = pd.DataFrame({"name" :os.listdir("../input/dogs-vs-cats-redux-kernels-edition/test"), 
                      "path" :pd.Series(os.listdir("../input/dogs-vs-cats-redux-kernels-edition/test")).apply(lambda x : "../input/dogs-vs-cats-redux-kernels-edition/test/" + str(x)), 
                     })
from keras.preprocessing.image import ImageDataGenerator
train_image_gen = ImageDataGenerator(rescale=1/255, horizontal_flip=True, validation_split=val_split)

train_generator = train_image_gen.flow_from_dataframe(dataframe = train, x_col = "path", y_col ="label",
                                                      target_size=(Image_width,Image_height), batch_size=batch_size,seed=19,subset='training',shuffle=True,class_mode='categorical')
val_generator = train_image_gen.flow_from_dataframe(dataframe = train, x_col = "path", y_col ="label",
                                                    target_size=(Image_width,Image_height), batch_size=batch_size,seed=19,subset='validation',shuffle=True,class_mode='categorical')

from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
InceptionV3_base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling = "avg") 

x = InceptionV3_base_model.output
x_dense = Dense(1024,activation='relu')(x)
final_pred = Dense(2,activation='softmax')(x_dense)
model = Model(inputs=InceptionV3_base_model.input, outputs=final_pred)

from keras.callbacks import ModelCheckpoint
cb_checkpoint = ModelCheckpoint('best.hd5', monitor = 'val_loss', save_best_only = True)

from keras.optimizers import Adam
from keras.clallbacks import ReDuceLRnplateau  #러닝레이트가 너무 큰 경우, 섬세한 튜닝할 경우, 최적의 순간에 맴돌 때 성능 높일 수 있음

# 1.파라미터 조정
#(일반적으로 300층 정도 쌓여 있다고 할때, 가중치를 수정하고 업데이트할 때 
#이미지데이터y를 잘 예측하는 그러한 가중치를 찾아 나가는 과정이 딥러닝 모델이 학습하는 방향이다.
#이미지넷에서 가중치를 가져와서 학습을 시작하는데, 처음 층 부터 ~150층까지는 직선, 명암 등의 물체와 나눠지는 배경정보를 학습하기 때문에
#이미지넷에서 쓰는걸 그대로 가져오고 임의로 바꿔주지 않는다.ex 고양이 강아지 모델 나누는 가장 중요한 피쳐를 찾아서 알아서 구분해준다.)
# 앞에서는 고정해주고 뒤에서는 고정해주지 않는다. fine 튜닝, 172라는 숫자는 어떻게 나온건가? 테스트해보기, 
# and, 레이어층들이 어떤 부분에 초점을 맞춰 학습하는지 확인할 수 있는 코드가 있다.(어떤 픽셀에 맞춰서)
#conv레이어들이 뒤로 갈 수록 추상적인 내용 학습. 
# 2. 러닝레이트 조정
#adam optimizer의 러닝레이트 0.001 너무 크다. 러닝 레이트 조절옵션
#reduce튜닝,  val 로스가 낮을 수족 점수가 높아짐. 
#ss

layer_to_Freeze=172 
for layer in model.layers[:layer_to_Freeze]:
    layer.trainable =False
for layer in model.layers[layer_to_Freeze:]:
    layer.trainable=True

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer=Adam(Ir=0.001,decay=1e-6), loss='categorical_crossentropy',metrics=['accuracy'])


#몇 번 참아줄거냐(reduce 관련)
reduceelr= ReduceLROnPl(patience=4,factor=0.1)
#에폭에 비해 patience는 반이다.에폭 결정이 어려울 경우 - 로스가 클 경우에는 페이션스 크게, 로스가 작으면 페이션스 작게한다.
#러닝레이트를 얼마나 낮출거냐 하는 옵션=factor, 

#val 가 가장 중요하다. 평가셋에서 잘 나와야 우리가 값을 잘 알 수 있다. 새로운 데이터에서 데이터가 잘 나와야 (val los는 새로운 데이터에서 당연히 로스가 안나온다)


model.fit_generator(train_generator, epochs=3,
                                                steps_per_epoch=n_train//batch_size,
                                                validation_data=val_generator,
                                                validation_steps=n_val//batch_size,
                                                verbose=1,
                                                callbacks=[reducelr],)
#earlystopping은 자동으로 과접합일어나기 전에 멈춘다. 그래서 patient 옵션을 써줘야한다. 
#성능이 오르지 않아도 참아주는게 필요함.이거는 reduc 보다 1,2정도 더 크다 
#modelchekpoint -> 최적의 상태를 저장할 수 있다(vel가 최적일 때마다 저장, 갱신함. 옵션 2개(파일패스 넣어주기, best_model.h5, save_best_only)
#callbacks에 반드시 옵션을 넣어줘야함
#예측하기 전에 하나가 빠졌다. 코드가 하나 더 들어가야함

model.load_weights('best.hd5')

test_image_gen = ImageDataGenerator(rescale=1/255)
test_generator = test_image_gen.flow_from_dataframe(dataframe = test, x_col = "path", y_col = None,
                                                    target_size=(Image_width,Image_height), batch_size=batch_size, seed=42, class_mode=None, shuffle=False)

model.load.weights("best_model.h5") # 저장한 가중치를 가져와야된다. 
y_pred = model.predict_generator(generator=test_generator, steps=int(np.ceil(len(test)/batch_size)), workers = 2, verbose=1)

submission = pd.DataFrame({'id':pd.Series(test_generator.filenames),'label':pd.Series(y_pred.clip(min=0.005, max=0.995)[:,1])})
submission["id"] = submission["id"].apply(lambda x: str(x).split("/")[-1].split(".")[0])
submission.to_csv('DogVsCats_submission.csv',index=False)


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra
import pandas as pd
Image_width,Image_height = 400,400
batch_size=16
val_split=0.01
n_train=25000*(1-val_split)
n_val=25000*val_split

train = pd.DataFrame({"name" :os.listdir("../input/dogs-vs-cats-redux-kernels-edition/train"), 
                      "path" :pd.Series(os.listdir("../input/dogs-vs-cats-redux-kernels-edition/train")).apply(lambda x : "../input/dogs-vs-cats-redux-kernels-edition/train/" + str(x)), 
                      "label" :pd.Series(os.listdir("../input/dogs-vs-cats-redux-kernels-edition/train")).apply(lambda x : str(x).split(".")[0])})

test = pd.DataFrame({"name" :os.listdir("../input/dogs-vs-cats-redux-kernels-edition/test"), 
                      "path" :pd.Series(os.listdir("../input/dogs-vs-cats-redux-kernels-edition/test")).apply(lambda x : "../input/dogs-vs-cats-redux-kernels-edition/test/" + str(x)), 
                     })
from keras.preprocessing.image import ImageDataGenerator
train_image_gen = ImageDataGenerator(rescale=1/255, horizontal_flip=True, validation_split=val_split)

train_generator = train_image_gen.flow_from_dataframe(dataframe = train, x_col = "path", y_col ="label",
                                                      target_size=(Image_width,Image_height), batch_size=batch_size,seed=19,subset='training',shuffle=True,class_mode='categorical')
val_generator = train_image_gen.flow_from_dataframe(dataframe = train, x_col = "path", y_col ="label",
                                                    target_size=(Image_width,Image_height), batch_size=batch_size,seed=19,subset='validation',shuffle=True,class_mode='categorical')

from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
InceptionV3_base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling = "avg") 

x = InceptionV3_base_model.output
x_dense = Dense(1024,activation='relu')(x)
final_pred = Dense(2,activation='softmax')(x_dense)
model = Model(inputs=InceptionV3_base_model.input, outputs=final_pred)

from keras.callbacks import ModelCheckpoint
cb_checkpoint = ModelCheckpoint('best.hd5', monitor = 'val_loss', save_best_only = True)

layer_to_Freeze=172 
for layer in model.layers[:layer_to_Freeze]:
    layer.trainable =False
for layer in model.layers[layer_to_Freeze:]:
    layer.trainable=True

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_generator, epochs=3,
                                                steps_per_epoch=n_train//batch_size,
                                                validation_data=val_generator,
                                                validation_steps=n_val//batch_size,
                                                verbose=1,
                                                callbacks=[cb_checkpoint],)



model.load_weights('best.hd5')

test_image_gen = ImageDataGenerator(rescale=1/255)
test_generator = test_image_gen.flow_from_dataframe(dataframe = test, x_col = "path", y_col = None,
                                                    target_size=(Image_width,Image_height), batch_size=batch_size, seed=42, class_mode=None, shuffle=False)

y_pred = model.predict_generator(generator=test_generator, steps=int(np.ceil(len(test)/batch_size)), workers = 2, verbose=1)

submission = pd.DataFrame({'id':pd.Series(test_generator.filenames),'label':pd.Series(y_pred.clip(min=0.005, max=0.995)[:,1])})
submission["id"] = submission["id"].apply(lambda x: str(x).split("/")[-1].split(".")[0])
submission.to_csv('DogVsCats_submission.csv',index=False)

batch_size = 50

train_path = "../input/dogs-vs-cats-redux-kernels-edition/train/"
train = pd.Series(os.listdir(train_path))
train = pd.DataFrame({"path" : train.apply(lambda x : train_path + x),
                      "label" : train.apply(lambda x : x.split("train/")[-1][:3])})

test_path = "../input/dogs-vs-cats-redux-kernels-edition/test/"
test = pd.Series(os.listdir(test_path))
test = pd.DataFrame({"path" : test.apply(lambda x : test_path + x)})

train.head()

from keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(validation_split=0.2)

train_generator = train_gen.flow_from_dataframe(train,
                                                x_col="path",
                                                y_col="label",
                                                batch_size=batch_size,
                                                target_size=(299, 299),
                                                subset="training")

val_generator = train_gen.flow_from_dataframe(train,
                                              x_col="path",
                                              y_col="label",
                                              batch_size=batch_size,
                                              target_size=(299, 299),
                                              subset="validation")

from keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(validation_split=0.2,
                               rescale=1/255,
                               horizontal_flip=True)

train_generator = train_gen.flow_from_dataframe(train,
                                                x_col="path",
                                                y_col="label",
                                                batch_size=batch_size,
                                                target_size=(299, 299),
                                                subset="training")

val_generator = train_gen.flow_from_dataframe(train,
                                              x_col="path",
                                              y_col="label",
                                              batch_size=batch_size,
                                              target_size=(299, 299),
                                              subset="validation")

from keras import Sequential
from keras.layers import Dense
from keras.applications import MobileNetV2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(MobileNetV2(weights = "imagenet", include_top = False, pooling = "avg")) #input
model.add(Dense(2, activation = "softmax")) #output

layer_to_Freeze=172
for layer in model.layers[:layer_to_Freeze]:
    layer.trainable = False
for layer in model.layers[layer_to_Freeze:]:
    layer.trainable = True

# model.compile(loss = "categorical_crossentropy", optimizer= Adam(lr=0.001, decay=1e-6), metrics = ["acc"])
model.compile(loss = "categorical_crossentropy", optimizer= "adam", metrics = ["acc"])

reducelr = ReduceLROnPlateau(patience=4, factor=0.1, verbose=1)
earlystop = EarlyStopping(patience=5, verbose=1)
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1)

model.fit_generator(train_generator, 
                    epochs = 10,
                    steps_per_epoch = int(np.ceil(train_generator.n/batch_size)),
                    validation_data=val_generator,
                    validation_steps=int(np.ceil(val_generator.n/batch_size)), 
                    callbacks=[reducelr, earlystop, checkpoint])

sub = pd.read_csv("../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv")
sub["path"] = sub["id"].apply(lambda x: "../input/dogs-vs-cats-redux-kernels-edition/test/" + str(x) + ".jpg")

test_gen = ImageDataGenerator()
test_generator = test_gen.flow_from_dataframe(sub,
                                              x_col="path",
                                              y_col=None,
                                              target_size=(299, 299),
                                              batch_size=batch_size,
                                              class_mode=None,
                                              shuffle=False)

model.load_weights("best_model.h5")
pred = model.predict_generator(test_generator,
                               steps=int(np.ceil(test_generator.n/batch_size)),
                               workers=2,
                               verbose=1)

pred = pred[:, 1]

sub.drop("path", 1, inplace=True)

sub["label"] = pred

train_generator.class_indices

test_generator.filenames

sub.to_csv("submission.csv", index=False)


# In[ ]:


#데이터가 고화질일수록 성능이 오른다. 고화질=타겟사이즈를 올려주면 더 성능이 오르다. 처음에 하면 안되고 뒤에서 해야한다. (픽셀을 더 늘리고 줄여준다)
#크롭: 이미지를 보고 싶은것만 확대한다.(레이블링해서 바운딩박스를 만들어서 그 자체를 학습해도 된다.)
#왜 데이터를 50개일때보다 100개씩 넣어줄 때 (배치사이즈,) 모델 점수도 잘나오고, 속도도 빠를까.  램크기에 따라서 최대로 가져올 수 있는 배치사이즈를 가져오는게 좋을까
# => 곤충을 분류할 때, 50 종류일 때 보다, 100종류의 매미를 구분할 때 더 잘나오는 이유는? 
#= 종류가 많으면 배치사이즈도 커야한다. 가중치가 100개 종류를 모두다 보면서 수정되어야하는데 배치사이즈가 50이면, 50개만 보고 학습되고 나머지는 학습 안된다. 
#종류가 100개이면 최소 250개 배치사이즈가 되면 각각 2,3개 정도는 들어간다.  이미지는 배치사이즈가 클수록 좋고, 텍스트는 작을 수록 좋다. 텍스트는 정답 클래스가 많은 경우가 없다(긍부정등)
#이미지가 바이너리이면 작게 해줘도 되는데, 속도생각해서 빠르게 해주는게 좋다. 
#VAL 0.9 <VAL 0.1 속도 높다. 모바일넷은 엄청 깊진 않아도 쓸만함 (미국성 OR 영국성인지 성능이 엄청 잘 나오지 않아도 된다. 그러면 모바일넷쓰면 좋다.대회할 때는 다른것 써도 좋다)
#러닝레이트를 높여주니까 성능이 좋아짐
#EPOCH : 모든 데이터를 한번 쭉 학습할 때 (1번), 2만개 전체 데이터 보면서 학습하면 / 에폭 1면 = 나무 개수 = 학습 횟수 = 에폭  400 = 2만개의 데이터에서 배치 50개씩 하기위해서는 총 400번을 넣어줘야지 2만번 모두 학습함. 
#-> 가중치가 바뀌면서 로스가 낮아지려고 노력함 
#딥러닝 모델은 학습을 이어서 할 수 있다. 에폭을 더 높이면 된다. 컴파일, 모델 선언 하지 않고 피쳐엔지니어링만 실행시키면 된다. 
#VAL 가장 낮게 나오고, PATIENCE 5번 참아주고  

