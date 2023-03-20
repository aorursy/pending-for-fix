#!/usr/bin/env python
# coding: utf-8



print("This is a code cell.")
print("Hello world")
print(f"Hello world, time is {3} o'clock")




# Simple variable assignment
# This is a comment
# x variable like 3x = 5
x = 5/3




y = x*3
print(y)




# simple calculations
# ** means exponential
print(2**5)
print(2**3*3) # computer 2**3 first, 8 is multiplied with 3
print(1/2+1) # 1/2 first, +1 computed secondly




print(2***5) # unfortunately tetration is not built-in Python




# browse the Cats vs dog competition




# == vs =
a = 2 # we let a be 2




type(a)




dir(a)




# == is checking whether a variable is equal to whatever on the right side of ==
a == 3




s = 'David'
print(type(s))




False is not True




a = 5.
type(a)




a = 7
type(a)




s1 = [a, s]
print(s1)




type({2,3})




dict1 = {'Shuhao': 'instructor'}




# simple logical checking == or using "is"
dict1['Shuhao'] == 'student'




1 is 0




# simple if-then condition
# flow control

if 1 is 1:
    print("1 is 1")

if 1 is 0:
    print("1 is 0")




# range(a,b) the integer greater than or equal to a but less than b
for i in range(0,5):
    if i > 2: # press Tab to indent
        print(i)




import os
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import random




dir(pandas)




dir(pd)




import site
site.getsitepackages()




help(pd)




print(os.listdir("../input/dogs-vs-cats/"))




get_ipython().system("unzip -q '../input/dogs-vs-cats/train.zip'")
get_ipython().system("unzip -q '../input/dogs-vs-cats/test1.zip'")




filenames = os.listdir("./train")
print(filenames[:10])




categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(0)
    else:
        categories.append(1)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
print(df.head(20))




len(filenames)




from keras.preprocessing.image import load_img




sample = random.choice(filenames)
image = load_img("./train/"+sample)
fig = plt.figure()
fig.set_size_inches(6,6)
plt.imshow(image);




from keras import layers, applications, optimizers, callbacks
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Model, load_model
from keras.utils import plot_model, to_categorical

image_size = 224
input_shape = (image_size, image_size, 3)

epochs = 6
batch_size = 16

pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
    
for layer in pre_trained_model.layers[:15]:
    layer.trainable = False

for layer in pre_trained_model.layers[15:]:
    layer.trainable = True
    
last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output
    
# Flatten the output layer to 1 dimension
x = GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)
# Add a dropout rate of 0.3
x = Dropout(0.3)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

model.summary()




plot_model(model, to_file='/model_vgg16.png', show_shapes=True)




df['category'] = df['category'].astype('str')




train_df, validate_df = train_test_split(df, test_size=0.1)
train_df = train_df.reset_index()
validate_df = validate_df.reset_index()

# validate_df = validate_df.sample(n=100).reset_index() # use for fast testing code purpose
# train_df = train_df.sample(n=1800).reset_index() # use for fast testing code purpose

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]




train_datagen = ImageDataGenerator(
    rotation_range=16,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "./train/", 
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(image_size, image_size),
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "./train/",  
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(image_size, image_size),
    batch_size=batch_size
)




example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "./train/", 
    x_col='filename',
    y_col='category',
#     class_mode='binary'
)
plt.figure(figsize=(12, 12))
for i in range(0, 9):
    plt.subplot(3, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()




test_filenames = os.listdir("./test1/")
test_df = pd.DataFrame({
    'filename': test_filenames[:128]
})

nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "./test1/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    batch_size=batch_size,
    target_size=(image_size, image_size),
    shuffle=False
)




# this may take a while
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
threshold = 0.5
test_df['category'] = np.where(predict > threshold, 1,0)




sample_test = test_df.sample(n=9).reset_index()
sample_test.head()
plt.figure(figsize=(12, 12))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("./test1/"+filename, target_size=(256, 256))
    plt.subplot(3, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()




model = load_model('../input/vgg16catsvsdogs/model_0_vgg16.h5')
model.summary()




# this may take a while
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
threshold = 0.5
test_df['category'] = np.where(predict > threshold, 1,0)




sample_test = test_df.sample(n=9).reset_index()
sample_test.head()
plt.figure(figsize=(12, 12))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("./test1/"+filename, target_size=(256, 256))
    plt.subplot(3, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()




# example of scalar, vector, matrix
s = 1.34992




v = [1, 3, 5.2]
print(v)




m = [[1,3], [8,9]]
print(m)




m = np.array(m)
print(type(m))
print(m)




# example of imshow
a = np.array([[0,4], [2,10]])
plt.imshow(a);




pokemon_filename = os.listdir("../input/pokemon-images-dataset/pokemon_jpg/pokemon_jpg/")




random_pokemon = random.choice(pokemon_filename)
G = plt.imread("../input/pokemon-images-dataset/pokemon_jpg/pokemon_jpg/"+random_pokemon)
plt.imshow(G)




G




m.shape




G.shape # 3 represents the Red, Green, Blue color 




# indexing
G1 = G[:,:,0] # take the red channel




G1.shape




# show only 1 color channel
plt.imshow(G1)




pkmn = pd.read_csv('../input/pokemon/Pokemon.csv')




pkmn.sample(20)




sns.countplot(pkmn['Generation']) # sns means seaborn




pkmn = pkmn.drop(['Generation', 'Legendary'],1)




sns.jointplot(x="HP", y="Speed", data=pkmn);




sns.jointplot(x="Attack", y="Defense", data=pkmn, kind='hex');

