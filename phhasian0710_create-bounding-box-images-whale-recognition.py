#!/usr/bin/env python
# coding: utf-8



import os

import PIL
from PIL import Image
from PIL.ImageDraw import Draw
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import load_model
from keras.preprocessing import image




ls ../input




MODEL_BASE = '../input/bbox-model-whale-recognition'
DATA = '../input/humpback-whale-identification'
TRAIN_DATA = os.path.join(DATA, 'train')
TEST_DATA = os.path.join(DATA, 'test')




model = load_model(os.path.join(MODEL_BASE, 'cropping.model'))




# # input: (128, 128, 1)
# model.summary()




train_paths = [os.path.join(TRAIN_DATA, img) for img in os.listdir(TRAIN_DATA)]
test_paths = [os.path.join(TEST_DATA, img) for img in os.listdir(TEST_DATA)]




train_paths[0]




img = image.load_img(train_paths[10])




img




img_arr = image.img_to_array(img)




img_arr.shape




rimg = img.resize((128, 128), PIL.Image.ANTIALIAS)




rimg




rimg_arr = image.img_to_array(rimg)




rimg_ = rimg.convert('L')




rimg_arr_ = image.img_to_array(rimg_)




rimg_arr_.shape




bbox = model.predict(np.expand_dims(rimg_arr_, axis=0))




bbox




draw = Draw(rimg_)




draw.rectangle(bbox, outline='red')




rimg_




rimg




img_crop = rimg_.crop(tuple(bbox[0]))




img_crop




def make_bbox_image(img_path):
    """
    :param img: path to image
    """
    main_img = image.load_img(img_path)
    r_img = main_img.resize((128, 128), PIL.Image.ANTIALIAS)
    # convert to 1d image
    rb_img = r_img.convert('L')
    rb_img_arr = image.img_to_array(rb_img)
    bbox = model.predict(np.expand_dims(rb_img_arr, axis=0))
    
    # draw rectangle
    # draw = Draw(rimg)
    # draw.rectangle(bbox, outline='red')
    
    img_crop = r_img.crop(tuple(bbox[0]))
    img_arr = image.img_to_array(img_crop)
    return img_crop




train_paths[10]




img = make_bbox_image(train_paths[10])




plt.imshow(img)






