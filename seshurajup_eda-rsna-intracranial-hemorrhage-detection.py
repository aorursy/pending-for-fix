#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import pandas as pd
import os
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import gc
import warnings
import pydicom
import cv2
from tqdm import tqdm
warnings.simplefilter(action = 'ignore')




get_ipython().system('ls ../input/*')




ls -R ../input/rsnasample/*




train_labels = pd.read_csv('../input/rsnasample/stage_1_train.csv')
train_labels.head()




train_labels = train_labels.drop_duplicates()
train_labels.info()




train_labels['ID'].value_counts(sort=True).head(10)




train_labels['Label'].plot.hist()




pd.DataFrame(train_labels.groupby('Label')['ID'].count())




plt.style.use('ggplot')
plot = train_labels.groupby('Label')     .count()['ID']     .plot(kind='bar', figsize=(10,4), rot=0)




def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print()
    
    print("Patient id..........:", dataset.PatientID )
    print("Patient's Age.......:", dataset.SOPInstanceUID )
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)
            
def plot_pixel_array(dataset, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()




get_ipython().system('ls ../input/rsnasample/stage_1_train_images/stage_1_train_images')




root_path = '../input/rsnasample/stage_1_train_images/stage_1_train_images/'
for r, d, files in os.walk(root_path):
    for dcm_file in files:
        file_path = os.path.join(root_path, dcm_file)
        dataset = pydicom.dcmread(file_path)
        print(dataset)




for r, d, files in os.walk(root_path):
    for dcm_file in files:
        file_path = os.path.join(root_path, dcm_file)
        dataset = pydicom.dcmread(file_path)
        show_dcm_info(dataset)
        plot_pixel_array(dataset)

