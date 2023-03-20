#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




import os
import dicom
INPUT_FOLDER = '../input/sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()




patients




type(patients)




len(patients)




# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices




s = dicom.read_file('../input/sample_images'+ '/' +'00cba091fa4ad62cc3200a657aeb957e')




'../input/sample_images'+ '/' +'00cba091fa4ad62cc3200a657aeb957e'




for s in os.listdir(INPUT_FOLDER)
   print(s)




patients[0]




slices = [dicom.read_file(INPUT_FOLDER + patients[0] + '/' + s) for s in os.listdir(INPUT_FOLDER + patients[0])]




len(slices)




slices




type(slices[0])




slices[0]




slices[0].SliceLocation




slices[1].SliceLocation




slices[0].ImagePositionPatient[2]




slices[0].dir('setup')




slices[0].PatientName




type(slices[1].PixelData)




slices[0].PixelData




slices[0].pixel_data






