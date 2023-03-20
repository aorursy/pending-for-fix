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




pip install -U git+https://github.com/lyft/nuscenes-devkit




get_ipython().system('pip install moviepy')
import pdb
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

# Load the SDK
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer, Quaternion, view_points
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

from moviepy.editor import ImageSequenceClip
from tqdm import tqdm_notebook as tqdm

get_ipython().run_line_magic('matplotlib', 'inline')





get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data data')




lyftdata = LyftDataset(data_path='.', json_path='data/', verbose=True)




lyftdata.list_scenes()





my_scene = lyftdata.scene[0]
my_scene




my_sample_token = my_scene["first_sample_token"]
# my_sample_token = level5data.get("sample", my_sample_token)["next"]  # proceed to next sample

lyftdata.render_sample(my_sample_token)




my_sample = lyftdata.get('sample', my_sample_token)
my_sample




lyftdata.list_sample(my_sample['token'])





lyftdata.render_pointcloud_in_image(sample_token = my_sample["token"],
                                      dot_size = 1,
                                      camera_channel = 'CAM_FRONT')




my_sample['data']





sensor_channel = 'CAM_FRONT'  # also try this e.g. with 'LIDAR_TOP'
my_sample_data = lyftdata.get('sample_data', my_sample['data'][sensor_channel])
my_sample_data




lyftdata.render_sample_data(my_sample_data['token'])




my_annotation_token = my_sample['anns'][16]
my_annotation =  my_sample_data.get('sample_annotation', my_annotation_token)
my_annotation




lyftdata.render_annotation(my_annotation_token)





my_instance = lyftdata.instance[100]
my_instance
print("First annotated sample of this instance:")
lyftdata.render_annotation(my_instance['first_annotation_token'])




print("Last annotated sample of this instance")
lyftdata.render_annotation(my_instance['last_annotation_token'])




lyftdata.list_categories()




lyftdata.category[2]

