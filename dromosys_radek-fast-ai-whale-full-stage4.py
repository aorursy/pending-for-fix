#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from skimage.util import montage
import pandas as pd
from torch import optim
import re

from utils import *


# In[ ]:


get_ipython().system('git clone https://github.com/radekosmulski/whale')


# In[ ]:


import sys
 # Add directory holding utility functions to path to allow importing utility funcitons
#sys.path.insert(0, '/kaggle/working/protein-atlas-fastai')
sys.path.append('/kaggle/working/whale')


# In[ ]:


from whale.utils import map5


# In[ ]:


import fastai
from fastprogress import force_console_behavior
import fastprogress
fastprogress.fastprogress.NO_BAR = True
master_bar, progress_bar = force_console_behavior()
fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


ls ../input


# In[ ]:


path = Path('../input/humpback-whale-identification/')
path_test = Path('../input/humpback-whale-identification/test')
path_train = Path('../input/humpback-whale-identification/train')


# In[ ]:


df = pd.read_csv(path/'train.csv')#.sample(frac=0.05)
df.head()
val_fns = {'69823499d.jpg'}


# In[ ]:


fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}
path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)


# In[ ]:


name = f'res50-full-train'


# In[ ]:


SZ = 224
BS = 64
NUM_WORKERS = 0
SEED=0


# In[ ]:


data = (
    ImageItemList
        .from_df(df[df.Id != 'new_whale'], '../input/humpback-whale-identification/train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder('../input/humpback-whale-identification/test'))
        .transform(get_transforms(do_flip=False), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='../input/humpback-whale-identification')
        .normalize(imagenet_stats)
)


# In[ ]:


MODEL_PATH = "/kaggle/working/"


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nlearn = create_cnn(data, models.resnet50, lin_ftrs=[2048], model_dir=MODEL_PATH)\nlearn.clip_grad();')


# In[ ]:


SZ = 224 * 2
BS = 64 // 4
NUM_WORKERS = 0
SEED=0


# In[ ]:


data = (
    ImageItemList
        .from_df(df[df.Id != 'new_whale'], '../input/humpback-whale-identification/train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder('../input/humpback-whale-identification/test'))
        .transform(get_transforms(do_flip=False), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='../input/humpback-whale-identification')
        .normalize(imagenet_stats)
)


# In[ ]:


get_ipython().system('cp ../input/radek-fast-ai-whale-full-stage3/res50-full-train-stage-3.pth /kaggle/working')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nlearn.load(f'{name}-stage-3')\n\nlearn.unfreeze()\n\nmax_lr = 1e-3 / 4\nlrs = [max_lr/100, max_lr/10, max_lr]\n\nlearn.fit_one_cycle(22, lrs)\nlearn.save(f'{name}-stage-4')")


# In[ ]:


get_ipython().system('rm -rf /kaggle/working/whale')


# In[ ]:





# In[ ]:



