#!/usr/bin/env python
# coding: utf-8



get_ipython().system('ls')




get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')




from fastai.conv_learner import *




PATH = '../input/planet-understanding-the-amazon-from-space/'




ls {PATH}




# Visualizing data:
from fastai.plots import *
list_paths = [f'{PATH}train-jpg/train_0.jpg', f'{PATH}train-jpg/train_1.jpg']
titles = ['haze primary', 'agriculture clear water primary']
plots_from_files(list_paths, titles=titles, maintitle='Multi-labeled classification')




# Helper function for f2 metric evaluation
import warnings
from sklearn.metrics import fbeta_score
def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                    for th in np.arange(start,end,step)])




metrics = [f2]
model = resnet34




# Perparing cross validation data:
label_csv = f'{PATH}train_v2.csv'
n = len(list(open(label_csv))) - 1
val_idxs = get_cv_idxs(n) # Automatically randomly selects 20% of validation indices
val_idxs.shape




def get_data(sz):
    # Allow for top down data augmentation here as images are satellite images
    tfms = tfms_from_model(model, sz, aug_tfms=transforms_top_down, max_zoom=1.1)
    return ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms, suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg-v2')




# Fetching data from above helper function:
data = get_data(256)
x, y = next(iter(data.val_dl))
y




# View first validation's classes and labels
list(zip(data.classes, y[0]))




sz = 64




data = get_data(sz)




#import pathlib
#data.path = pathlib.Path('.')
from os.path import expanduser, join, exists
from os import makedirs
cache_dir = expanduser(join('~', '.torch'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)




get_ipython().system('cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth')




import pathlib
data.path = pathlib.Path('.')
#data = data.resize(int(sz * 1.3), '.') # Tells to ignore training images more than sz * 1.3 to save time
learn = ConvLearner.pretrained(model, data, metrics=metrics)




# Using learning rate finder to find a good initial starting learning rate
lr = learn.lr_find()
learn.sched.plot()




# Surprisingly lr is too large for this dataset, anyways we will use 0.2 as loss is clearly decreasing
lr = 0.2
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)




# The next step is to unfreeze the layers and use differential learning rates for training:
lr_d = [lr/9, lr/3, lr] # The lr's are decreased by order of 3 per group towards initial layers of network
learn.unfreeze()




learn.fit(lr_d, 3, cycle_len=1, cycle_mult=2)




learn.sched.plot_loss()




# Training with images of size 128 now with layers of the model frozen and only the last f.c being trained
sz = 128
learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)




learn.unfreeze()
learn.fit(lr_d, 3, cycle_len=1, cycle_mult=2)
