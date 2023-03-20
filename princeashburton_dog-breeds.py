#!/usr/bin/env python
# coding: utf-8



###  majic calls
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')




# Fastai's Library
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
from fastai.imports import *




torch.cuda.set_device(1)




PATH = "../input/"
sz=224
arch=resnext101_64
bs=58




label_csv = f'{PATH}labels.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)




n




len(val_idxs)




get_ipython().system('ls {PATH}')




label_df = pd.read_csv(label_csv)




label_df.head()




label_df.info()




label_df.pivot_table(index='breed', aggfunc=len).sort_values('id',ascending=False)




tfms = tfms_from_model(arch,sz,aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv',test_name='test',
                                     val_idxs=val_idxs, suffix='.jpg', tfms=tfms,bs=bs)




fn = PATH+data.trn_ds.fnames[0]; fn




Here we need to find out how big the sizes are for the rows and columns of the images




fn = PATH+data.trn_ds.fnames[0]; fn




img = PIL.Image.open(fn); img




img.size




size_d = {k: PIL.Image.open(PATH+k).size for k in data.trn_ds.fnames}




row_sz,col_sz = list(zip(*size_d.values()))




row_sz=np.array(row_sz); col_sz=np.array(col_sz)
row_sz[:5]




plt.hist(row_sz)




plt.hist(row_sz[row_sz<1000])




len(data.trn_ds), len(data.test_ds)




len(data.classes), data.classes[:5]









def get_data(sz,bs):
    tfms = tfms_from_model(arch,sz,aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv',test_name='test', num_workers=4,
                                        val_idxs=val_idxs, suffix='.jpg', tfms=tfms,bs=bs)
 




data = get_data(sz,bs)




learn = ConvLearner.pretrained(arch, data, precompute=True) 




learn.fit(1e-2,5)




from sklearn import metric




data = get_data(sz,bs)




learn = ConvLearner.pretrained(arch, data, precompute=True,ps=0.5) 




learn.fit(1e-2,2)




learn.precompute=False




learn.fit(1e-2,5,cycle_len=1)




learn.save('224_pre')




learn.load('224_pre')




learn.set_data(get_data(299,bs))
learn.freeze()




learn.fit(1e-2,3,cycle_len=1)




learn.fit(1e-2,3,cycle_len=1,cycle_mult=2)




log_preds,y = learn.TTA()
probs = np.exp(log_preds)
accuracy(log_preds,y), metrics.log_loss(y,probs)




learn.save('299_pre')




learn.load('299_pre')




learn.fit(1e-2,3,cycle_len=1)




learn.save('299_pre')




log_preds,y = learn.TTA()
probs = np.exp(log_preds)
accuracy(log_preds,y), metrics.log_loss(y,probs)











