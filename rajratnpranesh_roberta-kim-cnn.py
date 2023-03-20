#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers
print('TF version',tf.__version__)


# In[2]:


def read_train():
    train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
    train['text']=train['text'].astype(str)
    train['selected_text']=train['selected_text'].astype(str)
    return train

def read_test():
    test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
    test['text']=test['text'].astype(str)
    return test

def read_submission():
    test=pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
    return test
    
train_df = read_train()
test_df = read_test()
submission_df = read_submission()


# In[3]:


train_df


# In[4]:


train_list = train_df.to_dict('records')


# In[5]:


MAX_SEQUENCE_LENGTH = 0

for image_data in train_list:
  MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH, len(image_data['text'].split(' ')))

print(MAX_SEQUENCE_LENGTH)


# In[6]:


MAX_SEQUENCE_LENGTH = 0

for image_data in train_list:
  MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH, len(image_data['selected_text'].split(' ')))

print(MAX_SEQUENCE_LENGTH)


# In[ ]:





# In[ ]:





# In[7]:


def jaccard(str1, str2): 
    a = set(str(str1).lower().split()) 
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[8]:


MAX_LEN = 96
PATH = '../input/tf-roberta/'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=PATH+'vocab-roberta-base.json', 
    merges_file=PATH+'merges-roberta-base.txt', 
    lowercase=True,
    add_prefix_space=True
)
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}


# In[9]:


ct = train_df.shape[0]
input_ids = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(train_df.shape[0]):
    
    # FIND OVERLAP
    text1 = " "+" ".join(train_df.loc[k,'text'].split())
    text2 = " ".join(train_df.loc[k,'selected_text'].split())
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx+len(text2)]=1
    if text1[idx-1]==' ': chars[idx-1] = 1 
    enc = tokenizer.encode(text1) 
        
    # ID_OFFSETS
    offsets = []; idx=0
    for t in enc.ids:
        w = tokenizer.decode([t])
        offsets.append((idx,idx+len(w)))
        idx += len(w)
    
    # START END TOKENS
    toks = []
    for i,(a,b) in enumerate(offsets):
        sm = np.sum(chars[a:b])
        if sm>0: toks.append(i) 
        
    s_tok = sentiment_id[train_df.loc[k,'sentiment']]
    input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask[k,:len(enc.ids)+5] = 1
    if len(toks)>0:
        start_tokens[k,toks[0]+1] = 1
        end_tokens[k,toks[-1]+1] = 1


# In[10]:


ct = test_df.shape[0]
input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(test_df.shape[0]):
        
    # INPUT_IDS
    text1 = " "+" ".join(test_df.loc[k,'text'].split())
    enc = tokenizer.encode(text1)                
    s_tok = sentiment_id[test_df.loc[k,'sentiment']]
    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask_t[k,:len(enc.ids)+5] = 1


# In[11]:


def scheduler(epoch):
    return 3e-5 * 0.2**epoch


# In[12]:


from keras import regularizers


# In[13]:



def build_model():
    
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
    x = bert_model(ids,attention_mask=att,token_type_ids=tok)
    
    
    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x1 = tf.keras.layers.Conv1D(128, 2,padding='same',activation='relu', kernel_regularizer=regularizers.l2(0.01))(x1)
    #x1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid')(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Conv1D(64, 2,padding='same',activation='relu', kernel_regularizer=regularizers.l2(0.01))(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Conv1D(32, 2,padding='same',activation='relu', kernel_regularizer=regularizers.l2(0.01))(x1)
    #x1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid')(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dropout(0.1)(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x2 = tf.keras.layers.Conv1D(128, 2,padding='same',activation='relu', kernel_regularizer=regularizers.l2(0.01))(x2)
    #x2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Conv1D(64, 2,padding='same',activation='relu', kernel_regularizer=regularizers.l2(0.01))(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Conv1D(32, 2,padding='same',activation='relu', kernel_regularizer=regularizers.l2(0.01))(x2)
    #x2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid')(x2)
    x2 = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Dropout(0.1)(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)  
    

    return model


# In[ ]:





# In[14]:


model.summary()


# In[15]:


n_splits = 5


# In[16]:


''''
jac = []; VER='v7'; DISPLAY=1 # USE display=1 FOR INTERACTIVE
oof_start = np.zeros((input_ids.shape[0],MAX_LEN))
oof_end = np.zeros((input_ids.shape[0],MAX_LEN))

skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=777)
for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train_df.sentiment.values)):

    print('#'*32)
    print('### FOLD %i'%(fold+1))
    print('#'*32)
    
    K.clear_session()
    model = build_model()
        
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    sv = tf.keras.callbacks.ModelCheckpoint(
        '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch')
        
    hist = model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]], 
        epochs=5, batch_size=8, verbose=DISPLAY, callbacks=[sv, reduce_lr],
        validation_data=([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]], 
        [start_tokens[idxV,], end_tokens[idxV,]]))
    
    print('Loading model...')
    model.load_weights('%s-roberta-%i.h5'%(VER,fold))
    
    print('Predicting OOF...')
    oof_start[idxV,],oof_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)
    
    # DISPLAY FOLD JACCARD
    all = []
    for k in idxV:
        a = np.argmax(oof_start[k,])
        b = np.argmax(oof_end[k,])
        if a>b: 
            st = train_df.loc[k,'text'] # IMPROVE CV/LB with better choice here
        else:
            text1 = " "+" ".join(train_df.loc[k,'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a-1:b])
        all.append(jaccard(st,train_df.loc[k,'selected_text']))
    jac.append(np.mean(all))
    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
    print()
''''


# In[17]:



preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))
DISPLAY=1
for i in range(5):
    print('#'*25)
    print('### MODEL %i'%(i+1))
    print('#'*25)
    
    K.clear_session()
    model = build_model()
    model.load_weights('../input/roberta-kim-cnn/v7-roberta-%i.h5'%i)

    print('Predicting Test...')
    preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
    preds_start += preds[0]/n_splits
    preds_end += preds[1]/n_splits


# In[18]:



all = []
for k in range(input_ids_t.shape[0]):
    a = np.argmax(preds_start[k,])
    b = np.argmax(preds_end[k,])
    if a>b: 
        st = test_df.loc[k,'text']
    else:
        text1 = " "+" ".join(test_df.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc.ids[a-1:b])
    all.append(st)


# In[19]:



test_df['selected_text'] = all
test_df[['textID','selected_text']].to_csv('submission.csv',index=False)


# In[ ]:




