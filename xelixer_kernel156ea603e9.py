#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install spacy')
get_ipython().system('pip3 install nltk')


# In[2]:



from google.colab import files
files.upload()


# In[3]:


get_ipython().system('pip3 install kaggle')


# In[4]:


get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')


# In[5]:


get_ipython().system('kaggle competitions download -c quora-insincere-questions-classification')


# In[6]:


chmod 600 /root/.kaggle/kaggle.json


# In[7]:


zip=ZipFile('train.csv.zip', 'r')
zip.extractall()


# In[8]:


zip=ZipFile('test.csv.zip', 'r')
zip.extractall()


# In[9]:


import spacy
import nltk
nlp=spacy.load('en')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
from zipfile import ZipFile
import gensim
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[10]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[11]:


train.head()


# In[12]:


from sklearn.utils import resample

sincere = train[train.target == 0]
insincere = train[train.target == 1]

train = pd.concat([resample(sincere,
                     replace = False,
                     n_samples = len(insincere)), insincere])


# In[13]:


contractions = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(contractions.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return contractions[match.group(0)]
    return c_re.sub(replace, text)


# In[14]:


from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short

CUSTOM_FILTERS = [lambda x: x.lower(), #lowercase
                  strip_tags, # remove html tags
                  strip_punctuation, # replace punctuation with space
                  strip_multiple_whitespaces,# remove repeating whitespaces
                  strip_non_alphanum, # remove non-alphanumeric characters
                  strip_numeric, # remove numbers
                  remove_stopwords,# remove stopwords
                  strip_short # remove words less than minsize=3 characters long
                 ]

def gensim_preprocess(docs, logging=True):
    docs = [expandContractions(doc) for doc in docs]
    docs = [preprocess_string(text, CUSTOM_FILTERS) for text in docs]
    texts_out = []
    for doc in docs:
    # https://spacy.io/usage/processing-pipelines
        doc = nlp((" ".join(doc)),  # doc = text to tokenize => creates doc
                  # disable parts of the language processing pipeline we don't need here to speed up processing
                  disable=['ner', # named entity recognition
                           'tagger', # part-of-speech tagger
                           'textcat', # document label categorizer
                          ])
        texts_out.append([tok.lemma_ for tok in doc if tok.lemma_ != '-PRON-'])
    return pd.Series(texts_out)

gensim_preprocess(train.question_text.iloc[10:15])


# In[15]:


get_ipython().run_line_magic('time', 'train_corpus = gensim_preprocess(train.question_text)')


# In[16]:


# create ngrams
ngram_phraser = models.Phrases(train_corpus, threshold=1)
ngram = models.phrases.Phraser(ngram_phraser)
#print example
print(ngram[train_corpus[0]])

# apply model to corpus
texts = [ngram[token] for token in train_corpus]


# In[17]:


# preparing ngrams for modeling
texts = [' '.join(text) for text in texts]
train['ngrams'] = texts
train.head()


# In[18]:


# represent features as BOW
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(train.ngrams)

# split into test and train sets
X_train, X_test, y_train, y_test = train_test_split(train.ngrams, train.target, test_size=0.2)


# In[19]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(vectorizer.transform(X_train), y_train)

print('Logistic Regression Score: ', lr.score(vectorizer.transform(X_test), y_test))


# In[20]:


y_ = lr.predict(vectorizer.transform(X_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_))


# In[21]:


from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test, y_))


# In[22]:


from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()
get_ipython().run_line_magic('time', 'bnb.fit(vectorizer.transform(X_train), y_train)')

print('Naive Bayes Score: ', bnb.score(vectorizer.transform(X_test), y_test))


# In[23]:


get_ipython().run_line_magic('time', 'bnb_y_ = bnb.predict(vectorizer.transform(X_test))')

print(classification_report(y_test, bnb_y_))


# In[24]:


pd.DataFrame(confusion_matrix(y_test, bnb_y_))


# In[25]:


import xgboost as xgb

xgb_model = xgb.XGBClassifier().fit(vectorizer.transform(X_train), y_train)

print('XGBoost Score: ', xgb_model.score(vectorizer.transform(X_test), y_test))


# In[26]:


xgb_y_ = xgb_model.predict(vectorizer.transform(X_test))

print(classification_report(y_test, xgb_y_))


# In[27]:


pd.DataFrame(confusion_matrix(y_test, xgb_y_))


# In[28]:


from sklearn.ensemble import VotingClassifier

#create submodels
estimators = []

model1 = lr
model2 = bnb
model3 = xgb_model


estimators.append(('logistic', model1))
estimators.append(('bernoulli', model2))
estimators.append(('xgboost', model3))


# create ensemble model
get_ipython().run_line_magic('time', 'ensemble = VotingClassifier(estimators).fit(vectorizer.transform(X_train), y_train)')
print('Ensemble Score: ', ensemble.score(vectorizer.transform(X_test), y_test))


# In[29]:


ensemble_y_ = ensemble.predict(vectorizer.transform(X_test))

print(classification_report(y_test, ensemble_y_))


# In[30]:


pd.DataFrame(confusion_matrix(y_test, ensemble_y_))


# In[31]:


# preprocessing/lemmatizing/stemming test data
get_ipython().run_line_magic('time', 'test_corpus = gensim_preprocess(test.question_text)')
test_texts = [ngram[token] for token in test_corpus]

test_texts = [' '.join(text) for text in test_texts]
test['ngrams'] = test_texts
test.head()


# In[32]:


#ensemble on test data
ensemble.fit(vectorizer.transform(train.ngrams), train.target)
prediction = ensemble.predict(vectorizer.transform(test.ngrams))

submission = pd.DataFrame({'qid':test.qid, 'prediction':prediction})
submission.to_csv('submission.csv', index=False)
submission.head()

