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




import nltk
import string
import re
import numpy as np
import pandas as pd
import pickle
#import lda

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words

from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
get_ipython().run_line_magic('matplotlib', 'inline')

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook
#from bokeh.transform import factor_cmap

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("lda").setLevel(logging.WARNING)




import os
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("lda").setLevel(logging.WARNING)

print(os.listdir("../input"))




pwd




ls ../input/mercari-price-suggestion-challenge/




import pandas as pd

train = pd.read_csv(f'../input/mercari/train.tsv', sep='\t') #sep='\t'でタブ区切り
test = pd.read_csv(f'../input/mercari/test.tsv', sep='\t')




print(train.shape)
print(test.shape)




train.info()




train.head()




train.dtypes




train.price.describe() #eの横の数字だけ10倍




import matplotlib.pyplot as plt

train['price'].hist(bins = 30, range=(0, 250))
plt.xlabel('price')
plt.ylabel('count')




#自然対数
import matplotlib.pyplot as plt
import numpy as np

(train['price'] + 1).apply(np.log).hist(bins = 30, range=(0, 7))
plt.xlabel('log(price + 1)')
plt.ylabel('count')
plt.xlim(1, 7)




plt.hist(train.loc[train['shipping'] == 0, 'price'].dropna(),
        range=(0, 250), bins=30, alpha=0.5, label='0')
plt.hist(train.loc[train['shipping'] == 1, 'price'].dropna(),
        range=(0, 250), bins=30, alpha=0.5, label='1')
plt.xlabel('price')
plt.ylabel('count')
plt.legend(title='shipping')




print(train.loc[train['shipping'] == 0, 'price'].mean()) #送料購入者
print(train.loc[train['shipping'] == 1, 'price'].mean()) #送料出品者




#自然対数
plt.hist((train.loc[train['shipping'] == 0, 'price'] + 1).apply(np.log).dropna(),
        range=(1, 7), bins=30, alpha=0.5, label='0')
plt.hist((train.loc[train['shipping'] == 1, 'price'] + 1).apply(np.log).dropna(),
        range=(1, 7), bins=30, alpha=0.5, label='1')
plt.xlabel('log(price + 1)')
plt.ylabel('count')
plt.legend(title='shipping')




print((train.loc[train['shipping'] == 0, 'price'] + 1).apply(np.log).mean()) #送料購入者
print((train.loc[train['shipping'] == 1, 'price'] + 1).apply(np.log).mean()) #送料出品者




#The number of category_name
train['category_name'].nunique()




#top 5
train['category_name'].value_counts()[:5]




#count null.
train['category_name'].isnull().sum()




#survied details of categories
def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No data", "No data", "No data")




#edit category_name
train['general_cat'], train['subcat1'], train['subcat2'] = zip(*train['category_name'].apply(lambda x : split_cat(x)))
train.head()




#repeat the same step for the test set
test['general_cat'], test['subcat1'], test['subcat2'] = zip(*test['category_name'].apply(lambda x : split_cat(x)))
test.head()




#The number of subcat1
train['subcat1'].nunique()




train['general_cat'].value_counts()




train['subcat1'].value_counts()[:5]




#The number of subcat2
train['subcat2'].nunique()




train['subcat2'].value_counts()[:5]




#valuesにすることで配列に変換
x = train['general_cat'].value_counts().index.values.astype(str) #大カテゴリーの配列
y = train['general_cat'].value_counts().values #大カテゴリー別件数の配列
unique_pct = [("%.2f"%(v*100))+"%"for v in (y/len(train))] #各general_catラベルの出現率




import plotly.offline as py
import plotly.graph_objs as go

trace1 = go.Bar(x=x, y=y, text=unique_pct)
layout = dict(title = 'Number of Items by general_cat',
             yaxis = dict(title = 'Count'),
             xaxis = dict(title = 'general_cat'))
fig = dict(data=[trace1], layout=layout)
py.iplot(fig)




#上から15件までのサブカテゴリ1を同様に出現率計算
x = train['subcat1'].value_counts().index.values.astype('str')[:15]
y = train['subcat1'].value_counts().values[:15]

subcat1_pct = [("%.2f"%(v*100))+"%" for v in (y/(len(train)))]




trace1 = go.Bar(x=x, y=y, text=subcat1_pct,
               marker=dict(
               color = y,colorscale='Portland', showscale=True,
               reversescale = False
               ))
layout = dict(title='Number of Items by subcat1(~15)',
             yaxis = dict(title='Count'),
             xaxis = dict(title='subcat1'))
fig = dict(data=[trace1], layout=layout)
py.iplot(fig)




#とりあえず箱ひげ図作る
general_cats = train['general_cat'].unique()
general_cats




#大カテゴリー別の値段をリストでまとめる(これが横軸)
x = [train.loc[train['general_cat'] == cat, 'price'] for cat in general_cats] 




#priceはlog(price + 1)で正規化する
data = [go.Box(x=np.log(x[i] + 1), name=general_cats[i]) for i in range(len(general_cats))]




layout = dict(title='Price distribution by general_cat',
             yaxis = dict(title='Category'),
             xaxis = dict(title='log(price + 1)'))
fig = dict(data=data, layout=layout)
py.iplot(fig)




train['brand_name'].nunique()




x = train['brand_name'].value_counts().index.values.astype('str')[:10]
y = train['brand_name'].value_counts().values[:10]




trace1 = go.Bar(x=x, y=y, 
                 marker=dict(
                 color = y,colorscale='Portland',showscale=True,
                 reversescale = False
                 ))
layout = dict(title= 'Top 10 Brand by Number of Items',
               yaxis = dict(title='Brand Name'),
               xaxis = dict(title='Count'))
fig=dict(data=[trace1], layout=layout)
py.iplot(fig)




import re
import string
from sklearn.feature_extraction import stop_words

#item_decriptionに含まれる単語長を調べる(字句解析は後ほど)
def wordCount(text):
    #小文字に変換して正規表現を取り除く
    #try:
        text = str(text).lower()
        #正規表現パターン文字列をコンパイルして正規表現パターンオブジェクトを作成
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]') #[!"#$%&'()*+,-./:;<=>?@[]^_`{|}~0-9\\r\\t\\n]
        #text内の特殊文字を半角空白で置換
        txt = regex.sub(" ", text)
        #tokenize
        #remove words in stop words
        words = [w for w in txt.split(" ")                 if not w in stop_words.ENGLISH_STOP_WORDS and len(w)>3] #内包表記(処理, ループ, 条件)
        #wordsには文字数3以上かつstop_words以外がリストとして格納されている
        return len(words)
    #except:
        #return 0




#ストップワードを取り除いた後の単語の総数をカウントした列を追加
train['desc_len'] = train['item_description'].apply(lambda x : wordCount(x))
test['desc_len'] = test['item_description'].apply(lambda x : wordCount(x))




train.head()




test.head()




df = train.groupby('desc_len')['price'].mean().reset_index()
df




trace1 = go.Scatter(
    x = df['desc_len'],
    y = np.log(df['price']), #dfで平均価格0のものが無いため,log(price)で良い(真数を+1しなくていい)
    mode = 'lines+markers',
    name = 'lines+markers'
)
layout = dict(title= 'Average log(price) by description length',
             yaxis = dict(title='Average log(price)'),
             xaxis = dict(title='Description length'))
fig = dict(data=[trace1], layout=layout)
py.iplot(fig)




#item_description内の欠損値を除く
train = train[pd.notnull(train['item_description'])]




from nltk.corpus import stopwords

stopwords.words('english')




#(準備)リストの演算
a = []
a += 't'
a += 'e'
a += 's'
a += 't'
a




from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from janome.tokenizer import Tokenizer

stop = set(stopwords.words('english'))

#tokenizerとは字句解析の意味で,文(item_description)を意味のあるトークン単位に分割する.
def tokenize(text):
    """
    sent_tokenize(): segment text into sentences
    word_tokenize(): break sentences into words
    """
    try:
        #正規表現パターン文字列をコンパイルして正規表現パターンオブジェクトを作成
        #[!"#$%&'()*+,-./:;<=>?@[]^_`{|}~0-9\\r\\t\\n]
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]') 
        #text内の特殊文字を半角空白で置換
        txt = regex.sub(" ", text)
        
        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        #filter(lambda式, イテラブルオブジェクト),イテラブルオブジェクトでlamda式を満たすものだけを抽出
        tokens = list(filter(lambda t : t.lower() not in stop, tokens))
        #数字などを含むトークンを除く
        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]
        
        return filtered_tokens 
        
    except TypeError as e:
        print(text, e)




"""
from collections import Counter


#item_descriptionをtokenizeしてみる(実行に時間かかる)


#単語辞書の作成
cat_dicts = dict()
for cat in general_cats:
    #general_catごとにitem_descriptionの文字列を抽出
    text = " ".join(train.loc[train['general_cat']==cat, 'item_description'].values)
    #sentenceからtokenに変換
    #{大カテゴリー:tokenオブジェクト化したitemdescription文字列}
    cat_dicts[cat] = tokenize(text)

    
# flat list of all words combined
flat_list = [item for sublist in list(cat_dicts.values()) for item in sublist]
#Counter()は出現回数が多い順に要素を取得
allWordCount = Counter(flat_list)
#上位20位まで取得
all_top10 = allWordCount.most_common(20)
x = [w[0] for w in all_top10]
y = [w[1] for w in all_top10]
"""




"""
trace1 = go.Bar(x=x, y=y, text=subcat1_pct)
layout = dict(title= 'Word Frequency',
              yaxis = dict(title='Count'),
              xaxis = dict(title='Word'))
fig=dict(data=[trace1], layout=layout)
py.iplot(fig)
"""




# apply the tokenizer into the item descriptipn column
train['token'] = train['item_description'].map(tokenize)
test['token'] = test['item_description'].map(tokenize)




#index再振り分け
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)




train.head()




test.head()




#一旦整形データ保存
#train.to_csv('re_train.csv', sep='\t')
#test.to_csv('re_test.csv', sep='\t')




#train = pd.read_csv('re_train.csv', sep='\t')
#test = pd.read_csv('re_test.csv', sep='\t')




for descriptions, tokens in zip(train['item_description'].head(), train['token'].head()):
    print('description:', descriptions)
    print('token:', tokens)
    print()




from collections import Counter


#item_descriptionをtokenizeしてみる(実行に時間かかる)


#単語辞書の作成
cat_dicts = dict()
for cat in general_cats:
    #general_catごとにitem_descriptionの文字列を抽出
    text = " ".join(train.loc[train['general_cat']==cat, 'item_description'].values)
    #sentenceからtokenに変換
    #{大カテゴリー:tokenオブジェクト化したitemdescription文字列}
    cat_dicts[cat] = tokenize(text)

#上位4位までのカテゴリーで最も出現回数の多い単語を見つける
women100 = Counter(cat_dicts['Women']).most_common(100)
beauty100 = Counter(cat_dicts['Beauty']).most_common(100)
kids100 = Counter(cat_dicts['Kids']).most_common(100)
electronics100 = Counter(cat_dicts['Electronics']).most_common(100)




from wordcloud import WordCloud

#wordcloudを生成する
def generate_wordcloud(tup):
    wordcloud = WordCloud(background_color = 'white',
                         max_words=50, max_font_size=40,
                         random_state=42
                         ).generate(str(tup))
    return wordcloud




fig, axes = plt.subplots(2, 2, figsize=(30, 15))

ax = axes[0, 0]
ax.imshow(generate_wordcloud(women100), interpolation='bilinear')
ax.axis('off')
ax.set_title('Women Top 100', fontsize=30)

ax = axes[0, 1]
ax.imshow(generate_wordcloud(beauty100))
ax.axis('off')
ax.set_title('Beauty Top 100', fontsize=30)

ax = axes[1, 0]
ax.imshow(generate_wordcloud(kids100))
ax.axis('off')
ax.set_title('Kids Top 100', fontsize=30)

ax = axes[1, 1]
ax.imshow(generate_wordcloud(electronics100))
ax.axis('off')
ax.set_title('Electronics Top 100', fontsize=30)




#インスタンス化
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10,
                            max_features=180000,
                            tokenizer=tokenize,
                            ngram_range=(1, 2))




all_desc = np.append(train['item_description'].values, test['item_description'].values)
#TFidfVectorizerクラスのfit_transformメソッドで単語をベクトル化
vz = vectorizer.fit_transform(list(all_desc))




#  create a dictionary mapping the tokens to their tfidf values
#{単語名:idf値}のdict生成
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_)) 
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
                    dict(tfidf), orient='index')
tfidf.columns = ['tfidf']




tfidf.sort_values(by=['tfidf'], ascending=True).head(10)




tfidf.sort_values(by=['tfidf'], ascending=False).head(10)




#trainとtestデータをコピー
trn = train.copy()
tst = test.copy()

#trainとtestでラベル分け
trn['is_train'] = 1
tst['is_test'] = 0

sample_sz = 15000
combined_df = pd.concat([trn, tst])
#ランダムに行を抽出
combined_sample = combined_df.sample(n=sample_sz)
vz_sample = vectorizer.fit_transform(list(combined_sample['item_description']))




from sklearn.decomposition import TruncatedSVD

n_comp=30
svd = TruncatedSVD(n_components=n_comp, random_state=42)
svd_tfidf = svd.fit_transform(vz_sample)




from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=42, n_iter=500)




tsne_tfidf = tsne_model.fit_transform(svd_tfidf)




tsne_tfidf




#可視化ライブラリbokeh
import bokeh.plotting as bp
from bokeh.plotting import show, figure, output_notebook

output_notebook() #notebook出力にはこの1行が必要
#オブジェクトの作成
plot_tfidf = bp.figure(plot_width=700, plot_height=600,
                      title='tfidf clustering of the item_description',
                      tools="pan,wheel_zoom,box_zoom,reset,hover",
                      x_axis_type=None, y_axis_type=None, min_border=1)




combined_sample.reset_index(inplace=True, drop=True)




tfidf_df = pd.DataFrame(tsne_tfidf, columns=['x', 'y']) #2次元データtnse_tfidfをx, yコラムに置換
tfidf_df['description'] = combined_sample['item_description']
tfidf_df['tokens'] = combined_sample['token']
tfidf_df['category'] = combined_sample['general_cat']




plot_tfidf.scatter(x='x', y='y', source=tfidf_df, alpha=0.7)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips = {"description": "@description", "tokens": "@tokens", "category": "@category"}
show(plot_tfidf)




from sklearn.cluster import MiniBatchKMeans

num_clusters = 30 #広くとった
#インスタンス化
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters,
                              init='k-means++',
                              n_init=1,
                              init_size=1000, batch_size=1000, verbose=0, max_iter=1000)




kmeans = kmeans_model.fit(vz) #Compute the centroids on X by chunking it into mini-batches.
kmeans_clusters = kmeans.predict(vz) #Compute cluster centers and predict cluster index for each sample.
kmeans_distance = kmeans.transform(vz) #Compute clustering and transform X to cluster-distance space.




kmeans




len(kmeans_clusters)




kmeans_distance




sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
len(sorted_centroids)




terms = vectorizer.get_feature_names()

for i in range(num_clusters):
    print("Cluster %d:" % i)
    aux = ''
    for j in range(i, 30):
        aux += terms[j] + ' | '
    print(aux)
    print() 




#repeat the same step for the sample
kmeans = kmeans_model.fit(vz_sample) #Compute the centroids on X by chunking it into mini-batches.
kmeans_clusters = kmeans.predict(vz_sample) #Compute cluster centers and predict cluster index for each sample.
kmeans_distance = kmeans.transform(vz_sample) #Compute clustering and transform X to cluster-distance space.
tsne_kmeans = tsne_model.fit_transform(kmeans_distance)




colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
"#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
"#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",
"#52697d", "#194196", "#d27c88", "#36422b", "#b68f79"])




kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y']) #2次元データtnse_kmeansをx, yコラムに置換
kmeans_df['cluster'] = kmeans_clusters
kmeans_df['description'] = combined_sample['item_description']
kmeans_df['category'] = combined_sample['general_cat']




output_notebook() #notebook出力にはこの1行が必要
#オブジェクトの作成
plot_kmeans = bp.figure(plot_width=700, plot_height=600,
                      title='KMeans clustering of the item_description',
                      tools="pan,wheel_zoom,box_zoom,reset,hover",
                      x_axis_type=None, y_axis_type=None, min_border=1)




#クラスターごとの色分けはsourceで設定した
source = ColumnDataSource(data=dict(x=kmeans_df['x'], y=kmeans_df['y'],
                                   color=colormap[kmeans_clusters],
                                   description=kmeans_df['description'],
                                   category=kmeans_df['category'],
                                   cluster=kmeans_df['cluster']))


plot_kmeans.scatter(x='x', y='y', color='color', source=source)
hover = plot_kmeans.select(dict(type=HoverTool))
hover.tooltips = {"cluster": "@cluster", "description": "@description", "category": "@category"}
show(plot_kmeans)




from sklearn.feature_extraction.text import CountVectorizer
cvectorizer = CountVectorizer(min_df=4,
                             max_features=180000,
                             tokenizer=tokenize,
                             ngram_range=(1, 2))




cvz = cvectorizer.fit_transform(combined_sample['item_description'])




from sklearn.decomposition import LatentDirichletAllocation

lda_model = LatentDirichletAllocation(n_components=20,
                                     learning_method='online',
                                     max_iter=20,
                                     random_state=42)




X_topics = lda_model.fit_transform(cvz)




n_top_words = 10
topic_summaries = []

topic_word = lda_model.components_  # get the topic words
vocab = cvectorizer.get_feature_names()

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i, ' | '.join(topic_words)))




# reduce dimension to 2 using tsne
tsne_lda = tsne_model.fit_transform(X_topics)




unnormalized = np.matrix(X_topics)
doc_topic = unnormalized/unnormalized.sum(axis=1)

lda_keys = []
for i, tweet in enumerate(combined_sample['item_description']):
    lda_keys += [doc_topic[i].argmax()]

lda_df = pd.DataFrame(tsne_lda, columns=['x','y'])
lda_df['description'] = combined_sample['item_description']
lda_df['category'] = combined_sample['general_cat']
lda_df['topic'] = lda_keys
lda_df['topic'] = lda_df['topic'].map(int)




#可視化
plot_lda = bp.figure(plot_width=700,
                     plot_height=600,
                     title="LDA topic visualization",
    tools="pan,wheel_zoom,box_zoom,reset,hover",
    x_axis_type=None, y_axis_type=None, min_border=1)




source = ColumnDataSource(data=dict(x=lda_df['x'], y=lda_df['y'],
                                    color=colormap[lda_keys],
                                    description=lda_df['description'],
                                    topic=lda_df['topic'],
                                    category=lda_df['category']))

plot_lda.scatter(source=source, x='x', y='y', color='color')
hover = plot_kmeans.select(dict(type=HoverTool))
hover = plot_lda.select(dict(type=HoverTool))
hover.tooltips={"description":"@description",
                "topic":"@topic", "category":"@category"}
show(plot_lda)




def prepareLDAData():
    data = {
        'vocab': vocab,
        'doc_topic_dists': doc_topic,
        'doc_lengths': list(lda_df['len_docs']),
        'term_frequency':cvectorizer.vocabulary_,
        'topic_term_dists': lda_model.components_
    } 
    return data




"""
import pyLDAvis

lda_df['len_docs'] = combined_sample['token'].map(len)
ldadata = prepareLDAData()
pyLDAvis.enable_notebook()
prepared_data = pyLDAvis.prepare(**ldadata)
"""




"""
import IPython.display
from IPython.core.display import display, HTML, Javascript

h = IPython.display.display(HTML(html_string))
IPython.display.display_HTML(h)
"""




#整形データ保存
train.to_csv('re_train.csv', sep='\t')
test.to_csv('re_test.csv', sep='\t')

