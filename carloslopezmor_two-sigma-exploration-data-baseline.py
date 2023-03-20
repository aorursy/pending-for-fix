#!/usr/bin/env python
# coding: utf-8



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import warnings
toy = True
from kaggle.competitions import twosigmanews
from itertools import chain
warnings.filterwarnings("ignore")

env = twosigmanews.make_env()
(marketdata, news) = env.get_training_data()
print('Done!')




if toy:
    marketdata = marketdata.tail(100_000)
    news = news.tail(300_000)
else:
    marketdata = marketdata.tail(3_000_000)
    news = news.tail(6_000_000)




print("Marketdata (Rows, Columns): ",marketdata.shape,"News (Rows, Columns): ",news.shape)




marketdata.info()




news.info()




marketdata.head()




news.head()




resumen_marketdata= marketdata.describe()
resumen_marketdata = resumen_marketdata.transpose()
resumen_marketdata




resumen_news= news.describe()
resumen_news = resumen_news.transpose()
resumen_news




# Load Visualization Libraries
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import plotly.figure_factory as ff
init_notebook_mode(connected=True) 




# Number of missing in each column
nulls_marketdata = pd.DataFrame(marketdata.isnull().sum()).rename(columns = {0: 'total'})
nulls_news = pd.DataFrame(news.isnull().sum()).rename(columns = {0: 'total'})

print("Missing Values. Marketdata: ",nulls_marketdata['total'].sum()," News: ",nulls_news['total'].sum())




nulls_marketdata['percentage'] = nulls_marketdata['total'] / len(marketdata)
columns_marketdata = nulls_marketdata[nulls_marketdata['percentage']>0].sort_values('percentage', ascending = False).head(10).reset_index().rename(index=str, columns={"index": "name"})




labels_marketdata = list(columns_marketdata['name'])
values_marketdata = list(columns_marketdata['percentage'])


fig = {'data': 
       [{'type':'pie',
                 'labels':labels_marketdata,
                 'domain': {"x": [0, 1],"y":[0,.9]},
                 'name': 'Marketdata',
                 'hoverinfo':'label+percent+name',
                 'values': values_marketdata
        }],
        'layout':
        {
          'title':'Missing Values',
           'annotations': [
            {
                'font': {
                    'size': 18
                },
                'showarrow': False,
                'text': 'Marketdata',
                'x': 0.5,
                'y': 1
            }]
        }
      }
               

iplot(fig)




marketdata_AAPL = marketdata[marketdata['assetCode']=='AAPL.O']




trace_high = go.Scatter(
                x=marketdata_AAPL['time'],
                y=marketdata_AAPL['returnsClosePrevRaw1'],
                name = "Open",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

trace_low = go.Scatter(
                x=marketdata_AAPL['time'],
                y=marketdata_AAPL['close'],
                name = "Close",
                line = dict(color = '#7F7F7F'),
                opacity = 0.8)

data = [trace_high,trace_low]

layout = dict(
    title = "Apple Inc.",
    xaxis = dict(
        range = ['2007-02-01','2016-12-30'])
)

fig = dict(data=data, layout=layout)
iplot(fig)




print("Unknown names: ", marketdata[marketdata["assetName"]=='Unknown'].size)




assetCode_Unknown = marketdata[marketdata['assetName'] == 'Unknown'].groupby('assetCode').size().reset_index('assetCode')
print("Asset Codes without names: ",assetCode_Unknown.shape[0])




assetCode_Unknown.head()




news[['assetCodes','assetName']].head()




marketdata['time'].dt.time.describe()




news['time'].dt.time.describe()




news['time'] = (news['time'] - np.timedelta64(22,'h')).dt.ceil('1D')
marketdata['time'] = marketdata['time'].dt.floor('1D')




news['assetCodes'] = news['assetCodes'].str.findall(f"'([\w\./]+)'")    
assetCodes_expanded = list(chain(*news['assetCodes']))
assetCodes_index = news.index.repeat( news['assetCodes'].apply(len) )
assert len(assetCodes_index) == len(assetCodes_expanded)
assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})




assetCodes.head()




news_cols_agg = {
    'urgency': ['min', 'count'],
    'takeSequence': ['max'],
    'bodySize': ['min', 'max', 'mean', 'std'],
    'wordCount': ['min', 'max', 'mean', 'std'],
    'sentenceCount': ['min', 'max', 'mean', 'std'],
    'companyCount': ['min', 'max', 'mean', 'std'],
    'marketCommentary': ['min', 'max', 'mean', 'std'],
    'relevance': ['min', 'max', 'mean', 'std'],
    'sentimentNegative': ['min', 'max', 'mean', 'std'],
    'sentimentNeutral': ['min', 'max', 'mean', 'std'],
    'sentimentPositive': ['min', 'max', 'mean', 'std'],
    'sentimentWordCount': ['min', 'max', 'mean', 'std'],
    'noveltyCount12H': ['min', 'max', 'mean', 'std'],
    'noveltyCount24H': ['min', 'max', 'mean', 'std'],
    'noveltyCount3D': ['min', 'max', 'mean', 'std'],
    'noveltyCount5D': ['min', 'max', 'mean', 'std'],
    'noveltyCount7D': ['min', 'max', 'mean', 'std'],
    'volumeCounts12H': ['min', 'max', 'mean', 'std'],
    'volumeCounts24H': ['min', 'max', 'mean', 'std'],
    'volumeCounts3D': ['min', 'max', 'mean', 'std'],
    'volumeCounts5D': ['min', 'max', 'mean', 'std'],
    'volumeCounts7D': ['min', 'max', 'mean', 'std']
}




news_cols = ['time', 'assetCodes'] + sorted(news_cols_agg.keys())









def getx(news, marketdata, le=None)
    news['time'] = (news['time'] - np.timedelta64(22,'h')).dt.ceil('1D')
    marketdata['time'] = marketdata['time'].dt.floor('1D')
    
    news_train_df_expanded = pd.merge(assetCodes, news[news_cols], left_on='level_0', right_index=True, suffixes=(['','_old']))

    # Aggregate numerical news features
    news_train_df_aggregated = news_train_df_expanded.groupby(['time', 'assetCode']).agg(news_cols_agg)
    news_train_df_aggregated = news_train_df_aggregated.apply(np.float32)
    # Flat columns
    news_train_df_aggregated.columns = ['_'.join(col).strip() for col in news_train_df_aggregated.columns.values]
    # Join with train
    market_train_df = marketdata.join(news_train_df_aggregated, on=['time', 'assetCode'])
    # Free memory
    del news_train_df_aggregated
    
    # Free memory
    del news_train_df_expanded
    
    x= market_train_df
    y = market_train_df['returnsOpenNextMktres10'].clip(-1, 1)
    
    try:
        x.drop(columns=['returnsOpenNextMktres10'], inplace=True)
    except:
        pass
    
    try:
        x.drop(columns=['universe'], inplace=True)
    except:
        pass

    x['dayofweek'], x['month'] = x.time.dt.dayofweek, x.time.dt.month
    x.drop(columns='time', inplace=True)
    
    def label_encode(series, min_count):
        vc = series.value_counts()
        le = {c:i for i, c in enumerate(vc.index[vc >= min_count])}
    return le

    le_assetCode = label_encode(x['assetCode'], min_count=10)
    le_assetName = label_encode(x['assetName'], min_count=5)
        
    x['assetCode'] = x['assetCode'].map(le_assetCode).fillna(-1).astype(int)
    x['assetName'] = x['assetName'].map(le_assetName).fillna(-1).astype(int)
    
    x['dayofweek'], x['month'] = x.time.dt.dayofweek, x.time.dt.month
    x.drop(columns='time', inplace=True)
    return (x,le_assetCode,le_assetName)




# Save universe data for latter use
  universe = marketdata['universe']
  time = marketdata['time']




n_train = int(x.shape[0] * 0.8)

x_train, y_train = x.iloc[:n_train], y.iloc[:n_train]
x_valid, y_valid = x.iloc[n_train:], y.iloc[n_train:]




# For valid data, keep only those with universe > 0. This will help calculate the metric
u_valid = (universe.iloc[n_train:] > 0)
t_valid = time.iloc[n_train:]

x_valid = x_valid[u_valid]
y_valid = y_valid[u_valid]
t_valid = t_valid[u_valid]
del u_valid




import lightgbm as lgb
# Creat lgb datasets
train_cols = x.columns.tolist()
categorical_cols = [] # ['assetCode', 'assetName', 'dayofweek', 'month']

# Note: y data is expected to be a pandas Series, as we will use its group_by function in `sigma_score`
dtrain = lgb.Dataset(x_train.values, y_train, feature_name=train_cols, categorical_feature=categorical_cols, free_raw_data=False)
dvalid = lgb.Dataset(x_valid.values, y_valid, feature_name=train_cols, categorical_feature=categorical_cols, free_raw_data=False)




dvalid.params = {
    'extra_time': t_valid.factorize()[0]
}




lgb_params = dict(
    objective = 'regression_l1',
    learning_rate = 0.1,
    num_leaves = 127,
    max_depth = -1,
#     min_data_in_leaf = 1000,
#     min_sum_hessian_in_leaf = 10,
    bagging_fraction = 0.75,
    bagging_freq = 2,
    feature_fraction = 0.5,
    lambda_l1 = 0.0,
    lambda_l2 = 1.0,
    metric = 'None', # This will ignore the loss objetive and use sigma_score instead,
    seed = 42 # Change for better luck! :)
)

def sigma_score(preds, valid_data):
    df_time = valid_data.params['extra_time']
    labels = valid_data.get_label()
    
#    assert len(labels) == len(df_time)

    x_t = preds * labels #  * df_valid['universe'] -> Here we take out the 'universe' term because we already keep only those equals to 1.
    
    # Here we take advantage of the fact that `labels` (used to calculate `x_t`)
    # is a pd.Series and call `group_by`
    x_t_sum = x_t.groupby(df_time).sum()
    score = x_t_sum.mean() / x_t_sum.std()

    return 'sigma_score', score, True

evals_result = {}
m = lgb.train(lgb_params, dtrain, num_boost_round=1000, valid_sets=(dvalid,), valid_names=('valid',), verbose_eval=25,
              early_stopping_rounds=100, feval=sigma_score, evals_result=evals_result)


df_result = pd.DataFrame(evals_result['valid'])




num_boost_round, valid_score = df_result['sigma_score'].idxmax()+1, df_result['sigma_score'].max()
print(lgb_params)
print(f'Best score was {valid_score:.5f} on round {num_boost_round}')




import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(14, 14))
lgb.plot_importance(m, ax=ax[0])
lgb.plot_importance(m, ax=ax[1], importance_type='gain')
fig.tight_layout()









dtrain_full = lgb.Dataset(x, y, feature_name=train_cols, categorical_feature=categorical_cols)

model = lgb.train(lgb_params, dtrain, num_boost_round=num_boost_round)




def make_predictions(predictions_template_df, market_obs_df, news_obs_df, le):
    x, _ = get_x(market_obs_df, news_obs_df, le)
    predictions_template_df.confidenceValue = np.clip(model.predict(x), -1, 1)




days = env.get_prediction_days()

for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_predictions(predictions_template_df, market_obs_df, news_obs_df, le)
    env.predict(predictions_template_df)
print('Done!')




env.write_submission_file()

