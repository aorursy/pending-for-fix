#!/usr/bin/env python
# coding: utf-8



# Run this cell and select the kaggle.json file downloaded
# from the Kaggle account settings page.
from google.colab import files
files.upload()




# Let's make sure the kaggle.json file is present.
get_ipython().system('ls -lha kaggle.json')




# Next, install the Kaggle API client.
get_ipython().system('pip install -q kaggle')




def submit_kaggle(submission_path, competition, message="submission"):
    get_ipython().system('kaggle competitions submit -c {competition} -f {submission_path} -m "{message}"')




# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')

# This permissions change avoids a warning on Kaggle tool startup.
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')




# List available datasets.
get_ipython().system('kaggle datasets list')




# Copy the stackoverflow data set locally.
#!kaggle datasets download -d stackoverflow/stack-overflow-2018-developer-survey
get_ipython().system('kaggle competitions download -c ashrae-energy-prediction -p /content/ashrae-energy-prediction')




ls /content/ashrae-energy-prediction




DATA = "/content/ashrae-energy-prediction/"




#!cd /content/ashrae-energy-prediction
#!unzip *.zip




get_ipython().system('unzip /content/ashrae-energy-prediction/sample_submission.csv.zip -d {DATA}')
get_ipython().system('unzip /content/ashrae-energy-prediction/test.csv.zip -d {DATA}')
get_ipython().system('unzip /content/ashrae-energy-prediction/train.csv.zip -d {DATA}')
get_ipython().system('unzip /content/ashrae-energy-prediction/weather_test.csv.zip -d {DATA}')
get_ipython().system('unzip /content/ashrae-energy-prediction/weather_train.csv.zip -d {DATA}')




ls /content/ashrae-energy-prediction




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import gc, math

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from datetime import datetime




sns.set(rc={'figure.figsize':(11,8)})
sns.set(style="whitegrid")




#DATA = "/content/"




ls




get_ipython().run_cell_magic('time', '', 'metadata_df = pd.read_csv(f"{DATA}building_metadata.csv")\ntrain_df = pd.read_csv(f"{DATA}train.csv", parse_dates=[\'timestamp\'])\ntest_df = pd.read_csv(f\'{DATA}test.csv\', parse_dates=[\'timestamp\'])\nweather_train_df = pd.read_csv(f\'{DATA}weather_train.csv\', parse_dates=[\'timestamp\'])\nweather_test_df = pd.read_csv(f\'{DATA}weather_test.csv\', parse_dates=[\'timestamp\'])')




pd.Series(np.log1p(train_df['meter_reading'])).hist()




train_df.head()




metadata_df.head()




weather_train_df.shape




weather_train_df.head()




test_df.head()




weather = pd.concat([weather_train_df,weather_test_df],ignore_index=True)
weather_key = ['site_id', 'timestamp']

temp_skeleton = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()

# calculate ranks of hourly temperatures within date/site_id chunks
temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.date])['air_temperature'].rank('average')

# create a dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)
df_2d = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)

# Subtract the columnID of temperature peak by 14, getting the timestamp alignment gap.
site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)
site_ids_offsets.index.name = 'site_id'

def timestamp_align(df):
    df['offset'] = df.site_id.map(site_ids_offsets)
    df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))
    df['timestamp'] = df['timestamp_aligned']
    del df['timestamp_aligned']
    return df




weather_train_df.tail()




weather_train_df = timestamp_align(weather_train_df)
weather_test_df = timestamp_align(weather_test_df)




del weather 
del df_2d
del temp_skeleton
del site_ids_offsets




weather_train_df.tail()




def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in cols:
        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]




add_lag_feature(weather_train_df, window=72)
add_lag_feature(weather_test_df, window=72)




weather_train_df.columns




weather_train_df.isna().sum()




weather_test_df.isna().sum()




weather_train_df = weather_train_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))




weather_test_df = weather_test_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))




weather_train_df.isna().sum()




weather_test_df.isna().sum()




# Since loss metric is RMSLE
train_df['meter_reading'] = np.log1p(train_df['meter_reading'])




weather_train_df.head()




## Function to reduce the memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df




le = LabelEncoder()
metadata_df['primary_use'] = le.fit_transform(metadata_df['primary_use'])




metadata_df = reduce_mem_usage(metadata_df)
train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)
weather_train_df = reduce_mem_usage(weather_train_df)
weather_test_df = reduce_mem_usage(weather_test_df)




print (f'Training data shape: {train_df.shape}')
print (f'Weather training shape: {weather_train_df.shape}')
print (f'Weather training shape: {weather_test_df.shape}')
print (f'Weather testing shape: {metadata_df.shape}')
print (f'Test data shape: {test_df.shape}')




train_df.head()




weather_train_df.head()




metadata_df.head()




test_df.head()




train_df.head()




get_ipython().run_cell_magic('time', '', "full_train_df = train_df.merge(metadata_df, on='building_id', how='left')\nfull_train_df = full_train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')")




full_train_df = full_train_df.loc[~(full_train_df['air_temperature'].isnull() & full_train_df['cloud_coverage'].isnull() & full_train_df['dew_temperature'].isnull() & full_train_df['precip_depth_1_hr'].isnull() & full_train_df['sea_level_pressure'].isnull() & full_train_df['wind_direction'].isnull() & full_train_df['wind_speed'].isnull() & full_train_df['offset'].isnull())]
#full_train_df.loc[(full_train_df['air_temperature'].isnull() & full_train_df['cloud_coverage'].isnull() & full_train_df['dew_temperature'].isnull() & full_train_df['precip_depth_1_hr'].isnull() & full_train_df['sea_level_pressure'].isnull() & full_train_df['wind_direction'].isnull() & full_train_df['wind_speed'].isnull() & full_train_df['offset'].isnull())] = -1




full_train_df.shape




# Delete unnecessary dataframes to decrease memory usage
del train_df
del weather_train_df
gc.collect()




get_ipython().run_cell_magic('time', '', "full_test_df = test_df.merge(metadata_df, on='building_id', how='left')\nfull_test_df = full_test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')")




full_test_df.shape




# Delete unnecessary dataframes to decrease memory usage
del metadata_df
del weather_test_df
del test_df
gc.collect()




ax = sns.barplot(pd.unique(full_train_df['primary_use']), full_train_df['primary_use'].value_counts())
ax.set(xlabel='Primary Usage', ylabel='# of records', title='Primary Usage vs. # of records')
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")
plt.show()




meter_types = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
ax = sns.barplot(np.vectorize(meter_types.get)(pd.unique(full_train_df['meter'])), full_train_df['meter'].value_counts())
ax.set(xlabel='Meter Type', ylabel='# of records', title='Meter type vs. # of records')
plt.show()




# Average meter reading
print (f'Average meter reading: {full_train_df.meter_reading.mean()} kWh')




ax = sns.barplot(np.vectorize(meter_types.get)(full_train_df.groupby(['meter'])['meter_reading'].mean().keys()), full_train_df.groupby(['meter'])['meter_reading'].mean())
ax.set(xlabel='Meter Type', ylabel='Meter reading', title='Meter type vs. Meter Reading')
plt.show()




fig, ax = plt.subplots(1,1,figsize=(14, 6))
ax.set(xlabel='Year Built', ylabel='# Of Buildings', title='Buildings built in each year')
full_train_df['year_built'].value_counts(dropna=False).sort_index().plot(ax=ax)
full_test_df['year_built'].value_counts(dropna=False).sort_index().plot(ax=ax)
ax.legend(['Train', 'Test']);




fig, ax = plt.subplots(1,1,figsize=(15, 7))
full_train_df.groupby(['building_id'])['square_feet'].mean().plot(ax=ax)
ax.set(xlabel='Building ID', ylabel='Area in Square Feet', title='Square Feet area of buildings')
plt.show()




pd.DataFrame(full_train_df.isna().sum().sort_values(ascending=False), columns=['NaN Count'])




def mean_without_overflow_fast(col):
    col /= len(col)
    return col.mean() * len(col)




missing_values = (100-full_train_df.count() / len(full_train_df) * 100).sort_values(ascending=False)




get_ipython().run_cell_magic('time', '', 'missing_features = full_train_df.loc[:, missing_values > 0.0]\nmissing_features = missing_features.apply(mean_without_overflow_fast)')




# Both train and test are interpolated with mean of train
for key in full_train_df.loc[:, missing_values > 0.0].keys():
    if key == 'year_built' or key == 'floor_count':
        full_train_df[key].fillna(math.floor(missing_features[key]), inplace=True)
        full_test_df[key].fillna(math.floor(missing_features[key]), inplace=True)
    else:
        full_train_df[key].fillna(missing_features[key], inplace=True)
        full_test_df[key].fillna(missing_features[key], inplace=True)




full_train_df.tail()




full_test_df.tail()




full_train_df.isna().sum().sum(), full_test_df.isna().sum().sum()




full_train_df['timestamp'].dtype




full_train_df["timestamp"] = pd.to_datetime(full_train_df["timestamp"])
full_test_df["timestamp"] = pd.to_datetime(full_test_df["timestamp"])




def transform(df):
    df['hour'] = np.uint8(df['timestamp'].dt.hour)
    df['day'] = np.uint8(df['timestamp'].dt.day)
    df['weekday'] = np.uint8(df['timestamp'].dt.weekday)
    df['month'] = np.uint8(df['timestamp'].dt.month)
    df['year'] = np.uint8(df['timestamp'].dt.year-1900)
    
    df['square_feet'] = np.log(df['square_feet'])
    
    return df




full_train_df = transform(full_train_df)
full_test_df = transform(full_test_df)




dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')
us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
full_train_df['is_holiday'] = (full_train_df['timestamp'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)
full_test_df['is_holiday'] = (full_test_df['timestamp'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)




# Assuming 5 days a week for all the given buildings
full_train_df.loc[(full_train_df['weekday'] == 5) | (full_train_df['weekday'] == 6) , 'is_holiday'] = 1
full_test_df.loc[(full_test_df['weekday']) == 5 | (full_test_df['weekday'] == 6) , 'is_holiday'] = 1




full_train_df.shape




full_train_df = full_train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')




full_train_df.shape




full_test_df = full_test_df.drop(['timestamp'], axis=1)
full_train_df = full_train_df.drop(['timestamp'], axis=1)
print (f'Shape of training dataset: {full_train_df.shape}')
print (f'Shape of testing dataset: {full_test_df.shape}')




full_train_df.tail()




## Reducing memory
full_train_df = reduce_mem_usage(full_train_df)
full_test_df = reduce_mem_usage(full_test_df)
gc.collect()




# def degToCompass(num):
#     val=int((num/22.5)+.5)
#     arr=[i for i in range(0,16)]
#     return arr[(val % 16)]




# full_train_df['wind_direction'] = full_train_df['wind_direction'].apply(degToCompass)




# beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 
#           (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]

# for item in beaufort:
#     full_train_df.loc[(full_train_df['wind_speed']>=item[1]) & (full_train_df['wind_speed']<item[2]), 'beaufort_scale'] = item[0]




# le = LabelEncoder()
# full_train_df['primary_use'] = le.fit_transform(full_train_df['primary_use'])

categoricals = ['site_id', 'building_id', 'primary_use', 'hour', 'weekday', 'meter',  'wind_direction', 'is_holiday']
# drop_cols = ['sea_level_pressure', 'wind_speed']
numericals = ['square_feet', 'year_built', 'air_temperature', 'cloud_coverage',
              'dew_temperature', 'precip_depth_1_hr', 'floor_count', 'air_temperature_mean_lag72',
       'cloud_coverage_mean_lag72', 'dew_temperature_mean_lag72',
       'precip_depth_1_hr_mean_lag72']

feat_cols = categoricals + numericals




full_train_df[numericals].describe()




full_train_df.tail()




full_train_df = reduce_mem_usage(full_train_df)
gc.collect()




cat_card = full_train_df[categoricals].nunique().apply(lambda x: min(x, 50)).to_dict()
cat_card




target = full_train_df["meter_reading"]
full_train_df.to_pickle('full_train_df.pkl')
#del full_train_df["meter_reading"]




# full_train_df.drop(drop_cols, axis=1)
# gc.collect()




# Save the testing dataset to freeup the RAM. We'll read after training
full_test_df.to_pickle('full_test_df.pkl')
del full_test_df
gc.collect()




import pickle




with open('full_train_df.pkl', 'rb') as f:
    full_train_df = pickle.load(f)




full_train_df.columns




from fastai.tabular import *




from fastai.basic_train import *




ls




#learner = learn.load(f'model1.bin')




full_train_df_sample = full_train_df.sample(frac=0.1)




full_train_df_sample.shape




procs = [FillMissing, Categorify] # Took out Normalize to address NaN prob when getting mean




valid_idx = range(len(full_train_df_sample)- int(full_train_df_sample.shape[0] * 0.1), len(full_train_df_sample))




#folds = 2
#seed = 666

#kf = StratifiedKFold(n_splits=folds, shuffle=False, random_state=seed)




#index_splits = next(kf.split(full_train_df, full_train_df['building_id']))




dep_var = "meter_reading"
cat_names = categoricals
path = DATA




#full_train_df[dep_var] = target




#del target




full_train_df_sample[dep_var]




# Not including test set first since takes too much space
data = TabularDataBunch.from_df(path, full_train_df_sample, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names) #, test_df=full_test_df)
print(data.train_ds.cont_names)  # `cont_names` defaults to: set(df)-set(cat_names)-{dep_var}




(cat_x,cont_x),y = next(iter(data.train_dl))
for o in (cat_x, cont_x, y): print(to_np(o[:5]))




learn = tabular_learner(data, layers=[200,100], emb_szs=cat_card, metrics=rmse)




learn.fit_one_cycle(1, 1e-2)




pred_batch1 = learn.pred_batch()
pred_batch1[: 5]




learn.save('model_sample')




learn.export('model_sample_export')




# Checking if predictions will be similar after loading
learn1 = tabular_learner(data, layers=[200,100], emb_szs=cat_card, metrics=rmse)




learn1.load('model_sample')




ls /content/ashrae-energy-prediction/




pred_batch2 = learn1.pred_batch()
pred_batch2[: 5]




learn2.data.train_ds




tabList_sample = TabularList.from_df(full_train_df.head(10), cat_names=cat_names, procs=procs)
learn2 = load_learner(DATA, 'model_sample_export', test=tabList_sample)




pred_batch3 = learn2.pred_batch(ds_type=DatasetType.Test)
pred_batch3[: 5]




full_test_df = pd.read_pickle('full_test_df.pkl')




nan_cols = full_test_df.columns.values[full_test_df.isna().sum().nonzero()[0]].tolist()




for col_ in nan_cols:
    median = full_test_df[col_].median()
    full_test_df[col_] = full_test_df[col_].fillna(median)




# Need to input data as tabList when learner is loaded
# Note that the learner is loaded without the datasets
tabList = TabularList.from_df(full_test_df, cat_names=cat_names, procs=procs)
learn2 = load_learner(DATA, 'model_sample_export', test=tabList)




del tabList
del full_test_df




pred_test = learn2.get_preds(ds_type=DatasetType.Test)




len(pred_test[0]), full_test_df.shape




pred_test_np = pred_test[0].numpy()




from google.colab import drive
drive.mount('/content/gdrive')




res = pred_test_np




submission = pd.read_csv(f'{DATA}sample_submission.csv')
submission.shape




submission = pd.read_csv(f'{DATA}sample_submission.csv')
# Remember, we predicted the log consumption, so we have to get the exponential to convert it back!
submission['meter_reading'] = np.expm1(res)
submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0
submission.to_csv('submission_fastai_20191103.csv', index=False)
submission




submission.head()




get_ipython().system("cp submission_fastai_20191103.csv /content/gdrive/'My Drive'/ashrae/submissions/submission_fastai_20191103.csv")




ls /content/gdrive/'My Drive'/ashrae/submissions/




get_ipython().system('kaggle competitions submit -c ashrae-energy-prediction -f submission_fastai_20191103.csv -m "Hello fastai tabular FCNN (corrected w/ exponential)"')




del pred_test
del learn2
del learn




from fastai.tabular import *




from fastai.basic_train import *




ls




#learner = learn.load(f'model1.bin')




full_train_df_sample = full_train_df.sample(frac=1)




full_train_df_sample.shape




del full_train_df




procs = [FillMissing, Categorify] # Took out Normalize to address NaN prob when getting mean




#valid_idx = range(len(full_train_df_sample)- int(full_train_df_sample.shape[0] * 0.1), len(full_train_df_sample))
valid_idx = np.random.permutation(len(full_train_df_sample))[: int(full_train_df_sample.shape[0] * 0.05)]




valid_idx




#folds = 2
#seed = 666

#kf = StratifiedKFold(n_splits=folds, shuffle=False, random_state=seed)




#index_splits = next(kf.split(full_train_df, full_train_df['building_id']))




dep_var = "meter_reading"
cat_names = categoricals
path = DATA




#full_train_df[dep_var] = target




#del target




full_train_df_sample[dep_var]




# Not including test set first since takes too much space
data = TabularDataBunch.from_df(path, full_train_df_sample, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names, bs=128) #, test_df=full_test_df)
print(data.train_ds.cont_names)  # `cont_names` defaults to: set(df)-set(cat_names)-{dep_var}




del full_train_df_sample




(cat_x,cont_x),y = next(iter(data.train_dl))
for o in (cat_x, cont_x, y): print(to_np(o[:5]))




learn = tabular_learner(data, layers=[200,100], emb_szs=cat_card, metrics=rmse)




learn.fit_one_cycle(1, 1e-2)




pred_batch1 = learn.pred_batch()
pred_batch1[: 5]




learn.export('model_all_export')




# Checking if predictions will be similar after loading
#learn1 = tabular_learner(data, layers=[200,100], emb_szs=cat_card, metrics=rmse)




#learn1.load('model_sample')




#ls /content/ashrae-energy-prediction/




#pred_batch2 = learn1.pred_batch()
#pred_batch2[: 5]




#learn2.data.train_ds




#tabList_sample = TabularList.from_df(full_train_df.head(10), cat_names=cat_names, procs=procs)
#learn2 = load_learner(DATA, 'model_sample_export', test=tabList_sample)




#pred_batch3 = learn2.pred_batch(ds_type=DatasetType.Test)
#pred_batch3[: 5]




full_test_df = pd.read_pickle('full_test_df.pkl')




nan_cols = full_test_df.columns.values[full_test_df.isna().sum().nonzero()[0]].tolist()




for col_ in nan_cols:
    median = full_test_df[col_].median()
    full_test_df[col_] = full_test_df[col_].fillna(median)




## Need to input data as tabList when learner is loaded
## Note that the learner is loaded without the datasets
tabList = TabularList.from_df(full_test_df, cat_names=cat_names, procs=procs)
learn2 = load_learner(DATA, 'model_all_export', test=tabList)




del tabList
del full_test_df




pred_test = learn2.get_preds(ds_type=DatasetType.Test)




len(pred_test[0])#, full_test_df.shape




pred_test_np = pred_test[0].numpy()




from google.colab import drive
drive.mount('/content/gdrive')




res = pred_test_np




submission = pd.read_csv(f'{DATA}sample_submission.csv')
submission.shape




submission = pd.read_csv(f'{DATA}sample_submission.csv')
submission['meter_reading'] = np.expm1(res)
submission.loc[submission['meter_reading'] < 0, 'meter_reading'] = 0
submission.to_csv('submission_fastai_full_20191103.csv', index=False)
submission





