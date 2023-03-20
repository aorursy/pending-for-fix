#!/usr/bin/env python
# coding: utf-8



import time
notebookstart = time.time()




import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)




import platform
import sys
import importlib
import multiprocessing
import random




import numpy as np
import pandas as pd

random.seed(321)
np.random.seed(321)

pd.options.display.max_columns = 9999




import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

get_ipython().run_line_magic('matplotlib', 'inline')

mpl.rc('figure', figsize=(15, 12))
plt.figure(figsize=(15, 12))
plt.rcParams['figure.facecolor'] = 'azure'
mpl.style.use('seaborn')
plt.style.use('seaborn')

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')




import seaborn as sns

sns.set(rc={'figure.figsize': (15, 12)})
sns.set(context='notebook', style='darkgrid', font='sans-serif',
        font_scale=1.1, rc={'figure.facecolor': 'azure',
        'axes.facecolor': 'azure', 'grid.color': 'steelblue'})
sns.color_palette(mcolors.TABLEAU_COLORS);




import missingno as msno




import scikitplot as skplt




import sklearn

from sklearn.model_selection import train_test_split,     cross_val_predict, cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit,     StratifiedKFold, GridSearchCV

from sklearn.preprocessing import StandardScaler, MinMaxScaler,     RobustScaler, MaxAbsScaler, Normalizer
from sklearn.preprocessing import LabelBinarizer, label_binarize

from sklearn.metrics import accuracy_score, confusion_matrix,     classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import average_precision_score,     precision_recall_fscore_support

from sklearn.utils import shuffle, resample
from sklearn.base import BaseEstimator, ClassifierMixin




from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier




from scipy import stats




import xgboost as xgb
from xgboost import XGBClassifier, plot_tree, plot_importance




import lightgbm as lgbm
from lightgbm import LGBMClassifier




import catboost
from catboost import CatBoostClassifier




import skopt
from skopt import BayesSearchCV




import imblearn
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek




print('Operating system version........', platform.platform())
print('Python version is............... %s.%s.%s' % sys.version_info[:3])
print('scikit-learn version is.........', sklearn.__version__)
print('pandas version is...............', pd.__version__)
print('numpy version is................', np.__version__)
print('matplotlib version is...........', mpl.__version__)
print('seaborn version is..............', sns.__version__)
print('scikit-plot version is..........', skplt.__version__)
print('missingno version is............', msno.__version__)
print('xgboost version is..............', xgb.__version__)
print('catboost version is.............', catboost.__version__)
print('lightgbm version is.............', lgbm.__version__)
print('scikit-optimize version is......', skopt.__version__)
print('imblearn version is.............', imblearn.__version__)




def getDatasetInformation(csv_filepath, is_corr_required=True):
    """
    Read CSV (comma-separated) file into DataFrame
    
    Returns,
    - DataFrame
    - DataFrame's shape
    - DataFrame's data types
    - DataFrame's describe
    - DataFrame's sorted unique value count
    - DataFrame's missing or NULL value count
    - DataFrame's correlation between numerical columns
    """

    dataset_tmp = pd.read_csv(csv_filepath)

    dataset_tmp_shape = pd.DataFrame(list(dataset_tmp.shape),
            index=['No of Rows', 'No of Columns'], columns=['Total'])
    dataset_tmp_shape = dataset_tmp_shape.reset_index()

    dataset_tmp_dtypes = dataset_tmp.dtypes.reset_index()
    dataset_tmp_dtypes.columns = ['Column Names', 'Column Data Types']

    dataset_tmp_desc = pd.DataFrame(dataset_tmp.describe())
    dataset_tmp_desc = dataset_tmp_desc.transpose()

    dataset_tmp_unique = dataset_tmp.nunique().reset_index()
    dataset_tmp_unique.columns = ['Column Name', 'Unique Value(s) Count'
                                  ]

    dataset_tmp_missing = dataset_tmp.isnull().sum(axis=0).reset_index()
    dataset_tmp_missing.columns = ['Column Names',
                                   'NULL value count per Column']
    dataset_tmp_missing =         dataset_tmp_missing.sort_values(by='NULL value count per Column'
            , ascending=False)

    if is_corr_required:
        dataset_tmp_corr = dataset_tmp.corr(method='spearman')
    else:
        dataset_tmp_corr = pd.DataFrame()

    return [
        dataset_tmp,
        dataset_tmp_shape,
        dataset_tmp_dtypes,
        dataset_tmp_desc,
        dataset_tmp_unique,
        dataset_tmp_missing,
        dataset_tmp_corr,
        ]




def getHighlyCorrelatedColumns(dataset, NoOfCols=6):
    df_corr = dataset.corr()

    # set the correlations on the diagonal or lower triangle to zero,
    # so they will not be reported as the highest ones

    df_corr *= np.tri(k=-1, *df_corr.values.shape).T
    df_corr = df_corr.stack()
    df_corr =         df_corr.reindex(df_corr.abs().sort_values(ascending=False).index).reset_index()
    return df_corr.head(NoOfCols)




def createFeatureEngineeredColumns(dataset):
    dataset_tmp = pd.DataFrame()

    dataset_tmp['CountOfZeroValues'] = (dataset == 0).sum(axis=1)
    dataset_tmp['CountOfNonZeroValues'] = (dataset != 0).sum(axis=1)

    weight = ((dataset != 0).sum() / len(dataset)).values
    dataset_tmp['WeightedCount'] = (dataset * weight).sum(axis=1)

    dataset_tmp['SumOfValues'] = dataset.sum(axis=1)

    dataset_tmp['VarianceOfValues'] = dataset.var(axis=1)
    dataset_tmp['MedianOfValues'] = dataset.median(axis=1)
    dataset_tmp['MeanOfValues'] = dataset.mean(axis=1)
    dataset_tmp['StandardDeviationOfValues'] = dataset.std(axis=1)

    # dataset_tmp['ModeOfValues'] = dataset.mode(axis=1)

    dataset_tmp['SkewOfValues'] = dataset.skew(axis=1)
    dataset_tmp['KurtosisOfValues'] = dataset.kurtosis(axis=1)

    dataset_tmp['MaxOfValues'] = dataset.max(axis=1)
    dataset_tmp['MinOfValues'] = dataset.min(axis=1)
    dataset_tmp['DiffOfMinMaxOfValues'] =         np.subtract(dataset_tmp['MaxOfValues'],
                    dataset_tmp['MinOfValues'])

    dataset_tmp['QuantilePointFiveOfValues'] = dataset[dataset
            > 0].quantile(0.5, axis=1)
    dataset_tmp['MovingAverage'] = dataset.apply(lambda x:             np.ma.average(x), axis=1)

    percentileList = [ 1, 2, 5, 10, 25, 50, 60, 75, 80, 85, 95, 99, ]
    for i in percentileList:
        dataset_tmp['Percentile' + str(i)] = dataset.apply(lambda x:                 np.percentile(x, i), axis=1)

    for column in dataset.columns:
        (hist, bin_edges) = np.histogram(dataset[column], bins=1000,
                density=True)
        dataset_tmp[column + '_histval'] =             [hist[np.searchsorted(bin_edges, ele) - 1] for ele in
             dataset[column]]

    for column in dataset.columns:
        dataset_tmp[column + '_log'] = np.log(dataset[column])

    for column in dataset.columns:
        dataset_tmp[column + '_square'] = np.square(dataset[column])

    dataset_tmp = dataset_tmp.fillna(0)
    dataset = pd.concat([dataset, dataset_tmp], axis=1)

    return dataset




def getZeroStdColumns(dataset):
    columnsWithZeroStd = dataset.columns[dataset.std() == 0].tolist()
    return columnsWithZeroStd




def getUniqueValueColumns(dataset, valueToCheck=0):
    columnsWithUniqueValue = dataset.columns[dataset.nunique()
            == valueToCheck].tolist()
    return columnsWithUniqueValue




def plotCategoricalVariableDistributionGraph(target_value, title='', xticksrotation=0):
    tmp_count = target_value.value_counts()
    
    fig=plt.figure()
    fig.suptitle(title, fontsize=18)
    
    ax1=fig.add_subplot(221)
    sns.pointplot(x=tmp_count.index, y=tmp_count, ax=ax1)
    ax1.set_title('Distribution Graph')
    plt.xticks(rotation=xticksrotation)
    
    ax2=fig.add_subplot(222)
    sns.barplot(x=tmp_count.index, y=tmp_count, ax=ax2)
    ax2.set_title('Distribution Graph - Bar')
    plt.xticks(rotation=xticksrotation)
    
    #ax3=fig.add_subplot(212)
    ax3=fig.add_subplot(223)
    ax3.pie(tmp_count, labels=tmp_count.index, autopct="%1.1f%%", shadow=True, startangle=195)
    ax3.axis('equal')
    ax3.set_title('Distribution Graph - Pie')
    
    ax4=fig.add_subplot(224)
    stats.probplot(target_value, plot=ax4)
    ax4.set_title('Distribution - Probability Plot')
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.show()




def plot_distplot(dataset):
    colors = mcolors.TABLEAU_COLORS

    dataset_fordist = dataset.select_dtypes([np.int, np.float])
    number_of_subplots = len(dataset_fordist.columns)
    number_of_columns = 3

    number_of_rows = number_of_subplots // number_of_columns
    number_of_rows += number_of_subplots % number_of_columns

    postion = range(1, number_of_subplots + 1)

    fig = plt.figure(1)
    for k in range(number_of_subplots):
        ax = fig.add_subplot(number_of_rows, number_of_columns,
                             postion[k])
        sns.distplot(dataset_fordist.iloc[:, k],
                     color=random.choice(list(colors.keys())), ax=ax)
    fig.tight_layout()
    plt.show()




def convertIntFloatToInt(dictObj):
    for (k, v) in dictObj.items():
        if float('Inf') == v:
            pass
        elif int(v) == v and isinstance(v, float):
            dictObj[k] = int(v)
    return dictObj




(
    dataset_sctp_train,
    df_train_shape,
    df_train_dtypes,
    df_train_describe,
    df_train_unique,
    df_train_missing,
    df_train_corr,
    ) = getDatasetInformation('../input/train.csv', False)

(
    dataset_sctp_test,
    df_test_shape,
    df_test_dtypes,
    df_test_describe,
    df_test_unique,
    df_test_missing,
    df_test_corr,
    ) = getDatasetInformation('../input/test.csv', False)




dataset_sctp_train.head()




df_train_shape




df_train_dtypes




df_train_describe




df_train_unique




df_train_missing




msno.matrix(dataset_sctp_train, color=(33 / 255, 102 / 255, 172 / 255));




dataset_sctp_test.head()




df_test_shape




df_test_dtypes




df_test_describe




df_test_unique




df_test_missing




msno.matrix(dataset_sctp_test, color=(33 / 255, 102 / 255, 172 / 255));




del(df_train_shape, df_train_dtypes, df_train_describe, df_train_unique, df_train_missing, df_train_corr)
del(df_test_shape, df_test_dtypes, df_test_describe, df_test_unique, df_test_missing, df_test_corr)




plot_distplot(dataset_sctp_train.iloc[:, 2:29])




plot_distplot(dataset_sctp_train.iloc[:, 29:56])




plot_distplot(dataset_sctp_train.iloc[:, 56:83])




plot_distplot(dataset_sctp_train.iloc[:, 83:110])




plot_distplot(dataset_sctp_train.iloc[:, 110:137])




plot_distplot(dataset_sctp_train.iloc[:, 137:164])




plot_distplot(dataset_sctp_train.iloc[:, 164:191])




plot_distplot(dataset_sctp_train.iloc[:, 191:204])




dataset_sctp_train.target.unique()




dataset_sctp_train.target.value_counts()




plotCategoricalVariableDistributionGraph(dataset_sctp_train.target, 'Target (feature) - Distribution', xticksrotation=90)




y = dataset_sctp_train['target']
X = dataset_sctp_train.drop(['ID_code', 'target'], axis=1)
X.astype('float32')

dataset_sctp_test_ID_code = dataset_sctp_test['ID_code']
Z = dataset_sctp_test.drop(['ID_code'], axis=1)
Z.astype('float32')

(X.shape, y.shape, Z.shape)

df_train_corr = X.corr(method='spearman')
df_train_corr#mask = np.zeros_like(df_train_corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True

sns.heatmap(
    df_train_corr,
    cmap='rainbow',
    annot=False,
    fmt='.2f',
    center=0,
    square=False,
    linewidths=.75,
    #mask=mask,
    );
plt.title('Correlation Matrix', fontsize=18)
plt.show()


getHighlyCorrelatedColumns(X, 10)

upper = df_train_corr.where(np.triu(np.ones(df_train_corr.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

print(f'Columns to drop from Train and Test datasets are {to_drop}.')

X.drop(columns=to_drop, axis=1, inplace=True)
Z.drop(columns=to_drop, axis=1, inplace=True)

(X.shape, Z.shape)del(df_train_corr)


X=createFeatureEngineeredColumns(X)
Z=createFeatureEngineeredColumns(Z)

(X.shape, y.shape, Z.shape)




columnsWithZeroStdToRemove = getZeroStdColumns(X)
print(f'Columns with Zero STD to drop from Train and Test dataset(s) are {columnsWithZeroStdToRemove}.')

X.drop(columnsWithZeroStdToRemove, axis=1, inplace=True)
Z.drop(columnsWithZeroStdToRemove, axis=1, inplace=True)

(X.shape, Z.shape)




X_columns_one_unique_value = getUniqueValueColumns(X, 1)
print(f'Columns with only 1 as value to drop from Train and Test datasets are {X_columns_one_unique_value}.')

X.drop(X_columns_one_unique_value, axis=1, inplace=True)
Z.drop(X_columns_one_unique_value, axis=1, inplace=True)

(X.shape, Z.shape)




X_columns_zero_unique_value = getUniqueValueColumns(X, 0)
print(f'Columns with only 0 as value to drop from Train and Test datasets are {X_columns_zero_unique_value}.')

X.drop(X_columns_zero_unique_value, axis=1, inplace=True)
Z.drop(X_columns_zero_unique_value, axis=1, inplace=True)

(X.shape, Z.shape)




n_cpus_avaliable = multiprocessing.cpu_count()

print(f'We\'ve got {n_cpus_avaliable} cpus to work with.')




def augment(x, y, t=2):
    (xs, xn) = ([], [])
    for i in range(t):
        mask = y > 0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:, c] = x1[ids][:, c]
        xs.append(x1)

    for i in range(t // 2):
        mask = y == 0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:, c] = x1[ids][:, c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x, xs, xn])
    y = np.concatenate([y, ys, yn])
    return (x, y)




lgbmparams = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "tree_learner": "serial",
    'n_jobs': n_cpus_avaliable,
}


n_splits = 7
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
sctp_predictions_lgbm_pp = np.zeros(len(Z))


for (index, (train_indices, val_indices)) in enumerate(skf.split(X, y)):
    print('Training on fold ' + str(index + 1) + '/' + str(n_splits) + '...')

    # Generate batches from indices
    (xtrain, xval) = (X.iloc[train_indices], X.iloc[val_indices])
    (ytrain, yval) = (y.iloc[train_indices], y.iloc[val_indices])

    (xtrain_aug, ytrain_aug) = augment(xtrain.values, ytrain.values,
            t=2)
    xtrain_aug = pd.DataFrame(xtrain_aug, columns=X.columns)
    ytrain_aug = [int(value) for value in ytrain_aug]
    ytrain_aug = pd.Series(ytrain_aug)

    train_set_lgbm = lgbm.Dataset(data=xtrain_aug, label=ytrain_aug)
    val_set_lgbm = lgbm.Dataset(data=xval, label=yval)
    watchlist = [train_set_lgbm, val_set_lgbm]

    evaluation_results_lgbm = {}

    lgbm_twoclass_model = lgbm.train(
        lgbmparams,
        train_set=train_set_lgbm,
        num_boost_round=10000,
        valid_sets=watchlist,
        early_stopping_rounds=1000,
        evals_result=evaluation_results_lgbm,
        verbose_eval=1000,
        )
    sctp_predictions_lgbm_pp += lgbm_twoclass_model.predict(Z,
            num_iteration=lgbm_twoclass_model.best_iteration) \
        / skf.n_splits




ax = lgbm.plot_metric(evaluation_results_lgbm)
plt.show()




ax = lgbm.plot_importance(lgbm_twoclass_model, max_num_features=50, color=mcolors.TABLEAU_COLORS)
plt.show()

xgbparams = {
    'booster': 'gbtree',
    'colsample_bylevel': 0.48973134715974026,
    'colsample_bytree': 0.46131821596707184,
    'device': 'cpu',
    'eval_metric': 'auc',
    'gamma': 8.0762512124703e-06,
    'learning_rate': 0.054722068464825926,
    'max_delta_step': 3,
    'max_depth': 29,
    'min_child_weight': 14,
    'missing': 'None',
    'n_jobs': n_cpus_avaliable,
    'objective': 'binary:logistic',
    'reg_lambda': 2.1021696940800796,
    'scale_pos_weight': 43.8858626195784,
    'subsample': 0.8113392946402368,
    'tree_method': 'approx',
    }n_splits = 2
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
sctp_predictions_xgb_pp = np.zeros(len(Z))


for (index, (train_indices, val_indices)) in enumerate(skf.split(X, y)):
    print('Training on fold ' + str(index + 1) + '/' + str(n_splits) + '...')
    
    (xtrain, xval) = (X.iloc[train_indices], X.iloc[val_indices])
    (ytrain, yval) = (y.iloc[train_indices], y.iloc[val_indices])

    (xtrain_aug, ytrain_aug) = augment(xtrain.values, ytrain.values,
            t=2)
    xtrain_aug = pd.DataFrame(xtrain_aug, columns=X.columns)
    ytrain_aug = [int(value) for value in ytrain_aug]
    ytrain_aug = pd.Series(ytrain_aug)

    train_set_xgb = xgb.DMatrix(data=xtrain_aug, label=ytrain_aug)
    val_set_xgb = xgb.DMatrix(data=xval, label=yval)
    watchlist = [(train_set_xgb, 'train'), (val_set_xgb, 'valid')]

    evaluation_results_xgb = {}

    xgb_twoclass_model = xgb.train(
        xgbparams,
        train_set_xgb,
        10000,
        watchlist,
        verbose_eval=1000,
        early_stopping_rounds=1000,
        evals_result=evaluation_results_xgb,
        )
    sctp_predictions_xgb_pp += xgb_twoclass_model.predict(xgb.DMatrix(Z)) \
        / skf.n_splitsplot_importance(xgb_twoclass_model, max_num_features=50, color=mcolors.TABLEAU_COLORS);


sctp_predictions_lgbm = lgbm_twoclass_model.predict(Z)
sctp_predictions_lgbm = [round(value) for value in sctp_predictions_lgbm]
sctp_predictions_lgbm = list(map(int, sctp_predictions_lgbm))
sctp_predictions_lgbm[:20]




dataset_submission = pd.DataFrame()
dataset_submission['ID_code'] = dataset_sctp_test_ID_code
dataset_submission['target'] = sctp_predictions_lgbm
dataset_submission.to_csv('lgbm_twoclass_model_submission.csv', index=False)




dataset_submission = pd.DataFrame()
dataset_submission['ID_code'] = dataset_sctp_test_ID_code
dataset_submission['target'] = sctp_predictions_lgbm_pp
dataset_submission.to_csv('lgbm_pp_twoclass_model_submission.csv', index=False)

sctp_predictions_xgb = xgb_twoclass_model.predict(xgb.DMatrix(Z))
sctp_predictions_xgb = [round(value) for value in sctp_predictions_xgb]
sctp_predictions_xgb = list(map(int, sctp_predictions_xgb))
sctp_predictions_xgb[:20]dataset_submission = pd.DataFrame()
dataset_submission['ID_code'] = dataset_sctp_test_ID_code
dataset_submission['target'] = sctp_predictions_xgb
dataset_submission.to_csv('xgb_twoclass_model_submission.csv', index=False)dataset_submission = pd.DataFrame()
dataset_submission['ID_code'] = dataset_sctp_test_ID_code
dataset_submission['target'] = sctp_predictions_xgb_pp
dataset_submission.to_csv('xgb_pp_twoclass_model_submission.csv', index=False)


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

