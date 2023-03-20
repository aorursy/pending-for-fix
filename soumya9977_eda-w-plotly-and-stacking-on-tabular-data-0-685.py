#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




get_ipython().system('pip install plotly')




get_ipython().system('pip install cufflinks')




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
import cufflinks as cf




train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv',na_values=['unknown'])
submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
train.head()




train.isnull().sum()




train.info()




py.offline.init_notebook_mode(connected = True)
cf.go_offline()




type(train.columns)




lists = ['anatom_site_general_challenge']




# cf.set_config_file(theme = 'solar')
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource

data_table = DataTable(
    columns=[TableColumn(field=Ci, title=Ci) for Ci in lists],
    source=ColumnDataSource(train),
    height=300,
)


count_anatom = train['anatom_site_general_challenge'].value_counts().plot_bokeh(kind = 'barh',color='green',title="count of location of imaged site", 
    alpha=0.6,show_figure=False)

pandas_bokeh.plot_grid([[data_table, count_anatom]], plot_width=400, plot_height=350)




train['benign_malignant'].value_counts().plot_bokeh(kind='bar',alpha=0.6,color= 'red',title="count of benign and malignant",ylabel='count',xlabel='benign_malignant')




train['diagnosis'].value_counts().plot_bokeh(kind='bar',color ='magenta',alpha=0.6,vertical_xlabel=True,title="count of diagnosis",ylabel='count',xlabel='diagnosis')




train['age_approx'].value_counts().plot_bokeh(kind='bar',color = 'blue',title="count of age_approx",vertical_xlabel=True,ylabel='count',xlabel='age_approx',alpha=0.6)




train['sex'].value_counts().plot_bokeh(kind='bar',alpha=0.6,colormap=["#009933"],title="count of sex",ylabel='count',xlabel='sex')




p_vs_img = train.groupby('patient_id').image_name.count().to_frame().reset_index()




p_vs_img_plot = p_vs_img.sort_values(by=['image_name'],ascending=False).iloc[0:50]
p_vs_img_plot.plot_bokeh(kind='bar',alpha=0.6,color="brown",title="count of top 50 patient_id",ylabel='count',xlabel='patient_id',vertical_xlabel=True)





# patient_id
train['patient_id'].value_counts().plot_bokeh(kind='bar',alpha=0.6,color="blue",title="count of full patient_id",ylabel='count',xlabel='patient_id',vertical_xlabel=True,figsize=(1000, 600))




(train.groupby('patient_id').image_name.count()).max()




train.head()




train.groupby(['target','sex']).count()




train.groupby(['target','anatom_site_general_challenge'])['benign_malignant'].count().iplot(kind = 'bar')




train.groupby(['target','sex'])['benign_malignant'].count().iplot(kind = 'bar',color='red')




train.groupby(['target','age_approx'])['benign_malignant'].count().iplot(kind = 'bar',color='green')




train.groupby(['sex','anatom_site_general_challenge'])['benign_malignant'].count().iplot(kind = 'bar',color = 'blue')




train.groupby(['target','diagnosis'])['benign_malignant'].count().iplot(kind = 'bar')




train.groupby(['sex','diagnosis'])['benign_malignant'].count().iplot(kind = 'bar',color ='magenta')









amount = train.groupby('anatom_site_general_challenge')['anatom_site_general_challenge'].transform('count')




import plotly.graph_objs as go
labels = set(train['anatom_site_general_challenge'])
labels_list = list(labels)
trace = go.Pie(values=amount,labels = labels_list,hole=0.3,pull=[0, 0, 0.2, 0])
iplot([trace])




train['diagnosis'].value_counts()




train['diagnosis'].value_counts().iplot(kind='bar',color='yellow')




train.head()




numerical_features = [i for i in train.columns if train[i].dtypes != 'O']
numerical_features




categorical_features = [i for i in train.columns if train[i].dtypes == 'O']
categorical_features




discreat_features = [i for i in numerical_features if len(train[i].unique())<25 ]
discreat_features




import gc
gc.collect()




train.head()




train.isnull().sum()




nan_features = [i for i in train.columns if train[i].isnull().sum()>=1]
nan_features




# pd.pandas.set_option('display.max_columns',None)
# pd.pandas.set_option('display.max_rows',None)

train['age_approx'] = train['age_approx'].fillna(train['age_approx'].median())
train['age_approx'].isnull().sum()




train['anatom_site_general_challenge'].mode()




train['anatom_site_general_challenge']=train['anatom_site_general_challenge'].fillna('torso') 
train['anatom_site_general_challenge'].isnull().sum()




train['sex'] = train['sex'].fillna(str(train['sex'].mode()))
train['diagnosis'] = train['diagnosis'].fillna(str(train['diagnosis'].mode()))
train['sex'].isnull().sum()
train['diagnosis'].isnull().sum()




train.isnull().sum()




from sklearn.preprocessing import LabelEncoder
label_encod = LabelEncoder()

for i in categorical_features:
    train[i]=label_encod.fit_transform(train[i])




train.head()




data = train.copy()
data = data.drop(['image_name','patient_id','diagnosis','benign_malignant'],axis = 1)
data.head()




Y = data['target']
X = data.drop(['target'],axis = 1)




SEED = 42
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(X,Y,test_size = 0.2,random_state = SEED)




random_grid = {'bootstrap': [True, False],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [130, 180, 230]}




from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

classifier_rf = RandomForestClassifier(random_state=SEED)
rf_random = RandomizedSearchCV(estimator = classifier_rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(x_train,y_train)




rf_random.best_estimator_




classifier_rf1 = RandomForestClassifier(max_depth=70, min_samples_leaf=4, min_samples_split=5,
                       n_estimators=180, random_state=42)
classifier_rf1.fit(x_train,y_train)




y_pred = classifier_rf1.predict_proba(x_val)
type(y_pred)









test.isnull().sum()




test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna(str(test['anatom_site_general_challenge'].mode())) 




test.isnull().sum()









categorical_fea_test = [i for i in test.columns if test[i].dtypes == 'O']
for i in categorical_fea_test:
    test[i] = label_encod.fit_transform(test[i])
    
test.head()




test = test.drop(['image_name','patient_id'],axis = 1)
test.head()




from sklearn.metrics import roc_curve, roc_auc_score,auc
print(roc_auc_score(y_val, y_pred[:,1]))

fpr, tpr, _ = roc_curve(y_val, y_pred[:,1])

plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()




y_pred




auc_rf = auc(fpr,tpr)
print(auc_rf)




y_pred_test_rf = classifier_rf1.predict_proba(test)
y_pred_test_rf[:,1].




submission_main = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')




img_sub = pd.read_csv('../input/1st-featuredcom-submission-baseline-keras-vgg16/submission.csv')
img_sub.head()




import xgboost as xgb
from scipy import stats
from scipy.stats import randint

xgb_clf = xgb.XGBClassifier()


param_dist = {'n_estimators': stats.randint(150, 1000),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4]
             }
clf_xgb = RandomizedSearchCV(xgb_clf, param_distributions = param_dist, n_iter = 25, scoring = 'roc_auc', error_score = 0, verbose = 3, n_jobs = -1)
clf_xgb.fit(x_train,y_train)




clf_xgb.best_estimator_




xgbo = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5791848525626986, gamma=0,
              gpu_id=-1, importance_type='gain', interaction_constraints='',
              learning_rate=0.47840118037023044, max_delta_step=0, max_depth=4,
              min_child_weight=4, monotone_constraints='()',
              n_estimators=585, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
              subsample=0.792826572247452, tree_method='exact',
              validate_parameters=1, verbosity=None)




xgbo.fit(x_train,y_train)
y_pred_xgb = xgbo.predict_proba(x_val)




print(roc_auc_score(y_val, y_pred_xgb[:,1]))




from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier

base_learners = [
                 ('rf_1', RandomForestClassifier(n_estimators=100, random_state=SEED)),
                 ('adb',AdaBoostClassifier(n_estimators=100, random_state=SEED)),
                 ('ext',ExtraTreesClassifier(n_estimators=100, random_state=SEED)),
                 ('gbc',GradientBoostingClassifier(n_estimators=100,random_state=SEED)),
                 ('svc', SVC())
    
    
                ]

# Initialize Stacking Classifier with the Meta Learner
stk_clf = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())

stk_clf.fit(x_train, y_train)




y_pred_stk = stk_clf.predict_proba(x_val)




print(roc_auc_score(y_val, y_pred_stk[:,1]))




get_ipython().system('pip install -U pandas_bokeh')




import pandas_bokeh
pd.set_option('plotting.backend', 'pandas_bokeh')
pandas_bokeh.output_notebook()




train['anatom_site_general_challenge'].value_counts().plot_bokeh(kind='barh')






