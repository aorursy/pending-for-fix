#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns




import scipy.stats as stats




df_train = pd.read_csv('../input/train.csv')
df_train.shape




df_test = pd.read_csv('../input/test.csv')
df_test.shape




df = pd.concat((df_train, df_test), ignore_index= True, sort = False)
df.shape




unique_hh_heads = (df_train['parentesco1'] == 1).sum()
unique_hh = len(df_train['idhogar'].unique())

print ('There are {} unique households and the dataset contain {} records of household heads'.format(unique_hh, unique_hh_heads))
print ('There is {} households without household head'.format(unique_hh - unique_hh_heads))




unique_hh_heads = (df_test['parentesco1'] == 1).sum()
unique_hh = len(df_test['idhogar'].unique())

print ('There are {} unique households and the dataset contain {} records of household heads'.format(unique_hh, unique_hh_heads))
print ('There is {} households without household head'.format(unique_hh - unique_hh_heads))




df.info(verbose=False)




df.columns[df.dtypes == object]




df.columns[df.dtypes == float]




# check columns for null values
df.isnull().sum()[df.isnull().sum() > 0]




df['Target'].value_counts(normalize = True)




sns.countplot(x = 'Target', data = df)




hh_heads = set(df['idhogar'][df['parentesco1'] == 1])
households = set(df['idhogar'])




'''
missing_hh =  households.difference(hh_heads)
rows_to_delete = df[df['idhogar'].isin(missing_hh)].index
df.drop(index= rows_to_delete, inplace = True) '''




# Number of records with no rent amount
df['v2a1'].isnull().sum()




# 
col = [i for i in df.columns if i.startswith('tipovivi')]
df.loc[:, col].sum()




# Create temporary column to identify the home ownership type.
df['temp_tipovivi'] = df[col].idxmax(axis = 1)




## Identify the home ownership status of the hh with zero rent
df['temp_tipovivi'][df['v2a1'].isnull()].value_counts()




df['v2a1'].fillna(value = 0, inplace = True)




df['temp_tipovivi'][df['v2a1'] == 0].value_counts()




# Change the homeownership type to  be consistant with the rent amount.
tipovivi2 = (df['v2a1'] == 0)&(df['tipovivi2'] == 1)
tipovivi3 = (df['v2a1'] == 0)&(df['tipovivi3'] == 1)




df.loc[tipovivi2,'tipovivi1'] = 1
df.loc[tipovivi3,'tipovivi1'] = 1

df.loc[tipovivi2,'tipovivi2'] = 0
df.loc[tipovivi3,'tipovivi3'] = 0




## Update temp_tipovivi to reflect change
df['temp_tipovivi'] = df[col].idxmax(axis = 1)




df[col][df['v2a1'] == 0].sum()




sns.distplot(df['v2a1'],fit = stats.norm)




## Seperate out the records where the households does not pay rent
df['RentPaying'] = (df['v2a1'] > 0)*1
## Log transfrom to make distribution normal
df['v2a1'] = np.log1p(df['v2a1'])
sns.distplot(df['v2a1'][df['RentPaying'] == 1],fit = stats.norm)









df.pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_tipovivi', aggfunc= 'count', margins= True)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_tipovivi', aggfunc= 'count')
cat = df['temp_tipovivi'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)




df['v18q'][df['v18q1'].isnull()].value_counts()




df['v18q1'].fillna(value = 0, inplace = True)




df[df['parentesco1'] == 1].groupby('Target')['v18q'].mean()




temp = df[(df['parentesco1'] == 1)&(df['Target'].notnull())].pivot_table(index = 'Target', columns = 'v18q1', values = 'idhogar', aggfunc='count')
cat = df['v18q1'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)




#### Years behind in school
df['rez_esc'].value_counts(dropna = False)




df['age'][df['rez_esc'].notna()].value_counts().sort_index()




df['rez_esc'].fillna(value = 0, inplace = True)




## Fixing the large age behind in school value
df.loc[df['rez_esc'] > 50, 'rez_esc'] = 0




## Obtain a list of households where the average years schooled is NA
na_mean_households = df['idhogar'][df['meaneduc'].isna()].unique()




## Checking if there are 18+ persons in households 
df[df['meaneduc'].isna()].groupby('idhogar')['age'].max()




## recompute meaneduc for households.
mapper = df[df['age'] >= 18].groupby('idhogar')['escolari'].mean().to_dict()




df['meaneduc'] = df.apply(lambda x: mapper.get('idhogar', 0) if np.isnan(x['meaneduc']) else x['meaneduc'], axis = 1)




df['meaneduc'].isna().sum()




df['SQBmeaned'] = df['meaneduc']**2




sns.countplot(x = 'Target', data = df[df['hacdor'] == 1])




df[df['parentesco1'] == 1].groupby('Target')['hacdor'].mean()




## Since overcrowing occures only ~3% of the time this is a possible candidates for deletion
## df.drop(column = ['hacdor', 'hacapo'])









### IGNORE !!!
### For the time being we'll calculate the likelihood based on the household head only.
temp = df[df['Target'].notnull()].pivot_table(index = 'Target', columns = 'v18q1', values = 'idhogar', aggfunc='count')
cat = df['v18q1'][df['Target'].notnull()].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)




df[['tamhog', 'tamviv', 'r4t3', 'hhsize', 'hogar_total']][df['r4t3'] != df['hhsize']].head()




df.drop(columns= ['tamhog', 'hogar_total', 'r4t3'], inplace = True)




(df['v14a'][df['parentesco1']== 1]).mean()




df[df['parentesco1'] == 1].groupby('Target')['v14a'].mean()




(df['refrig'][df['parentesco1']== 1]).mean()




df[df['parentesco1'] == 1].groupby('Target')['refrig'].mean()




col = [i for i in df.columns if i.startswith('pared')]
df.loc[:, col].sum()




df['temp_pared'] = df[col].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_pared', aggfunc= 'count')
cat = df['temp_pared'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)









col = [i for i in df.columns if i.startswith('piso')]
df.loc[:, col].sum()




df['temp_piso'] = df[col].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_piso', aggfunc= 'count')
cat = df['temp_piso'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)









col = [i for i in df.columns if i.startswith('techo')]
df.loc[:, col].sum()




df['temp_techo'] = df[col].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_techo', aggfunc= 'count')
cat = df['temp_techo'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)









df[df['parentesco1'] == 1].groupby('Target')['cielorazo'].mean()




col = [i for i in df.columns if i.startswith('abastagua')]
df.loc[:, col].sum()




df['temp_abastagua'] = df[col].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_abastagua', aggfunc= 'count')
cat = df['temp_abastagua'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)









df['temp_electricity'] = df[['public', 'planpri', 'noelec', 'coopele']].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_electricity', aggfunc= 'count')
cat = df['temp_electricity'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)









col = [i for i in df.columns if i.startswith('sanit')]
df.loc[:, col].sum()




df['temp_sanitario'] = df[col].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_sanitario', aggfunc= 'count')
cat = df['temp_sanitario'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)









col = [i for i in df.columns if i.startswith('energcocinar')]
df.loc[:, col].sum()




df['temp_energcocinar'] = df[col].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_energcocinar', aggfunc= 'count')
cat = df['temp_energcocinar'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)




df['cooking_lowEng'] = ((df['energcocinar1'] == 1)|(df['energcocinar4'] == 1))*1




col = [i for i in df.columns if i.startswith('elimbasu')]
df.loc[:, col].sum()




df['temp_elimbasu'] = df[col].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_elimbasu', aggfunc= 'count')
cat = df['temp_elimbasu'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)









col = [i for i in df.columns if i.startswith('epared')]
df.loc[:, col].sum()




df['temp_epared'] = df[col].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_epared', aggfunc= 'count')
cat = df['temp_epared'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)









col = [i for i in df.columns if i.startswith('etecho')]
df.loc[:, col].sum()




df['temp_etecho'] = df[col].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_etecho', aggfunc= 'count')
cat = df['temp_etecho'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)









col = [i for i in df.columns if i.startswith('eviv')]
df.loc[:, col].sum()




df['temp_eviv'] = df[col].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_eviv', aggfunc= 'count')
cat = df['temp_eviv'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)









df[df['parentesco1'] == 1].groupby('Target')['dis'].mean()




df[df['parentesco1'] == 1].groupby('Target')['male'].mean()




df.drop(columns= 'female', inplace = True)









col = [i for i in df.columns if i.startswith('estadocivil')]
df.loc[:, col].sum()




df['temp_estadocivil'] = df[col].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_estadocivil', aggfunc= 'count')
cat = df['temp_estadocivil'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)














col = [i for i in df.columns if i.startswith('instlevel')]
df.loc[:, col].sum()




df['temp_instlevel'] = df[col].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_instlevel', aggfunc= 'count')
cat = df['temp_instlevel'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)









col = [i for i in df.columns if i.startswith('lugar')]
df.loc[:, col].sum()




df['temp_lugar'] = df[col].idxmax(axis = 1)




temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_lugar', aggfunc= 'count')
cat = df['temp_lugar'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/cat.T, vmin= 0, vmax= 1, cmap = 'viridis', annot= True)









df[df['parentesco1'] == 1].groupby('Target')['area1'].mean()




df.drop(columns= 'area2', inplace = True)




df['hogar_workingAge'] = df['hogar_adul'] - df['hogar_mayor']
df['hogar_dependent'] = df['hogar_nin'] + df['hogar_mayor']




## df[['hogar_nin', 'hogar_adul','hogar_mayor', 'hogar_workingAge', 'hogar_dependent','dependency']][df['dependency'] == 'no']

df[['hogar_nin', 'hogar_adul','hogar_mayor', 'hogar_workingAge', 'hogar_dependent','dependency']][df['dependency'] == '8']




df['dependency'] = df['dependency'].replace({'yes': 1, 'no': 0}).astype(float)














df['edjefe'] = df['edjefe'].replace({'no': 0, 'yes': 1})
df['edjefa'] = df['edjefa'].replace({'no': 0, 'yes': 1})




df['edjefe'] = df['edjefe'].astype(int)
df['edjefa'] = df['edjefa'].astype(int)




df['median_schooling'] = df['escolari'].groupby(df['idhogar']).transform('median')
df['max_schooling'] = df['escolari'].groupby(df['idhogar']).transform('max')




df['eduForHeadofHH'] = 0
df.loc[(df['parentesco1']== 1), 'eduForHeadofHH'] = df['escolari']




df['eduForHeadofHH'] = df['eduForHeadofHH'].groupby(df['idhogar']).transform('max')




df['SecondaryEduLess'] = ((df[['instlevel1','instlevel2', 'instlevel3', 'instlevel4']] == 1).any(axis = 1)&(df['age'] > 19))*1
df['SecondaryEduMore'] = ((df[['instlevel5','instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']] == 1).any(axis = 1)&(df['age'] > 19))*1




df['MembersWithSecEdu']  = df['SecondaryEduMore'].groupby(df['idhogar']).transform('sum')
df['MembersWithPrimEdu']  = df['SecondaryEduLess'].groupby(df['idhogar']).transform('sum')




df['Educated_Gap'] = (df['MembersWithSecEdu'] - df['MembersWithPrimEdu'])
























df['marital_status'] = (((df['estadocivil3'] ==1)|(df['estadocivil4'] == 1))&(df['parentesco1'] == 1))*1

df['marital_status'] = df['marital_status'].groupby(df['idhogar']).transform('max')




df['FemaleHousehold'] = ((df['male'] == 0)&(df['parentesco1'] == 1))*1
df['FemaleHousehold'] = df['FemaleHousehold'].groupby(df['idhogar']).transform('max')




df['phones_percap'] = df['qmobilephone'] / df['tamviv']
df['tablets_percap'] = df['v18q1'] / df['tamviv']
df['rooms_percap'] = df['rooms'] / df['tamviv']
df['rent_percap'] = df['v2a1'] / df['tamviv']




df['minors_ratio'] = df['hogar_nin']/df['tamviv']
df['elder_ratio'] = df['hogar_mayor']/df['tamviv']




df['child_ratio'] = df['r4t1']/ df['tamviv']
df['malefemale_ratio'] = df['r4h3'] -  df['r4m3'] 




df['ismale_only'] =  (df['r4m3'] == 0)*1
df['isfemale_only'] = (df['r4h3'] == 0)*1
df['no_adultmale'] = (df['r4h2'] == 0)*1
df['no_adultfemale'] = (df['r4m2'] == 0)*1




df['rent_per_room'] = df['v2a1']/df['rooms']
df['bedroom_per_room'] = df['bedrooms']/df['rooms']




df['rent_per_room'] = df['v2a1'] / df['rooms']




df['total_disabled'] = df.groupby('idhogar')['dis'].transform(lambda x: x.sum())




df['average_age'] = df.groupby('idhogar')['age'].transform(lambda x: x.mean())




df['disable_ratio'] = df['total_disabled']/df['tamviv']




df['info_accessibility'] = df[['mobilephone', 'television', 'computer', 'v18q']].any(axis = 1)




df.select_dtypes(include = 'number').columns




df.to_csv('./processed.csv')




df.drop(columns=['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace = True)




training_df = df.select_dtypes(include = 'number')[df['Target'].notnull()]
test_df = df.select_dtypes(include = 'number')[df['Target'].isnull()]




training_df.shape




features = [col for col in training_df.columns if col != 'Target']
X, y = training_df[features], training_df['Target']




test_df.drop(columns = 'Target', inplace = True)




from sklearn.metrics import f1_score, make_scorer, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier




tree = DecisionTreeClassifier(max_features= 75, class_weight='balanced')
tree.fit(X,y)









model = RandomForestClassifier(n_estimators=100, random_state=10, max_features= 75, n_jobs = -1 ,class_weight= 'balanced')
cv_score = cross_val_score(model, X, y, cv = 10, scoring = 'f1_macro')
cv_score.mean()




from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier




ets = []
for i in range(10):
    rf = RandomForestClassifier(random_state=217+i, n_jobs=4, n_estimators=700, min_impurity_decrease=1e-3, min_samples_leaf=2, verbose=0, class_weight= 'balanced')
    ets.append(('rf{}'.format(i), rf)) 




vclf = VotingClassifier(ets, voting= 'soft')




### Score CV results
cv_score = cross_val_score(vclf, X, y, cv= 5, scoring = 'f1_macro')




cv_score.mean()




cv_predict = cross_val_predict(vclf, X, y, cv = 5)




confusion_matrix(y, cv_predict)




f1_score()














vclf = VotingClassifier(ets, voting= 'hard')
cv_score = cross_val_score(vclf, X, y, cv = 5, scoring = 'f1_macro')
cv_score




cv_score.mean()




vclf.fit(X,y)
vclf_hardvoting = vclf.predict(test_df)




len(vclf_hardvoting)




import lightgbm as lgb




##clf = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                                 random_state=None, silent=True, metric='None', 
                                 n_jobs=4, n_estimators=500, class_weight='balanced',
                                 colsample_bytree =  0.89, min_child_samples = 90, num_leaves = 56, subsample = 0.96)




## cv_score = cross_val_score(clf, X, y, cv = 3, scoring = 'f1_macro')
## cv_score









### prediction = [model].predict(test_df)




submit=pd.DataFrame({'Id': df['Id'][df['Target'].isna()] , 'Target': vclf_hardvoting.astype(int)})




submit['Target'].value_counts(normalize = True)




submit.to_csv('./submission.csv', index= False)




##training_df_hhO = df.select_dtypes(include = 'number')[(df['Target'].notnull())&(df['parentesco1'] == 1)]
##test_df_hhO = df.select_dtypes(include = 'number')[(df['Target'].isnull())&(df['parentesco1'] == 1)]




##features = [col for col in training_df.columns if col != 'Target']
##X_hhO, y_hhO = training_df_hhO[features], training_df_hhO['Target']




##cv_score = cross_val_score(vclf, X_hhO, y_hhO, cv = 5, scoring = 'f1_macro')




##cv_score

array([0.4596173 , 0.42617731, 0.38660971, 0.35653499, 0.37714824])  
array([0.46729109, 0.42841538, 0.38395209, 0.35942352, 0.37480922])


## np.array([0.4596173 , 0.42617731, 0.38660971, 0.35653499, 0.37714824]).mean()






