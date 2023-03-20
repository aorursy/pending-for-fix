#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from scipy.stats import skew
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




teams = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeams.csv')




def get_team_dict(df):
    group = df.groupby(['TeamID','TeamName']).size().reset_index()
    team_ids = list(group['TeamID'])
    team_names = list(group['TeamName'])
    return dict((x,y) for x,y in list(zip(team_ids, team_names)))

team_dict = get_team_dict(teams)




players = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MPlayers.csv')




def get_player_dict(df):
    lol = players.groupby(['PlayerID','LastName','FirstName']).size().reset_index()
    lol['Name'] = lol['FirstName'] + ['_']*lol.shape[0] + lol['LastName']
    player_ids = list(lol['PlayerID'])
    player_names = list(lol['Name'])
    return dict((x,y) for x,y in list(zip(player_ids, player_names)))




results = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')
results = results.drop('WLoc', axis = 1) #useless col (1115 'N' values)




def plot_kde(results, col):
    
    win_col = 'W' + col
    lost_col = "L" + col
    
    plt.figure(figsize = (12,5))
    plt.title("Team avg per match " + col)
    sns.kdeplot(results[win_col])
    sns.kdeplot(results[lost_col])
    
w_l_cols = [re.sub('(L|W){1}', "", i) for i in results.columns if re.match('[LW]{1}', i)]
stat_cols = list(set(w_l_cols))
stat_cols.remove('TeamID')

for col in stat_cols:
    plot_kde(results, col)




def plot_kde(results, col):
    
    win_col = 'W' + col
    lost_col = "L" + col
    
    plt.figure(figsize = (12,5))
    sns.kdeplot(results[win_col])
    sns.kdeplot(results[lost_col])




team_dict = get_team_dict(teams)
results['WTeam'] = results.WTeamID.map(lambda x:team_dict[x])
results['LTeam'] = results.LTeamID.map(lambda x:team_dict[x])




def basic_eda(train):
    tr_nulls = train.isnull().sum()
    for col in train.columns[1:]:
        if train[col].dtype != object:
            plt.figure(figsize = (12,4))
            plt.title('%s has %s nulls and %s nunique, %s dtype, %s skew' %(col,tr_nulls[col], train[col].nunique(), train[col].dtype, skew(train[col])))
            sns.distplot(train[col].dropna())
        if train[col].dtype == object:
            print ('%s has %s nuniq' %(col, train[col].nunique()))
            print (train[col].unique()[:10])




ev_16 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2016.csv')
ev_17 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2017.csv')
ev_18 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2018.csv')
ev_19 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2019.csv')
players = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MPlayers.csv')




ev_full = pd.concat([ev_16, ev_17, ev_18, ev_19], ignore_index = True)
del ev_16, ev_17, ev_18, ev_19




ev_full['win_bool'] = ev_full['WTeamID'] - ev_full['EventTeamID']
ev_full['win_bool'] = ev_full['win_bool'].map(lambda x: 1 if x == 0 else 0)

ev_full['event_name'] = ev_full['EventType'] + ['_'] * ev_full.shape[0] + ev_full['EventSubType']

ev_full.loc[ev_full['event_name'] == 'jumpb_start', 'EventTeamID'] = 1

#np.where to replace event_names whose EventTeamID == 0
a = np.array(['id0_error'] * ev_full.shape[0]) #error arr
b = np.array(list(ev_full['EventTeamID']))
c = np.array(list(ev_full['event_name']))
ev_full['event_name'] = np.where(b == 0, a, c)

#ignore player id 0 when eda player stats




ev_full = ev_full.sort_values(['Season','DayNum','WTeamID','WCurrentScore','ElapsedSeconds']).reset_index().drop('index', axis = 1)




val_cnts = ev_full['event_name'].value_counts()




val_cnts[-30:-20]









val_cnts[:45]




win = ev_full[ev_full.win_bool == 1].groupby(['Season',"DayNum", "WTeamID", "event_name"]).size()
lost = ev_full[ev_full.win_bool == 0].groupby(['Season',"DayNum", "WTeamID", "event_name"]).size()




lol = ev_full.iloc[7897547:7897557, :]
lol.head(20)




lol = ev_full.iloc[7904503:7904509, :]
lol.head(20)




ev_full[ev_full.event_name == "miss1_1of3"]





evname = 'made2_stepb'
plt.figure(figsize = (15,4))




for event in ['s':
    if match_group[match_group.event_name == event]['ev_count'].nunique() > 10:
    
        plt.figure(figsize = (10,4))
        plt.title(event)
        win_ax = sns.kdeplot(match_group[match_group.win_bool == 0][match_group.event_name == event].rename(columns = {'ev_count':'lost'})['lost'].dropna())
    
        lost_ax = sns.kdeplot(match_group[match_group.win_bool == 1][match_group.event_name == event].rename(columns = {'ev_count':'won'})['won'].dropna()) 




ev_full[ev_full.event_name == 'jumpb_won']




ev_name = 'jumpb_won'

lol_win = ev_full[ev_full.win_bool == 1].groupby(['Season','DayNum','WTeamID','event_name']).size().reset_index()
lol_lost = ev_full[ev_full.win_bool == 0].groupby(['Season','DayNum','WTeamID','event_name']).size().reset_index()
lol_win.columns = ['Season', 'DayNum', 'WTeamID', 'event_name', 'win_count']
lol_lost.columns = ['Season', 'DayNum', 'WTeamID', 'event_name', 'lost_count']




for i in lol_win.event_name.unique():
    if 'jumpb' in i:
        print (i)




ev_name = 'reb_def'




lol_lost[lol_lost.event_name == ev_name]['lost_count'].value_counts()




lol_win[lol_win.event_name == ev_name]['win_count'].value_counts()









ev_name = 'made2_stepb'

plt.figure(figsize = (15,6))
sns.distplot(lol_win[lol_win.event_name == ev_name]['win_count'])
sns.plot(lol_lost[lol_lost.event_name == ev_name]['lost_count'])




year = 2017
count_num = 15

lol_win = ev_full[ev_full.Season == year][ev_full.win_bool == 1].groupby(['DayNum','WTeamID','event_name'])size().reset_index()
lol_lost = ev_full[ev_full.Season == year][ev_full.win_bool == 0].groupby(['DayNum','WTeamID','event_name']).size().reset_index()

win1 = lol_win.groupby('event_name')['count'].mean().sort_values(ascending = False).reset_index()
lost1 = lol_lost.groupby('event_name')['count'].mean().sort_values(ascending = False).reset_index()

win_top = win1.head(count_num)
lost_top = lost1.head(count_num)

plt.figure(figsize = (20,6))

plt.subplot(1,2,1)
plt.title('win top %s of %s' %(count_num, year))
sns.barplot(x = 'event_name', y = 'count', data = win_top)
plt.xticks(rotation = 45)

plt.subplot(1,2,2)
plt.title('lost top %s of %s' %(count_num, year))
sns.barplot(x = 'event_name', y = 'count', data = lost_top)
plt.xticks(rotation = 45)




full_diff_df = pd.DataFrame({'event_name':list(ev_full.event_name.unique())})


for year in [2016,2017,2018,2019]:

    lol_win = ev_full[ev_full.Season == year][ev_full.win_bool == 1].groupby(['DayNum','WTeamID','event_name'])['ElapsedSeconds'].agg(['count','mean']).reset_index()
    lol_lost = ev_full[ev_full.Season == year][ev_full.win_bool == 0].groupby(['DayNum','WTeamID','event_name'])['ElapsedSeconds'].agg(['count','mean']).reset_index()

    win1 = lol_win.groupby('event_name')['count'].mean().sort_values(ascending = False).reset_index()
    win1.columns = ['event_name', 'win_count']
    lost1 = lol_lost.groupby('event_name')['count'].mean().sort_values(ascending = False).reset_index()
    lost1.columns = ['event_name','lost_count']

    diff_df = pd.merge(win1, lost1, how = 'left', on = 'event_name')
    diff_df['diff'] = diff_df['win_count'] / diff_df['lost_count']
    diff_df = diff_df[['event_name','diff']]
    diff_df.columns = ['event_name', 'diff_' + str(year)]
    
    full_diff_df = pd.merge(full_diff_df, diff_df, how = 'left', on = 'event_name')




event_names = full_diff_df.sort_values('diff_2017').head(23).event_name.unique()




match_group = ev_full.groupby(['Season','DayNum','EventTeamID','event_name','win_bool'])['ElapsedSeconds'].count().reset_index()




ev_full.columns




match_group.columns = ['Season', 'DayNum', 'EventTeamID', 'event_name', 'win_bool',
       'ev_count']




match_group.head()




for event in event_names:
    if match_group[match_group.event_name == event]['ev_count'].nunique() > 20:
    
        plt.figure(figsize = (10,4))
        plt.title(event)
        win_ax = sns.kdeplot(match_group[match_group.win_bool == 0][match_group.event_name == event].rename(columns = {'ev_count':'lost'})['lost'].dropna())
    
        lost_ax = sns.kdeplot(match_group[match_group.win_bool == 1][match_group.event_name == event].rename(columns = {'ev_count':'won'})['won'].dropna())
    
    
    
    




match_group = ev_full.groupby(['Season','DayNum','WTeamID','LTeamID'])['']




#all of the nan event_names only exists in 2019
lol_list = list(lol[lol.diff_2016.isnull()].event_name)
lol_list.remove(np.nan)
ev_full[ev_full.event_name.isin(lol_list)].Season.unique()




ev_full.isnull().sum()




ev_full.EventSubType.unique()




ev_full[ev_full.EventSubType == 'coate'].DayNum.unique()




ev_names = list(ev_full.event_name.unique())




ev_full.event_name.nunique()




diff_df['diff'] = diff_df['win_count'] / diff_df['lost_count']




pd.DataFrame({'event_name':list(ev_full.event_name.unique())})




diff_df.sort_values('diff')




lol_top = lol1.head(10)
plt.figure()
sns.barplot(x = 'event_name', y = 'count', data = lol_top)
plt.xticks(rotation = 45)




lol1.head(1)




lol.tail(3)




for season in ev_full.Season.unique():
    lol = ev_full[ev_full.Season == 2017].groupby(['event_name'])['win_bool'].agg(['count','mean']).sort_values('mean').reset_index()
    lol['name_count'] = lol['event_name'] + ['_'].shape[0] + lol['count'].astype('str')
    top10 = lol.tail(10)
    bottom1




sample_df = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')




sample_df






