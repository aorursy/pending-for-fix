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




df_all = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2020/train.csv")




df_all.head()




df_all.columns




df_all[['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'A',
        'Dis', 'Orientation', 'Dir', 'NflId', 'DisplayName', 'JerseyNumber']].head()




df_all[['Season', 'YardLine', 'Quarter', 'GameClock', 
        'PossessionTeam', 'Down', 'Distance', 'FieldPosition']].head()




df_all[['HomeScoreBeforePlay', 'VisitorScoreBeforePlay',
       'NflIdRusher', 'OffenseFormation', 'OffensePersonnel',
       'DefendersInTheBox', 'DefensePersonnel']].head()




df_all[['PlayDirection', 'TimeHandoff','TimeSnap', 'Yards']].head()




df_all[['PlayerHeight', 'PlayerWeight', 'PlayerBirthDate',
       'PlayerCollegeName', 'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr']].head()




df_all[['Week', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather',
       'Temperature', 'Humidity', 'WindSpeed', 'WindDirection']].head()




# 選手一人毎の情報がある列
# 他の列は同一PlayId内では同じ
personal_columms = ['X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 
                    'NflId', 'DisplayName', 'JerseyNumber', 
                   'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate','PlayerCollegeName', 'Position']




df_all.shape




pd.set_option("display.max_columns", 80)




df_all.describe()




import matplotlib.pyplot as plt




plt.tight_layout() 

df_all.hist()
#plt.show()




df_all.columns




# ボールを持っている選手の行のみを選択
df_play = df_all[df_all["NflId"]==df_all["NflIdRusher"]]




df_play.tail()




df_play["Position"].value_counts()




df_play[df_play["Position"].isin(["DT", "DE", "CB"])]




personal_columms




import seaborn as sns




sns.jointplot('A', 'Yards', data=df_play)




sns.jointplot('YardLine', 'Yards', data=df_play[df_play["PossessionTeam"] == df_play["FieldPosition"]])




sns.jointplot('YardLine', 'Yards', data=df_play[df_play["PossessionTeam"] != df_play["FieldPosition"]])




sns.jointplot('X', 'Yards', data=df_all[df_all["Position"]=="HB"])




df_play["Yards"].hist(range=(-20,30),bins=50)




df_play["Yards"]




from math import erf




erf(1)




df_all["Position"].unique()




# オフェンスチーム
# QB
# RB（FB、HB、TB）
# WR（SE、FL、SB、WB）
# TE
# OL（C、G、T、E）
# ディフェンスチーム
# DL（DT（NT）、DE）
# LB（ILB（MLB）、OLB（LOLB、ROLB、SLB、WLB））
# DB（CB、S（SS、FS））
# スペシャルチーム
# K、P、LS、H、KR/PR

offence_position = ['WR', 'TE', 'T', 'QB', 'RB', 'G', 'C', 'FB', 'HB',  'OT', 'OG', ]
#OL_position = ['T', 'G', 'C', 'OT', 'OG']
#RB_position = ['RB', 'FB', 'HB']

defence_position = ['SS', 'DE', 'ILB', 'FS', 'CB', 'DT', 'OLB', 'NT', 'MLB', 'LB', 'S', 'DL', 'DB', 'SAF']
#DL_position = ['DL', 'DT', 'DE']
#LB_position = ['LB', 'ILB', ]




import math




df_all["offence"] = 0
df_all.loc[df_all["Position"].isin(offence_position), "offence"] = 1




df_all.head(23)[["Team", "Position", "offence"]]




position_count=df_all.groupby(["PlayId", "Position"]).count()
position_count




df_position = position_count["GameId"].unstack().fillna(0).astype(int)
df_position




pd.merge(df_play, df_position, on="PlayId").corr()["Yards"].sort_values()




df_play




df_play.loc[:, "OffensePersonnel"] = df_play["OffensePersonnel"].apply(
    lambda x : { i.split(" ")[-1]:int(i.split(" ")[-2]) for i in x.split(",")} )




df_play.loc[:, "DefensePersonnel"] = df_play["DefensePersonnel"].apply(
    lambda x : { i.split(" ")[-1]:int(i.split(" ")[-2]) for i in x.split(",")} )




for position in ["DL", "LB", "DB"]:
    df_play.loc[:, position] = [ d[position] for d in df_play["DefensePersonnel"]]
for position in ["RB", "TE", "WR"]:
    df_play.loc[:, position] = [ d[position] for d in df_play["OffensePersonnel"]]    




df_play




df_play.corr()["Yards"].sort_values()




df_play[df_play["Team"]=="home"].groupby("HomeTeamAbbr").mean()["Yards"].sort_values()




team_home = df_play[df_play["Team"]=="home"].groupby("HomeTeamAbbr")




df_play["HomeTeamAbbr"].unique()




team_yards = dict(list(df_play[df_play["Team"]=="home"].groupby("HomeTeamAbbr")["Yards"]))




plt.hist(team_yards["CAR"], bins=30)
plt.hist(team_yards["WAS"], bins=30)




dict




df_play[df_play["Team"]=="home"].groupby("HomeTeamAbbr").mean()["Yards"].hist()




df_play[df_play["Team"]=="away"].groupby("VisitorTeamAbbr").mean()["Yards"].hist()




np.array([[1,2,2,5]]).mean()




df["OffenseFormation"].unique()




df["OffensePersonnel"].unique()




df["StadiumType"]




df["stadiumsype"] = df["StadiumType"].isin()




df["PlayDirection"].unique()




personal_yards = df_play[["NflId", "Yards"]].groupby("NflId").agg(["mean", "std", "max", "min", "count"])["Yards"]
personal_yards.dropna(inplace=True)




personal_yards




df_play = df_play.merge(personal_yards, on="NflId", how="left")




df_play.corr()["Yards"].sort_values()




len(df_play.index)




df_play["Yards"].value_counts().head(30)




pd.




df_play[df_play["Yards"]>50].sort_values("Yards")[["YardLine", "Yards"]]









len(df_play[df_play["Yards"]+df_play["YardLine"]==100].index)




df_play[df_play["Yards"]>20].corr()["Yards"].sort_values()




df_play.loc[:, "over20"] = (df_play["Yards"]>20)*1




df_play.loc[:, "touchdown"] = (df_play["Yards"]+df_play["YardLine"]==100)




(df_play["Yards"]+df_play["YardLine"]==100)




df_play[["over20", "touchdown"]]




df_play.corr()["touchdown"].sort_values()




df_play.corr()["over20"].sort_values()






