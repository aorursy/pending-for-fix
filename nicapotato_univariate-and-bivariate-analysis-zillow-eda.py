#!/usr/bin/env python
# coding: utf-8



# General
import numpy as np
import pandas as pd
import os

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




# Load
properties = pd.read_csv("../input/properties_2016.csv", index_col="parcelid")
transactions = pd.read_csv("../input/train_2016_v2.csv",parse_dates=["transactiondate"])




transactions.head()




# Figure Size
plt.rcParams['figure.figsize'] = (10, 5)
# Density Plot
ax = sns.kdeplot(transactions["logerror"], shade=True)
ax.set_xlabel("Log Error")
ax.set_ylabel("Density")
ax.set_title("Log Error Density")

ax.legend_.remove()
ax.axvline(x=0, color='r', linestyle='-')
plt.show()




# Parcing - http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
def mean_count_plot(data, by, title):
    t1= data[["transactiondate","logerror"]].resample(by,on='transactiondate').agg({"mean","count"})
    t1.columns = t1.columns.droplevel()
    ax = t1.plot(secondary_y=["mean"])
    ax.set_title("Transaction Count and Mean by {}".format(title))




for x,y in [("7D","Seven Days"),("15D","Fifteen Days"),("M","Month"),("3M","Three Months")]:
    mean_count_plot(transactions, by=x, title=y)




# By Day in Month
ax = transactions[["logerror","transactiondate"]].groupby(
    [transactions['transactiondate'].dt.day ]).mean().plot()
ax.set_ylabel("Log Error")
ax.set_xlabel("Day in the Month")
ax.legend_.remove()
ax.set_title("Average Log Error by Day in the Month")
plt.show()

# By Weekday
ax = transactions[["logerror","transactiondate"]].groupby(
    [transactions['transactiondate'].dt.weekday ]).mean().plot()
ax.set_ylabel("Log Error")
ax.set_xlabel("Week Day")
ax.legend_.remove()
ax.set_title("Average Log Error by Weekday")
plt.show()




print("Shape (Heigth, Width): \n{}".format(properties.shape))
print("\nColumn Data Types:")
properties.dtypes.value_counts()




missing= (properties.isnull().sum()/properties.shape[0]*100).sort_values(ascending=False).reset_index()
missing.columns = ["Column Name","Percent Missing"]




pd.concat([missing[:29], missing[29:].reset_index(drop=True)], axis=1)




# Remove Highly Missing Columns
high_miss = missing.loc[missing["Percent Missing"] >34,["Column Name"]]
properties.drop([item for sublist in high_miss.values for item in sublist], axis=1, inplace=True)




# Impute
#for col in (x for x in missing["Column Name"] if x is not high_miss):
for col in properties.columns.values:
    if properties[col].dtypes == float:
        properties[col] = properties[col].fillna(properties[col].mean())
        properties[col] = properties[col].astype(int)
    elif properties[col].dtypes == object:
        properties[col] = properties[col].fillna(properties[col].mode().iloc[0])
    elif properties[col].dtypes == int:
        properties[col] = properties[col].fillna(properties[col].median())




# Excluding the obj_columns
obj_cols = properties.loc[:,properties.dtypes == object].columns.values
plot_worthy = [x for x in properties.columns.values if x not in obj_cols]

# Number of Unique values by Column
uniqlen = {}
for x in plot_worthy:
    temp = len(properties[x].unique())
    uniqlen[x] = temp

# Columns for Bar Plotting
bar_cols = []
for col,val in uniqlen.items():
    if val < 40: bar_cols.append(col)

# Columns for KDE plotting
hist_cols = [x for x in plot_worthy if x not in bar_cols]




print("Missing Values? -> {}".format(properties.isnull().values.any()))
print("\nColumn Data Types:")
properties.dtypes.value_counts()




non_log = ["latitude","longitude","yearbuilt"]
log_hist = [x for x in hist_cols if x not in non_log]




grid = sns.FacetGrid(pd.melt(np.log10(properties.loc[:,log_hist]), value_vars=log_hist),
    col="variable",  col_wrap=3 , size=5.0, aspect=0.8, sharex=False, sharey=False)
grid.map(sns.distplot, "value")
plt.show()




grid = sns.FacetGrid(pd.melt(properties.loc[:,non_log], value_vars=non_log), col="variable",
                     col_wrap=5 , size=4.0, aspect=0.8,sharex=False, sharey=False)
grid.map(sns.distplot, "value")
plt.show()




# Not Worth Plotting Object Types
print(obj_cols)
for x in obj_cols:
    print(len(properties[x].unique()))
properties.loc[:,obj_cols].sample(5)




df = pd.merge(transactions,
   properties.loc[:,log_hist].reset_index(),on="parcelid",how="left")




# Log
log_df = pd.merge(transactions, np.log10(properties.loc[:,log_hist]).reset_index(),
                on="parcelid",how="right")
# Melt
melt_log_df = pd.melt(log_df, id_vars="logerror",value_vars=log_hist)

# New Variable to see under/over predictions
melt_log_df["Over/Under"] = melt_log_df.logerror >= 0




grid = sns.FacetGrid(melt_log_df, hue="Over/Under"
    ,col="variable",  col_wrap=5 , size=4.0,aspect=0.8,sharex=False, sharey=False)
grid.map(plt.scatter, "value", "logerror")
plt.show()




log_df["Over/Under"] = log_df.logerror >= 0
log_df.set_index("parcelid", inplace=True)




print("x")




plt.savefig("seaborn_pairplot.png")




## Explore Correlations

