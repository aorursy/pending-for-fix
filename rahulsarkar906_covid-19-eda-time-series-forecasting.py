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




train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
test = pd.read_csv(r'../input/covid19-global-forecasting-week-1/test.csv')
train.head()




train.describe()




test.head()




# rename columns
train = train.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region'})
test = test.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region'})




train['Date'].max(), test['Date'].min()




train.head()




test.head()




# Remove the overlapping train and test data

valid = train[train['Date'] >= test['Date'].min()] # set as validation data
train = train[train['Date'] < test['Date'].min()]
train.shape, valid.shape




# Standard plotly imports
#import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import iplot, init_notebook_mode, plot
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)




train_total = train[['Country_Region','Province_State','ConfirmedCases','Fatalities']]
train_total['Province_State'] = train_total['Province_State'].fillna(train_total['Country_Region']) # replace NaN States with country name
train_total = train_total.groupby(['Country_Region','Province_State'],as_index=False).agg({'ConfirmedCases': 'max', 'Fatalities': 'max'})




# pio.renderers.default = 'vscode'
pio.renderers.default = 'kaggle'

fig = px.treemap(train_total.sort_values(by='ConfirmedCases', ascending=False).reset_index(drop=True), 
                 path=["Country_Region", "Province_State"], values="ConfirmedCases", height=600, width=800,
                 title='Number of Confirmed Cases',
                 color_discrete_sequence = px.colors.qualitative.Prism)
fig.data[0].textinfo = 'label+text+value'
fig.show()

fig = px.treemap(train_total.sort_values(by='Fatalities', ascending=False).reset_index(drop=True), 
                 path=["Country_Region", "Province_State"], values="Fatalities", height=600, width=800,
                 title='Number of Deaths',
                 color_discrete_sequence = px.colors.qualitative.Prism)
fig.data[0].textinfo = 'label+text+value'
fig.show()




# Sum countries with states, not dealing with states for now
train_agg= train[['Country_Region','Date','ConfirmedCases','Fatalities']].groupby(['Country_Region','Date'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum'})

# change to datetime format
train_agg['Date'] = pd.to_datetime(train_agg['Date'])




pip install pycountry




pip install pycountry-convert




#import pycountry_convert as pc
import pycountry
# function for getting the iso code through fuzzy search
def do_fuzzy_search(country):
    try:
        result = pycountry.countries.search_fuzzy(country)
    except Exception:
        return np.nan
    else:
        return result[0].alpha_2

train_continent = train_agg
# manually change name of some countries
train_continent.loc[train_continent['Country_Region'] == 'Korea, South', 'Country_Region'] = 'Korea, Republic of'
train_continent.loc[train_continent['Country_Region'] == 'Taiwan*', 'Country_Region'] = 'Taiwan'
# create iso mapping for countries in df
iso_map = {country: do_fuzzy_search(country) for country in train_continent['Country_Region'].unique()}
# apply the mapping to df
train_continent['iso'] = train_continent['Country_Region'].map(iso_map)
#train_continent['Continent'] = [pc.country_alpha2_to_continent_code(iso) for iso in train_continent['iso']]




def alpha2_to_continent(iso):
    try: cont = pc.country_alpha2_to_continent_code(iso)
    except: cont = float('NaN')
    return cont

train_continent['Continent'] = train_continent['iso'].apply(alpha2_to_continent) # get continent code
train_continent.loc[train_continent['iso'] == 'CN', 'Continent'] = 'CN' # Replace China's continent value as we want to keep it separate

train_continent = train_continent[['Continent','Date','ConfirmedCases','Fatalities']].groupby(['Continent','Date'],as_index=False).agg({'ConfirmedCases':'sum','Fatalities':'sum'})
train_continent['Continent'] = train_continent['Continent'].map({'AF':'Africa','AS':'Asia','CN':'China','EU':'Europe','NA':'North America','OC':'Oceania','SA':'South America'})




long = pd.melt(train_continent, id_vars=['Continent','Date'], value_vars=['ConfirmedCases','Fatalities'], var_name='Case', value_name='Count').sort_values(['Date','Count'])
long['Date'] = long['Date'].astype('str')




pio.renderers.default = 'kaggle' # does not work on vscode

# color palette
cnf = '#393e46' # confirmed - grey
dth = '#ff2e63' # death - red
# rec = '#21bf73' # recovered - cyan
# act = '#fe9801' # active case - yellow

fig = px.bar(long, y='Continent', x='Count', color='Case', barmode='group', orientation='h', text='Count', title='Counts by Continent', animation_frame='Date',
             color_discrete_sequence= [dth,cnf], range_x=[0, 100000])
fig.update_traces(textposition='outside')




# Interactive time series plot of confirmed cases
fig = px.line(train_agg, x='Date', y='ConfirmedCases', color="Country_Region", hover_name="Country_Region")
fig.update_layout(autosize=False,width=1000,height=500,title='Confirmed Cases Over Time for Each Country')
fig.show()




# Interactive time series plot of fatalities
fig = px.line(train_agg, x='Date', y='Fatalities', color="Country_Region", hover_name="Country_Region")
fig.update_layout(autosize=False,width=1000,height=500,title='Fatalities Over Time for Each Country')
fig.show()




## Load Natural Earth Map Data

import geopandas as gpd # for reading vector-based spatial data format
#shapefile = '.../input/natural-earth-maps/ne_110m_admin_0_countries.shp'
#shapefile = r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\110m_cultural\ne_110m_admin_0_countries.shp'

# Read shapefile using Geopandas
#gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
#gdf = gpd.read_file(shapefile)

# Drop row corresponding to 'Antarctica'
#gdf = gdf.drop(gdf.index[159])




## Get the ISO 3166-1 alpha-3 Country Codes

import pycountry
# function for getting the iso code through fuzzy search
def do_fuzzy_search(country):
    try:
        result = pycountry.countries.search_fuzzy(country)
    except Exception:
        return np.nan
    else:
        return result[0].alpha_3

# manually change name of some countries
train_agg.loc[train_agg['Country_Region'] == 'Korea, South', 'Country_Region'] = 'Korea, Republic of'
train_agg.loc[train_agg['Country_Region'] == 'Taiwan*', 'Country_Region'] = 'Taiwan'
# create iso mapping for countries in df
iso_map = {country: do_fuzzy_search(country) for country in train_agg['Country_Region'].unique()}
# apply the mapping to df
train_agg['iso'] = train_agg['Country_Region'].map(iso_map)




# countries with no iso
noiso = train_agg[train_agg['iso'].isna()]['Country_Region'].unique()
# get other iso from natural earth data, create the mapping and add to our old mapping
#otheriso = gdf[gdf['SOVEREIGNT'].isin(noiso)][['SOVEREIGNT','SOV_A3']]
#otheriso = dict(zip(otheriso.SOVEREIGNT, otheriso.SOV_A3))
#iso_map.update(otheriso)




# apply mapping and find countries with no iso again
train_agg['iso'] = train_agg['Country_Region'].map(iso_map)
train_agg[train_agg['iso'].isna()]['Country_Region'].unique()




# change date to string, not sure why plotly cannot accept datetime format
train_agg['Date'] = train_agg['Date'].dt.strftime('%Y-%m-%d')




# apply log10 so that color changes are more prominent
import numpy as np
train_agg['ConfirmedCases_log10'] = np.log10(train_agg['ConfirmedCases']).replace(-np.inf, 0) # log10 changes 0 to -inf so change back




# Interactive Map of Confirmed Cases Over Time

#pio.renderers.default = 'browser' # does not work on vscode
pio.renderers.default = 'kaggle'
fig = px.choropleth(train_agg, locations='iso', color='ConfirmedCases_log10', hover_name='Country_Region', animation_frame='Date', color_continuous_scale='reds')
fig.show()




pip install fbprophet




from fbprophet import Prophet




train.columns




train.dtypes




train.query('Country_Region=="India"').groupby("Date")[['ConfirmedCases', 'Fatalities']].sum().reset_index()




train.groupby("Country_Region")[['ConfirmedCases', 'Fatalities']].sum().reset_index()




confirmed = train.query('Country_Region=="India"').groupby("Date")[['ConfirmedCases']].sum().reset_index()
#confirmed.columns=['ds','y']
#confirmed['Fatalities'] = confirmed['Fatalities']
confirmed




confirmed['Fatalities'] = train['Fatalities']




confirmed['y'] = confirmed['ConfirmedCases']




confirmed['ds']= confirmed['Date']




#confirmed.columns=['ds','y','Fatalities']
confirmed.drop(['ConfirmedCases', 'Date'], axis=1, inplace=True)
confirmed




#train.set_index('Date')




import fbprophet




from fbprophet import Prophet
m = Prophet()
m.add_regressor('Fatalities')
m.fit(confirmed)




future = m.make_future_dataframe(periods=60)
future.tail()




#forecast = m.predict(future)
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper','Fatalities']].tail()
forecast = m.predict(confirmed.drop(columns="y"))




fig1 = m.plot(forecast)




confirmed_test = test.query('Country_Region=="India"').groupby("Date")[['ForecastId']].sum().reset_index()
#confirmed_test.columns=['ds','y']
#confirmed['ds'] = confirmed['ds'].Date
confirmed_test




confirmed_test['Fatalities'] =  np.nan
confirmed_test['Fatalities'].astype('float')
confirmed_test




confirmed_test['y'] = confirmed_test['ForecastId']
confirmed_test['ds']= confirmed_test['Date']
confirmed_test.drop(['Date'], axis=1, inplace=True)
confirmed_test




#test_forecast['Fatalities'].astype('float')
#test_forecast = m.predict(confirmed_test)
test_forecast.reset_index(drop=True)
test_forecast




confirmed_test['Fatalities'] = 0.0




test_forecast = m.predict(confirmed_test)




test_forecast.head()




test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()




test_forecastplot = m.plot(test_forecast)




test_forecast.head()




test_forecast.dtypes




#submission_forecast = test_forecast['ForecastId','yhat']




#test_forecast.to_csv('prophet_sub.csv',index = False)
#test_forecast.groupby("ForecastID")[['multiplicative_terms', 'multiplicative_terms_upper']].sum().reset_index()
#test_forecast.drop(['yhat','yhat_lower','yhat_upper'],axis = 1, inplace = True)
#submission_forecast = test_forecast[['ForecastId','ConfirmedCases','Fatalities']]
#test_forecast['ConfirmedCases'] = test_forecast['yhat']
#test_forecast['Fatalities']= test_forecast['yhat_lower']
#test_forecast.drop(['yhat','yhat_lower','yhat_upper'],axis = 1)
test_forecast.to_csv('submission.csv',index = False)




fig2 = m.plot_components(test_forecast)




# Python
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, test_forecast)  # This returns a plotly Figure
py.iplot(fig)

