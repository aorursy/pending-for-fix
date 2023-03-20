#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np




data = pd.read_csv(".data/sensors,smooth.csv.gz")




data




from tsfresh import extract_features
# extracted_features = extract_features(timeseries, column_id="id", column_sort="time")




features = extract_features(data.drop(columns = "AC"), column_id = "FLIGHT", column_sort = "TIME")




from tsfresh.utilities.dataframe_functions import impute




impute(features)




features.to_csv(".data/sensors,features.csv")




.data/sensors,smooth.csv.gz

