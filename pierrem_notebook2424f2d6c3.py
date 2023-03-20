#!/usr/bin/env python
# coding: utf-8



import pandas as pd




train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

train.head(10)




headers = train.columns.values
print(headers)

def count_occ(cat, headers):
    print("Number of ", cat, ": ", len([h for h in ]))
print("Number of cont:")






