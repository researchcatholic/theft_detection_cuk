# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 09:06:03 2018

@author: p0p
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics 
from sklearn import preprocessing

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

from datetime import date,timedelta
import datetime
from dateutil.rrule import rrule, DAILY

# from statsmodels.tsa.seasonal import seasonal_decompose
# from PyAstronomy import pyasl
# import keras
# from keras.layers import Input, Dense
# from keras.models import Model,Sequential
# from keras.callbacks import TensorBoard
# from keras.models import load_model
# from tensorflow import set_random_seed
import itertools
from itertools import chain, repeat
import os

data_1 = pd.read_csv("C:/Users/p0p/Desktop/anaomaly/1year_Example-1.csv")
data_2 = pd.read_csv("C:/Users/p0p/Desktop/anaomaly/1year_Example-2.csv")
data_3 = pd.read_csv("C:/Users/p0p/Desktop/anaomaly/1year_Example-3.csv")
data_4 = pd.read_csv("C:/Users/p0p/Desktop/anaomaly/1year_Example-4.csv")

anomalies_1 = ['2012-09-10','2012-09-11','2012-09-12','2012-09-15','2012-09-16','2012-10-30','2012-11-16','2012-11-30','2012-12-02','2012-12-03','2012-12-04','2012-12-22']
anomalies_2 = ['2011-03-04','2011-03-06','2011-04-17','2011-04-22','2011-05-06','2011-05-19','2011-05-20','2011-05-22','2011-05-23','2011-05-24']
anomalies_3 = ['2014-01-17','2014-01-29','2014-02-26','2014-04-04','2014-04-05','2014-04-11','2014-04-12','2014-04-22','2014-04-30','2014-05-12','2014-05-13','2014-05-26']
anomalies_4 = ['2014-05-01','2014-06-10','2014-06-17','2014-07-08','2014-07-10','2014-09-02','2014-10-17','2014-10-23','2014-10-24']

def preprocess(data_ ):
    print("processing dataset ")
    data_['time'] = pd.to_datetime(data_.time)
    data_['date'] = pd.DatetimeIndex(data_.time).normalize()
    data_ = data_.set_index('time') 
    del data_['example']
    print("does data set have null value? :", data_['value'].isnull().any())

    print("deleting days with null values")
    nanlist = data_[data_['value'].isnull()].date.tolist()
    nanlist = list(map(lambda x: x.date(), nanlist))
    nanlist = list(set(nanlist))
    # nanlist

    data = data_.copy()
    for n in nanlist:
        data = data[data.date != n.strftime("%Y-%m-%d")]

    # data['value'] = data['value'].fillna(0)
    print("null status :", data['value'].isnull().any())
    del data['date']

    plt.figure(figsize=(90,3))
    plt.plot(data['value'])
    plt.title("plot without nulls")
    plt.show()
    # a.strftime("%Y-%m-%d")
    return data

def createtraintest(data,anomalies,file_no):
    start_date = data.head(1).index.date[0]
    end_date = data.tail(1).index.date[0]
    print('first date :%s and last date :%s '%(start_date.strftime("%Y-%m-%d"),end_date.strftime("%Y-%m-%d"))) 
    middle_date = start_date + (end_date-start_date)/2
    #takes in the entire dataset, anomalies during training period
#     print(data.head(5))
    if file_no == 1:
        print("file no", file_no)
        train_data = data[middle_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")].copy()
        # train_data =  train_data[train_data.index.weekday < 6]
        mix_data = train_data.copy()
        mask = ~np.in1d(train_data.index.date, pd.to_datetime(anomalies).date)
        anomaly_data = train_data.loc[~mask,:]
        train_data = train_data.loc[mask, :]
        test_data = data[start_date.strftime("%Y-%m-%d"):middle_date.strftime("%Y-%m-%d")].copy()
        # test_data =  test_data[test_data.index.weekday < 6]
    elif file_no == 3:
        print("file no", file_no)
#         print("ssss11")
        train_data = data[start_date.strftime("%Y-%m-%d"):middle_date.strftime("%Y-%m-%d")].copy()
        train_data =  train_data[train_data.index.weekday < 6]
        mix_data = train_data.copy()
        mask = ~np.in1d(train_data.index.date, pd.to_datetime(anomalies).date)
        anomaly_data = train_data.loc[~mask,:]
        train_data = train_data.loc[mask, :]
        test_data = data[middle_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")].copy()
        test_data =  test_data[test_data.index.weekday < 6]
    elif file_no == 4:
        print("file no", file_no)
        train_data = data[start_date.strftime("%Y-%m-%d"):middle_date.strftime("%Y-%m-%d")].copy()
#         print(train_data.head(5))
        mix_data = train_data.copy()
        mask = ~np.in1d(train_data.index.date, pd.to_datetime(anomalies).date)        
#        anomaly_data = train_data.loc[~mask,:]
        anomaly_data = train_data.copy()
        train_data = train_data.loc[mask, :]
        test_data = data[middle_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")].copy()        
    else:
        print("file no", file_no)
        train_data = data[start_date.strftime("%Y-%m-%d"):middle_date.strftime("%Y-%m-%d")].copy()
#         print(train_data.head(5))
        mix_data = train_data.copy()
        mask = ~np.in1d(train_data.index.date, pd.to_datetime(anomalies).date)        
        anomaly_data = train_data.loc[~mask,:]
        train_data = train_data.loc[mask, :]
        test_data = data[middle_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")].copy()        
               
        
    datatotrain = train_data['value'].values.reshape(-1,144)
    datatotrain_normalized = preprocessing.normalize(datatotrain, norm='l2')
    #    datatotest = with anomaly test dataset
    datatotest = test_data['value'].values.reshape(-1,144)
    datatotest_normalized = preprocessing.normalize(datatotest, norm='l2')
    dataanomaly = anomaly_data['value'].values.reshape(-1,144)     
    dataanomaly_normalized = preprocessing.normalize(dataanomaly, norm='l2')
    return datatotrain,datatotest,datatotrain_normalized,datatotest_normalized,dataanomaly_normalized,train_data,test_data,mix_data

#%%
file_no = 1
data = preprocess(data_1)    
anomalies = anomalies_1
start_date = data.head(1).index.date[0]
end_date = data.tail(1).index.date[0]
print('first date :%s and last date :%s '%(start_date.strftime("%Y-%m-%d"),end_date.strftime("%Y-%m-%d"))) 
middle_date = start_date + (end_date-start_date)/2
#takes in the entire dataset, anomalies during training period
#     print(data.head(5))
#if file_no == 1:
#    print("file no", file_no)
#    train_data = data[middle_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")].copy()
#    # train_data =  train_data[train_data.index.weekday < 6]
#    mask = ~np.in1d(train_data.index.date, pd.to_datetime(anomalies).date)
#    train_data = train_data.loc[mask, :]
#    test_data = data[start_date.strftime("%Y-%m-%d"):middle_date.strftime("%Y-%m-%d")].copy()
#    # test_data =  test_data[test_data.index.weekday < 6]
if file_no == 3:
    print("file no", file_no)
#         print("ssss11")
    train_data = data[start_date.strftime("%Y-%m-%d"):middle_date.strftime("%Y-%m-%d")].copy()
    train_data =  train_data[train_data.index.weekday < 6]
    mix_data = train_data.copy()
    mask = ~np.in1d(train_data.index.date, pd.to_datetime(anomalies).date)
    anomaly_data = train_data.loc[~mask,:]
    train_data = train_data.loc[mask, :]
    test_data = data[middle_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")].copy()
    test_data =  test_data[test_data.index.weekday < 6]
else:
    print("file no", file_no)
    train_data = data[start_date.strftime("%Y-%m-%d"):middle_date.strftime("%Y-%m-%d")].copy()
#         print(train_data.head(5))
    mix_data = train_data.copy()
    mask = ~np.in1d(train_data.index.date, pd.to_datetime(anomalies).date)        
    anomaly_data = train_data.loc[~mask,:]
    train_data = train_data.loc[mask, :]
    test_data = data[middle_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")].copy()   
datatotrain = train_data['value'].values.reshape(-1,144)
datatotrain_normalized = preprocessing.normalize(datatotrain, norm='l2')
#    datatotest = with anomaly test dataset
datatotest = test_data['value'].values.reshape(-1,144)
datatotest_normalized = preprocessing.normalize(datatotest, norm='l2')
dataanomaly = anomaly_data['value'].values.reshape(-1,144)     
dataanomaly_normalized = preprocessing.normalize(dataanomaly, norm='l2')

#%%
#y_pred_train = clf.predict(X_train)
#y_pred_test = clf.predict(X_test)
#y_pred_outliers = clf.predict(X_outliers)
#n_error_train = y_pred_train[y_pred_train == -1].size
#n_error_test = y_pred_test[y_pred_test == -1].size
#n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
##%%
#colors = {1 : 'b',
#          -1 : 'r',}
#c = [colors[val] for val in test_data['ocsvm_score']]
##ax.scatter(idx_src, idx_cty, year, s=avg, c=c)

#%%
file_no = 3
anomalies = anomalies_3
data = preprocess(data_3)

start_date = data.head(1).index.date[0]
end_date = data.tail(1).index.date[0]
middle_date = start_date + (end_date-start_date)/2
datatotrain,datatotest,datatotrain_normalized,datatotest_normalized,dataanomaly_normalized,train_data,test_data,mix_data = createtraintest(data,anomalies,file_no)
X_train,X_test,X_outliers = datatotrain_normalized,datatotest_normalized,dataanomaly_normalized
#X_train,X_test,X_outliers = datatotrain,datatotest,dataanomaly_normalized
##%%
#clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
#clf.fit(X_train)
##%%
#
#clf = KMeans(n_clusters=3, random_state=42).fit(X_train)
##%%
#clf = IsolationForest(behaviour='new',
#                      random_state=42, contamination='auto')
#clf.fit(X_train)
#%%
data =  data[data.index.weekday < 6]
datax = data['value'].values.reshape(-1,144)
data_n = preprocessing.normalize(datax, norm='l2')
clf = LocalOutlierFactor(n_neighbors=10,contamination=0.05)
y_pred = clf.fit_predict(data_n)
i=0
for dt in rrule(DAILY, dtstart=start_date, until=end_date):
    try:
        if (data.loc[dt.strftime("%Y-%m-%d")]['value']).values.size != 0:
            data.loc[dt.strftime("%Y-%m-%d"),'ocsvm_score'] =y_pred[i]
            i = i + 1
    except:
        print(dt,i)
        continue
data.loc[:,['value','ocsvm_score']].plot(figsize=(144, 4))
for dt in rrule(DAILY, dtstart=start_date, until=end_date):
    plt.axvline(x=dt.strftime("%Y-%m-%d"))
    if dt.strftime("%Y-%m-%d") in anomalies: 
        plt.axvspan(dt, dt+ timedelta(days=1), color='r', alpha=0.2)

OSCVMS = np.unique(data[data.ocsvm_score == -1].index.date.astype(str))
print("LOCAL OUTLIER FACTOR:")
print(OSCVMS)
#%%
#for dt in rrule(DAILY, dtstart=start_date, until=middle_date):
#    try:
#        train_data.loc[dt.strftime("%Y-%m-%d"),'ocsvm_score'] = clf.predict(preprocessing.normalize(train_data.loc[dt.strftime("%Y-%m-%d")].value.values.reshape(1, -1)))
#    except:
#        continue
#train_data.loc[:,['value','ocsvm_score']].plot(figsize=(144, 4))
#for dt in rrule(DAILY, dtstart=start_date, until=middle_date):
#    plt.axvline(x=dt.strftime("%Y-%m-%d"))
#    if dt.strftime("%Y-%m-%d") in anomalies_3: 
#        plt.axvspan(dt, dt+ timedelta(days=1), color='r', alpha=0.2)
##%%
#for dt in rrule(DAILY, dtstart=start_date, until=middle_date):
#    try:
#        mix_data.loc[dt.strftime("%Y-%m-%d"),'ocsvm_score'] = clf.predict(preprocessing.normalize(mix_data.loc[dt.strftime("%Y-%m-%d")].value.values.reshape(1, -1)))
#    except:
#        continue
#mix_data.loc[:,['value','ocsvm_score']].plot(figsize=(144, 4))
#for dt in rrule(DAILY, dtstart=start_date, until=middle_date):
#    plt.axvline(x=dt.strftime("%Y-%m-%d"))
#    if dt.strftime("%Y-%m-%d") in anomalies_3: 
#        plt.axvspan(dt, dt+ timedelta(days=1), color='r', alpha=0.2)
#        #%%
#for dt in rrule(DAILY, dtstart=start_date, until=middle_date):
#    try:
#        test_data.loc[dt.strftime("%Y-%m-%d"),'ocsvm_score'] = clf.predict(preprocessing.normalize(test_data.loc[dt.strftime("%Y-%m-%d")].value.values.reshape(1, -1)))
#    except:
#        continue
#test_data.loc[:,['value','ocsvm_score']].plot(figsize=(144, 4))
#
#for dt in rrule(DAILY, dtstart=start_date, until=middle_date):
#    plt.axvline(x=dt.strftime("%Y-%m-%d"))
#    if dt.strftime("%Y-%m-%d") in anomalies_3: 
#        plt.axvspan(dt, dt+ timedelta(days=1), color='r', alpha=0.2)
#%%
for dt in rrule(DAILY, dtstart=start_date, until=end_date):
    try:
        data.loc[dt.strftime("%Y-%m-%d"),'ocsvm_score'] = clf.predict(preprocessing.normalize(data.loc[dt.strftime("%Y-%m-%d")].value.values.reshape(1, -1)))
    except:
        continue
data.loc[:,['value','ocsvm_score']].plot(figsize=(144, 4))
for dt in rrule(DAILY, dtstart=start_date, until=end_date):
    plt.axvline(x=dt.strftime("%Y-%m-%d"))
    if dt.strftime("%Y-%m-%d") in anomalies: 
        plt.axvspan(dt, dt+ timedelta(days=1), color='r', alpha=0.2)

OSCVMS = np.unique(data[data.ocsvm_score == -1].index.date.astype(str))
print("ISOLATION FOREST:")
print(OSCVMS)