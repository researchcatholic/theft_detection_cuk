# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 15:37:04 2019

@author: p0p
"""

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

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP

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
#%%
data_1 = pd.read_csv("C:/Users/p0p/Desktop/anaomaly/1year_Example-1.csv")
data_2 = pd.read_csv("C:/Users/p0p/Desktop/anaomaly/1year_Example-2.csv")
data_3 = pd.read_csv("C:/Users/p0p/Desktop/anaomaly/1year_Example-3.csv")
data_4 = pd.read_csv("C:/Users/p0p/Desktop/anaomaly/1year_Example-4.csv")

anomalies_1 = ['2012-09-10','2012-09-11','2012-09-12','2012-09-15','2012-09-16','2012-10-30','2012-11-16','2012-11-30','2012-12-02','2012-12-03','2012-12-04','2012-12-22']
anomalies_2 = ['2011-03-04','2011-03-06','2011-04-17','2011-04-22','2011-05-06','2011-05-19','2011-05-20','2011-05-22','2011-05-23','2011-05-24']
anomalies_3 = ['2014-01-17','2014-01-29','2014-02-26','2014-04-04','2014-04-05','2014-04-11','2014-04-12','2014-04-22','2014-04-30','2014-05-12','2014-05-13','2014-05-26']
anomalies_4 = ['2014-05-01','2014-06-10','2014-06-17','2014-07-08','2014-07-10','2014-09-02','2014-10-17','2014-10-23','2014-10-24']

datadict = {1:(data_1,anomalies_1), 2:(data_2,anomalies_2), 3:(data_3,anomalies_3), 4:(data_4,anomalies_4)}
#%%
def preprocess(data_, plot = 0 ):
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
    if plot == 1:
        plt.figure(figsize=(90,3))
        plt.plot(data['value'])
        plt.title("plot without nulls")
        plt.show()
    # a.strftime("%Y-%m-%d")
    return data
#%%
def createtraintest(data):
    start_date = data.head(1).index.date[0]
    end_date = data.tail(1).index.date[0]
    print('first date :%s and last date :%s '%(start_date.strftime("%Y-%m-%d"),end_date.strftime("%Y-%m-%d"))) 
    middle_date = start_date + (end_date-start_date)/2
    #takes in the entire dataset, anomalies during training period\
    train_data = data[start_date.strftime("%Y-%m-%d"):middle_date.strftime("%Y-%m-%d")].copy()
    test_data = data[(middle_date+timedelta(days= 1)).strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")].copy()
    # train_data =  train_data[train_data.index.weekday < 6]
#    mix_data = train_data.copy()
#    mask = ~np.in1d(train_data.index.date, pd.to_datetime(anomalies).date)
#    anomaly_data = train_data.loc[~mask,:]
#    train_data = train_data.loc[mask, :]
    datatotrain = train_data['value'].values.reshape(-1,144)
    datatotrain_normalized = preprocessing.normalize(datatotrain, norm='l2')
    #    datatotest = with anomaly test dataset
    datatotest = test_data['value'].values.reshape(-1,144)
    datatotest_normalized = preprocessing.normalize(datatotest, norm='l2')
#    dataanomaly = anomaly_data['value'].values.reshape(-1,144)     
#    dataanomaly_normalized = preprocessing.normalize(dataanomaly, norm='l2')
    return datatotrain,datatotest,datatotrain_normalized,datatotest_normalized,train_data,test_data
#%%
#outliers_fraction = 0.2
#random_state = np.random.RandomState(42)
## Define nine outlier detection tools to be compared
## initialize a set of detectors for LSCP
#detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
#                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
#                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
#                 LOF(n_neighbors=50)]
#classifiers = {
#    'Angle-based Outlier Detector (ABOD)':
#        ABOD(contamination=outliers_fraction),
#    'Cluster-based Local Outlier Factor (CBLOF)':
#        CBLOF(contamination=outliers_fraction,
#              check_estimator=False, random_state=random_state),
#    'Feature Bagging':
#        FeatureBagging(LOF(n_neighbors=35),
#                       contamination=outliers_fraction,
#                       check_estimator=False,
#                       random_state=random_state),
#    'Histogram-base Outlier Detection (HBOS)': HBOS(
#        contamination=outliers_fraction),
#    'Isolation Forest': IForest(contamination=outliers_fraction,
#                                random_state=random_state),
#    'K Nearest Neighbors (KNN)': KNN(
#        contamination=outliers_fraction),
#    'Average KNN': KNN(method='mean',
#                       contamination=outliers_fraction),
#    # 'Median KNN': KNN(method='median',
#    #                   contamination=outliers_fraction),
#    'Local Outlier Factor (LOF)':
#        LOF(n_neighbors=35, contamination=outliers_fraction),
#    # 'Local Correlation Integral (LOCI)':
#    #     LOCI(contamination=outliers_fraction),
#    'Minimum Covariance Determinant (MCD)': MCD(
#        contamination=outliers_fraction, random_state=random_state),
#    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction,
#                                   random_state=random_state),
#    'Principal Component Analysis (PCA)': PCA(
#        contamination=outliers_fraction, random_state=random_state),
#    # 'Stochastic Outlier Selection (SOS)': SOS(
#    #     contamination=outliers_fraction),
#    'Locally Selective Combination (LSCP)': LSCP(
#        detector_list, contamination=outliers_fraction,
#        random_state=random_state)
#}
#%%
#file_no = 3
#data = preprocess(data_3)    
#data = data[data.index.weekday < 6]
#%%
outliers_fraction = 0.1
file_no = 3
data = preprocess(datadict[file_no][0]) 
anomalies = datadict[file_no][1]  
if file_no == 3: 
    data = data[data.index.weekday < 6]
start_date = data.head(1).index.date[0]
end_date = data.tail(1).index.date[0]
datatotrain,datatotest,datatotrain_normalized,datatotest_normalized,train_data,test_data = createtraintest(data)
X_train,X_test = datatotrain_normalized,datatotest_normalized
datax = data['value'].values.reshape(-1,144)
data_n = preprocessing.normalize(datax, norm='l2')
#%%
#from pandas.tools.plotting import scatter_matrix

#df = pd.DataFrame(datax)
#
##scatter_matrix(df, alpha=0.2, diagonal='kde')
#from pandas.tools.plotting import lag_plot
#lag_plot(data.value)
#%%
#clf =  LOF(n_neighbors=10, contamination=0.1)
#clf.fit(data_n)
#clf1 =  LOF(n_neighbors=10, contamination=0.1)
#clf1.fit(data_n[:,0:48])
#clf2 =  LOF(n_neighbors=10, contamination=0.1)
#clf2.fit(data_n[:,48:96])
#clf3 =  LOF(n_neighbors=10, contamination=0.1)
#clf3.fit(data_n[:,96:144])
#%%
clf =  LOF(n_neighbors=10, contamination=0.1)
clf.fit(X_train)
clf1 =  LOF(n_neighbors=10, contamination=0.1)
clf1.fit(X_train[:,0:48])
clf2 =  LOF(n_neighbors=10, contamination=0.1)
clf2.fit(X_train[:,48:96])
clf3 =  LOF(n_neighbors=10, contamination=0.1)
clf3.fit(X_train[:,96:144])
 
#%%
clf =  HBOS(contamination=outliers_fraction)
clf.fit(X_train)
clf1 =  HBOS(contamination=outliers_fraction)
clf1.fit(X_train[:,0:48])
clf2 =  HBOS(contamination=outliers_fraction)
clf2.fit(X_train[:,48:96])
clf3 =  HBOS(contamination=outliers_fraction)
clf3.fit(X_train[:,96:144])
 
#%%

#y_pred = clf.predict(data_n)
i=0
for dt in rrule(DAILY, dtstart=start_date, until=end_date):
    if (data.loc[dt.strftime("%Y-%m-%d")]['value']).values.size != 0:
        data.loc[dt.strftime("%Y-%m-%d"),'ocsvm_score'] =clf.predict(preprocessing.normalize(data.loc[dt.strftime("%Y-%m-%d")].value.values.reshape(1, -1)))
        data.loc[dt.strftime("%Y-%m-%d"),'ocsvm_score1'] =clf1.predict(preprocessing.normalize(data.loc[dt.strftime("%Y-%m-%d")].value.values.reshape(1, -1))[0][0:48].reshape(1,-1))
        data.loc[dt.strftime("%Y-%m-%d"),'ocsvm_score2'] =clf2.predict(preprocessing.normalize(data.loc[dt.strftime("%Y-%m-%d")].value.values.reshape(1, -1))[0][48:96].reshape(1,-1))
        data.loc[dt.strftime("%Y-%m-%d"),'ocsvm_score3'] =clf3.predict(preprocessing.normalize(data.loc[dt.strftime("%Y-%m-%d")].value.values.reshape(1, -1))[0][96:144].reshape(1,-1))
        i = i + 1
    else:
#        print(dt)
        continue
data.loc[:,['value','ocsvm_score']].plot(figsize=(144, 4))
for dt in rrule(DAILY, dtstart=start_date, until=end_date):
    plt.axvline(x=dt.strftime("%Y-%m-%d"))
    if (data.loc[dt.strftime("%Y-%m-%d")]['value']).values.size != 0:
        if dt.strftime("%Y-%m-%d") in anomalies: 
            plt.axvspan(dt, dt+ timedelta(days=1), color='yellow', alpha=0.2)
        if data.loc[dt.strftime("%Y-%m-%d")].ocsvm_score1.all() == 1:
            plt.axvspan(dt, dt+ timedelta(days=1/3), color='red', alpha=0.2)
        if data.loc[dt.strftime("%Y-%m-%d")].ocsvm_score2.all() == 1:
            plt.axvspan(dt+ timedelta(days=1/3), dt+ timedelta(days=2/3), color='green', alpha=0.2)
        if data.loc[dt.strftime("%Y-%m-%d")].ocsvm_score3.all() == 1:
            plt.axvspan(dt+ timedelta(days=2/3), dt+ timedelta(days=1), color='blue', alpha=0.2)
    else:
        continue

LOFS = np.unique(data[data.ocsvm_score == 1].index.date.astype(str))
print("LOCAL OUTLIER FACTOR:")
print(LOFS)
#%%
data['doubt'] = (data['ocsvm_score']+data['ocsvm_score1']+data['ocsvm_score2']+data['ocsvm_score3'])
data['doubt'] = np.where(data['doubt']>0,1,0)
#%%
data.loc[:,['value','doubt']].plot(figsize=(144, 4))

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
#for dt in rrule(DAILY, dtstart=start_date, until=end_date):
#    try:
#        data.loc[dt.strftime("%Y-%m-%d"),'ocsvm_score'] = clf.predict(preprocessing.normalize(data.loc[dt.strftime("%Y-%m-%d")].value.values.reshape(1, -1)))
#    except:
#        continue
#data.loc[:,['value','ocsvm_score']].plot(figsize=(144, 4))
#for dt in rrule(DAILY, dtstart=start_date, until=end_date):
#    plt.axvline(x=dt.strftime("%Y-%m-%d"))
#    if dt.strftime("%Y-%m-%d") in anomalies: 
#        plt.axvspan(dt, dt+ timedelta(days=1), color='r', alpha=0.2)
#
#OSCVMS = np.unique(data[data.ocsvm_score == -1].index.date.astype(str))
#print("ISOLATION FOREST:")
#print(OSCVMS)
