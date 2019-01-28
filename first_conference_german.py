# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 12:02:22 2019

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
from sklearn.metrics import confusion_matrix

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
#%% LOCAL OUTLIER FACTOR
M = np.array([],dtype=int).reshape(0,256)
#M = np.array([],dtype=int).reshape(0,128)
number_of_neighbors = 3
for _ in range(1,number_of_neighbors):
    clf =  LOF(n_neighbors=_, contamination=outliers_fraction)
    clf.fit(X_train)    
    y_pred = clf.predict(data_n)
#    y_pred = clf.predict(X_test)
    del clf
    M = np.vstack((M,y_pred))
#    print(y_pred.shape)
drawss = plt.pcolor(M,cmap=('Reds'),edgecolor='black')
#%%
M = np.array([],dtype=int).reshape(0,256)
M2 = np.array([],dtype=int).reshape(0,256)
#M = np.array([],dtype=int).reshape(0,128)
number_of_neighbors = 10
clf =  LOF(n_neighbors=_, contamination=outliers_fraction)
clf1 = HBOS(contamination=outliers_fraction)
clf.fit(X_train)  
clf1.fit(X_train)  
y_pred = clf.predict(data_n)
y_pred1 = clf1.predict(data_n)
M = np.vstack((M,y_pred))
M2= np.vstack((M2,y_pred1))
#%%
ano1 = y_pred[y_pred==1].size
print('LOCAL outlier factor' , ano1/y_pred.size*100)
ano2 = y_pred1[y_pred1==1].size
print('histogram based', ano2/y_pred.size*100 )
#%%
fig, (ax0, ax1) = plt.subplots(2,1)
c = ax0.pcolor(M,cmap=('Oranges'),edgecolor='black')
ax0.set_title('default: no edges')

c = ax1.pcolor(M2,cmap=('Reds'),edgecolor='black')
ax1.set_title('thick edges')

fig.tight_layout()
plt.show()
#%%
clf =  LOF(n_neighbors=_, contamination=outliers_fraction)
clf.fit(X_train)
clf.predict(data_n)
w = clf.predict_proba(data_n)
#%%
#w[:,0].shape
fig, (ax0, ax1) = plt.subplots(2,1)
c = ax0.pcolor(w[:,1].reshape(1,-1),cmap=('Oranges'),edgecolor='black')
ax0.set_title('probabilities')
c = ax1.pcolor(M,cmap=('Reds'),edgecolor='black')
ax1.set_title('original')
fig.tight_layout()
plt.show()
#%%
plt.pcolor()