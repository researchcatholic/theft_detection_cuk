# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 12:02:22 2019

@author: p0p
"""
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(sum(map(ord, 'calmap')))
import pandas as pd
import calmap
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from scipy.stats import scoreatpercentile
from sklearn.preprocessing import MinMaxScaler

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
#old 
#anomalies_1 = ['2012-09-10','2012-09-11','2012-09-12','2012-09-15','2012-09-16','2012-10-30','2012-11-16','2012-11-30','2012-12-02','2012-12-03','2012-12-04','2012-12-22']
#anomalies_2 = ['2011-03-04','2011-03-06','2011-04-17','2011-04-22','2011-05-06','2011-05-19','2011-05-20','2011-05-22','2011-05-23','2011-05-24']
#anomalies_3 = ['2014-01-17','2014-01-29','2014-02-26','2014-04-04','2014-04-05','2014-04-11','2014-04-12','2014-04-22','2014-04-30','2014-05-12','2014-05-13','2014-05-26']
#anomalies_4 = ['2014-05-01','2014-06-10','2014-06-17','2014-07-08','2014-07-10','2014-09-02','2014-10-17','2014-10-23','2014-10-24']

#new
anomalies_1 = ["2012-02-04", "2012-07-18", "2012-12-03", "2012-12-04", "2012-11-16", "2012-12-02", "2012-02-05", "2012-05-14", "2012-05-15", "2012-05-16", "2012-07-18", "2012-09-10", "2012-09-11", "2012-09-12", "2012-09-15", "2012-09-16", "2012-10-30", "2012-11-16", "2012-12-02", "2012-12-03", "2012-12-04"]
anomalies_2 = ["2011-02-21", "2011-01-08", "2011-07-02",  "2011-09-10", "2011-02-17", "2011-02-20", "2011-02-27", "2011-03-04", "2011-03-06", "2011-04-17", "2011-04-22", "2011-05-01", "2011-05-06", "2011-05-31", "2011-06-20", "2011-10-09", "2011-10-16", "2011-10-27", "2011-05-19", "2011-05-20", "2011-05-21", "2011-05-22", "2011-05-23", "2011-05-24", "2011-06-08", "2011-08-14", "2011-08-27"]
anomalies_3 = ["2014-01-17", "2014-01-29", "2014-02-26", "2014-04-22", "2014-04-30", "2014-05-26", "2014-09-22", "2014-10-23", "2014-10-24", "2014-01-30", "2014-01-31", "2014-04-04", "2014-04-05", "2014-04-11", "2014-04-12", "2014-04-13", "2014-05-12", "2014-05-13", "2014-06-25", "2014-06-26", "2014-08-24", "2014-09-11", "2014-09-12", "2014-10-05"]
anomalies_4 = ["2013-10-07", "2013-10-11",  "2012-11-04", "2013-11-06", "2013-11-13", "2013-11-21", "2014-04-15", "2014-05-24", "2014-05-25", "2013-05-03", "2013-10-29"]



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
file_no = 1
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
#M = np.array([],dtype=int).reshape(0,256)
##M = np.array([],dtype=int).reshape(0,128)
#number_of_neighbors = 4
#for _ in range(1,number_of_neighbors):
#    clf =  LOF(n_neighbors=_, contamination=outliers_fraction)
#    clf.fit(X_train)    
#    y_pred = clf.predict(data_n)
##    y_pred = clf.predict(X_test)
#    del clf
#    M = np.vstack((M,y_pred))
##    print(y_pred.shape)
#drawss = plt.pcolor(M,cmap=('Reds'),edgecolor='black')
##%%
#M = np.array([],dtype=int).reshape(0,256)
#M2 = np.array([],dtype=int).reshape(0,256)
##M = np.array([],dtype=int).reshape(0,128)
#number_of_neighbors = 10
#clf =  LOF(n_neighbors=_, contamination=outliers_fraction)
#clf1 = HBOS(contamination=outliers_fraction)
#clf.fit(X_train)  
#clf1.fit(X_train)  
#y_pred = clf.predict(data_n)
#y_pred1 = clf1.predict(data_n)
#M = np.vstack((M,y_pred))
#M2= np.vstack((M2,y_pred1))
##%%
#ano1 = y_pred[y_pred==1].size
#print('LOCAL outlier factor' , ano1/y_pred.size*100)
#ano2 = y_pred1[y_pred1==1].size
#print('histogram based', ano2/y_pred.size*100 )
##%%
#fig, (ax0, ax1) = plt.subplots(2,1)
#c = ax0.pcolor(M,cmap=('Oranges'),edgecolor='black')
#ax0.set_title('default: no edges')
#
#c = ax1.pcolor(M2,cmap=('Reds'),edgecolor='black')
#ax1.set_title('thick edges')
#
#fig.tight_layout()
#plt.show()
##%%
#tset = M.reshape(16,16)
#
#plt.pcolor(tset,cmap=('Reds'),edgecolor='black')
##%%
#
#anos = np.zeros(len(data_n))
#for _ in range(0,len(anomalies)):
#    try:
#        anos[(data.index.get_loc(anomalies[_] + ' 00:00:00')//144)] = 1
#    except:
#        continue
#fig, (ax0,ax1,ax2,ax3,ax4) = plt.subplots(5,1)
#random_state = np.random.RandomState(42)
### Define nine outlier detection tools to be compared
### initialize a set of detectors for LSCP
##detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
##                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
##                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
##                 LOF(n_neighbors=50)]
#classifiers = {
##    'Angle-based Outlier Detector (ABOD)':
##        ABOD(contamination=outliers_fraction),
#    'Cluster-based Local Outlier Factor (CBLOF)':
#        (CBLOF(contamination=outliers_fraction,
#              check_estimator=False, random_state=random_state),ax0),
##    'Feature Bagging':
##        FeatureBagging(LOF(n_neighbors=35),
##                       contamination=outliers_fraction,
##                       check_estimator=False,
##                       random_state=random_state),
#    'Histogram-base Outlier Detection (HBOS)': (HBOS(
#        contamination=outliers_fraction),ax1),
##    'Isolation Forest': IForest(contamination=outliers_fraction,
##                                random_state=random_state),
#    'K Nearest Neighbors (KNN)': (KNN(
#        contamination=outliers_fraction),ax2),
##    'Average KNN': (KNN(method='mean',
##                       contamination=outliers_fraction),ax3),
#    # 'Median KNN': KNN(method='median',
#    #                   contamination=outliers_fraction),
#    'Local Outlier Factor (LOF)':
#        (LOF(n_neighbors=35, contamination=outliers_fraction),ax3)
#    # 'Local Correlation Integral (LOCI)':
#    #     LOCI(contamination=outliers_fraction),
##    'Minimum Covariance Determinant (MCD)': MCD(
##        contamination=outliers_fraction, random_state=random_state),
##    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction,
##                                   random_state=random_state),
##    'Principal Component Analysis (PCA)': PCA(
##        contamination=outliers_fraction, random_state=random_state),
##    # 'Stochastic Outlier Selection (SOS)': SOS(
##    #     contamination=outliers_fraction),
##    'Locally Selective Combination (LSCP)': LSCP(
##        detector_list, contamination=outliers_fraction,
##        random_state=random_state)
#}
#
##clf =  LOF(n_neighbors=_, contamination=outliers_fraction)
#for _ in classifiers:
#    clf = classifiers[_][0]
#    clf.fit(X_train)
#    clf.predict(data_n)
#    w = clf.predict_proba(data_n)
#    #w[:,0].shape
#    c = classifiers[_][1].pcolor(w[:,1].reshape(1,-1),cmap=('Oranges'),edgecolor='black')
#    classifiers[_][1].set_title(_)
##    classifiers[_][1].legend()
#    #c = ax1.pcolor(M,cmap=('Reds'),edgecolor='black')
#    #ax1.set_title('original')
#    
#anoss = ax4.pcolor(anos.reshape(1,-1),cmap=('Oranges'),edgecolor='black')
#fig.tight_layout()
#plt.show()
##%%
#fig, (ax0,ax1,ax2,ax3,ax4) = plt.subplots(5,1)
##clf =  LOF(n_neighbors=_, contamination=outliers_fraction)
#random_state = np.random.RandomState(42)
#classifierss = {
##    'Angle-based Outlier Detector (ABOD)':
##        ABOD(contamination=outliers_fraction),
#    'Cluster-based Local Outlier Factor (CBLOF)':
#        (CBLOF(contamination=outliers_fraction,
#              check_estimator=False, random_state=random_state),ax0),
##    'Feature Bagging':
##        FeatureBagging(LOF(n_neighbors=35),
##                       contamination=outliers_fraction,
##                       check_estimator=False,
##                       random_state=random_state),
#    'Histogram-base Outlier Detection (HBOS)': (HBOS(
#        contamination=outliers_fraction),ax1),
##    'Isolation Forest': IForest(contamination=outliers_fraction,
##                                random_state=random_state),
#    'K Nearest Neighbors (KNN)': (KNN(
#        contamination=outliers_fraction),ax2),
##    'Average KNN': (KNN(method='mean',
##                       contamination=outliers_fraction),ax3),
#    # 'Median KNN': KNN(method='median',
#    #                   contamination=outliers_fraction),
#    'Local Outlier Factor (LOF)':
#        (LOF(n_neighbors=35, contamination=outliers_fraction),ax3)
#    # 'Local Correlation Integral (LOCI)':
#    #     LOCI(contamination=outliers_fraction),
##    'Minimum Covariance Determinant (MCD)': MCD(
##        contamination=outliers_fraction, random_state=random_state),
##    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction,
##                                   random_state=random_state),
##    'Principal Component Analysis (PCA)': PCA(
##        contamination=outliers_fraction, random_state=random_state),
##    # 'Stochastic Outlier Selection (SOS)': SOS(
##    #     contamination=outliers_fraction),
##    'Locally Selective Combination (LSCP)': LSCP(
##        detector_list, contamination=outliers_fraction,
##        random_state=random_state)
#}
#for _ in classifierss:
#    clf = classifierss[_][0]
#    clf.fit(X_train)
#    clf.predict(data_n)
#    w = clf.predict(data_n)
#    #w[:,0].shape
#    c = classifierss[_][1].pcolor(w.reshape(1,-1),cmap=('Oranges'),edgecolor='black')
#    classifierss[_][1].set_title(_)
##    classifiers[_][1].legend()
#    #c = ax1.pcolor(M,cmap=('Reds'),edgecolor='black')
#    #ax1.set_title('original')
#    
#anoss = ax4.pcolor(anos.reshape(1,-1),cmap=('Oranges'),edgecolor='black')
#fig.tight_layout()
#plt.show()

#%%


#all_days = pd.date_range('1/1/2014', periods=365, freq='D')
#days = np.random.choice(all_days, 40)

##% year plot
#dayss = []
#for _ in np.where(w==1):
#    dayss.append(data.iloc[_*144].index)
##    print(_)
#dayss = dayss[0]
#events = pd.Series(np.ones(len(dayss)), index=dayss)
#calmap.yearplot(events, year=start_date.year)
#%%
random_state = np.random.RandomState(42)
classi = {
    'Cluster-based Local Outlier Factor (CBLOF)':
        (CBLOF(contamination=outliers_fraction,
              check_estimator=False, random_state=random_state)),
    'Histogram-base Outlier Detection (HBOS)': (HBOS(
        contamination=outliers_fraction)),
    'K Nearest Neighbors (KNN)': (KNN(
        contamination=outliers_fraction)),
    'Local Outlier Factor (LOF)':
        (LOF(n_neighbors=10, contamination=outliers_fraction))
}
data_analysed = data.copy()
for jo in classi:
    clf = classi[jo]
    clf.fit(X_train)
    for dt in rrule(DAILY, dtstart=start_date, until=end_date):
        try:
            data_analysed.loc[dt.strftime("%Y-%m-%d"),(str(classi[jo]).split("(")[0])] = clf.predict_proba(preprocessing.normalize(data_analysed.loc[dt.strftime("%Y-%m-%d")].value.values.reshape(1, -1)))[0][1]
        except:
            continue
#    data.loc[:,['value',(str(classifiers[jo][0]).split("(")[0])]].plot(figsize=(144, 4))
#    for dt in rrule(DAILY, dtstart=start_date, until=end_date):
#        plt.axvline(x=dt.strftime("%Y-%m-%d"))
#        if dt.strftime("%Y-%m-%d") in anomalies: 
#            plt.axvspan(dt, dt+ timedelta(days=1), color='r', alpha=0.2)
#    OSCVMS = np.unique(data_analysed[data_analysed[(str(classifiers[jo][0]).split("(")[0])] == 1].index.date.astype(str))
#    print((str(classifiers[jo][0]).split("(")[0]))
#    print(OSCVMS)
#%%
for _ in range(1,5):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    events = pd.Series(data_analysed.iloc[:,(_)].values, index=data_analysed.index)
    cax = calmap.yearplot(events, year=start_date.year,linecolor='grey',cmap='Oranges', ax=ax)
    fig.colorbar(cax.get_children()[1], ax=cax, orientation='horizontal')
#    fig.show()
#%%
for jo in classi:
    clf.fit(X_train)
    b = 0
    ans = 0
    for i in range(1,100):
        a = np.sum(((clf.predict_proba(X_test)[:,1]) >=(i/100)).astype(int) ==  clf.predict(X_test))
        if a > b:
            b = a
            ans = i
    plt.plot(((clf.predict_proba(X_test)[:,1]) >=ans/100).astype(int) ==  clf.predict(X_test))
    train_scores = clf.decision_scores_
    test_scores = clf.decision_function(X_test)
    threshold = scoreatpercentile(train_scores, 100 * (1 - outliers_fraction))
    predsss = (test_scores > threshold).astype('int')
    print(predsss == clf.predict(X_test))
#
#
#
#probs = np.zeros([X_test.shape[0], int(clf._classes)])
#scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
#probs[:, 1] = scaler.transform(test_scores.reshape(-1, 1)).ravel().clip(0, 1)
#probs[:, 0] = 1 - probs[:, 1]
#%%
#
#random_state = np.random.RandomState(42)
#classi = {
#    'Cluster-based Local Outlier Factor (CBLOF)':
#        (CBLOF(contamination=outliers_fraction,
#              check_estimator=False, random_state=random_state)),
#    'Histogram-base Outlier Detection (HBOS)': (HBOS(
#        contamination=outliers_fraction)),
#    'K Nearest Neighbors (KNN)': (KNN(
#        contamination=outliers_fraction)),
#    'Local Outlier Factor (LOF)':
#        (LOF(n_neighbors=10, contamination=outliers_fraction))
#}
#outliers_fraction = 0.1
#file_no = 4
#data = preprocess(datadict[file_no][0]) 
#anomalies = datadict[file_no][1]  
#if file_no == 3: 
#    data = data[data.index.weekday < 6]
#start_date = data.head(1).index.date[0]
#end_date = data.tail(1).index.date[0]
##%%
#datatotrain,datatotest,datatotrain_normalized,datatotest_normalized,train_data,test_data = createtraintest(data)
#X_train,X_test = datatotrain_normalized,datatotest_normalized
#datax = data['value'].values.reshape(-1,144)
#data_n = preprocessing.normalize(datax, norm='l2')
#data_analysed = data.copy()
##%%
#for jo in classi:
#    clf = classi[jo]
#    clf.fit(X_train)
#    for dt in rrule(DAILY, dtstart=start_date, until=end_date):
#        try:
#            data_analysed.loc[dt.strftime("%Y-%m-%d"),(str(classi[jo]).split("(")[0])] = clf.predict_proba(preprocessing.normalize(data_analysed.loc[dt.strftime("%Y-%m-%d")].value.values.reshape(1, -1)))[0][1]
#        except:
#            continue
##%%
#for _ in range(1,5):
##    fig = plt.figure(figsize=(10,4))
##    ax = fig.add_subplot(111)
#    events = pd.Series(data_analysed.iloc[:,(_)].values, index=data_analysed.index)
##    cax = calmap.yearplot(events, year=start_date.year,linecolor='grey',cmap='Oranges', ax=ax)
#    calmap.calendarplot(events)
##    fig.colorbar(cax.get_children()[1], ax=cax, orientation='horizontal')
#%%
#
#for jo in classi:
#    clf = classi[jo]
#    clf.fit(X_train)
#    b = 0
#    ans = 0
#    for i in range(1,100):
#        a = np.sum(((clf.predict_proba(X_test)[:,1]) >=(i/100)).astype(int) ==  clf.predict(X_test))
#        if a > b:
#            b = a
#            ans = i
#    print(jo,ans)
#    plt.plot(((clf.predict_proba(X_test)[:,1]) >=ans/100).astype(int) ==  clf.predict(X_test))
#    train_scores = clf.decision_scores_
#    test_scores = clf.decision_function(X_test)
#    threshold = scoreatpercentile(train_scores, 100 * (1 - outliers_fraction))
#    predsss = (test_scores > threshold).astype('int')
#    print(predsss == clf.predict(X_test))
##%%
##all_days = pd.date_range(start=start_date,end=end_date,freq = 'D')
#events = pd.Series(np.zeros(len(data_analysed.index)), index=data_analysed.index)
##events = events.replace()
#events[events.index.isin(anomalies)] = 1
##events2 = pd.Series(data_analysed.iloc[:,(_)].values, index=data_analysed.index)
##plt.figure()
##calmap.yearplot(events2, year=start_date.year)
##plt.figure()
#calmap.calendarplot(events)

#%%
#
#for jo in classi:
#    clf = classi[jo]
#    clf.fit(X_train)
#    b = 0
#    ans = 0
#    for i in range(1,100):
#        a = np.sum(((clf.predict_proba(X_test)[:,1]) >=(i/100)).astype(int) ==  clf.predict(X_test))
#        if a > b:
#            b = a
#            ans = i
#    print(jo,ans)
#    plt.plot(((clf.predict_proba(X_test)[:,1]) >=ans/100).astype(int) ==  clf.predict(X_test))
#    train_scores = clf.decision_scores_
#    test_scores = clf.decision_function(X_test)
#    threshold = scoreatpercentile(train_scores, 100 * (1 - outliers_fraction))
#    predsss = (test_scores > threshold).astype('int')
#    print(predsss == clf.predict(X_test))
###%%
#for jo in classi:
#    clf = classi[jo]
#    clf.fit(X_train)
#    for dt in rrule(DAILY, dtstart=start_date, until=end_date):
#        try:
#            data_analysed.loc[dt.strftime("%Y-%m-%d"),(str(classi[jo]).split("(")[0])] = clf.predict_proba(preprocessing.normalize(data_analysed.loc[dt.strftime("%Y-%m-%d")].value.values.reshape(1, -1)))[0][1]
#        except:
#            continue
#%%

    
#%%

random_state = np.random.RandomState(42)
classi = {
    'Cluster-based Local Outlier Factor (CBLOF)':
        (CBLOF(contamination=outliers_fraction,
              check_estimator=False, random_state=random_state)),
    'Histogram-base Outlier Detection (HBOS)': (HBOS(
        contamination=outliers_fraction)),
    'K Nearest Neighbors (KNN)': (KNN(
        contamination=outliers_fraction)),
    'Local Outlier Factor (LOF)':
        (LOF(n_neighbors=10, contamination=outliers_fraction))
}
outliers_fraction = 0.05
file_no = 2
data = preprocess(datadict[file_no][0]) 
anomalies = datadict[file_no][1]  
if file_no == 3: 
    data = data[data.index.weekday < 6]
start_date = data.head(1).index.date[0]
end_date = data.tail(1).index.date[0]
#%%
datatotrain,datatotest,datatotrain_normalized,datatotest_normalized,train_data,test_data = createtraintest(data)
X_train,X_test = datatotrain_normalized,datatotest_normalized
datax = data['value'].values.reshape(-1,144)
data_n = preprocessing.normalize(datax, norm='l2')
#%%
b = {}
for dt in rrule(DAILY, dtstart=start_date, until=end_date):
    try:
#        b[dt.strftime("%Y-%m-%d")] = (data[dt.strftime("%Y-%m-%d")].values.reshape(1,144))
        b[dt.strftime("%Y-%m-%d")] = data[dt.strftime("%Y-%m-%d")].values.flatten()
    except:
        continue
datao = pd.DataFrame(list(b.items()),columns=['date','value'])
datao = datao.set_index('date')
datao.index = pd.to_datetime(datao.index)
datao.loc[datao.index.isin(anomalies),'anomaly'] = 1
datao['anomaly'] = datao['anomaly'].fillna(0)
#%%
#X_train = datao.iloc[0:len(datao)//2,0].values
#X_train = [x for x in X_train if x.size != 0]
#X_train = preprocessing.normalize(X_train)
#X_test = datao.iloc[len(datao)//2:,0].values
#X_test = [x for x in X_test if x.size != 0]
#X_test = preprocessing.normalize(X_test)
#%%
dataoc = datao.copy()
for jo in classi:
    clf = classi[jo]
    clf.fit(data_n)
    for dt in rrule(DAILY, dtstart=start_date, until=end_date):
        try:
            datao.loc[dt.strftime("%Y-%m-%d"),(str(classi[jo]).split("(")[0])] = clf.predict_proba(preprocessing.normalize([datao.loc[dt.strftime("%Y-%m-%d")].value]))[0][1]
            dataoc.loc[dt.strftime("%Y-%m-%d"),(str(classi[jo]).split("(")[0])] = clf.predict(preprocessing.normalize([dataoc.loc[dt.strftime("%Y-%m-%d")].value]))[0]
        except:
            continue
#%%
events = datao.iloc[:,3]
calmap.calendarplot(events)
##%%
#for _ in range(2,6):
#    b = 0
#    ans = 0
#    for i in range(1,80):
#        a = np.nansum((datao.iloc[:,_].values >=(i/100)).astype(int) ==  datao.anomaly.values)
#        if a > b:
#            b = a
#            ans = i
#    print(ans,b)
#%%
for _ in range(2,6):    
    a = np.sum(dataoc.iloc[:,_].values ==  dataoc.anomaly.values)
    print(a)
#%%
datao.loc[datao.index.isin(anomalies),('LOF','CBLOF','KNN','HBOS','anomaly')]

#%%
datao[datao.index.isin(anomalies)].iloc[:,2:6].sum(axis=1)/4
#%%
events = dataoc.iloc[:,4]
calmap.calendarplot(events)
#%%
from sklearn.decomposition import PCA
#%%
plt.figure()
pca1 = PCA(2)  # project from 144 to 2 dimensions
projected1 = pca1.fit_transform(datax)
print(datax.shape)
print(projected1.shape)

plt.scatter(projected1[:, 0], projected1[:, 1],
            edgecolor='none', alpha=0.5,color='grey')
plt.xlabel('component 1')
plt.ylabel('component 2')

for _ in anomalies:
    try:
        a = pca1.transform([data[_].value.values])
        plt.scatter(a[:,0],a[:,1],alpha = 1,color = 'red')
    except:
        continue
lof = dataoc[dataoc.iloc[:,3] == 1]['value'].values
for _ in lof:
    try:
        a_ = pca1.transform([_])
        plt.scatter(a_[:,0],a_[:,1],alpha = 0.3,color = 'blue')
    except:
        continue
plt.show()
#%%
plt.figure()
pca = PCA(2)  # project from 144 to 2 dimensions
projected = pca.fit_transform(data_n)
print(data_n.shape)
print(projected.shape)

plt.scatter(projected[:, 0], projected[:, 1],
            edgecolor='none', alpha=0.5,color = 'grey')
plt.xlabel('component 1')
plt.ylabel('component 2')

for _ in anomalies:
    try:
        aa = pca.transform(preprocessing.normalize([data[_].value.values]))
        plt.scatter(aa[:,0],aa[:,1],alpha = 1,color = 'red')
    except:
        continue

for _ in lof:
    try:    
        aa_ = pca.transform(preprocessing.normalize([_]))
        plt.scatter(aa_[:,0],aa_[:,1],alpha = 0.3,color = 'blue')
    except:
        continue
plt.show()
#%%
pca.transform(data.values.reshape(-1,144))
#%%

#data.index.time()
#data.index.date()

data['date'] = data.index.date
data['time'] = data.index.time
#%%
table = data.pivot_table(index = ["date"], columns = ["time"], values = "value")
table.tail()
#%%
table = table.fillna(0).reset_index()
#%%
preprocessing.normalize(table[table.columns[1:]],axis=0)[235]
#%%
clf = KNN()
table["KNN"] = clf.fit_predict(preprocessing.normalize(table[table.columns[1:]]))
#%%
table

