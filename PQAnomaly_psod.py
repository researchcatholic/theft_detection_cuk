# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 04:16:09 2019

@author: p0p
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager

# Import all models
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

from sklearn import metrics 
from sklearn import preprocessing

from datetime import date,timedelta
import datetime
from dateutil.rrule import rrule, DAILY
#%%
data_1 = pd.read_csv("C:/Users/p0p/Desktop/anaomaly/1year_Example-1.csv")
data_2 = pd.read_csv("C:/Users/p0p/Desktop/anaomaly/1year_Example-2.csv")
data_3 = pd.read_csv("C:/Users/p0p/Desktop/anaomaly/1year_Example-3.csv")
data_4 = pd.read_csv("C:/Users/p0p/Desktop/anaomaly/1year_Example-4.csv")

anomalies_1 = ['2012-09-10','2012-09-11','2012-09-12','2012-09-15','2012-09-16','2012-10-30','2012-11-16','2012-11-30','2012-12-02','2012-12-03','2012-12-04','2012-12-22']
anomalies_2 = ['2011-03-04','2011-03-06','2011-04-17','2011-04-22','2011-05-06','2011-05-19','2011-05-20','2011-05-22','2011-05-23','2011-05-24']
anomalies_3 = ['2014-01-17','2014-01-29','2014-02-26','2014-04-04','2014-04-05','2014-04-11','2014-04-12','2014-04-22','2014-04-30','2014-05-12','2014-05-13','2014-05-26']
anomalies_4 = ['2014-05-01','2014-06-10','2014-06-17','2014-07-08','2014-07-10','2014-09-02','2014-10-17','2014-10-23','2014-10-24']
#%%
def preprocess(data_ ,plotp = 0):
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
    if plotp == 1:
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
file_no,plotp = 3,0
anomalies = anomalies_3
data = preprocess(data_3,plotp)

start_date = data.head(1).index.date[0]
end_date = data.tail(1).index.date[0]
middle_date = start_date + (end_date-start_date)/2
datatotrain,datatotest,datatotrain_normalized,datatotest_normalized,dataanomaly_normalized,train_data,test_data,mix_data = createtraintest(data,anomalies,file_no)
X_train,X_test,X_outliers = datatotrain_normalized,datatotest_normalized,dataanomaly_normalized
#%%
clf =  LOF(n_neighbors=10, contamination=0.1)
clf.fit(X_train)
#%%
#y_pred = clf.fit_predict(X_test)
i=0
for dt in rrule(DAILY, dtstart=start_date, until=end_date):
    try:
        if (data.loc[dt.strftime("%Y-%m-%d")]['value']).values.size != 0:
            data.loc[dt.strftime("%Y-%m-%d"),'ocsvm_score'] =clf.predict(preprocessing.normalize(data.loc[dt.strftime("%Y-%m-%d")].value.values.reshape(1, -1)))
            i = i + 1
    except:
        print(dt,i)
        continue
data.loc[:,['value','ocsvm_score']].plot(figsize=(144, 4))
for dt in rrule(DAILY, dtstart=start_date, until=end_date):
    plt.axvline(x=dt.strftime("%Y-%m-%d"))
    if dt.strftime("%Y-%m-%d") in anomalies: 
        plt.axvspan(dt, dt+ timedelta(days=1), color='r', alpha=0.2)

OSCVMS = np.unique(data[data.ocsvm_score == 1].index.date.astype(str))
#print("LOCAL OUTLIER FACTOR:")
print(OSCVMS)