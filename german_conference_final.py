# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:59:12 2019

@author: p0p
"""

import matplotlib.pyplot as plt
import numpy as np; np.random.seed(sum(map(ord, 'calmap')))
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from scipy.stats import scoreatpercentile
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
data_1 = pd.read_csv("C:/Users/User/Desktop/German anomaly/1year_Example-1.csv")
data_2 = pd.read_csv("C:/Users/User/Desktop/German anomaly/1year_Example-2.csv")
data_3 = pd.read_csv("C:/Users/User/Desktop/German anomaly/1year_Example-3.csv")
data_4 = pd.read_csv("C:/Users/User/Desktop/German anomaly/1year_Example-4.csv")

#new
anomalies_1 = ["2012-02-04", "2012-07-18", "2012-12-03", "2012-12-04", "2012-11-16", "2012-12-02", "2012-02-05", "2012-05-14", "2012-05-15", "2012-05-16", "2012-07-18", "2012-09-10", "2012-09-11", "2012-09-12", "2012-09-15", "2012-09-16", "2012-10-30", "2012-11-16", "2012-12-02", "2012-12-03", "2012-12-04"]
anomalies_2 = ["2011-02-21", "2011-01-08", "2011-07-02",  "2011-09-10", "2011-02-17", "2011-02-20", "2011-02-27", "2011-03-04", "2011-03-06", "2011-04-17", "2011-04-22", "2011-05-01", "2011-05-06", "2011-05-31", "2011-06-20", "2011-10-09", "2011-10-16", "2011-10-27", "2011-05-19", "2011-05-20", "2011-05-21", "2011-05-22", "2011-05-23", "2011-05-24", "2011-06-08", "2011-08-14", "2011-08-27"]
anomalies_3 = ["2014-01-17", "2014-01-29", "2014-02-26", "2014-04-22", "2014-04-30", "2014-05-26", "2014-09-22", "2014-10-23", "2014-10-24", "2014-01-30", "2014-01-31", "2014-04-04", "2014-04-05", "2014-04-11", "2014-04-12", "2014-04-13", "2014-05-12", "2014-05-13", "2014-06-25", "2014-06-26", "2014-08-24", "2014-09-11", "2014-09-12", "2014-10-05"]
anomalies_4 = ["2013-10-07", "2013-10-11",  "2012-11-04", "2013-11-06", "2013-11-13", "2013-11-21", "2014-04-15", "2014-05-24", "2014-05-25", "2013-05-03", "2013-10-29"]



datadict = {1:(data_1,anomalies_1), 2:(data_2,anomalies_2), 3:(data_3,anomalies_3), 4:(data_4,anomalies_4)}
#%% ano type labelling
def ano_lbl(anomalies):
    lbl_ano = {}
    for __ in anomalies:
        lbl_ano[__] = np.random.randint(0,2)
    return lbl_ano
#%%
ano_lbl_1 = ano_lbl(anomalies_1)
ano_lbl_2 = ano_lbl(anomalies_2)
ano_lbl_3 = ano_lbl(anomalies_3)
ano_lbl_4 = ano_lbl(anomalies_4)
ano_type_dict = {1:ano_lbl_1,2:ano_lbl_2,3:ano_lbl_3,4:ano_lbl_4}
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
def plot_confusion_matrix(cm, classes=('0','1'),
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
#%%
results_table1_ = {}
results_table2_ = {}
#%%
file_no = 1
outliers_fraction = 0.3
random_state = np.random.RandomState(42)
classi = {
    'Cluster-based Local Outlier Factor (CBLOF)':
        (CBLOF(contamination=outliers_fraction,
              check_estimator=False, random_state=random_state,n_clusters=40)),
    'Histogram-base Outlier Detection (HBOD)': (HBOS(
        contamination=outliers_fraction)),
    'K Nearest Neighbors (KNN)': (KNN(
        contamination=outliers_fraction,n_neighbors=10)),
    'Local Outlier Factor (LOF)':
        (LOF(n_neighbors=20, contamination=outliers_fraction))
}
    #%%
data = preprocess(datadict[file_no][0]) 
anomalies = datadict[file_no][1]
ano_type_lbl = ano_type_dict[file_no]
if file_no == 3: 
    data = data[data.index.weekday < 6]
start_date = data.head(1).index.date[0]
end_date = data.tail(1).index.date[0]
#%%
data['date'] = pd.DatetimeIndex(data.index.date)
data['time'] = data.index.time
table = data.pivot_table(index = ["date"], columns = ["time"], values = "value")
table = table.fillna(0).reset_index()
table.tail()
table['anomalies'] = 0
for _ in anomalies:
    table['anomalies'][table['date'] == _] = 1
#%%
table['ano_type'] = 'not'
for __ in ano_type_lbl:
    table.loc[table['date'] == __, 'ano_type'] = ano_type_lbl[__]
#%%
X = preprocessing.normalize(table[table.columns[1:145]].values)
y = table[table.columns[145]].values
##%%
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=12)
#%%
#%%
outliers_fraction = 0.1
classifiers = ['CBLOF', 'HBOS', 'KNN', 'LOF']
para_range = {'CBLOF': (' 10', '50'), 'HBOS': (' 4', '50'), 'KNN': ('4', '30'), 'LOF': ('4', '20')}
answers_dict = {}
#%%
def ranger(parameter,classifier):
    __ = parameter
    classi__ = {
    'CBLOF':
        (CBLOF(contamination=outliers_fraction,
              check_estimator=False, random_state=random_state,n_clusters=__)),
    'HBOS': (HBOS(
        contamination=outliers_fraction, n_bins=__)),
    'KNN': (KNN(
        contamination=outliers_fraction,n_neighbors=__)),
    'LOF':
        (LOF(n_neighbors=__, contamination=outliers_fraction))
        }
    return classi__[classifier] 
#%%
#classifier = classifiers[3]
for clf_no in range(0,4):
    classifier = classifiers[clf_no]
    a,b = para_range[classifier]
    answers_dict[classifier] = {}
    answers_dict[classifier]['graph'] = []
    maxx = 0
    answers_dict[classifier]['best'] = []
    for __ in range(int(a),int(b)):
        clf = ranger(__,classifier)
        y_pred = clf.fit_predict(X)
        ano_de = confusion_matrix(y, y_pred)[1][1]
        answers_dict[classifier]['graph'].append(ano_de) 
        if ano_de > maxx:
            maxx = ano_de
            par = __
#    plt.figure()
#    plt.scatter(np.arange(int(a),int(b)),answers_dict[classifier]['graph'])
#    plt.xticks(np.arange(int(a),int(b)))
#    plt.xlabel("range of the parameter")
#    plt.ylabel("%s"%(classifier))
#    plt.show()
    answers_dict[classifier]['best'] = (par, maxx)
    clf = ranger(par,classifier)
    y_pred = clf.fit_predict(X)
    ano_de = confusion_matrix(y, y_pred)[1][1]
    table['pred_by %s'%(classifier)] = y_pred
    answers_dict[classifier]['single'] = table[(table['pred_by %s'%(classifier)] == 1)&(table['ano_type'] == 1)].shape[0]
    answers_dict[classifier]['multiple'] = table[(table['pred_by %s'%(classifier)] == 1)&(table['ano_type'] == 1)].shape[1]

#%%
for clf_no in range(0,4):
    classifier = classifiers[clf_no]
    plt.figure()
    plt.title("%s" %(classifier))
    for __ in table.index[(table['ano_type'] == 1)]:
        plt.plot(table.iloc[__][1:145].values,color = 'black',linestyle='-.')
    for __ in table.index[(table['pred_by %s'%(classifier)] == 1)&(table['ano_type'] == 1)]:
        plt.plot(table.iloc[__][1:145].values,color = 'black')
    
    #%%
    for __ in table.index[(table['ano_type'] == 0)]:
        plt.plot(table.iloc[__][1:145].values,color = 'black',linestyle='-.')
    for __ in table.index[(table['pred_by %s'%(classifier)] == 1)&(table['ano_type'] == 0)]:
        plt.plot(table.iloc[__][1:145].values,color = 'black')
        
    #%%
    for __ in table.index[(table['pred_by %s'%(classifier)] == 1)&(table['anomalies'] == 0)]:
        plt.plot(table.iloc[__][1:145].values,color = 'black')
        
#%%
    



#%%
for jo in classi:
    clf = classi[jo]
    table[(str(classi[jo]).split("(")[0])] = clf.fit_predict(preprocessing.normalize(table[table.columns[1:145]].values))
#    del clf
#    table[(str(classi[jo]).split("(")[0])] = clf.fit_predict((table[table.columns[1:145]].values))

for c in [-1,-2,-3,-4]:
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()
    for _ in table.index[table[table.columns[c]] == 1]:
        axs[0,0].plot(table.iloc[_][1:145].values,color= 'y')
    for _ in table.index[table.anomalies == 1]:
        axs[0,1].plot(table.iloc[_][1:145].values,color= 'r')
    for _ in table.index[table.anomalies == 1]:
        if table.iloc[_][table.columns[c]] == 1:
            axs[1,0].plot(table.iloc[_][1:145].values,color= 'g')
        if table.iloc[_][table.columns[c]] == 0:
            axs[1,1].plot(table.iloc[_][1:145].values,color= 'b')
    axs[0,0].set_title('Predicted by the algorithm : %d'%((table.index[table[table.columns[c]] == 1].shape[0])))
    axs[0,1].set_title('Assumed anomalies: %d'%((table.index[table[table.columns[-5]] == 1].shape[0])))
    axs[1,0].set_title('Assumed and predicted right: %d'%(sum(table[table[table.columns[-5]] == 1][table.columns[c]] == 1)))
    axs[1,1].set_title('Assumed but not predicted: %d'%(sum(table[table[table.columns[-5]] == 1][table.columns[c]] == 0)))
    fig.suptitle(table.columns[c])
    plt.show()
    plt.savefig('%d' %(file_no) + table.columns[c] + '%d' %(outliers_fraction) + '.png')
#%%
for c in [-1,-2,-3,-4]:
    y_pred = table[table.columns[c]].values
    y_true = table.anomalies.values
    print(classification_report(y_true, y_pred))
    cnf = confusion_matrix(y_true,y_pred)
    plt.figure()
    print('%d' %(file_no) + table.columns[c] + '%d' %(outliers_fraction))
    plot_confusion_matrix(cnf)
    plt.title(table.columns[c] + ' file no. %d' %(file_no))
    plt.show()
    print('accuracy score = %.2f' %(accuracy_score(y_true,y_pred)))
    plt.savefig('%d' %(file_no) + table.columns[c]  + '%d' %(outliers_fraction) + 'cm.png')
#%%

#%%
file_no = 4
outliers_fraction = 0.3
random_state = np.random.RandomState(42)
classi = {
    'Cluster-based Local Outlier Factor (CBLOF)':
        (CBLOF(contamination=outliers_fraction,
              check_estimator=False, random_state=random_state,n_clusters=10)),
    'Histogram-base Outlier Detection (HBOS)': (HBOS(
        contamination=outliers_fraction)),
    'K Nearest Neighbors (KNN)': (KNN(
        contamination=outliers_fraction,n_neighbors=10)),
    'Local Outlier Factor (LOF)':
        (LOF(n_neighbors=20, contamination=outliers_fraction))
}
data = preprocess(datadict[file_no][0]) 
anomalies = datadict[file_no][1]  
if file_no == 3: 
    data = data[data.index.weekday < 6]
start_date = data.head(1).index.date[0]
end_date = data.tail(1).index.date[0]

data['date'] = pd.DatetimeIndex(data.index.date)
data['time'] = data.index.time
table = data.pivot_table(index = ["date"], columns = ["time"], values = "value")
table = table.fillna(0).reset_index()
table.tail()

table['anomalies'] = 0
for _ in anomalies:
    table['anomalies'][table['date'] == _] = 1
    
for jo in classi:
    clf = classi[jo]
    table[(str(classi[jo]).split("(")[0])] = clf.fit_predict(preprocessing.normalize(table[table.columns[1:145]].values))
#    del clf
    
fig, axs = plt.subplots(4, 3)    
#fig.tight_layout()
c = -1
for _ in table.index[table[table.columns[c]] == 1]:
    axs[0,0].plot(table.iloc[_][1:145].values,color= 'y')
for _ in table.index[table.anomalies == 1]:
    if table.iloc[_][table.columns[c]] == 1:
        axs[0,1].plot(table.iloc[_][1:145].values,color= 'g')
    if table.iloc[_][table.columns[c]] == 0:
        axs[0,2].plot(table.iloc[_][1:145].values,color= 'b')
axs[0,0].set_title('Predicted by the %s : %d'%( table.columns[c], (table.index[table[table.columns[c]] == 1].shape[0])))
#axs[0,1].set_title('Assumed anomalies: %d'%((table.index[table[table.columns[-5]] == 1].shape[0])))
axs[0,1].set_title('Assumed and predicted right: %d'%(sum(table[table[table.columns[-5]] == 1][table.columns[c]] == 1)))
axs[0,2].set_title('Assumed but not predicted: %d'%(sum(table[table[table.columns[-5]] == 1][table.columns[c]] == 0)))

c = -2
for _ in table.index[table[table.columns[c]] == 1]:
    axs[1,0].plot(table.iloc[_][1:145].values,color= 'y')
for _ in table.index[table.anomalies == 1]:
    if table.iloc[_][table.columns[c]] == 1:
        axs[1,1].plot(table.iloc[_][1:145].values,color= 'g')
    if table.iloc[_][table.columns[c]] == 0:
        axs[1,2].plot(table.iloc[_][1:145].values,color= 'b')
axs[1,0].set_title('Predicted by the %s : %d'%( table.columns[c], (table.index[table[table.columns[c]] == 1].shape[0])))
#axs[1,1].set_title('Assumed anomalies: %d'%((table.index[table[table.columns[-5]] == 1].shape[0])))
axs[1,1].set_title('Assumed and predicted right: %d'%(sum(table[table[table.columns[-5]] == 1][table.columns[c]] == 1)))
axs[1,2].set_title('Assumed but not predicted: %d'%(sum(table[table[table.columns[-5]] == 1][table.columns[c]] == 0)))

c = -3
for _ in table.index[table[table.columns[c]] == 1]:
    axs[2,0].plot(table.iloc[_][1:145].values,color= 'y')
for _ in table.index[table.anomalies == 1]:
    if table.iloc[_][table.columns[c]] == 1:
        axs[2,1].plot(table.iloc[_][1:145].values,color= 'g')
    if table.iloc[_][table.columns[c]] == 0:
        axs[2,2].plot(table.iloc[_][1:145].values,color= 'b')
axs[2,0].set_title('Predicted by the %s : %d'%( table.columns[c], (table.index[table[table.columns[c]] == 1].shape[0])))
#axs[2,1].set_title('Assumed anomalies: %d'%((table.index[table[table.columns[-5]] == 1].shape[0])))
axs[2,1].set_title('Assumed and predicted right: %d'%(sum(table[table[table.columns[-5]] == 1][table.columns[c]] == 1)))
axs[2,2].set_title('Assumed but not predicted: %d'%(sum(table[table[table.columns[-5]] == 1][table.columns[c]] == 0)))

c = -4
for _ in table.index[table[table.columns[c]] == 1]:
    axs[3,0].plot(table.iloc[_][1:145].values,color= 'y')
for _ in table.index[table.anomalies == 1]:
    if table.iloc[_][table.columns[c]] == 1:
        axs[3,1].plot(table.iloc[_][1:145].values,color= 'g')
    if table.iloc[_][table.columns[c]] == 0:
        axs[3,2].plot(table.iloc[_][1:145].values,color= 'b')
axs[3,0].set_title('Predicted by the %s : %d'%( table.columns[c], (table.index[table[table.columns[c]] == 1].shape[0])))
#axs[0,1].set_title('Assumed anomalies: %d'%((table.index[table[table.columns[-5]] == 1].shape[0])))
axs[3,1].set_title('Assumed and predicted right: %d'%(sum(table[table[table.columns[-5]] == 1][table.columns[c]] == 1)))
axs[3,2].set_title('Assumed but not predicted: %d'%(sum(table[table[table.columns[-5]] == 1][table.columns[c]] == 0)))

plt.show()
#plt.savefig('%d' %(file_no) + table.columns[c] + '%d' %(outliers_fraction) + '.png')
#%%
plt.figure()
plt.title('Assumed anomalies: %d'%((table.index[table[table.columns[-5]] == 1].shape[0])))
for _ in table.index[table.anomalies == 1]:
    plt.plot(table.iloc[_][1:145].values,color= 'r')
#%%
table['TOTAL'] = table['CBLOF'] + table['LOF'] + table['KNN'] + table['HBOS']
fig, axs = plt.subplots(2, 2)
fig.tight_layout()
for _ in table.index[table[table.columns[-1]] == 4]:
    axs[0,0].plot(table.iloc[_][1:145].values,color= 'r')
for _ in table.index[table[table.columns[-1]] == 3]:
    axs[0,1].plot(table.iloc[_][1:145].values,color= 'b')
for _ in table.index[table[table.columns[-1]] == 2]:
    axs[1,0].plot(table.iloc[_][1:145].values,color= 'g')
for _ in table.index[table[table.columns[-1]] == 1]:
    axs[1,1].plot(table.iloc[_][1:145].values,color= 'y')
axs[0,0].set_title('ALL 4')
axs[0,1].set_title('ANY 3')
axs[1,0].set_title('ANY 2')
axs[1,1].set_title('ANY 1')
fig.suptitle('weighted predictions')
plt.show()
#%%
#days = np.random.choice(all_days, 500)
#events = pd.Series(table.anomalies.values, index=table.loc[:,'date'])
#calmap.calendarplot(events)
##%%
#del events
#events = pd.Series(table.LOF.values, index=table.loc[:,'date'])
#calmap.calendarplot(events)
#%%
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
#outliers_fraction = 0.05