# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:09:45 2019

@author: p0p
"""
import pandas as pd
import random
import numpy as np
import timeit
import math
from sklearn import preprocessing 
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from xgboost import train
from xgboost import DMatrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import plot_importance
from sklearn import svm
import lightgbm
from scipy import stats
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import warnings
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from datetime import timedelta
import datetime
from sklearn.preprocessing import PowerTransformer
from dateutil.rrule import rrule, DAILY
import glob,os
##%%
##filepath = "C:/Users/p0p/Desktop/Power-Networks-LCL-June2015(withAcornGps)v2.csv"
#filepath = "C:/Users/p0p/Desktop/anaomaly\London carbon/Power-Networks-LCL-June2015(withAcornGps).csv_Pieces/Power-Networks-LCL-June2015(withAcornGps).csv_Pieces/Power-Networks-LCL-June2015(withAcornGps)v2_137.csv"
#df = pd.read_csv(filepath)
##%%
#df.isnull().any()
#df.columns
#df = df.replace('Null', 0)
##df['KWH/hh (per half hour) '].astype('float64')
#df['KWH/hh (per half hour) '] = pd.to_numeric(df['KWH/hh (per half hour) '])
#df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S.%f')
#df = df.rename(columns={"KWH/hh (per half hour) ": "kwh"})
#df['kwh'].dtype
#c_no1 = df[df['stdorToU'] == 'ToU'].LCLid.unique()
##%%
#df = df[df['LCLid'].isin(c_no1)]
#%%file2
dir = "C:/Users/p0p/Desktop/anaomaly\London carbon/Power-Networks-LCL-June2015(withAcornGps).csv_Pieces/test"
#%%
def dataframe(filepath):
    df2 = pd.read_csv(filepath)
    #df2.isnull().any()
    #df2.columns
    df2 = df2.replace('Null', 0)
    #df['KWH/hh (per half hour) '].astype('float64')
    df2['KWH/hh (per half hour) '] = pd.to_numeric(df2['KWH/hh (per half hour) '])
    df2['DateTime'] = pd.to_datetime(df2['DateTime'], format='%Y-%m-%d %H:%M:%S.%f')
    df2 = df2.rename(columns={"KWH/hh (per half hour) ": "kwh"})
    #df2['kwh'].dtype
    c_no2 = df2[df2['stdorToU'] == 'ToU'].LCLid.unique()
    df2 = df2[df2['LCLid'].isin(c_no2)]
    return df2,c_no2
#%%
frames = []
c_no1 = []
os.chdir(dir)
for _ in glob.glob("*.csv"):
    temp,c_not = dataframe(_)
    frames.append(temp)
    c_no1.extend(c_not.tolist())
df = pd.concat(frames)
len(c_no1)
#%%
#%%
tariff = pd.read_excel("C:/Users/p0p/Desktop/anaomaly/London carbon/tariffs.xlsx")
#tariff2 = pd.read_excel("C:/Users/p0p/Desktop/anaomaly/London carbon/tariffs.xlsx")
#%%
prices = {'High' : 67.20,'Low' : 3.99,'Normal' : 11.76}
#%%
def price(row):
    return prices[row['Tariff']]
#%%
tariff['prices'] = tariff.apply(price,axis = 1)
#%%
tstart = tariff.iloc[0]['TariffDateTime'].date()
tend = tariff.iloc[-1]['TariffDateTime'].date()
tariff.replace(['Normal','High', 'Low'], [1,1,0], inplace = True)
#tariff2.replace(['Normal','High', 'Low'], [1,1,-1], inplace = True)
#%%
total_price_day = []
high_med_price = []
low_price = []
k = df[(df['LCLid'] == c_no1[2])]
start_date = tariff.iloc[0].TariffDateTime.date()
end_date = tariff.iloc[-1].TariffDateTime.date()
#%%
for _ in rrule(DAILY, dtstart=start_date, until=end_date):
    a = (k[k['DateTime'].dt.date == _.date()].kwh.values)
    if a.shape[0] == 48:
        b = (tariff[tariff['TariffDateTime'].dt.date == _.date()].Tariff.values)
        c = (tariff[tariff['TariffDateTime'].dt.date == _.date()].prices.values)
        total_price_day.append(sum(a*c))
        high_med_price.append(sum(a*c*b))
        low_price.append(sum(a*c) - sum(a*c*b))
    else:
        continue
#%%5 vectors
        
def tarray(lister):
    transformed_array = np.array([np.min(lister), np.mean(lister), np.std(lister), np.max(lister),np.size(lister)])
    return transformed_array.reshape(-1,5)
def features_5_generator(no):
    k = df[(df['LCLid'] == c_no1[no])]
    X = np.array([]).reshape(0,5)
    y = np.array([])
    
    #clf = SVC(kernel='rbf',probability=True)
    j = 0
    print(c_no1[no], "using statistic 5 feature")
    for dat in rrule(DAILY, dtstart=tstart, until=tend):
        j = j + 1
        a = k[k['DateTime'].dt.date == dat.date()].kwh.values
        price_of_day = tariff[tariff['TariffDateTime'].dt.date == dat.date()].Tariff.values
        if a.shape[0] == 48:
            listr = []
            for i in range(0,len(price_of_day)):
                if price_of_day[i] == 1:
                    listr.append((price_of_day[i]*a[i]))
            if len(listr) != 0:
                random.seed(j)
                p = (random.randint(1,9))/10
                b=np.array(listr)
                t1 =  np.array([x * p for x in b])
                t2 = np.array([z * random.randint(0,1) for z in b])
                t3 = np.array([y * (random.randint(1,10)/10) for y in b])
                t4 =  np.array([np.mean(b)*(random.randint(1,10))/10 for z in b])
                t5 =  np.array([np.mean(b) for t in b])

                b,t4,t5,t3,t2,t1= tarray(b),tarray(t4),tarray(t5),tarray(t3),tarray(t2),tarray(t1)
                X= np.concatenate((X, b,t1,t2,t3,t4,t5), axis=0)
                y = np.append(y,0)
                for i in range(0,5):   
                    y = np.append(y,1)
        else:
            continue
    return X,y
#%%48-vectors
def features_48_generator(no):
    k = df[(df['LCLid'] == c_no1[no])]
    X = np.array([]).reshape(0,48)
#    X = np.array([]).reshape(0,52)
    y = np.array([])

    #clf = SVC(kernel='rbf',probability=True)
    j = 0
    print(c_no1[no], "using 48 features")
    for dat in rrule(DAILY, dtstart=start_date, until=end_date):
        j = j + 1
        sub = k[k['DateTime'].dt.date == dat.date()]
        price_of_day = tariff[tariff['TariffDateTime'].dt.date == dat.date()].Tariff.values
        if sub.shape[0] == 48:
            random.seed(j)
            p = (random.randint(1,9))/10
            b=sub.kwh.values.reshape(-1,48)
            t1,t2,t3,t4,t5 = b.copy(), b.copy(), b.copy(), b.copy(), b.copy()
            for _ in range(0,len(b[0])):
                if price_of_day[_] == 1:
                    t1[0][_] = b[0][_]*p
                    t2[0][_] = b[0][_]*random.randint(0,1)
                    t3[0][_] = b[0][_]*(random.randint(1,9)/10)
                    t4[0][_] = b[0][_]*np.mean(b[0][price_of_day == 1])*(random.randint(1,9))/10
                    t5[0][_] = b[0][_]*np.mean(b[0][price_of_day == 1])
            X= np.concatenate((X, b,t1,t2,t3,t4,t5), axis=0)
            y = np.append(y,0)
            for i in range(0,5):   
                y = np.append(y,1)
        else:
            continue
    return X,y
#%%52 vectors
def features_52_generator(no):
    k = df[(df['LCLid'] == c_no1[no])]
    X = np.array([]).reshape(0,52)
    y = np.array([])

    #clf = SVC(kernel='rbf',probability=True)
    j = 0
    print(c_no1[no], "using 52 features")
    for dat in rrule(DAILY, dtstart=start_date, until=end_date):
        j = j + 1
        sub = k[k['DateTime'].dt.date == dat.date()]
        price_of_day = tariff[tariff['TariffDateTime'].dt.date == dat.date()].Tariff.values
        if sub.shape[0] == 48:
            random.seed(j)
            p = (random.randint(1,9))/10
            b=sub.kwh.values.reshape(-1,48)
            t1,t2,t3,t4,t5 = b.copy(), b.copy(), b.copy(), b.copy(), b.copy()
            for _ in range(0,len(b[0])):
                if price_of_day[_] == 1:
                    t1[0][_] = b[0][_]*p
                    t2[0][_] = b[0][_]*random.randint(0,1)
                    t3[0][_] = b[0][_]*(random.randint(1,9)/10)
                    t4[0][_] = b[0][_]*np.mean(b[0][price_of_day == 1])*(random.randint(1,9))/10
                    t5[0][_] = b[0][_]*np.mean(b[0][price_of_day == 1])

            b = np.append(b,[(b.mean()),b.std(),b.min(),b.max()]).reshape(1,-1)
            t1 = np.append(t1,[t1.mean(),t1.std(),t1.min(),t1.max()]).reshape(1,-1)
            t2 = np.append(t2,[t2.mean(),t2.std(),t2.min(),t2.max()]).reshape(1,-1)
            t3 = np.append(t3,[t3.mean(),t3.std(),t3.min(),t3.max()]).reshape(1,-1)
            t4 = np.append(t4,[t4.mean(),t4.std(),t4.min(),t4.max()]).reshape(1,-1)
            t5 = np.append(t5,[t5.mean(),t5.std(),t5.min(),t5.max()]).reshape(1,-1)
            X= np.concatenate((X, b,t1,t2,t3,t4,t5), axis=0)
            y = np.append(y,0)
            for i in range(0,5):   
                y = np.append(y,1)
        else:
            continue
    return X,y
#%%48original
#def features_48_generator_full(no):
#    k = df[(df['LCLid'] == c_no1[no])]
#    X = np.array([]).reshape(0,48)
#    y = np.array([])
#
#    #clf = SVC(kernel='rbf',probability=True)
#    j = 0
#    print(c_no1[no], "using 48 features")
#    for dat in rrule(DAILY, dtstart=start_date, until=end_date):
#        j = j + 1
#        sub = k[k['DateTime'].dt.date == dat.date()]
#        price_of_day = tariff2[tariff2['TariffDateTime'].dt.date == dat.date()].Tariff.values
#        if sub.shape[0] == 48:
#            random.seed(j)
#            p = (random.randint(1,9))/10
#            b=sub.kwh.values.reshape(-1,48)
#            t1,t2,t3,t4,t5 = b.copy(), b.copy(), b.copy(), b.copy(), b.copy()
#            for _ in range(0,len(b[0])):
#                t1[0][_] = b[0][_]*p
#                t2[0][_] = b[0][_]*random.randint(0,1)
#                t3[0][_] = b[0][_]*(random.randint(1,9)/10)
#                t4[0][_] = b[0][_]*np.mean(b[0][price_of_day == 1])*(random.randint(1,9))/10
#                t5[0][_] = b[0][_]*np.mean(b[0][price_of_day == 1])
#        
##            b = b*price_of_day
##            t1 = t1*price_of_day
##            t2 = t2*price_of_day
##            t3 = t3*price_of_day
##            t4 = t4*price_of_day
##            t5 = t5*price_of_day
##            t6 = t6*price_of_day
#                    
##            b,t1,t2,t3,t4,t5= t2array(b),t2array(t1),t2array(t2),t2array(t3),t2array(t4),t2array(t5)
##            t6 = t2array(t6)
##             print(b.shape,t1.shape,t2.shape,t3.shape,t4.shape,t5.shape,t6.shape)
##                     b = np.append(b,[(b.mean()),b.std()])
##                     t1 = np.append(t1,[t1.mean(),t1.std()])
##                     t2 = np.append(t2,[t2.mean(),t2.std()])
##                     t3 = np.append(t3,[t3.mean(),t3.std()])
##                     t4 = np.append(t4,[t4.mean(),t4.std()])
##                     t5 = np.append(t5,[t5.mean(),t5.std()])
##                     t6 = np.append(t6,[t6.mean(),t6.std()])
#            X= np.concatenate((X, b,t1,t2,t3,t4,t5), axis=0)
#            y = np.append(y,0)
#            for i in range(0,5):   
#                y = np.append(y,1)
#        else:
#            continue
#    return X,y

#%%detector
def features_f_detector(no,clf,f):
    customer_meter = c_no1[no]
    if f == 48:
        X,y = features_48_generator(no)
    elif f == 52:
        X,y = features_52_generator(no)
    elif f == 5:
        X,y = features_5_generator(no)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.14, random_state=0)
    sm = SMOTE(random_state=42)
    X_res_train, y_res_train = sm.fit_sample(X_train, y_train)
    X_res_test, y_res_test = sm.fit_sample(X_test, y_test)
    ts = timeit.default_timer()
    clf.fit(X_res_train, y_res_train)
    score = clf.score(X_res_test, y_res_test)
    #print(Counter(y),Counter(y_train),Counter(y_test),Counter(y_res_train),Counter(y_res_test))
    #print("The score for customer :", customer_input, " is ",  score)
    y_pred = clf.predict(X_res_test)
    probs = clf.predict_proba(X_res_test)
    preds = probs[:,1]
#    print(confusion_matrix(y_res_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_res_test, y_pred).ravel()
#    print("tn, fp, fn, tp",tn, fp, fn, tp)
    specificity = tn / (tn+fp)
    sensitivity =  tp/ (tp+fn)
    fpr =  1 - specificity
    print ("sensi = %.2f" %sensitivity, "fpr= %.2f" % fpr )
    total =sensitivity
    print("The score for customer :", customer_meter, " is %.2f" %  total)
    te = timeit.default_timer()
    tim = te- ts
#    plot_importance(clf,importance_type="weight", ax=plt.gca())
    return sensitivity,fpr,tim
#%%
clf = XGBClassifier()
#    clf = SVC(kernel='rbf',probability=True)
#    clf = LGBMClassifier()
#    clf = CatBoostClassifier(logging_level = "Silent")
de, fe, tim = {},{},{}
for f in (48,52,5):
    oo=0
    de[f],fe[f],tim[f] = [],[],[]
    for o in range(0,100):
        try:
            oo = oo + 1
            a,b, tima = features_f_detector(o,clf,f)
            de[f].append(a)
            fe[f].append(b)
            tim[f].append(tima)
        except:
            continue
#%%
#for f in (48,52,5):
#    ts = timeit.default_timer()
#    print(features_f_detector(1,clf,f))
#    te = timeit.default_timer()
#    print(ts-te)
#%%
for _ in de:
#    print(np.mean(de[_]),_)
    plt.scatter( np.arange(1,len(de[_])+1),de[_],label = ("%0.2f"%np.mean(de[_]),_))
    plt.legend()
    plt.show()
#%%
for _ in de:
#    print(np.mean(de[_]),_)
    plt.plot(de[_], label = ("%0.2f"%np.mean(de[_]),_))
    plt.legend()
    plt.show()
#%%
plt.figure()
for _ in fe:
#    print(np.mean(fe[_]),_)
    plt.plot(fe[_], label = ("%0.2f"%np.mean(fe[_]),_))
    plt.legend()
    plt.show()
#%%
plt.figure()
for _ in tim:
#    print(np.mean(tim[_]),_)
    plt.plot(tim[_], label = ("%0.2f"%np.mean(tim[_]),_))
    plt.legend()
    plt.show()
#%%