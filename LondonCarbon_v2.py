
# coding: utf-8

# In[1]:


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

# In[3]:


file = pd.read_csv("C:/Users/p0p/Desktop/anaomaly\London carbon/Power-Networks-LCL-June2015(withAcornGps).csv_Pieces/Power-Networks-LCL-June2015(withAcornGps).csv_Pieces/Power-Networks-LCL-June2015(withAcornGps)v2_135.csv")
file.isnull().any()
file.columns
file = file.replace('Null', 0)
file['KWH/hh (per half hour) '] = pd.to_numeric(file['KWH/hh (per half hour) '])
file['DateTime'] = pd.to_datetime(file['DateTime'], format='%Y-%m-%d %H:%M:%S.%f')
file = file.rename({"KWH/hh (per half hour) ": "kwh"},axis='columns')
file['kwh'].dtype
#file[file['stdorToU'] == 'ToU'].LCLid.unique()[0]

## In[14]:
#
#
#k = file[(file['LCLid'] == 'MAC000005')]
#
#
## In[15]:
#
#
#k.drop_duplicates(inplace=True)
#
#
## In[16]:
#
#
#k[k['DateTime'].dt.date == datetime.date(2012, 11, 12)].DateTime
## 2012-11-12
#
#
## In[17]:
#
#
#sub = k[k['DateTime'].dt.date == datetime.date(2012, 11, 12)]
#
#
## In[18]:
#
#
#sub.DateTime.plot()
#
#
## In[19]:
#
#
#if (sub.DateTime.diff().dt.seconds > 1800).any():
#    print(sub[sub.DateTime.diff().dt.seconds > 1800].DateTime.index)
## sub[(sub.DateTime.diff().dt.seconds < 1800)
#
#
## In[20]:
#
#
#ixx = (sub[sub.DateTime.diff().dt.seconds > 1800].DateTime.index)[0]
#

# In[21]: c_no is the customers with TOU


c_no = file[file['stdorToU'] == 'ToU'].LCLid.unique()

#%%
file1 = pd.read_csv("C:/Users/p0p/Desktop/anaomaly\London carbon/Power-Networks-LCL-June2015(withAcornGps).csv_Pieces/Power-Networks-LCL-June2015(withAcornGps).csv_Pieces/Power-Networks-LCL-June2015(withAcornGps)v2_136.csv")
file1.isnull().any()
file1.columns
file1 = file1.replace('Null', 0)
file1['KWH/hh (per half hour) '] = pd.to_numeric(file1['KWH/hh (per half hour) '])
file1['DateTime'] = pd.to_datetime(file1['DateTime'], format='%Y-%m-%d %H:%M:%S.%f')
file1 = file1.rename({"KWH/hh (per half hour) ": "kwh"},axis='columns')
file1['kwh'].dtype
c_no1 = file1[file1['stdorToU'] == 'ToU'].LCLid.unique()

# In[22]:k is cusomer 8 database


#k = file[(file['LCLid'] == c_no[8])]


# In[23]:


#itd = (k[k['DateTime'].dt.date == datetime.date(2011, 12, 6)]).index.astype(int)


# In[24]:


#k.drop(itd, inplace = True)


# In[25]:


k = file[(file['LCLid'] == c_no[2])]
start_date = k.iloc[0].DateTime.date()
end_date = k.iloc[-1].DateTime.date()
#for dat in rrule(DAILY, dtstart=start_date, until=end_date):
#    sub = k[k['DateTime'].dt.date == dat.date()]
#    days =  sub.kwh.shape[0]
#    if days != 48:
#        print(dat, days)
#        itd = (k[k['DateTime'].dt.date == dat.date()]).index.astype(int)
#        k.drop(itd, inplace = True)


# In[26]:


#for dat in rrule(DAILY, dtstart=start_date, until=end_date):
#    sub = k[k['DateTime'].dt.date == dat.date()]
#    days =  sub.kwh.shape[0]
#    if days != 48:
#        print(dat, days)


# In[27]:


tariff = pd.read_excel("C:/Users/p0p/Desktop/anaomaly/London carbon/tariffs.xlsx")
tariff2 = pd.read_excel("C:/Users/p0p/Desktop/anaomaly/London carbon/tariffs.xlsx")
tstart = tariff.iloc[0]['TariffDateTime'].date()
tend = tariff.iloc[-1]['TariffDateTime'].date()
tariff.replace(['Normal','High', 'Low'], [1,1,0], inplace = True)
tariff2.replace(['Normal','High', 'Low'], [1,1,-1], inplace = True)

# In[30]:


#a = (k[k['DateTime'].dt.date == datetime.date(2013, 1,10)].kwh.values)
#b = (tariff[tariff['TariffDateTime'].dt.date == datetime.date(2013, 1,10)].Tariff.values)

#%%
#list = []
#for i in range(0,len(b)):
#    if b[i] == 1:
#        list.append((b[i]*a[i]))
#transformed_array = np.array([np.min(list), np.mean(list), np.std(list), np.max(list),np.size(list)])
#%%
#c =  np.array([]).reshape(0,5)
#dat = datetime.datetime(2013, 3, 29)
#a = k[k['DateTime'].dt.date == dat.date()].kwh.values
##    if a.size != 48:
##        print(dat.date(),a.size,b.size)
#
#if a.size == 48:
#    b = tariff[tariff['TariffDateTime'].dt.date == dat.date()].Tariff.values
#    
#    list = []
#    for i in range(0,len(b)):
#        if b[i] == 1:
#           list.append((b[i]*a[i]))
#    print(list)
##    transformed_array = np.array([np.min(list), np.mean(list), np.std(list), np.max(list),np.size(list)])
#    c = np.append(c,transformed_array)
#else:
#    print('nope')
#%%
def tarray(lister):
    transformed_array = np.array([np.min(lister), np.mean(lister), np.std(lister), np.max(lister),np.size(lister)])
    return transformed_array.reshape(-1,5)
#%%
#    for dat in rrule(DAILY, dtstart=tstart, until=tend):
#        a = k[k['DateTime'].dt.date == dat.date()].kwh.values
#        price_of_day = tariff[tariff['TariffDateTime'].dt.date == dat.date()].Tariff.values
#        if a.shape[0] == 48:
#            list = []
#            for i in range(0,len(price_of_day)):
#                if price_of_day[i] == 1:
#                    list.append((price_of_day[i]*a[i]))
#            p = (random.randint(1,10))/10
#            b=np.array(list)
#            t1 =  np.array([x * p for x in b])
#            t2 = np.array([z * (random.randint(1,9))/10 for z in b])
#            t3 = np.array([y * (random.randint(1,10)/10) for y in b])
##            t4 =  np.array([np.mean(b)*(random.randint(1,10))/10 for z in b[0]]).reshape(-1,48)
##            t5 =  np.array([np.mean(b) for t in b[0]]).reshape(-1,48)
#            t6 = b[::-1]
#            print(dat)
#%%
def ccnc(no):
    k = file[(file['LCLid'] == c_no[no])]
    X = np.array([]).reshape(0,5)
    y = np.array([])
    
    #clf = SVC(kernel='rbf',probability=True)
    
    print(c_no[no], "using statistic 5 feature")
    for dat in rrule(DAILY, dtstart=tstart, until=tend):
        a = k[k['DateTime'].dt.date == dat.date()].kwh.values
        price_of_day = tariff[tariff['TariffDateTime'].dt.date == dat.date()].Tariff.values
        if a.shape[0] == 48:
            listr = []
            for i in range(0,len(price_of_day)):
                if price_of_day[i] == 1:
                    listr.append((price_of_day[i]*a[i]))
            if len(listr) != 0:
#                random.seed(42)
                p = (random.randint(1,9))/10
                b=np.array(listr)
                t1 =  np.array([x * p for x in b])
                t2 = np.array([z * random.randint(0,1) for z in b])
                t3 = np.array([y * (random.randint(1,10)/10) for y in b])
                t4 =  np.array([np.mean(b)*(random.randint(1,10))/10 for z in b])
                t5 =  np.array([np.mean(b) for t in b])
#                t6 = b[::-1]
#                print(dat)
                b,t4,t5,t3,t2,t1= tarray(b),tarray(t4),tarray(t5),tarray(t3),tarray(t2),tarray(t1)
                
    #             print(b.shape,t1.shape,t2.shape,t3.shape,t4.shape,t5.shape,t6.shape)
    #                     b = np.append(b,[(b.mean()),b.std()])
    #                     t1 = np.append(t1,[t1.mean(),t1.std()])
    #                     t2 = np.append(t2,[t2.mean(),t2.std()])
    #                     t3 = np.append(t3,[t3.mean(),t3.std()])
    #                     t4 = np.append(t4,[t4.mean(),t4.std()])
    #                     t5 = np.append(t5,[t5.mean(),t5.std()])
    #                     t6 = np.append(t6,[t6.mean(),t6.std()])
                X= np.concatenate((X, b,t1,t2,t3,t4,t5), axis=0)
                y = np.append(y,0)
                for i in range(0,5):   
                    y = np.append(y,1)
        else:
            continue
    return X,y

# In[274]:

def tdetect(no,clf):
    customer_meter = c_no[no]
    X,y = ccnc(no)
#    clf = XGBClassifier()
#     clf = SVC(kernel='rbf',probability=True)
#    clf = LGBMClassifier()
#    clf = CatBoostClassifier(logging_level = "Silent")
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.14, random_state=0)
    sm = SMOTE(random_state=42)
    X_res_train, y_res_train = sm.fit_sample(X_train, y_train)
    X_res_test, y_res_test = sm.fit_sample(X_test, y_test)
    clf.fit(X_res_train, y_res_train)
#    score = clf.score(X_res_test, y_res_test)
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
#    plot_importance(clf,importance_type="weight", ax=plt.gca())
    return sensitivity,fpr
#%%
def t2array(b):
    for n, i in enumerate(b[0]):
        if i < 0:
            b[0][n] = -1
    return(b[0].reshape(-1,48))
#%%

def ccnc2(no):
    k = file[(file['LCLid'] == c_no[no])]
    X = np.array([]).reshape(0,48)
    y = np.array([])

    #clf = SVC(kernel='rbf',probability=True)
    j = 0
    print(c_no[no], "using 48 features -1")
    for dat in rrule(DAILY, dtstart=tstart, until=tend):
        sub = k[k['DateTime'].dt.date == dat.date()]
        price_of_day = tariff2[tariff2['TariffDateTime'].dt.date == dat.date()].Tariff.values
        if sub.shape[0] == 48:
#            random.seed(42)
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
        
#            b = b*price_of_day
#            t1 = t1*price_of_day
#            t2 = t2*price_of_day
#            t3 = t3*price_of_day
#            t4 = t4*price_of_day
#            t5 = t5*price_of_day
#            t6 = t6*price_of_day
                    
#            b,t1,t2,t3,t4,t5= t2array(b),t2array(t1),t2array(t2),t2array(t3),t2array(t4),t2array(t5)
#            t6 = t2array(t6)
#             print(b.shape,t1.shape,t2.shape,t3.shape,t4.shape,t5.shape,t6.shape)
#                     b = np.append(b,[(b.mean()),b.std()])
#                     t1 = np.append(t1,[t1.mean(),t1.std()])
#                     t2 = np.append(t2,[t2.mean(),t2.std()])
#                     t3 = np.append(t3,[t3.mean(),t3.std()])
#                     t4 = np.append(t4,[t4.mean(),t4.std()])
#                     t5 = np.append(t5,[t5.mean(),t5.std()])
#                     t6 = np.append(t6,[t6.mean(),t6.std()])
            X= np.concatenate((X, b,t1,t2,t3,t4,t5), axis=0)
#            X= np.concatenate((X, b,t6), axis=0)
            y = np.append(y,0)
            for i in range(0,5):   
                y = np.append(y,1)
        else:
            continue
    return X,y


def tdetect2(no,clf):
    customer_meter = c_no[no]
    X,y = ccnc2(no)
#    clf = XGBClassifier()
#    clf = SVC(kernel='rbf',probability=True)
#    clf = LGBMClassifier()
    clf = CatBoostClassifier(logging_level = "Silent")
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.14, random_state=0)
    sm = SMOTE(random_state=42)
    X_res_train, y_res_train = sm.fit_sample(X_train, y_train)
    X_res_test, y_res_test = sm.fit_sample(X_test, y_test)
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
#    plot_importance(clf,importance_type="weight", ax=plt.gca())
    return sensitivity,fpr

#%%
dat = datetime.datetime(2013, 1, 19)
sub = k[k['DateTime'].dt.date == dat.date()]
price_of_day = tariff2[tariff2['TariffDateTime'].dt.date == dat.date()].Tariff.values
if sub.shape[0] == 48:
#    random.seed(42)
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
#    t1 =  np.array([x * p for x in b])
#    t2 = np.array([z * random.randint(0,1) for z in b[0]]).reshape(-1,48)
#    t3 = np.array([y * (random.randint(1,10)/10) for y in b[0]]).reshape(-1,48)
#    t4 =  np.array([np.mean(b)*(random.randint(1,10))/10 for z in b[0]]).reshape(-1,48)
#    t5 =  np.array([np.mean(b) for t in b[0]]).reshape(-1,48)
#    t6 = b[::-1]

#    b = b*price_of_day 
#    t1 = t1*price_of_day
#    t2 = t2*price_of_day
#    t3 = t3*price_of_day
#            t4 = t4*price_of_day
#            t5 = t5*price_of_day
#    t6 = t6*price_of_day
#    b,t3= t2array(b),t2array(t1)

#plt.plot(t1[0], label = 'theft')
#plt.plot(b[0],label = 'real')
#plt.legend()
#plt.show()


fig, ax1 = plt.subplots()
ax1.plot(t1[0], label = 'theft 1')
ax1.plot(t2[0], label = 'theft 2')
ax1.plot(t3[0], label = 'theft 3')
ax1.plot(t4[0], label = 'theft 4')
ax1.plot(t5[0], label = 'theft 5')
ax1.plot(b[0], label = 'real')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('consumption', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()

ax2.plot(price_of_day,color = 'r',label = 'ToU price',linestyle = "--")
ax2.set_ylabel('price', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
fig.legend(loc='upper center')
plt.show()
#%%
plt.plot(price_of_day, label = 'ToU price pattern')
plt.xlabel('t')
plt.ylabel('ToU price')
plt.legend()
#%%
a = k[k['DateTime'].dt.date == dat.date()].kwh.values
price_of_day = tariff[tariff['TariffDateTime'].dt.date == dat.date()].Tariff.values
if a.shape[0] == 48:
    list = []
    for i in range(0,len(price_of_day)):
        if price_of_day[i] == 1:
            list.append((price_of_day[i]*a[i]))
    if len(list) != 0:
#        random.seed(42)
        p = (random.randint(1,9))/10
        b=np.array(list)
        print(b)
        t1 =  np.array([x * p for x in b])
        t2 = np.array([z * random.randint(0,1) for z in b])
        t3 = np.array([y * (random.randint(1,10)/10) for y in b])
        t4 =  np.array([np.mean(b)*(random.randint(1,10))/10 for z in b])
        t5 =  np.array([np.mean(b) for t in b])
#        t6 = b[::-1]
#                print(dat)
#        b,t1,t2,t3= tarray(b),tarray(t1),tarray(t2),tarray(t3)
data = [b,t1,t2,t3,t4,t5]
fig7, ax7 = plt.subplots()
ax7.set_title('Real vs theft boxplot')
ax7.boxplot(data,showmeans=True)
plt.legend()
plt.show()
#plt.plot(t2)
#plt.plot(t3)
#plt.plot(b)   
#%%
def ccnc_c(no):
    k = file[(file['LCLid'] == c_no[no])]
    X = np.array([]).reshape(0,48)
    y = np.array([])

    #clf = SVC(kernel='rbf',probability=True)
    j = 0
    print(c_no[no], "using consumption")
    for dat in rrule(DAILY, dtstart=tstart, until=tend):
        sub = k[k['DateTime'].dt.date == dat.date()]
        price_of_day = tariff2[tariff2['TariffDateTime'].dt.date == dat.date()].Tariff.values
        if sub.shape[0] == 48:
#            random.seed(42)
            p = (random.randint(1,9))/10
            b=sub.kwh.values.reshape(-1,48)
            t1 =  np.array([x * p for x in b])
            t2 = np.array([z * random.randint(0,1) for z in b[0]]).reshape(-1,48)
            t3 = np.array([y * (random.randint(1,10)/10) for y in b[0]]).reshape(-1,48)
            t4 =  np.array([np.mean(b)*(random.randint(1,10))/10 for z in b[0]]).reshape(-1,48)
            t5 =  np.array([np.mean(b) for t in b[0]]).reshape(-1,48)
#            t6 = b[::-1]
        
            
#            t6 = t2array(t6)
#             print(b.shape,t1.shape,t2.shape,t3.shape,t4.shape,t5.shape,t6.shape)
#                     b = np.append(b,[(b.mean()),b.std()])
#                     t1 = np.append(t1,[t1.mean(),t1.std()])
#                     t2 = np.append(t2,[t2.mean(),t2.std()])
#                     t3 = np.append(t3,[t3.mean(),t3.std()])
#                     t4 = np.append(t4,[t4.mean(),t4.std()])
#                     t5 = np.append(t5,[t5.mean(),t5.std()])
#                     t6 = np.append(t6,[t6.mean(),t6.std()])
            X= np.concatenate((X, b,t1,t2,t3,t4,t5), axis=0)
#            X= np.concatenate((X, b,t6), axis=0)
            y = np.append(y,0)
            for i in range(0,5):   
                y = np.append(y,1)
        else:
            continue
    return X,y

def tdetect_c(no):
    customer_meter = c_no[no]
    X,y = ccnc_c(no)
#    clf = XGBClassifier()
    clf = SVC(kernel='rbf',probability=True)
#    clf = LGBMClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.14, random_state=42)
    sm = SMOTE(random_state=42)
    X_res_train, y_res_train = sm.fit_sample(X_train, y_train)
    X_res_test, y_res_test = sm.fit_sample(X_test, y_test)
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
    return sensitivity,fpr
#%%
k = file[(file['LCLid'] == c_no[1])]
X = np.array([]).reshape(0,48)
y = np.array([])

#clf = SVC(kernel='rbf',probability=True)
j = 0
print(c_no[1], "using 48 features -1")
for dat in rrule(DAILY, dtstart=tstart, until=tstart +timedelta(days = 19)):
    sub = k[k['DateTime'].dt.date == dat.date()]
    price_of_day = tariff2[tariff2['TariffDateTime'].dt.date == dat.date()].Tariff.values
    if sub.shape[0] == 48:
#        random.seed(42)
        p = (random.randint(1,9))/10
        b=sub.kwh.values.reshape(-1,48)
        t1 =  np.array([x * p for x in b])
        t2 = np.array([z * random.randint(0,1) for z in b[0]]).reshape(-1,48)
        t3 = np.array([y * (random.randint(1,10)/10) for y in b[0]]).reshape(-1,48)
        t4 =  np.array([np.mean(b)*(random.randint(1,10))/10 for z in b[0]]).reshape(-1,48)
        t5 =  np.array([np.mean(b) for t in b[0]]).reshape(-1,48)
#            t6 = b[::-1]
    
#        b = b*price_of_day
#        t1 = t1*price_of_day
#        t2 = t2*price_of_day
#        t3 = t3*price_of_day
#        t4 = t4*price_of_day
#        t5 = t5*price_of_day
#            t6 = t6*price_of_day
        b,t1,t2,t3,t4,t5= t2array(b),t2array(t1),t2array(t2),t2array(t3),t2array(t4),t2array(t5)
#            t6 = t2array(t6)
#             print(b.shape,t1.shape,t2.shape,t3.shape,t4.shape,t5.shape,t6.shape)
#                     b = np.append(b,[(b.mean()),b.std()])
#                     t1 = np.append(t1,[t1.mean(),t1.std()])
#                     t2 = np.append(t2,[t2.mean(),t2.std()])
#                     t3 = np.append(t3,[t3.mean(),t3.std()])
#                     t4 = np.append(t4,[t4.mean(),t4.std()])
#                     t5 = np.append(t5,[t5.mean(),t5.std()])
#                     t6 = np.append(t6,[t6.mean(),t6.std()])
        X= np.concatenate((X, b,t1,t2,t3,t4,t5), axis=0)
#            X= np.concatenate((X, b,t6), axis=0)
        y = np.append(y,-1)
        for i in range(0,5):   
            y = np.append(y,1)
    else:
        continue
#%% get low usage dates
x = tariff2[tariff2.Tariff < 0].TariffDateTime
x = pd.to_datetime(x)
x = [str(i.date()) for i in x]
#%%
a = np.arange(2,10)
b = np.random.randint(2, size=8)
c = np.ones(8) + np.negative(b)*2
d = a.copy()
#%%
tclf = []
t2clf = []
classifiers = [
    
#     KNeighborsClassifier(5),
    #DecisionTreeClassifier(max_depth=20),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #MLPClassifier(alpha=1),
    #AdaBoostClassifier(),
   XGBClassifier(),
#     SVC(kernel='rbf',probability=True)
     CatBoostClassifier(logging_level = "Silent"),
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis(),
     LGBMClassifier(),
]
j = 0
for c in classifiers:
    j = j + 1
    print("classifier no." ,j)
    for _ in range(0,5):
#    tdetect2(1,c)
        a = timeit.default_timer()
        tdetect(_,c)
        b = timeit.default_timer()
        tclf.append(b-a)
        a1 = timeit.default_timer()
        tdetect2(_,c)
        b1 = timeit.default_timer()
        t2clf.append(b1-a1)
#%%
dat = datetime.datetime(2013, 1, 19)
sub = k[k['DateTime'].dt.date == dat.date()]
price_of_day = tariff2[tariff2['TariffDateTime'].dt.date == dat.date()].Tariff.values
if sub.shape[0] == 48:
#            random.seed(42)
    p = (random.randint(1,9))/10
    b=sub.kwh.values.reshape(-1,48)
    t1 =  np.array([x * p for x in b])
    t2 = np.array([z * random.randint(0,1) for z in b[0]]).reshape(-1,48)
    t3 = np.array([y * (random.randint(1,10)/10) for y in b[0]]).reshape(-1,48)
    t4 =  np.array([np.mean(b)*(random.randint(1,10))/10 for z in b[0]]).reshape(-1,48)
    t5 =  np.array([np.mean(b) for t in b[0]]).reshape(-1,48)
#            t6 = b[::-1]

    
#            t6 = t2array(t6)
#             print(b.shape,t1.shape,t2.shape,t3.shape,t4.shape,t5.shape,t6.shape)
#                     b = np.append(b,[(b.mean()),b.std()])
#                     t1 = np.append(t1,[t1.mean(),t1.std()])
#                     t2 = np.append(t2,[t2.mean(),t2.std()])
#                     t3 = np.append(t3,[t3.mean(),t3.std()])
#                     t4 = np.append(t4,[t4.mean(),t4.std()])
#                     t5 = np.append(t5,[t5.mean(),t5.std()])
#                     t6 = np.append(t6,[t6.mean(),t6.std()])
    X= np.concatenate((X, b,t1,t2,t3,t4,t5), axis=0)
#            X= np.concatenate((X, b,t6), axis=0)
    y = np.append(y,0)
    for i in range(0,5):   
        y = np.append(y,1)

fig, ax1 = plt.subplots()
ax1.plot(t1[0], label = 'theft 1')
ax1.plot(t2[0], label = 'theft 2')
ax1.plot(t3[0], label = 'theft 3')
ax1.plot(t4[0], label = 'theft 4')
ax1.plot(t5[0], label = 'theft 5')
ax1.plot(b[0], label = 'real')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('consumption')
ax1.tick_params('y')



fig.tight_layout()
fig.legend(loc=' right')
plt.show()