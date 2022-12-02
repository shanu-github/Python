# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 05:44:58 2022

@author: shanu
"""
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

data= pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week9\\13_kmeans\\income.csv")

data.columns

sns.relplot(data['Age'], data['Income($)'])

#As we three cluster can be formed set k=3
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(data[['Age','Income($)']])

data['cluster']=y_predicted

sns.relplot(data['Age'], data['Income($)'], hue= data['cluster'])

km.cluster_centers_

#cluster are looking inappropriate need scaling biased to income scale
#--------------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(data[['Income($)']])
data['Income($)'] = scaler.transform(data[['Income($)']])

scaler.fit(data[['Age']])
data['Age'] = scaler.transform(data[['Age']])

#As we three cluster can be formed set k=3
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(data[['Age','Income($)']])

data['cluster']=y_predicted

sns.relplot(data['Age'], data['Income($)'], hue= data['cluster'])

#---------------------------------------------------------------
#choosing value of K

WSSE= 0
k= 3
cluster_center = pd.DataFrame({'cluster': range(0, k, 1), 
              'center_age': km.cluster_centers_[:,0],
              'center_income': km.cluster_centers_[:,1]})

wcss_cal= pd.merge(data,cluster_center)

for k in  range(1, 10, 1):
    km = KMeans(n_clusters=3)
    data['cluster'] = km.fit_predict(data[['Age','Income($)']])
    cluster_center = pd.DataFrame({'cluster': range(0, k, 1), 
                  'center_age': km.cluster_centers_[:,0],
                  'center_income': km.cluster_centers_[:,1]})
    error_cal = pd.merge(data, cluster_center)
    data['Square Error'] =











