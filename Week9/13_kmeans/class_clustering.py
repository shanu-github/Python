# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 17:13:24 2022

@author: shanu
"""

import pandas as pd
import seaborn as sns

data = pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week9\\13_kmeans\\income.csv")

sns.relplot(data['Age'], data['Income($)'])

from sklearn.cluster import KMeans

model= KMeans(n_clusters=3)
xtrain = data[['Age','Income($)']]
model.fit(xtrain)

data['cluster']= model.predict(xtrain)

sns.relplot(data['Age'], data['Income($)'], hue= data['cluster'])

from sklearn.preprocessing import MinMaxScaler

#(x-min)/(max-min)

age_scale = MinMaxScaler()
age_scale.fit(data[['Age']])

data['scaled_Age'] = age_scale.transform(data[['Age']])

income_scale = MinMaxScaler()
income_scale.fit(data[['Income($)']])

data['scaled_income'] = income_scale.transform(data[['Income($)']])


'''
income_scale = MinMaxScaler()
income_scale.fit(data[['Income($)','Age']])

data['scaled_income', 'scaled_age'] = income_scale.transform(data[['Income($)','Age']])
'''
from sklearn.cluster import KMeans

model= KMeans(n_clusters=3)
xtrain = data[['scaled_Age','scaled_income']]
model.fit(xtrain)

data['scaled_cluster']= model.predict(xtrain)

sns.relplot(data['Age'], data['Income($)'], hue= data['scaled_cluster'])

model.cluster_centers_

model.inertia_

#how to choode the value of K
k_selection = pd.DataFrame()
for k in range(1, 10, 1):
    model= KMeans(n_clusters=k)
    xtrain = data[['scaled_Age','scaled_income']]
    model.fit(xtrain)
    wcss= model.inertia_
    k_selection= k_selection.append({'K': k, 'WCSS': wcss}, ignore_index=True)
    
    
sns.relplot(k_selection['K'], k_selection['WCSS'],kind="line")



