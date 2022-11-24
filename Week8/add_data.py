# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 19:52:29 2022

@author: shanu
"""

import pandas as pd
import seaborn as sns

ad_data= pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week8\\Advertising.csv")

#Exploratary Data Analysis

ad_data.describe()

ad_data.isnull().sum()

ad_data.dtypes

ad_data.head()

sns.distplot(ad_data['sales'])

sns.relplot(data= ad_data, x= 'TV', y='sales' )
sns.lmplot(data= ad_data, x= 'TV', y='sales')

from scipy.stats.stats import pearsonr

#calculation correlation coefficient and p-value between x and y
pearsonr(ad_data['TV'], ad_data['sales'])


sns.relplot(data= ad_data, x= 'radio', y='sales' )

sns.relplot(data= ad_data, x= 'newspaper', y='sales' )
pearsonr(ad_data['newspaper'], ad_data['sales'])

sns.distplot(ad_data['sales'])
import numpy as np
np.corrcoef(ad_data['sales'], ad_data['TV'])
np.corrcoef(ad_data['sales'], ad_data['radio'])
np.corrcoef(ad_data['sales'], ad_data['newspaper'])

ad_data.corr()

from sklearn.model_selection import train_test_split
train_data, test_data= train_test_split(ad_data,test_size=0.3)

#model building
x_train= train_data[['TV','radio']]
y_train = train_data['sales']

from sklearn.linear_model import LinearRegression

# with sklearn
regr = LinearRegression()
regr.fit(x_train, y_train)

print('Intercept: \n', regr.intercept_)
print(pd.DataFrame({'Features': x_train.columns,'Coeffiecient': regr.coef_}))

test_data['predicted_sales']=regr.predict(test_data[['TV','radio']])

regr.predict([[230.1,37.8]])


sns.relplot(x='sales', y= 'predicted_sales', data=test_data)

#Residual RMSE : root mean square error
np.sqrt(sum((test_data['predicted_sales']-test_data['sales'])**2)/len(test_data))

SS_Residual = sum((test_data['sales']-test_data['predicted_sales'])**2)
SS_Total = sum((test_data['sales']-np.mean(train_data['sales']))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
print("R-squared", r_squared )

train_data['predicted_sales']= regr.predict(x_train)
SS_Residual = sum((train_data['sales']-train_data['predicted_sales'])**2)
SS_Total = sum((train_data['sales']-np.mean(train_data['sales']))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
print("R-squared", r_squared )

#To check the model robustness
from sklearn.model_selection import KFold, cross_val_score
regr = LinearRegression()

k_folds = KFold(n_splits = 5)

scores = cross_val_score(regr,x_train, y_train, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

#to get entire summary of regression Model
import statsmodels.api as sm
X_train1 = sm.add_constant(x_train)
reg_model = sm.OLS(y_train, X_train1)
reg_model = reg_model.fit()
print(reg_model.summary())

#to get the variable importance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# transform data
scaled = scaler.fit_transform(x_train)
X_train1 = sm.add_constant(scaled)
reg_model = sm.OLS(y_train, X_train1)
reg_model = reg_model.fit()
print(reg_model.summary())