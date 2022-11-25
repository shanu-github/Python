# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 17:16:35 2022

@author: shanu
"""

import pandas as pd
import seaborn as sns

insurance_data = pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week8\\buy_insurance.csv")

sns.relplot(insurance_data['age'], insurance_data['bought_insurance'] , hue=insurance_data['bought_insurance'])

from sklearn.linear_model import LinearRegression

x_train = insurance_data[['age']]

y_train =  insurance_data['bought_insurance']

model= LinearRegression()
model.fit(x_train, y_train)

insurance_data['predicted_binsurance']= model.predict(x_train)

insurance_data['predicted_binsurance_class']= 0
insurance_data.loc[insurance_data['predicted_binsurance']>0.5,'predicted_binsurance_class'] =  1

model.score(x_train, y_train)


sns.relplot(insurance_data['age'], insurance_data['predicted_binsurance'] ,
            hue=insurance_data['bought_insurance'], style=  insurance_data['predicted_binsurance_class'])

pd.crosstab(insurance_data['bought_insurance'], insurance_data['predicted_binsurance_class'])

from sklearn.linear_model import LogisticRegression

model= LogisticRegression()
model.fit(x_train, y_train)

insurance_data['predicted_binsurance']= model.predict(x_train)

insurance_data[['prob_0', 'prob1']]= model.predict_proba(x_train)

pd.crosstab(insurance_data['bought_insurance'], insurance_data['predicted_binsurance'])

model.score(x_train, y_train)

pd.DataFrame(model.coef_, columns= x_train.columns)

model.intercept_

#ln(p/(1-p))= -5.26+ 0.13*age


model.classes_

import numpy as np
np.arange(0.0, 1.0, 0.1)

acuracy_cal= pd.DataFrame()
for threshold in np.arange(0.0, 1.0, 0.1):
    insurance_data['predicted_class']= np.where(insurance_data['prob1']>threshold, 1, 0)
    total_pred= len(insurance_data[insurance_data['predicted_class']==1])
    total_act= len(insurance_data[insurance_data['bought_insurance']==1])
    total_cprediction= len(insurance_data[(insurance_data['predicted_class']==1) &
                                          (insurance_data['bought_insurance']==1)])
    precision= total_cprediction/total_pred
    recall =  total_cprediction/total_act
    f1_score= (2*precision*recall) /(precision+ recall)
    acuracy_cal= acuracy_cal.append(pd.DataFrame([[threshold, precision, recall, f1_score]], 
                                                 columns= ['Threshold','Precision','Recall','F1_Score']))

default_data= pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week8\\Default.csv")

default_data['student']
sns.relplot(default_data['balance'], default_data['default'], hue= default_data['default'] )

from sklearn.linear_model import LogisticRegression

xtrain= default_data[['balance']]

ytrain = default_data['default']

model = LogisticRegression()
model.fit(xtrain, ytrain)

model.predict_proba(default_data[['balance']][0:3])
pd.DataFrame(model.predict_proba(default_data[['balance']]), columns= model.classes_)

default_data['predicted_default']= model.predict(default_data[['balance']])

pd.crosstab(default_data['default'],default_data['predicted_default'])

from sklearn.metrics import classification_report
print(classification_report(default_data['predicted_default'], default_data['default']))