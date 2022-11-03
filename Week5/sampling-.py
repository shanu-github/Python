# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:07:06 2022

@author: shanu
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

emp_data= pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week3\\employee_data.csv")

mean_data= pd.DataFrame()
for ite in range(1000):
    mean_5= emp_data['CTC'].sample(5).mean()
    mean_10= emp_data['CTC'].sample(10).mean()
    mean_50= emp_data['CTC'].sample(50).mean()
    mean_100= emp_data['CTC'].sample(100).mean()
    df= pd.DataFrame([{'Iteration': ite, 'Mean_5':mean_5, 'Mean_10':mean_10, 'Mean_50':mean_50,
     'Mean_100': mean_100}])
    mean_data= mean_data.append(df)
    
    
sns.distplot(emp_data['CTC'])

sns.distplot(mean_data['Mean_5'])
sns.distplot(mean_data['Mean_10'])
sns.distplot(mean_data['Mean_50'])
sns.distplot(mean_data['Mean_100'])

mean_data.describe()
emp_data['CTC'].mean()

    
stats.binom.pmf(2, n=5, p=0.1)
stats.binom.cdf(2, n=5, p=0.1)