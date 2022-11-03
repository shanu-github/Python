# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 03:40:08 2022

@author: shanu
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

state = pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week5\\state.csv")

# Compute the mean, trimmed mean, and median for Population. 
#For `mean` and `median` we can use the _pandas_ methods of the data frame. 
#The trimmed mean requires the `trim_mean` function in _scipy.stats_.
state['Population'].mean()
from scipy.stats import trim_mean
trim_mean(state['Population'],  0.1)
state['Population'].median()

# Weighted mean is available with numpy.
import numpy as np
print(state['Stadium'].mean())

print(np.average(state['Stadium'], weights=state['Population']))



## Estimates of Variability

state['Population'].std()
# Interquartile range is calculated as the difference of the 75% and 25% quantile.
state['Population'].quantile(0.75) - state['Population'].quantile(0.25)
from statsmodels import robust
robust.scale.mad(state['Population'])
### Percentiles and Boxplots
state['Stadium'].quantile([0.05, 0.25, 0.5, 0.75, 0.95])

sns.boxplot(data=state,x='Population')

#Frequency table & Histogram
#A frequency table of a variable divides up the variable range into
# equally spaced segments and tells us how many values fall within each segment.

pd.cut(state['Population'], bins=5).value_counts(sort = False)

pd.cut(state['Population']/1000000, bins=[0,0.5, 1, 2.5,5, 10, 15,20, 25]).value_counts(sort = False)

sns.histplot(state['Population']/1000000, bins=[0,0.5, 1,2.5 ,5, 10, 15,20,25])
sns.histplot(state['Population']/1000000, bins=5)
sns.histplot(state['Population'], binwidth= 5000000)

#Density plot, shows the distribution of data values as a continuous line.
# A density plot can be thought of as a smoothed histogram, 
#although it is typically computed directly from the data through a kernel density estimate

sns.displot(data=state, x='Population', kde=True)
sns.kdeplot(data=state, x='Population',shade=True, color='b')

emp_data= pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week3\\employee_data.csv")
emp_data['DEPT'].value_counts()
emp_data['DEPT'].mode()
sns.catplot(data=emp_data, kind="count", y="DEPT")
sns.catplot(data=emp_data, kind="count", x="LOCATION")

#Probabilty of each category
emp_data['DEPT'].value_counts()/ sum(emp_data['DEPT'].value_counts())*100

count_data= emp_data['LOCATION'].value_counts()
plt.pie(count_data,labels = count_data.index)
plt.show()

ax = state.plot.hexbin(x='Population', y='Stadium',
 gridsize=30, sharex=False, figsize=(5, 4))
ax.set_xlabel('Population')
ax.set_ylabel('Stadium')
ax = sns.kdeplot(data= state, x='Population', y='Stadium', ax=ax)
ax.set_xlabel('Finished Square Feet')
ax.set_ylabel('Tax-Assessed Value')

