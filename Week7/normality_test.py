# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 04:57:04 2022

@author: shanu
"""

'''
https://www.statology.org/normality-test-python/
Many statistical tests make the assumption that datasets are normally distributed.

There are four common ways to check this assumption in Python:

1. (Visual Method) Create a histogram.

If the histogram is roughly “bell-shaped”, then the data is assumed to be normally distributed.
2. (Visual Method) Create a Q-Q plot.

If the points in the plot roughly fall along a straight diagonal line, then the data is assumed to be normally distributed.
3. (Formal Statistical Test) Perform a Shapiro-Wilk Test.
If the p-value of the test is greater than α = .05, then the data is assumed to be normally distributed.
4. (Formal Statistical Test) Perform a Kolmogorov-Smirnov Test.

If the p-value of the test is greater than α = .05, then the data is assumed to be normally distributed.
Method 1: Create a Histogram
The following code shows how to create a histogram for a dataset that follows a log-normal distribution:
'''
import math
import numpy as np
from scipy.stats import lognorm
import seaborn as sns


#make this example reproducible
np.random.seed(1)

#generate dataset that contains 1000 log-normal distributed values
lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)

#create histogram to visualize values in dataset
sns.distplot(lognorm_dataset)

'''
By simply looking at this histogram, we can tell the dataset
 does not exhibit a “bell-shape” and is not normally distributed.
 Method 2: Create a Q-Q plot
The following code shows how to create a Q-Q plot for a dataset that 
follows a log-normal distribution:
'''

import statsmodels.api as sm
#create Q-Q plot with 45-degree line added to plot
fig = sm.qqplot(lognorm_dataset, line='45')

'''
If the points on the plot fall roughly along a straight diagonal line, 
then we typically assume a dataset is normally distributed.

However, the points on this plot clearly don’t fall along the red line, 
so we would not assume that this dataset is normally distributed.

This should make sense considering we generated the data using a 
log-normal distribution function.

Method 3: Perform a Shapiro-Wilk Test
The following code shows how to perform a Shapiro-Wilk for a dataset 
that follows a log-normal distribution:
'''
from scipy.stats import shapiro 
#perform Shapiro-Wilk test for normality
shapiro(lognorm_dataset)

'''
From the output we can see that the test statistic is 0.857 and the
corresponding p-value is 3.88e-29 (extremely close to zero).

Since the p-value is less than .05, we reject the null hypothesis of the 
Shapiro-Wilk test.

This means we have sufficient evidence to say that the sample data does not 
come from a normal distribution.

Method 4: Perform a Kolmogorov-Smirnov Test
The following code shows how to perform a Kolmogorov-Smirnov test for a
 dataset that follows a log-normal distribution:
'''
from scipy.stats import kstest
#perform Kolmogorov-Smirnov test for normality
kstest(lognorm_dataset, 'norm')    

'''
From the output we can see that the test statistic is 0.841 and 
the corresponding p-value is 0.0.

Since the p-value is less than .05, we reject the null hypothesis of
 the Kolmogorov-Smirnov test.

This means we have sufficient evidence to say that the sample data
 does not come from a normal distribution.
 '''
 
'''
How to Handle Non-Normal Data
If a given dataset is not normally distributed,
 we can often perform one of the following transformations to make 
 it more normally distributed:

1. Log Transformation: Transform the values from x to log(x).
2. Square Root Transformation: Transform the values from x to √x.

3. Cube Root Transformation: Transform the values from x to x1/3.

By performing these transformations, the dataset typically becomes more 
normally distributed.

Log Transformation in Python
The following code shows how to perform a log transformation on a 
variable and create side-by-side plots to view the original distribution 
and the log-transformed distribution of the data:

 '''
transformed_data= np.log(lognorm_dataset )
sns.distplot(transformed_data)
shapiro(transformed_data)

 
import pandas as pd
loan_data = pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week3\\Loan_Prediction.csv")
sns.distplot(loan_data['LoanAmount'])

import numpy as np
#create log-transformed data
loan_data['T_loanAmount']=  np.log(loan_data['LoanAmount'])
sns.distplot(loan_data['T_loanAmount'])

from scipy.stats import shapiro 
#perform Shapiro-Wilk test for normality
shapiro(loan_data['T_loanAmount'])

loan_data= loan_data.dropna(subset = ['LoanAmount','T_loanAmount'])
shapiro(loan_data['T_loanAmount'])

shapiro(loan_data['LoanAmount'])

'''
Notice how the log-transformed distribution is more normally distributed 
compared to the original distribution.
It’s still not a perfect “bell shape” but it’s closer to a normal distribution 
that the original distribution.

Square Root Transformation in Python
The following code shows how to perform a square root transformation on a 
variable :
'''

#create log-transformed data
loan_data['T_loanAmount']=  np.sqrt(loan_data['LoanAmount'])
sns.distplot(loan_data['T_loanAmount'])
shapiro(loan_data['T_loanAmount'])

'''
Notice how the square root transformed data is much more normally 
distributed than the original data.

Cube Root Transformation in Python
The following code shows how to perform a cube root transformation on a variable 
'''
loan_data['T_loanAmount']=  np.cbrt(loan_data['LoanAmount'])
sns.distplot(loan_data['T_loanAmount'])
shapiro(loan_data['T_loanAmount'])

