# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 05:49:29 2022

@author: shanu
"""
'''
Covariance provides the a measure of strength of correlation between two
 variable or more set of variables.

If COV(xi, xj) = 0 then variables are uncorrelated
If COV(xi, xj) > 0 then variables positively correlated
If COV(xi, xj) > < 0 then variables negatively correlated

'''
import pandas as pd
loan_data = pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week3\\Loan_Prediction.csv")

loan_data= loan_data.dropna(subset = ['LoanAmount','ApplicantIncome'])

loan_data['LoanAmount'].std()
loan_data['LoanAmount'].var()

import numpy as np
np.cov(loan_data['LoanAmount'],loan_data['ApplicantIncome'])
np.corrcoef(loan_data['LoanAmount'],loan_data['ApplicantIncome'])


x = [1.23, 2.12, 3.34, 4.5]

y = [2.56, 2.89, 3.76, 3.95]

np.cov(x,y)

np.var(x)
np.var(y)

np.std(x)


'''
Example: Correlation Test in Python
To determine if the correlation coefficient between two variables is
 statistically significant, you can perform a correlation test in Python using 
 the pearsonr function from the SciPy library.

This function returns the correlation coefficient between two variables
 along with the two-tailed p-value.

For example, suppose we have the following two arrays in Python:
    '''
import seaborn as sns

#create two arrays
x = [3, 4, 4, 5, 7, 8, 10, 12, 13, 15]
y = [2, 4, 4, 5, 4, 7, 8, 19, 14, 10]
from scipy.stats.stats import pearsonr

#calculation correlation coefficient and p-value between x and y
pearsonr(x, y)

sns.relplot(x,y)

'''
Here’s how to interpret the output:

Pearson correlation coefficient (r): 0.8076
Two-tailed p-value: 0.0047
Since the correlation coefficient is close to 1, this tells us that 
there is a strong positive association between the two variables.

And since the corresponding p-value is less than .05, we conclude that 
there is a statistically significant association between the two variables.
'''
import pandas as pd
loan_data = pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week3\\Loan_Prediction.csv")

loan_data= loan_data.dropna(subset = ['LoanAmount','ApplicantIncome'])

sns.relplot(loan_data['ApplicantIncome'],loan_data['LoanAmount'])

pearsonr(loan_data['ApplicantIncome'],loan_data['LoanAmount'])

sns.lmplot(data=loan_data, x='ApplicantIncome', y='LoanAmount')
