# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:07:06 2022

@author: shanu
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import norm
norm(loc = 0 , scale = 1).cdf(0)
norm(loc = 0 , scale = 1).cdf(1.93)
norm(loc = 0 , scale = 1).cdf(-1.65)

#calculate probability 
#P(0 < Z ≤ 1) = 0.3413

norm(loc=0, scale=1).cdf(1)- norm(loc=0, scale=1).cdf(0)

#P(-1.65 < Z ≤ 1.93) =0.92


#P(0.85 < Z ≤ 2.23) =0.1848
#P(Z > 1.75) =0.0401 =  1-P(Z<=1.75)
#P(Z ≤ -0.69)= 0.2451
#P(-1.27 < Z ≤ 0)=0.398   =  P(Z<=0)- Z(Z<= -1.27)
#P(Z  > -2.64) =0.9959
#P(Z  ≤ 0.96)=0.8315

#get the z value according to probability value
norm(loc = 0 , scale = 1).ppf(0.5)
norm(loc = 0 , scale = 1).ppf(0.96)





emp_data= pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week3\\employee_data.csv")

mean_data= pd.DataFrame()
for ite in range(1000):
    mean_5= emp_data['CTC'].sample(5).mean()
    mean_10= emp_data['CTC'].sample(10).mean()
    mean_50= emp_data['CTC'].sample(50).mean()
    mean_100= emp_data['CTC'].sample(100).mean()
    df= pd.DataFrame([{'Iteration': ite, 'Mean_5':mean_5, 'Mean_10':mean_10, 'Mean_50':mean_50,
     'Mean_100': mean_100}])
    mean_data= pd.concat([mean_data,df])
    
    
sns.distplot(emp_data['CTC'])

sns.distplot(mean_data['Mean_5'])
sns.distplot(mean_data['Mean_10'])
sns.distplot(mean_data['Mean_50'])
sns.distplot(mean_data['Mean_100'])

mean_data.describe()
emp_data['CTC'].mean()


'''
https://www.statology.org/z-test-python/
Suppose the IQ in a certain population is normally distributed with
 a mean of μ = 100 and standard deviation of σ = 15.

A researcher wants to know if a new drug affects IQ levels, 
so he recruits 20 patients to try it and records their IQ levels.
The following code shows how to perform a one sample z-test
'''
from statsmodels.stats.weightstats import ztest as ztest

#enter IQ levels for 20 patients
data = [88, 92, 94, 94, 96, 97, 97, 97, 99, 99,
        105, 109, 109, 109, 110, 112, 112, 113, 114, 115]

#perform one sample z-test
ztest(data, value=100)

'''Since this p-value is not less than .05, 
we do not have sufficient evidence to reject the null hypothesis.
 In other words, the new drug does not significantly affect IQ level'''
 
 
'''
 Two Sample Z-Test in Python
Suppose the IQ levels among individuals in two different cities
 are known to be normally distributed with known standard deviations.

A researcher wants to know if the mean IQ level between individuals
 in city A and city B are different, so she selects a simple random sample
 of  20 individuals from each city and records their IQ levels.

The following code shows how to perform a two sample z-test to
 determine if the mean IQ level is different between the two cities

'''
from statsmodels.stats.weightstats import ztest as ztest

#enter IQ levels for 20 individuals from each city
cityA = [82, 84, 85, 89, 91, 91, 92, 94, 99, 99,
         105, 109, 109, 109, 110, 112, 112, 113, 114, 114]

cityB = [90, 91, 91, 91, 95, 95, 99, 99, 108, 109,
         109, 114, 115, 116, 117, 117, 128, 129, 130, 133]

#perform two sample z-test
ztest(cityA, cityB, value=0,alternative='two-sided') 
#'two-sided': H1: difference in means not equal to value (default)
# 'larger' : H1: difference in means larger than value 
#'smaller' : H1: difference in means smaller than value
'''
(-1.9953236073282115, 0.046007596761332065)
Since this p-value is less than .05,
 we have sufficient evidence to reject the null hypothesis.
 In other words, the mean IQ level is significantly different 
 between the two cities.
'''

'''
https://www.statology.org/one-sample-t-test-python/
Suppose a botanist wants to know if the mean height of a certain 
species of plant is equal to 15 inches. She collects a random sample 
of 12 plants and records each of their heights in inches.
Use the following steps to conduct a one sample t-test to
 determine if the mean height for this species of plant is
 actually equal to 15 inches.
 '''
data = [14, 14, 16, 13, 12, 17, 15, 14, 15, 13, 15, 14]
import scipy.stats as stats

#perform one sample t-test
stats.ttest_1samp(a=data, popmean=15)

'''
(statistic=-1.6848, pvalue=0.1201)
The t test statistic is -1.6848 and the corresponding two-sided p-value is 0.1201.

Step 3: Interpret the results.

The two hypotheses for this particular one sample t-test are as follows:

H0: µ = 15 (the mean height for this species of plant is 15 inches)

HA: µ ≠15 (the mean height is not 15 inches)

Because the p-value of our test (0.1201) is greater than alpha = 0.05,
 we fail to reject the null hypothesis of the test. 
 We do not have sufficient evidence to say that the mean 
 height for this particular species of plant is different from 15 inches.
 '''
 
'''
 https://www.statology.org/two-sample-t-test-python/
 Example: Two Sample t-Test in Python
Researchers want to know whether or not two different species of plants have the same mean height. To test this, they collect a simple random sample of 20 plants from each species.

Use the following steps to conduct a two sample t-test to determine if the two species of plants have the same height.
'''
import numpy as np

group1 = np.array([14, 15, 15, 16, 13, 8, 14, 17, 16, 14, 19, 20, 21, 15, 15, 16, 16, 13, 14, 12])
group2 = np.array([15, 17, 14, 17, 14, 8, 12, 19, 19, 14, 17, 22, 24, 16, 13, 16, 13, 18, 15, 13])

#find variance for each group
print(np.var(group1), np.var(group2))
#The ratio of the larger sample variance to the smaller sample variance 
#is 12.26 / 7.73 = 1.586, which is less than 4. This means
# we can assume that the population variances are equal.
import scipy.stats as stats

#perform two sample t-test with equal variances
stats.ttest_ind(a=group1, b=group2, equal_var=True,alternative='two-sided')

'''
The t test statistic is -0.6337 and the corresponding two-sided p-value is 0.53005.

The two hypotheses for this particular two sample t-test are as follows:

H0: µ1 = µ2 (the two population means are equal)

HA: µ1 ≠µ2 (the two population means are not equal)

Because the p-value of our test (0.53005) is greater than alpha = 0.05, 
we fail to reject the null hypothesis of the test. 
We do not have sufficient evidence to say that the mean height of plants
 between the two populations is different.
 '''

'''
https://www.statology.org/paired-samples-t-test-python/
Example: Paired Samples T-Test in Python
Suppose we want to know whether a certain study program significantly impacts
 student performance on a particular exam. To test this, we have 15 students in a
 class take a pre-test. Then, we have each of the students participate in the 
 study program for two weeks. Then, the students retake a test of similar difficulty.

To compare the difference between the mean scores on the first and second test, 
we use a paired samples t-test because for each student their first test score can 
be paired with their second test score. 

'''
pre = [88, 82, 84, 93, 75, 78, 84, 87, 95, 91, 83, 89, 77, 68, 91]
post = [91, 84, 88, 90, 79, 80, 88, 90, 90, 96, 88, 89, 81, 74, 92]

import scipy.stats as stats

#perform the paired samples t-test
stats.ttest_rel(pre, post,alternative='two-sided')

'''
The test statistic is -2.9732 and the corresponding two-sided p-value is 0.0101.

In this example, the paired samples t-test uses the following null and alternative hypotheses:

H0: The mean pre-test and post-test scores are equal

HA:The mean pre-test and post-test scores are not equal

Since the p-value (0.0101) is less than 0.05, we reject the null hypothesis. 
We have sufficient evidence to say that the true mean test score is 
different for students before and after participating in the study program.
'''

'''
https://www.statology.org/one-way-anova-python/
Example 1: One-Way ANOVA
Suppose we want to know whether or not three different exam prep 
programs lead to different mean scores on a certain exam. 
To test this, we recruit 30 students to participate in a study and 
split them into three groups.

The students in each group are randomly assigned to use one of the 
three exam prep programs for the next three weeks to prepare for an exam. 
At the end of the three weeks, all of the students take the same exam. 

The exam scores for each group are shown below:
'''
#enter exam scores for each group
group1 = [85, 86, 88, 75, 78, 94, 98, 79, 71, 80]
group2 = [91, 92, 93, 85, 87, 84, 82, 88, 95, 96]
group3 = [79, 78, 88, 94, 92, 85, 83, 85, 82, 81]

from scipy.stats import f_oneway

#perform one-way ANOVA
f_oneway(group1, group2, group3)

'''
(statistic=2.3575, pvalue=0.1138)
A one-way ANOVA uses the following null and alternative hypotheses:

H0 (null hypothesis): μ1 = μ2 = μ3 = … = μk (all the population means are equal)
H1 (null hypothesis): at least one population mean is different from the rest
The F test statistic is 2.3575 and the corresponding p-value is 0.1138. Since the p-value is not less than .05, we fail to reject the null hypothesis.

This means we do not have sufficient evidence to say that there 
is a difference in exam scores among the three studying techniques.
'''

'''
https://www.statology.org/chi-square-test-real-life-examples/
https://www.statology.org/chi-square-test-of-independence-python/
Example: Chi-Square Test of Independence in Python
Suppose we want to know whether or not gender is associated with 
political party preference. We take a simple random sample of 500 voters 
and survey them on their political party preference. 
The following table shows the results of the survey:
	Republican	Democrat	Independent	Total
Male	120	90	40	250
Female	110	95	45	250
Total	230	185	85	500
Use the following steps to perform a Chi-Square Test of Independence in Python to determine if gender is associated with political party preference.
'''

data = [[120, 90, 40],
        [110, 95, 45]]

import scipy.stats as stats

#perform the Chi-Square Test of Independence
stats.chi2_contingency(data)   

'''
The way to interpret the output is as follows:

Chi-Square Test Statistic: 0.864
p-value: 0.649
Degrees of freedom: 2 (calculated as #rows-1 * #columns-1)
Array: The last array displays the expected values for
 each cell in the contingency table.
 Recall that the Chi-Square Test of Independence uses the following null 
 and alternative hypotheses:

H0: (null hypothesis) The two variables are independent.
H1: (alternative hypothesis) The two variables are not independent.
Since the p-value (.649) of the test is not less than 0.05, we 
fail to reject the null hypothesis. This means we do not have 
sufficient evidence to say that there is an association between
 gender and political party preference.

In other words, gender and political party preference are independent.
 '''
 '''
 https://www.statology.org/correlation-test-in-python/