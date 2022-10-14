
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 00:45:34 2022

@author: shanu
"""
#Histogram chart
#Bar chart

#density plot
#scatter plot
#box plot
#Line Plot

# Import seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Apply the default theme
sns.set_theme()

'''
Statistical analyses require knowledge about the distribution of variables in your dataset.
The seaborn function displot() supports several approaches to visualizing distributions. 
These include classic techniques like histograms and
computationally-intensive approaches like kernel density estimation

Distplot stands for distribution plot, it takes as input an array and 
plots a curve corresponding to the distribution of points in the array.
Only for numeric data
'''

emp_data= pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week3\\employee_data.csv")

sns.distplot(emp_data['CTC'])
sns.displot(data=emp_data, x="CTC")
sns.displot(data=emp_data, x="CTC", kde=True)

#how is distribution for male and female employees
sns.displot(data=emp_data, x="CTC",col='GENDER', kde=True)
#on top of eachother
sns.displot(data=emp_data, x="CTC",hue= 'GENDER', kde=True)

#Is the above analysis true for all locations
sns.displot(data=emp_data, x="CTC",col='GENDER',hue= 'LOCATION', kde=True)

#cumulative distribution
sns.displot(data=emp_data, kind="ecdf", x="CTC", hue="GENDER", rug=True)


sns.kdeplot(data=emp_data,x="CTC",shade=True, color='b')

sns.kdeplot(data=emp_data,x="CTC",hue='GENDER',shade=True, color='b')

sns.factorplot(data=emp_data, x='ANNUAL PERFORMANCE RATING',kind="count", color='steelblue')
sns.displot(data=emp_data, x="ANNUAL PERFORMANCE RATING", kde=False)


#how is joining of employees in years
emp_data['DATE OF JOINING']=emp_data['DATE OF JOINING'].apply(lambda x: pd.to_datetime(x,format='%A, %d %B %Y'))
emp_data['year']= emp_data['DATE OF JOINING'].dt.year
g = sns.factorplot(data=emp_data,x="year", kind="count", color='steelblue')
g.set_xticklabels(step=2)

#Factor plots can be useful for this kind of visualization as well.
# This allows you to view the distribution of a parameter within bins 
#defined by any other parameter
sns.factorplot(data=emp_data,x="year", kind="count", hue='GENDER')

sns.factorplot(data=emp_data,x="LOCATION", y='CTC',kind="box", hue='GENDER')

#--------------------------------------------------------------------------------
'''
Several specialized plot types in seaborn are oriented towards visualizing categorical data.
They can be accessed through catplot(). These plots offer different levels of granularity.
'''
#bar plot using catplot()
sns.catplot(data=emp_data, kind="count", x="LOCATION")
sns.catplot(data=emp_data, kind="count", x="LOCATION", col= 'GENDER')
sns.catplot(data=emp_data, kind="count", x="LOCATION", col= 'DEPT')

sns.catplot(data=emp_data, kind="count", x="DEPT")

ax = sns.catplot(data=emp_data, kind="count", x="DEPT")
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)

#with barplot
employee_count= emp_data.groupby(['DEPT'], as_index=True).agg(total_employee=('EMP ID', 'count')).reset_index()
sns.barplot(data=employee_count,  x="DEPT", y= 'total_employee')

ax = sns.barplot(data=employee_count,  x="DEPT", y= 'total_employee')
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)

sns.factorplot(data=emp_data, kind="count", x="LOCATION")



#plots With cat and numeric
#you could show only the mean value and its confidence interval within each nested category:
sns.catplot(data=emp_data, kind="bar", x="LOCATION", y="CTC")
sns.catplot(data=emp_data, kind="bar", x="LOCATION", y="CTC", hue='GENDER')

sns.boxplot(data=emp_data,x="LOCATION", y="CTC", palette='rainbow')
sns.boxplot(data=emp_data,x="LOCATION", y="CTC", hue='GENDER')


#with overall distribution
#At the finest level, you may wish to see every observation by drawing 
#a “swarm” plot: a scatter plot that adjusts the positions of the points
# along the categorical axis so that they don’t overlap
sns.catplot(data=emp_data, kind="swarm", x="LOCATION", y="CTC")

sns.catplot(data=emp_data, kind="swarm", x="LOCATION", y="CTC", hue='GENDER')

#Alternately, you could use kernel density estimation to represent the 
#underlying distribution that the points are sampled from:
sns.catplot(data=emp_data, kind="violin",  x="LOCATION", y="CTC")

sns.catplot(data=emp_data, kind="violin", x="LOCATION", y="CTC", hue='GENDER')
sns.catplot(data=emp_data, kind="violin", x="LOCATION", y="CTC", hue='GENDER',split=True)
#--------------------------------------------------------------------------------

'''
relational plots This plot shows the relationship between variables
using seaborn function relplot().
'''
loan_data = pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week3\\Loan_Prediction.csv")
# Create a visualization
sns.relplot(data=loan_data,x='ApplicantIncome', y='LoanAmount')

sns.relplot(
    data=loan_data,
    x='ApplicantIncome', y='LoanAmount', col="Married"
)

sns.relplot(
    data=loan_data,
    x='ApplicantIncome', y='LoanAmount', hue="Married"
)

sns.relplot(
    data=loan_data,
    x='ApplicantIncome', y='LoanAmount', hue="Loan_Status"
)


sns.relplot(
    data=loan_data,
    x='ApplicantIncome', y='LoanAmount', col="Married",
    hue="Gender",
)
sns.relplot(
    data=loan_data,
    x='ApplicantIncome', y='LoanAmount', hue="Married",
    style="Gender",
)

sns.relplot(
    data=loan_data,
    x='ApplicantIncome', y='LoanAmount', col="Married",
    hue="Gender", style="Gender"
)

sns.relplot(
    data=loan_data,
    x='ApplicantIncome', y='LoanAmount', col="Married",
    hue="Gender", style="Gender", size= 'Loan_Amount_Term'
)

#Statistical estimation in seaborn goes beyond descriptive statistics.
# For example, it is possible to enhance a scatterplot by including a linear regression
# model (and its uncertainty) using lmplot()

sns.lmplot(data=loan_data, x='ApplicantIncome', y='LoanAmount')
sns.lmplot(data=loan_data, x='ApplicantIncome', y='LoanAmount', hue='Gender'
          )
#-------------------------------------------
#line charts with time data
sns.relplot(data=emp_data, kind="line", x="DATE OF JOINING", y= 'CTC' )

sns.relplot(data=emp_data, kind="line", x="DATE OF JOINING", y= 'CTC', hue='GENDER' )


g.set_xticklabels(step=2)

dots = sns.load_dataset("dots")
sns.relplot(
    data=dots, kind="line",
    x="time", y="firing_rate",facet_kws=dict(sharex=False),)

sns.relplot(
    data=dots, kind="line",
    x="time", y="firing_rate", col="align"
)

sns.relplot(
    data=dots, kind="line",
    x="time", y="firing_rate", col="align",
    hue="choice", size="coherence", style="choice",
    facet_kws=dict(sharex=False),
)

#Statistical estimation
'''
Often, we are interested in the average value of one variable as a 
#function of other variables. Many seaborn functions will automatically 
#perform the statistical estimation that is necessary to answer these questions:
When statistical values are estimated, seaborn will use bootstrapping 
to compute confidence intervals and draw error bars representing 
the uncertainty of the estimate.
'''
fmri = sns.load_dataset("fmri")
sns.relplot(
    data=fmri, kind="line",
    x="timepoint", y="signal", col="region",
    hue="event", style="event",
)
#----------------------------------------------------------------------------
#Multivariate views on complex datasets
'''
Some seaborn functions combine multiple kinds of plots to quickly 
give informative summaries of a dataset. One, jointplot(),
 focuses on a single relationship. It plots the joint distribution
 between two variables along with each variable’s marginal distribution:
'''
sns.jointplot(data=loan_data, x='ApplicantIncome', y='LoanAmount')

sns.jointplot(data=loan_data, x='ApplicantIncome', y='LoanAmount', hue= 'Loan_Status')


'''
The other, pairplot(), takes a broader view:
    it shows joint and marginal distributions for all pairwise relationships 
    and for each variable, respectively   
    
'''
sns.pairplot(data=loan_data)

sns.pairplot(data=loan_data, hue= 'Loan_Status')

# plotting correlation heatmap
sns.heatmap(loan_data.corr(), cmap="YlGnBu", annot=True)

## Draw the heatmap with the mask and correct aspect ratio
#sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#            square=True, linewidths=.5, cbar_kws={"shrink": .5})  

#reference links
#https://seaborn.pydata.org/tutorial/introduction
#https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html
#https://vitalflux.com/correlation-heatmap-with-seaborn-pandas/

#Dump from class
