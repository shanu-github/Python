# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:07:24 2022

@author: shanu
"""

'''
Goal is to model that predict how likely a student is to pass their high school final exam.
school - student's school (binary: "GP" or "MS")
sex - student's sex (binary: "F" - female or "M" - male)
age - student's age (numeric: from 15 to 22)
address - student's home address type (binary: "U" - urban or "R" - rural)
famsize - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
Pstatus - parent's cohabitation status (binary: "T" - living together or "A" - apart)
Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
Mjob - mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
Fjob - father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
reason - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
guardian - student's guardian (nominal: "mother", "father" or "other")
traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
failures - number of past class failures (numeric: n if 1<=n<3, else 4)
schoolsup - extra educational support (binary: yes or no)
famsup - family educational support (binary: yes or no)
paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
activities - extra-curricular activities (binary: yes or no)
nursery - attended nursery school (binary: yes or no)
higher - wants to take higher education (binary: yes or no)
internet - Internet access at home (binary: yes or no)
romantic - with a romantic relationship (binary: yes or no)
famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
freetime - free time after school (numeric: from 1 - very low to 5 - very high)
goout - going out with friends (numeric: from 1 - very low to 5 - very high)
Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
health - current health status (numeric: from 1 - very bad to 5 - very good)
absences - number of school absences (numeric: from 0 to 93)
passed - did the student pass the final exam (binary: yes or no)
'''
import pandas as pd
import seaborn as sns

st_data= pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week9\\student-data.csv")

st_data.isnull().sum()

from sklearn.model_selection import train_test_split
#random split for  train data(80%) and test 20%)
train_data, test_data = train_test_split(st_data, test_size=0.2, random_state=31)


xtrain = train_data[st_data._get_numeric_data().columns]
ytrain = train_data['passed']

from sklearn.linear_model import LogisticRegression

model= LogisticRegression()
model.fit(xtrain, ytrain)

train_data['NcolPrediction']=  model.predict(xtrain)

pd.crosstab(ytrain, train_data['NcolPrediction'])

#-------------------------------------------------------------------
#model2
sns.boxplot(st_data['age'], st_data['passed'] )

import scipy.stats as stats

#perform two sample t-test with equal variances
stats.ttest_ind(a= st_data.loc[st_data['passed']=='no', 'age'],
                b= st_data.loc[st_data['passed']=='yes', 'age'],
                alternative='two-sided')

for num_col in st_data._get_numeric_data().columns :
    print ("Testing for ", num_col)
    sns.boxplot( st_data['passed'],st_data[num_col] )
    #perform two sample t-test with equal variances
    print(stats.ttest_ind(a= st_data.loc[st_data['passed']=='no', num_col],
                    b= st_data.loc[st_data['passed']=='yes', num_col],
                    alternative='two-sided'))
    
xtrain = train_data[['age','Medu','Fedu','failures','goout','absences','studytime']]
ytrain = train_data['passed']

from sklearn.linear_model import LogisticRegression

model= LogisticRegression()
model.fit(xtrain, ytrain)

train_data['NcolPrediction']=  model.predict(xtrain)

pd.crosstab(ytrain, train_data['NcolPrediction'])

#------------------------------------------------------
#model3
xtrain = train_data[['age','Medu','Fedu','failures','goout','absences','studytime']]
ytrain = train_data['passed']
obj_cols= ['school', 'sex', 'address', 'famsize', 'Pstatus', 
       'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 
       'activities', 'nursery','higher', 'internet', 'romantic' ]

for cat_var in obj_cols:
    dummy_data= pd.get_dummies(train_data[cat_var])
    cols= dummy_data.columns
    new_cols= []
    for col in cols:
        new_cols.append(cat_var+col)
        
    dummy_data.columns= new_cols
    xtrain = pd.concat([xtrain, dummy_data], axis =1)
    
from sklearn.linear_model import LogisticRegression

model= LogisticRegression()
model.fit(xtrain, ytrain)

train_data['prob_'+model.classes_[0]]=  model.predict_proba(xtrain)[:,0]


sns.boxplot(train_data['prob_no'],train_data['passed'])

sns.relplot(train_data['age'],train_data['prob_no'], hue= train_data['passed'])


import numpy as np
np.arange(0.0, 1.0, 0.1)

acuracy_cal= pd.DataFrame()
for threshold in np.arange(0.0, 1.0, 0.1):
    train_data['predicted_class']= np.where(train_data['prob_no']>threshold, 'no','yes')
    total_pred= len(train_data[train_data['predicted_class']=='no'])
    total_act= len(train_data[train_data['passed']=='no'])
    total_cprediction= len(train_data[(train_data['predicted_class']=='no') &
                                          (train_data['passed']=='no')])
    precision= total_cprediction/total_pred
    recall =  total_cprediction/total_act
    f1_score= (2*precision*recall) /(precision+ recall)
    acuracy_cal= acuracy_cal.append(pd.DataFrame([[threshold, precision, recall, f1_score]], 
                                                 columns= ['Threshold','Precision','Recall','F1_Score']))


train_data['NcolPrediction']= np.where(train_data['prob_no']>0.3, 'no','yes')
pd.crosstab(ytrain, train_data['NcolPrediction'])

from sklearn.metrics import classification_report
print(classification_report(train_data['NcolPrediction'],train_data['passed']))

pd.DataFrame(model.coef_, columns= xtrain.columns).
print(pd.DataFrame({'Features': xtrain.columns,'Coeffiecient': model.coef_}))


model.intercept_


st_data['passed'].value_counts()

#age, failures is important
#Medu, Fedu,studytime is less important
#traveltime is not important






#Exploratory Data Analysis

st_data.describe()

st_data.dtypes

#get numeric columns
cols = st_data.columns

num_cols= st_data._get_numeric_data().columns
list(set(cols) - set(num_cols))





#By default gives only for numeric columns
emp_data.describe()
# list of dtypes to include
emp_data.describe(include= ['object'])



from scipy import stats
stats.ttest_ind(input_data.loc[input_data['passed']=='yes','traveltime'],
                input_data.loc[input_data['passed']=='no','traveltime'])


sns.boxplot(data=emp_data,x="LOCATION", y="CTC", hue='GENDER')


cont = pd.crosstab(input_data['passed'],input_data['sex'])
import scipy.stats
scipy.stats.chi2_contingency(cont)

#selected continous variables
con_vars= ['age',"goout","famrel","failures","studytime","Fedu"]
X_train = train_data[con_vars] 

#Create dummy variables
#That is variables with only two values, zero and one.
print(cat_vars)
for var in cat_vars:
    cat_list = pd.get_dummies(train_data[var], prefix=var)
    X_train= X_train.join(cat_list)
Y_train= train_data['passed']

'''
#school: student from 'GP' are more likely to pass the exam, but not very strong predictor for passed.
#sex: male student are more likely to pass the exam.
#address: There is no difference in performance of students whether he belong to urban or rural
#famsize: There is no difference in performance of students whether there family size is less than 3 or greater than 3
#Pstatus: There is no difference in performance of students whether his parents stays together or apart
#Medu : mother education is important for student education
#Fedu : father education is also important for student education
#Mjob : Mother job is not very important for student education
#Fjob : father job is important for student education, failing only when at_home , can reduce number of categories
#reason is not very important
#If the gurdian is other, student passed 50% chance., it is important
# If students are taking extra class, means they will fail. schoolsup is important
#famsup: family educational support doesn't seem very important
#paid is important
#activities not important
#higher is important
#internet not so important
#romantic is important
'''
