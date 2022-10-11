# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 18:59:38 2022

@author: shanu
"""
# Data frame is a two-dimensional data structure,
# i.e., data is aligned in a tabular fashion in rows and columns.
#PANDAS (PANel DAta) is a high-level data manipulation tool used for analysing data.
# The primary two components of pandas are the Series and DataFrame.
# A Series is essentially a column, and 
# a DataFrame is a multi-dimensional table made up of a collection of Series.

#Create an Empty DataFrame
import pandas as pd
df = pd.DataFrame()
print(df)


#Create a DataFrame from Lists
#The DataFrame can be created using a single list or a list of lists

data = [1,2,3,4,5]
df = pd.DataFrame(data)
print(df)

data = [['Rahul',10],['Rohan',12],['Seema',13]]
df = pd.DataFrame(data,columns=['Name','Age'])
print(df)

#Create a DataFrame from Dict
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data)
print(df)

#create a dataframe by reading files
emp_data= pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week3\\employee_data.csv")

#get the column names
print(emp_data.columns)

#get the first 5 rows of data
emp_data.head()
emp_data.tail()

#to know the data type for each column
emp_data.dtypes

#to get the information
emp_data.info()

emp_data.describe()

#to know rows/columns
emp_data.shape

len(emp_data)
len(emp_data.columns)

emp_data.columns.values 

#-----------------------------------------------------------
#Data Selection using column names
emp_data['NAME']
emp_data[['NAME','GENDER']]


#Row Selection using index
emp_data.loc[0]
emp_data.loc[0:2]

#when using .loc it should be actual row index
sub_emp_data= emp_data[100:]
sub_emp_data.loc[0:3] #will provide empty dataframe

#using .iloc can be accessed when not sure about index
sub_emp_data.iloc[0:3]

#accessing columns and rows together
emp_data.loc[1:5, 'NAME']

emp_data.loc[1:5, ['NAME','GENDER']]

#.iloc should be used with indexes only
sub_emp_data.iloc[0:3, ['NAME','GENDER']] #gives an error
sub_emp_data.iloc[0:3, [1,5]]

#reset the index and can use .loc
sub_emp_data=sub_emp_data.reset_index(drop=True)
sub_emp_data.loc[0:3, ['NAME','GENDER']] #gives an error

#Conditional selections
emp_data.loc[emp_data['LOCATION']=='Bangalore']

#Selected columns
emp_data.loc[emp_data['LOCATION']=='Bangalore',['EMP ID','DEPT']]

emp_data.loc[(emp_data['LOCATION']=='Bangalore') & (emp_data['DEPT']=='Media Relations'),]
emp_data.loc[(emp_data['LOCATION']=='Bangalore') | (emp_data['DEPT']=='Media Relations'),]
emp_data.loc[~(emp_data['LOCATION']=='Bangalore'),]

#Filter employee for Bangalore and Kolkata location
sub_data= emp_data[(emp_data['LOCATION']=='Bangalore') | (emp_data['LOCATION']== 'Kolkata')]

#Filter employee not form Bangalore and Kolkata location
emp_data.loc[~(emp_data['LOCATION'].isin(['Bangalore','Kolkata']))]


#suppose we want to get employee have more than 3 rating
#suppose we want to filter kolkata employee in accounts dept
#filter the employee data for employees where CTC> 8 lakhs
#filter the employee data for employees where CTC> 8 lakhs,extract  Name, Gender
#-----------------------------------------------------------
#get summary the data like min, max, mean, std, count
emp_data['CTC'].max()
emp_data['CTC'].min()
df.apply(pd.Series.value_counts) # Unique values and counts for all columns
df.describe() # Summary statistics for numerical columns
df.mean() # Returns the mean of all columns
df.corr() # Returns the correlation between columns in a DataFrame
df.count() # Returns the number of non-null values in each DataFrame column
df.max() # Returns the highest value in each column
df.min() # Returns the lowest value in each column
df.median() # Returns the median of each column
df.std() # Returns the standard deviation of each column
df.columns = ['a','b','c'] # Renames columns
pd.isnull() # Checks for null Values, Returns Boolean Array
pd.notnull() # Opposite of s.isnull()
df.dropna() # Drops all rows that contain null values
df.dropna(axis=1) # Drops all columns that contain null values
df.dropna(axis=1,thresh=n) # Drops all rows have have less than n non null values
df.fillna(x) # Replaces all null values with x
s.fillna(s.mean()) # Replaces all null values with the mean (mean can be replaced with almost any function from the statistics section)
s.astype(float) # Converts the datatype of the series to float
s.replace(1,'one') # Replaces all values equal to 1 with 'one'
s.replace([1,3],['one','three']) # Replaces all 1 with 'one' and 3 with 'three'
df.rename(columns=lambda x: x + 1) # Mass renaming of columns
df.rename(columns={'old_name': 'new_ name'}) # Selective renaming
df.set_index('column_one') # Changes the index
df.rename(index=lambda x: x + 1) # Mass renaming of index

emp_data['LOCATION'].unique()
#suppose we want to know average salary of kolkata employee in accounts dept
kolkata_emp= emp_data[(emp_data['LOCATION']=='Kolkata') & (emp_data['DEPT']=='Accounting')]
kolkata_emp['CTC'].mean()


#By default gives only for numeric columns
emp_data.describe()
# list of dtypes to include
emp_data.describe(include= ['object'])

# we can call describe method for specific column as well
emp_data["GENDER"].describe()
emp_data['CTC'].describe()
# we can call count for specific column as well
emp_data["GENDER"].value_counts()
pd.crosstab(emp_data["GENDER"],emp_data['LOCATION'])
pd.crosstab(emp_data["GENDER"],emp_data['LOCATION'], margins=True)
pd.crosstab(emp_data["GENDER"],emp_data['LOCATION'], normalize=True)
pd.crosstab(emp_data["GENDER"],emp_data['LOCATION'], normalize='index')

pd.crosstab(emp_data["GENDER"],emp_data['LOCATION'], values= emp_data['CTC'], aggfunc= max)

#count the employees in each location
emp_data['LOCATION'].unique()
emp_data['LOCATION'].nunique()
#maximum salary, minimum salary, average salary in each location, in each dept
#maximum and minimum performance rating for each location

#Dept and Gender wise average CTC
emp_data.groupby(['DEPT','GENDER'], as_index=True).agg(
    Avg_CTC=('CTC', 'mean')).reset_index()

#suppose for each dept, gender, location wise employee count
emp_data.groupby(['DEPT','LOCATION','GENDER'], as_index=True).agg(
    total_employee=('EMP ID', 'count'),
    MAX_CTC= ('CTC','max')).reset_index()

def diffminmax(x):
    return (x.max() - x.min())


df3= emp_data.groupby(['LOCATION'])['CTC'].apply(diffminmax).reset_index()

#Minimum salary for each location
#Maximum salary for each location
#want to find maximum CTC for each location and each dept
#I want to know location and Gender wise average salary, maximum salary, minimum salary
#I want to know location and department wise average salary, maximum salary, minimum salary

#Find Average salary for employees located in Bangalore
#Is all department available in all location nunique
#How many employee are working in each location
#how many male & female employee
#how many male & female employee each location




#-------------------------------------------------------------------------
#Data cleaning & Preparation
#creating new column for bonus using CTC, it isCTC 10 % of CTC


#Data Type Transformation in python
emp_data.dtypes

emp_data['NAME']= emp_data['NAME'].astype('str')
emp_data['CTC']= emp_data['CTC'].astype('float')

emp_data['CTC']= emp_data['CTC'].apply(lambda x: float(x))

#Correct date column
#date format is very important for correct transformation
#https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
emp_data['DATE OF JOINING'].unique()
emp_data['DOJ']=emp_data['DATE OF JOINING'].apply(lambda x: pd.to_datetime(x,format='%A, %d %B %Y'))

#get new column
emp_data['experience']=pd.datetime.now()-emp_data['DOJ']
emp_data['experience']= emp_data['experience'].dt.days/365

emp_data['experience'].describe()

emp_data.isnull().sum()
# removing null values
emp_data.dropna(inplace = True)
len(emp_data)

emp_data.dropna(subset = ['CTC','EMP ID'])
emp_data.dropna(how='all')
emp_data.dropna(thresh=2) #2 valid value to keep row

#filling NA with some value
emp_data['ANNUAL PERFORMANCE RATING'].fillna(0, inplace=True)

'''
Imputation
Imputation is a conventional feature engineering technique used to keep 
valuable data that have null values.There may be instances where dropping 
every row with a null value removes too big a chunk from your dataset, 
so instead we can impute that null with another value, 
usually the mean or the median of that column.

'''
#filling NA with some value
emp_data['ANNUAL PERFORMANCE RATING'].fillna(emp_data['ANNUAL PERFORMANCE RATING'].mean(), inplace=True)

loan_data= pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\\Week3\\Loan_Prediction.csv")
loan_data.isnull().sum()
iloan_data=loan_data.interpolate()
loan_data.replace([-99999999, -777777777], np.NaN)
#replacing all negative CTC column value with zero

def rating_function(x):
    if x >= 3.0:
        return "good"
    else:
        return "bad"

emp_data["rating_category"] = emp_data["ANNUAL PERFORMANCE RATING"].apply(rating_function)   
emp_data["rating_category"] = emp_data["ANNUAL PERFORMANCE RATING"].apply(lambda x: rating_function(x))   
    
#removing the duplicate values
emp_data=emp_data.drop_duplicates()

#dropping particular column
emp_data.drop(['DEPT'],axis=1)

df1.dropna(thresh = 3) # drop any row containing  < 3 number of observations

#column renaming
emp_data.rename(columns={'ANNUAL PERFORMANCE RATING': 'ANNUAL RATING', 
        'EMP ID': 'EMPLOYEE ID'
    }, inplace=True)

emp_data.columns = ['EMP ID', 'NAME', 'DATE OF JOINING', 'LOCATION', 'DEPT', 'GENDER',
       'CTC', 'ANNUAL PERFORMANCE RATING', 'DOJ', 'experience']

#Reshaping of data
#suppose you need LOCATION in columns
employee_summary= emp_data.pivot_table(index=['DEPT','GENDER'],
                                  columns='LOCATION',values='EMP ID',
                                 fill_value=0, aggfunc='count').reset_index()

# Transforming LOCATION column to rows
temployee_summary = pd.melt(employee_summary, id_vars=['DEPT','GENDER'],
                            var_name="LOCATION", value_name="total_employee")

#suppose for each dept, gender, location wise employee count
emp_data.groupby(['DEPT','LOCATION','GENDER'], as_index=True).agg(
    total_employee=('EMP ID', 'count')).reset_index()

emp_data.sort_values('DOJ')
sorted_data=emp_data.sort_values(["CTC", 'ANNUAL PERFORMANCE RATING'], ascending=[False, True])

## Create a messy dataset
growth_data =pd.DataFrame({  'country':["A", "B", "C"],
  'q1_2017':[0.03, 0.05, 0.01],
  'q2_2017' : [0.05, 0.07, 0.02],
  'q3_2017':[0.04, 0.05, 0.01],
  'q4_2017' : [0.03, 0.02, 0.04]})

melted_data= pd.melt(  growth_data, id_vars=['country'],
                            var_name="quarter_year", value_name="growth")

melted_data[['quarter','year']] = melted_data['quarter_year'].str.split("_",expand=True)

melted_data['quarter_year1'] = melted_data['quarter']+melted_data['year']
#-----------------------------------------------------------------------------------
#Data Joining
df1.append(df2) # Adds the rows in df1 to the end of df2 (columns should be identical)
pd.concat([df1, df2],axis=1) # Adds the columns in df1 to the end of df2 (rows should be identical)
df1.join(df2,on=col1,how='inner') # SQL-style joins the columns in df1 with the columns on df2 where the rows for col have identical values. how can be one of 'left', 'right', 'outer', 'inner' 

#Joining operations
# data frame 1
df1 = pd.DataFrame({'CustomerId' :[1,2,3,4,5,6],'Product' : ["Oven","Radio", "Oven","Television",
                                                  "Oven","Television"]})
print(df1)
# data frame 2
df2 = pd.DataFrame({'CustomerId' :[2, 4, 6, 8], 'Place':["Mumbai","Kolkata",
                                                    "Bangalore","Mumbai"]})

print(df2)

pd.merge(df1, df2, how = 'inner')

pd.merge(df1, df2, how = 'left')

pd.merge(df1, df2, how = 'right')

pd.merge(df1, df2, how = 'outer')

#When right and left label different
pd.merge(df1, df2, how="inner", left_on=('CustomerId'), right_on=('CustomerId'))

#binding two dataframe into one
emp_data.loc[emp_data['LOCATION']=='Bangalore'].append(emp_data.loc[emp_data['LOCATION']=='Kolkata'])



#get numeric columns
cols = emp_data.columns

num_cols= emp_data._get_numeric_data().columns
list(set(cols) - set(num_cols))


emp_data.select_dtypes(exclude=["number","bool_","object_"])

emp_data.select_dtypes(exclude=["object_"])

emp_data.select_dtypes(include=['object_'])

emp_data.select_dtypes(include=['object']).columns.tolist()

gender_code= {'Male':0, 'Female':1}

emp_data['CGender']= emp_data['GENDER'].apply(lambda x: gender_code[x])

#reference links
#https://elitedatascience.com/python-cheat-sheet
#https://www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/
#https://blog.finxter.com/pandas-cheat-sheets/
#https://www.youtube.com/playlist?list=PLeo1K3hjS3uuASpe-1LjfG5f14Bnozjwy
#https://www.youtube.com/watch?v=bPrmA1SEN2k&list=PLZoTAELRMXVNUL99R4bDlVYsncUNvwUBB

#DUMP from class
#create a dataframe by reading files
emp_data= pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week3\\employee_data.csv")
emp_data.info()
len(emp_data)
emp_data.shape
emp_data['DEPT']
emp_data[['DEPT','NAME']]
emp_data.loc[6]
emp_data.loc[6:10]
emp_data.loc[6:10, 'CTC']
emp_data.loc[6:10, ['CTC','NAME']]
emp_data['DEPT'].unique()
emp_data.groupby(['LOCATION', 'GENDER'], as_index=True).agg(max_ctc= ('CTC','max').reset_index()
)
emp_data.groupby(['LOCATION', 'GENDER'], as_index=True).agg(max_ctc= ('CTC','max')).reset_index()
emp_data.groupby(['LOCATION', 'GENDER'], as_index=True).agg(max_ctc= ('CTC','count')).reset_index()
emp_data.groupby(['LOCATION', 'GENDER'], as_index=True).agg(employee_count= ('CTC','count')).reset_index()
def diffminmax(x):
    return (x.max() - x.min())
emp_data.groupby(['LOCATION', 'GENDER'], as_index=True)['CTC'].max().reset_index()
emp_data.groupby(['LOCATION', 'GENDER'], as_index=True)['CTC'].max()
emp_data.groupby(['LOCATION', 'GENDER'], as_index=True)['CTC'].max().reset_index()
emp_data.groupby(['LOCATION', 'GENDER'], as_index=True)['CTC'].mean().reset_index()
emp_data.groupby(['LOCATION', 'GENDER'], as_index=True)['CTC'].sum().reset_index()
emp_data.groupby(['LOCATION', 'GENDER'], as_index=False)['CTC'].sum().reset_index()
emp_data.groupby(['LOCATION', 'GENDER'])['CTC'].sum().reset_index()
emp_data.groupby(['LOCATION', 'GENDER'])['CTC'].apply(diffminmax).reset_index()
summary_data =emp_data.groupby(['LOCATION', 'GENDER'])['CTC'].apply(diffminmax).reset_index()
summary_data.columns= ['LOCATION', 'GENDER', 'MINMAXDiff']
summary_data
summary_data_max_ctc =emp_data.groupby(['LOCATION', 'GENDER'])['CTC'].max().reset_index()
summary_data_max_ctc.columns= ['LOCATION', 'GENDER', 'MAXCTC']
pd.merge(summary_data, summary_data_max_ctc, how='inner')
summary_data_max_ctc =emp_data.groupby(['LOCATION', 'GENDER'])['CTC'].max().reset_index()
summary_data_max_ctc
summary_data_max_ctc.columns
summary_data_max_ctc.columns=[ 'Location', 'GENDER', 'maxCTC']
summary_data_max_ctc.reanme(columns= {'Location': 'LOCATION'})
summary_data_max_ctc.rename(columns= {'Location': 'LOCATION'})
summary_data_max_ctc= summary_data_max_ctc.rename(columns= {'Location': 'LOCATION'})
summary_data_max_ctc.rename(columns= {'Location': 'LOCATION'}, inplace= True)
employee_data['DEPT'].unqiue()
employee_data['DEPT'].unique()
emp_data['DEPT'].unique()
emp_data['DEPT'].nunique()
emp_data['Bonus']= emp_data['CTC']*0.10
emo_dat['Tpotal Payable']= emp_data['CTC']+ emp_data['Bonus']
emp_data['Tpotal Payable']= emp_data['CTC']+ emp_data['Bonus']
emp_data.colums
emp_data.columns
emp_data['new Bonus']=  emp_data['CTC']* emp_data['ANNUAL PERFORMANCE RATING']/100
emp_data['CTC']= emp_data['CTC']. astype('float')
emp_data.dtypes
emp_data['DATE OF JOINING']
pd.to_datetime(emp_data['DATE OF JOINING'])
emp_data['DOJ']= pd.to_datetime(emp_data['DATE OF JOINING'])
emp_data['Year_DOJ']= emp_data['DOJ'].dt.year
emp_data['Year_DOJ']= emp_data['DOJ'].dt.month
pd.datetime.now()
emp_data['Experience']= pd.datetime.now()- emp_data['DOJ']
emp_data['Experience']= emp_data['Experience'].dt.days/365
import math
emp_data['Experience']= emp_data['Experience'].apply(lambda x: math.floor(x))
emp_data['Experience']= emp_data['Experience'].dt.days/365
emp_data['Experience']= pd.datetime.now()- emp_data['DOJ']
emp_data['Experience']= emp_data['Experience'].dt.days/365
emp_data['Experience']= emp_data['Experience'].astype('int')
emp_data.isnull().sum()
loan_data= pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\\Week3\\Loan_Prediction.csv")
loan_data.isnull().sum()
emp_data.dropna(['Married'])
loan_data.dropna(['Married'])
loan_data.dropna(subset=['Married'])
loan_data.dropna(subset=['Married', 'Loan_Status'])
loan_data.dropna()
loan_data_woutna=loan_data.dropna()
loan_data.columns
loan_data['LoanAmount']= loan_data['LoanAmount'].fillna( loan_data['LoanAmount'].mean()
)
emp_data.drop(['NAME'], axis=1)
 emp_data.pivot_table(index= ['DEPT'], columns='GENDER', values='EMP ID', aggfunc='count').reset_index()
 emp_data.pivot_table(index= ['DEPT','LOCATION'], columns='GENDER', values='EMP ID', aggfunc='count').reset_index()
 emp_data.pivot_table(index= ['DEPT','LOCATION'], columns='GENDER', values='EMP ID', aggfunc='count', fillna= 0).reset_index()
 emp_data.pivot_table(index= ['DEPT','LOCATION'], columns='GENDER', values='EMP ID', aggfunc='count', fill_value= 0).reset_index()
 summary_data =emp_data.pivot_table(index= ['DEPT','LOCATION'], columns='GENDER', values='EMP ID', aggfunc='count', fill_value= 0).reset_index()
pd.melt(summary_data, id_vars=['DEPT','LOCATION'], var_name='gender', value_name='total employye')
emp_data.columns()
emp_data.columns
emp_data.pivot_table(index=['DEPT'], columns='LOCATION').reset_index()
df= emp_data.pivot_table(index=['DEPT'], columns='LOCATION').reset_index()
df= emp_data.pivot_table(index=['DEPT'], columns='LOCATION', values='LOCATION', aggfunc='count').reset_index()
df= emp_data.pivot_table(index=['DEPT'], columns='LOCATION', values='DEPT', aggfunc='count').reset_index()
df= emp_data.pivot_table(index=['DEPT'], columns='LOCATION', values='NAME', aggfunc='count').reset_index()
growth_data =pd.DataFrame({  'country':["A", "B", "C"],
  'q1_2017':[0.03, 0.05, 0.01],
  'q2_2017' : [0.05, 0.07, 0.02],
  'q3_2017':[0.04, 0.05, 0.01],
  'q4_2017' : [0.03, 0.02, 0.04]})
pd.melt(growth_data, id_vars=['country'], var_name='quarter_year', value_name='growth')
melted_data =pd.melt(growth_data, id_vars=['country'], var_name='quarter_year', value_name='growth')
melted_data[['quarter', 'year']]= melted_data['quarter_year'].apply(lambda x: x.split('_') )
melted_data[['quarter', 'year']]= melted_data['quarter_year'].apply(lambda x: str(x).split('_') )
melted_data[['quarter', 'year']]= melted_data['quarter_year'].apply(lambda x: str(x).split('_', expand=True) )
melted_data['quarter_year'].apply(lambda x: str(x).split('_') )
pd.Dataframe(melted_data['quarter_year'].apply(lambda x: str(x).split('_') ))
pd.DataFrame(melted_data['quarter_year'].apply(lambda x: str(x).split('_') ))
melted_data[['quarter','year']] = melted_data['quarter_year'].str.split("_",expand=True)
melted_data['quarter_year'].str.split("_",expand=True)
melted_data[['quarter','year']] = melted_data['quarter_year'].str.split("_",expand=True)
cols= emp_data.columns
cols
emp_data._get_numeric_data()
emp_data._get_numeric_data().columns
numeric_cols= emp_data._get_numeric_data().columns
list(set(cols) - set(num_cols))

cols = emp_data.columns

num_cols= emp_data._get_numeric_data().columns
list(set(cols) - set(num_cols))
emp_data.select_dtypes(exclude=["number","bool_","object_"])
emp_data.select_dtypes(exclude=["object_"])
ta.select_dtypes(include=['object']).columns.tolist()
emp_data.select_dtypes(include=['object']).columns.tolist()
_data.select_dtypes(include=['object_'])
emp_data.select_dtypes(include=['object_'])
emp_data.select_dtypes(include=['object_']).columns
gender_code= {'Male':0, 'Female':1}

emp_data['CGender']= emp_data['GENDER'].apply(lambda x: gender_code[x])