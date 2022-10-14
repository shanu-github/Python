# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 05:57:31 2022

@author: shanu
"""
import pandas as pd
'''
The Pandas library offers a wide range of possibilities for saving your data
 to files and loading data from files. 
 In this section, you’ll learn more about working with CSV and Excel files.
 You’ll also see how to use other types of files, like 
 JSON, web pages, databases, and Python pickle files.
'''
#Reading of CSV files (comma separated files)
'''
Syntax: pd.read_csv(filepath, sep=’ ,’ , header=’infer’,
                    index_col=None, usecols=None,names = optional, 
                    skiprows=None, nrows=None) 

Parameters: 

filepath: It is the location of the file which is to be retrieved using this function. It accepts any string path or URL of the file.
sep: It stands for separator, default is ‘, ‘ as in CSV(comma separated values).
sep='[:, |_]'
header: It accepts int, a list of int, row numbers to use as the column names,
 and the start of the data. If no names are passed, 
 i.e., header=None, then,  it will display the first column as 0, 
 the second as 1, and so on.
usecols: It is used to retrieve only selected columns from the CSV file.
names: List of column names to use. If the file contains a header row,
 then you should explicitly pass header=0 to override the column names.
 Duplicates in this list are not allowed.
nrows: It means a number of rows to be displayed from the dataset.
index_col: If None, there are no index numbers displayed along with records.  
skiprows: Skips passed rows in the new data frame.
'''
file_path="C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week4\\Loan_Prediction.csv"
df = pd.read_csv(file_path)
df = pd.read_csv(file_path, skiprows=1)
## skiprows and header are kind of same
df = pd.read_csv(file_path, header=1) 
df = pd.read_csv(file_path, header=None, skiprows=1,
                 names = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
                        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'])
df = pd.read_csv(file_path,  nrows=2)
df = pd.read_csv(file_path, na_values=["n.a.", "not available"])


df = pd.read_csv(file_path, usecols=['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome'])
df = pd.read_csv(file_path, index_col=0, skiprows=range(1, 20, 2))
df = pd.read_csv(file_path,index_col=0, usecols=[0, 1, 3])

df = pd.read_csv('data.csv.zip', index_col=0,   parse_dates=['IND_DAY'])
dtypes = {'POP': 'float32', 'AREA': 'float32', 'GDP': 'float32'}
df = pd.read_csv('data.csv', index_col=0, dtype=dtypes,
                 parse_dates=['IND_DAY'])

'''
Use Chunks to Iterate Through Files
Another way to deal with very large datasets is to split the 
data into smaller chunks and process one chunk at a time.
 If you use read_csv(), read_json() or read_sql(), 
 then you can specify the optional parameter chunksize
'''

for df_chunk in pd.read_csv(file_path, index_col=0, chunksize=300):
     print(df_chunk, end='\n\n')
     print('memory:', df_chunk.memory_usage().sum(), 'bytes', end='\n\n\n')
     
df.to_csv('data.csv.zip')
df.to_csv("new.csv", index=False)
df.to_csv("new.csv",header=False)
 
df.to_csv('formatted-data.csv', date_format='%B %d, %Y')
'''
 The format of the dates is different now. The format '%B %d, %Y' means the date will first display the full name of the month, then the day followed by a comma, and finally the full year.

There are several other optional parameters that you can use with .to_csv():

sep denotes a values separator.
decimal indicates a decimal separator.
encoding sets the file encoding.
header specifies whether you want to write column labels in the file.
 '''
 
df = pd.read_excel("stock_data.xlsx","Sheet1")


df.to_excel('C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week4\\data-shifted.xlsx', sheet_name='COUNTRIES',
            startrow=2, startcol=4)

df.to_excel("new.xlsx", sheet_name="stocks", index=False, startrow=2, startcol=1)

#In case of multiple sheets
xls_file = pd.ExcelFile("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week4\\Loan_Prediction.xlsx")
#extracting all sheet names
sheets=xls_file.sheet_names
#read each sheet one by one and do processing and combine in one
full_df= pd.DataFrame()
for name in sheets:
    if name=='_com.sap.ip.bi.xl.hiddensheet':
        continue
    df=xls_file.parse(name)
    full_df= full_df.append(df)

df_stocks = pd.DataFrame({
    'tickers': ['GOOGLE', 'WMT', 'MSFT'],
    'price': [845, 65, 64 ],
    'pe': [30.37, 14.26, 30.97],
    'eps': [27.82, 4.61, 2.12]
})

df_weather =  pd.DataFrame({
    'day': ['1/1/2017','1/2/2017','1/3/2017'],
    'temperature': [32,35,28],
    'event': ['Rain', 'Sunny', 'Snow']
})
with pd.ExcelWriter('C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week4\\stocks_weather.xlsx') as writer:
    df_stocks.to_excel(writer, sheet_name="stocks")
    df_weather.to_excel(writer, sheet_name="weather")
    
'''
JSON stands for JavaScript object notation. 
JSON files are plaintext files used for data interchange, 
and humans can read them easily and use the .json extension.
 Python and Pandas work well with JSON files, 
 as Python’s json library offers built-in support for them.
'''    
df = pd.read_json('data.json')
df.to_json('data-time.json')

'''
HTML Files
An HTML is a plaintext file that uses hypertext markup language 
to help browsers render web pages. 
The extensions for HTML files are .html and .htm. 
You’ll need to install an HTML parser library like lxml or html5lib to
 be able to work with HTML files:
'''
df = pd.read_html('data.html', index_col=0, parse_dates=['IND_DAY'])
df.to_html('data.html')

'''
Pickle Files
Pickling is the act of converting Python objects into byte streams.
 Unpickling is the inverse process. 
 Python pickle files are the binary files that keep the data and 
 hierarchy of Python objects. They usually have the extension .pickle or .pkl.
'''
import pickle
df.to_pickle('data.pickle')
df = pd.read_pickle('data.pickle')

# Its important to use binary mode
dbfile = open('examplePickle', 'ab')

# source, destination
pickle.dump(db, dbfile)                     
dbfile.close()

dbfile = open('examplePickle', 'rb')     
db = pickle.load(dbfile)

pickle.dump(model, open('model.pkl', 'wb'))
pickled_model = pickle.load(open('model.pkl', 'rb'))
pickled_model.predict(X_test)
#https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
'''
SQL Files
Pandas IO tools can also read and write databases.
 In this next example, you’ll write your data to a database called data.db. 
 To get started, you’ll need the SQLAlchemy package.
 You’ll also need the database driver. Python has a built-in driver for SQLite.
https://www.youtube.com/watch?v=M-4EpNdlSuY
https://github.com/codebasics/py/blob/master/pandas/21_sql/pandas_sql.ipynb
https://www.geeksforgeeks.org/working-with-database-using-pandas/
https://datatofish.com/sql-to-pandas-dataframe/
'''

import pandas as pd
  
from sqlalchemy import create_engine
engine = create_engine('sqlite:///data.db', echo=False)
df = pd.read_sql('data.db', con=engine, index_col='ID')
df.to_sql('data.db', con=engine, index_label='ID')

# create a connection
import sqlite3
con = sqlite3.connect('Diabetes.db')
  
# read data from SQL to pandas dataframe.
data = pd.read_sql_query('Select * from Diabetes;', con)
  
#Oracle Database
#https://cx-oracle.readthedocs.io/en/latest/user_guide/connection_handling.html
#https://cx-oracle.readthedocs.io/en/latest/user_guide/introduction.html
#https://www.geeksforgeeks.org/oracle-database-connection-in-python/
import cx_Oracle
cx_Oracle.init_oracle_client(lib_dir=r"C:\instantclient_19_8")
dsn = cx_Oracle.makedsn("dbhost.example.com", 1521, service_name="orclpdb1")
connection = cx_Oracle.connect(user="hr", password=userpwd, dsn=dsn,
                               encoding="UTF-8")
cursor = connection.cursor()
query = 'SELECT DISTINCT "RunId", "UploadDate" FROM ' + forecast_tablename +
 ' WHERE "ModelName"=\'Metalearner\' AND "KeyfigureCode" =' + plant

df_runid = pd.read_sql(query, con=connection)

# Data for binding
manager_id = 145
first_name = "Peter"

# Execute the query
sql = """SELECT first_name, last_name
         FROM employees
         WHERE manager_id = :mid AND first_name = :fn"""
cursor.execute(sql, mid=manager_id, fn=first_name)

# Loop over the result set
for row in cursor:
    print(row)

#Writing back to database
#establish databse connection

doc_post = doc_post[["RELEASEKEY", "FCID", "POSTPROID", "GCOMMENT", "KEEPFLAG", "UPLOADDATE",
                     "RUNID", "PLANT", "FC_LOG_FLAG"]]
colnames = [i for i in doc_post.columns]
colnames = str(colnames).replace('\'', '')
colnames = str(colnames).replace('[', '(')
colnames = str(colnames).replace(']', ')')

colindex = [':' + str(i) for i in list(range(1, len(doc_post.columns) + 1))]
colindex = str(colindex).replace('\'', '')
colindex = str(colindex).replace('[', '(')
colindex = str(colindex).replace(']', ')')
query = 'INSERT INTO ' + constants.POSTPROCESSING_TRACK_TABLE + ' ' +
 colnames + ' VALUES ' + colindex
rows = [tuple(x) for x in doc_post.values]
cursor.executemany(query, rows)
connection.commit()

'''
Getting data in to pandas from many different file formats or data sources is supported by read_* functions.

Exporting data out of pandas is provided by different to_*methods.
'''

 #Reference links
 https://realpython.com/pandas-read-write-files/