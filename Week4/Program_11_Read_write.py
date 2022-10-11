# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 05:57:31 2022

@author: shanu
"""
'''
Syntax: pd.read_csv(filepath_or_buffer, sep=’ ,’ , header=’infer’,  index_col=None, usecols=None, engine=None, skiprows=None, nrows=None) 

Parameters: 

filepath_or_buffer: It is the location of the file which is to be retrieved using this function. It accepts any string path or URL of the file.
sep: It stands for separator, default is ‘, ‘ as in CSV(comma separated values).
header: It accepts int, a list of int, row numbers to use as the column names, and the start of the data. If no names are passed, i.e., header=None, then,  it will display the first column as 0, the second as 1, and so on.
usecols: It is used to retrieve only selected columns from the CSV file.
nrows: It means a number of rows to be displayed from the dataset.
index_col: If None, there are no index numbers displayed along with records.  
skiprows: Skips passed rows in the new data frame.
'''
df = pd.read_csv("stock_data.csv")
df = pd.read_csv("stock_data.csv", skiprows=1)
df = pd.read_csv("stock_data.csv", header=1) # skiprows and header are kind of same
df = pd.read_csv("stock_data.csv", header=None, names = ["ticker","eps","revenue","people"])
df = pd.read_csv("stock_data.csv",  nrows=2)
df = pd.read_csv("stock_data.csv", na_values=["n.a.", "not available"])
df = pd.read_csv("stock_data.csv",  na_values={
        'eps': ['not available'],
        'revenue': [-1],
        'people': ['not available','n.a.']
    })
df
df = pd.read_csv('data.csv', usecols=['COUNTRY', 'AREA'])
df = pd.read_csv('data.csv', index_col=0, skiprows=range(1, 20, 2))
df = pd.read_csv('data.csv',index_col=0, usecols=[0, 1, 3])

df = pd.read_csv('data.csv.zip', index_col=0,
...                  parse_dates=['IND_DAY'])
dtypes = {'POP': 'float32', 'AREA': 'float32', 'GDP': 'float32'}
df = pd.read_csv('data.csv', index_col=0, dtype=dtypes,
...                  parse_dates=['IND_DAY'])

data_chunk = pd.read_csv('data.csv', index_col=0, chunksize=8)
for df_chunk in pd.read_csv('data.csv', index_col=0, chunksize=8):
     print(df_chunk, end='\n\n')
     print('memory:', df_chunk.memory_usage().sum(), 'bytes', end='\n\n\n')
     
 df.to_csv('data.csv.zip')
 df.to_csv("new.csv", index=False)
 df.to_csv("new.csv",header=False)
 
 
df = pd.read_excel("stock_data.xlsx","Sheet1")
df.to_excel("new.xlsx", sheet_name="stocks", index=False, startrow=2, startcol=1)
with pd.ExcelWriter('stocks_weather.xlsx') as writer:
    df_stocks.to_excel(writer, sheet_name="stocks")
    df_weather.to_excel(writer, sheet_name="weather")
 #Reference links
 https://realpython.com/pandas-read-write-files/