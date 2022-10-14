# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 05:56:55 2022

@author: shanu
"""
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay

df = pd.DataFrame({'date': ['3/10/2000', '3/11/2000', '3/12/2000'],
                   'value': [2, 3, 4]})
df['date'] = pd.to_datetime(df['date'])

df = pd.DataFrame({'date': ['2016-6-10 20:30:0', 
                            '2016-7-1 19:45:30', 
                            '2013-10-12 4:5:1'],
                   'value': [2, 3, 4]})
df['date'] = pd.to_datetime(df['date'], format="%Y-%d-%m %H:%M:%S")

df['date'] = pd.to_datetime(df['date'], errors='ignore')

df['date'] = pd.to_datetime(df['date'], errors='coerce')

df = pd.DataFrame({'name': ['Tom', 'Andy', 'Lucas'],
                 'DoB': ['08-05-1997', '04-28-1996', '12-16-1995']})
df['DoB'] = pd.to_datetime(df['DoB'])
df['year']= df['DoB'].dt.year
df['month']= df['DoB'].dt.month
df['day']= df['DoB'].dt.day

df['week_of_year'] = df['DoB'].dt.week
df['day_of_week'] = df['DoB'].dt.dayofweek
df['is_leap_year'] = df['DoB'].dt.is_leap_year

dw_mapping={
    0: 'Monday', 
    1: 'Tuesday', 
    2: 'Wednesday', 
    3: 'Thursday', 
    4: 'Friday',
    5: 'Saturday', 
    6: 'Sunday'
} 

df['day_of_week_name']=df['DoB'].dt.weekday.map(dw_mapping)

#Function to calculate 'Time to order' using First day of demand week and lead time 
def subtract_bu_days(date_val, day_val):
    date_val=date_val-BDay(day_val)
    return date_val



#Function to find difference b/w 'Time to order' and current date
def find_bu_days(order_date, current_date):
    #days_to_order= len(pd.date_range(order_date, current_date, freq=BDay()))-1
    days_to_order=np.busday_count( current_date,order_date)
    return days_to_order


#https://towardsdatascience.com/working-with-datetime-in-pandas-dataframe-663f7af6c587
#https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
#https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html