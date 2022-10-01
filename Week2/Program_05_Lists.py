# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 04:40:10 2022

@author: shanu
"""
myList = [10,20,30,40]
myList.append([50,60])
print(myList)
myList.extend([80,90])
print(myList)

'''
Program to increment the elements of a 
list. The list is passed as an argument to 
a function.
'''
def incre_list(list1, incre_value):
    for i in range(len(list1)):
        list1[i]= list1[i]+incre_value
    return(list1)

'''
Write a user-defined function to check if a 
number is present in the list or not. If the 
number is present, return the position of 
the number. Print an appropriate message 
if the number is not present in the list
'''
def presence_num(list1, num_value):
    flag=0
    for i in range(len(list1)):
        if list1[i]==num_value:
            flag=1
    if flag==0:
        print('number is not present')
    else:
        print("number is present at index ", i)

def presence_num(list1, numb1):
    flag=0
    for i in range(len(list1)) :
        if list1[i]==numb1:
            flag=1 
            print("Number is present in the list",i)    
    if flag==0:
            print("Number is not present in the list")




'''
The record of a student (Name, Roll No., 
Marks in five subjects and percentage of 
marks) is stored in the following list:
stRecord = ['Raman','A-36',[56,98,99,72,69], 78.8]
Write Python statements to retrieve the following 
information from the list stRecord.
a) Percentage of the student
b) Marks in the fifth subject
c) Maximum marks of the student
d) Roll no. of the student
e) Change the name of the student from 
‘Raman’ to ‘Raghav’
'''

'''
Write a function to return the second largest number 
from a list of numbers.
'''