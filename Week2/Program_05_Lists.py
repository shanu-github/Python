# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 04:40:10 2022

@author: shanu
"""
#list
#List: All values have same meaning (homogenous)
#Suppose you want to go to grocery store, before going you are deciding on items to buy
#bread, butter, fruits, vegetables
item1= "bread"
item2= "butter"
item3= "fruits"
item4= "vegetables"

#end up creating many variables, is there a way to store in just one variable
items= ["bread","butter","fruits","vegetables"]
print(items)
#items will be stored in sequence of memory location

#We can access each item using python index
items[0]
items[3]

#what happens if outside of list index
items[5] #will throw an error

#to access range of items
items[0:3]

#If we want to know last item in items
items[-1]

#suppose you want to add extra item in list
items.append("biscuit")
items

#if you want to insert at specific index
items.insert(2, "chocolate")
items

#If you want to join two list
#your wife has given two list of items want to make one
food= ["bread","vegetables","egg"]
bathroom= ["shampoo","soap"]

items= food+bathroom

items

items+"chips" #throws an error can not add list with string

#while adding both should be list
items+["chips"]

# another way to join two lists
food.extend(bathroom)


#to count the number of items in list
len(items)

#to check the presence of particular item in list/ lookup
"chips" in items

"soda" in items

#to remove the specific value
items.remove("soda")

#lists are mutable, means we can change the value for specific index
daily_expense= [100, 300, 1000, 500, 200]
daily_expense[0]= 90
print(daily_expense)

daily_expense[0]= daily_expense[0]+90
print(daily_expense)

daily_expense.append(900)

#Copy a list
list2= daily_expense.copy()

myList = [10,20,30,40]
myList.append([50,60])
print(myList)
myList.extend([80,90])
print(myList)

min(myList)
max(myList)
sum(myList)

#List Traversal
for item in myList:
    print(item)

for i in range(len(myList)):
    print(myList[i])

#filtering in List
scores = [70, 60, 80, 90, 50]
filtered = filter(lambda x: x >= 70, scores)
print(list(filtered))

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
myList = [10,20,30,40, 31, 70, 60]

sorted_list= sorted(myList)
sorted_list[-2]