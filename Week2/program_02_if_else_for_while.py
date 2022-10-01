# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 18:26:03 2022

@author: shanu
"""
#print the positive difference of two numbers
num1= int(input("Enter first number"))
num2= int(input("Enter Second number"))
if num1 >num2 :
    diff= num1-num2
else:
    diff= num2-num1
print(diff)

#Program to print even numbers in a given 
#sequence using for loop

num_list= [5,7,3,4,10,8, 5]

for num in num_list:
    #print(num)
    if num%2 ==0:
        print(num)
    
#range example
for num in range(0,len(num_list),2):
    print("In position ", num ,"num_list value", num_list[num])

#program to print first 6 multiple of 9

for num in range(6):
    print((num+1)*9)
    
#Program to find the factors of a whole 
#number using while loop

number = 16
count = 1
while count <= number:
    if number % count ==0:
        print(count)
    count= count+1

#using for loop
for count in range(1,16):
    if number % count ==0:
        print(count)

        
#example of break
#Write a Python program to check if a given number is prime or not.
num = int(input("Enter the number to be checked: "))
flag = 0 #presume num is a prime number
if num > 1 :
    for i in range(2, int(num / 2)):
         if (num % i == 0):
             flag = 1 #num is a not prime number
             break #no need to check any further
    if flag == 1:
        print(num , "is not a prime number")
    else:
        print(num , "is a prime number")
else :
 print("Entered number is <= 1, execute again!")
 

#example of continue
#Prints values from 0 to 6 except 3
num = 0
for num in range(6):
     num = num + 1
     if num == 3:
         continue
     print('Num has value ' + str(num))
print('End of loop')

#Assignment
'''
Write a program to find the grade of a student when 
 grades are allocated as given in the table below. 
Percentage of Marks Grade
Above 90% A
80% to 90% B
70% to 80% C
60% to 70% D
Below 60% E
 Percentage of the marks obtained by the student is input 
to the program.

'''
'''
Write a program to calculate the factorial 
of a given number
Hint : 5!= 5*4*3*2*1
range( num, 0, -1) or while
'''

import Program_03

Program_03.