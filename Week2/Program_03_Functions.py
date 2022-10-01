# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:23:09 2022

@author: shanu
"""
'''
Write a program using a user defined 
function that displays sum of first n 
natural numbers, where n is passed as 
an argument.
'''
def sum_natural(n):
    sum= 0
    for num in range(1,n+1):
        sum= sum+num
    return(sum)
        
sum_natural(6)



'''
Write a program using a user defined 
function calcFact() to calculate and 
display the factorial of a number num 
passed as an argument.
6!
'''

'''
Write a program using a user defined 
function myMean() to calculate the mean 
of floating values stored in a list
'''
def myMean(num_list):
    sum=0
    count=0
    for num in num_list:
        sum= sum+num
        count= count+1
    list_mean= sum/count
    return(list_mean)

print(myMean([1,2,3,4,5]))


'''
XYZ store plans to give festival discount to its 
customers. The store management has decided to 
give discount on the following criteria:
Shopping Amount Discount Offered
>=500 and <1000 5%
>=1000 and <2000 8%
>=2000 10%
Create a program 
using user defined function that accepts the shopping 
amount as a parameter and calculates discount and net 
amount payable on the basis of the following conditions:
Net Payable Amount = Total Shopping Amount – 
Discount. 
'''
'''
Write a program that implements a user defined 
function that accepts Principal Amount, Rate, 
Time, Number of Times the interest is compounded 
to calculate and displays compound interest. 
(Hint: CI=P*(1+r/n)nt)
'''
'''
Write a program to check the divisibility of a 
number by 7 that is passed as a parameter to the 
user defined function.
'''
'''
Write a program that uses a user defined function 
that accepts name and gender (as M for Male,
F for Female) and prefixes Mr/Ms on the basis of 
the gender.
'''
#Inbuilt functions
sum([ 5,6,7])
max([6, 9, 5])
pow(4, 3)

#Inbuilt modules
import math

math.sqrt(8)
math.floor(7.8)

from math import ceil
math.ceil(8.9)

import random
random.random()


random.randint(7,20)


import statistics
statistics.mean([8,9,2,3])

#can call other python program 
import program_02
program_02.