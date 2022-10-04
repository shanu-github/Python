# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 20:01:45 2022

@author: shanu
"""
#---------------------------------------------------------------------------------
#String In Python: Used to store text information
#Strings are represented by quotes in python (single, double, tripple)
#python stores each character of string in different memory location index starts with 0

string= "Hello World"

#Single/ double quotation
text="hello"
text='text'
text= 'let's learn python' #throw error
#if single quote inside then use double quote outside viceversa
text= "let's learn python"

text = 'hello "world"'

#When multi line use triple quote
address= '''123, House No 209
Ashoka palace'''


string[0]
string[5]
len(string)
string[len(string)]
string[len(string)-1]
string[-2]
string[-1]
string[2:5]
string[0]= 8 #if you want to change specific character
#you will get error, as strings are immutable can not be changed partially, you can change whole

string[:5]
string*4
string[0:5]*4
#String slice with step
string[3::2]
string[::]
#String functions
len(string)
string.replace('o','8')
string.upper()
str.lower(string)
str.count(string,'l')
str.count(string,'lo')
str.partition(string, ' ') #provides three parts
str.split(string, 'World') #provide parts without spiltvalue

#copying string to another string
string2= string[:] or string[::]
print(id(string))
print(id(string2))

#------------------------------------------------------------
#Operations in String
#Concatination in strings
s1="Good"
s2="Morning"

s1+s2
#want to add space
s1+' '+s2

#want to concat number with string
s1+123  #wrong

#convert number to string first and then concat
str(123) #to know whether converted or not
s1+ str(123)

'''
Write a program with a user defined 
function with string as a parameter which 
replaces all vowels in the string with '*'
'''
def replace_vowel(string):
    for ch in string:
        if ch in 'aeiouAEIOU':
            string = string.replace(ch, '*')
    return string

print(replace_vowel('Hello'))


'''
Write a function to return the sum of digits present in this string
'''

'''
Write a program which reverses a string 
passed as parameter and stores the 
reversed string in a new string. Use a user 
defined function for reversing the string
'''

'''
Write a program to input line(s) of text to Count the total number of 
characters in the text (including white spaces),total 
number of alphabets, total number of digits, total 
number of special symbols and total number of 
words in the given text. (Assume that each word is 
separated by one space
'''

def count_char(string):
    print("Total number of characters present", len(string))
    print("Total number of words present", len(str.split(string," ")))
    alpha_str= 'abcdefghijklmnopqrstuvwxyz'
    count_alpha=0
    count_digit=0
    count_ssymbol=0
    for ch in string:
        if ch in alpha_str:
            count_alpha= count_alpha+1
        elif ch in alpha_str.upper():
            count_alpha= count_alpha+1
        elif ch in '0123456789':
            count_digit= count_digit+1
        else:
            count_ssymbol= count_ssymbol+1
    print("Total number of alphabets present", count_alpha)
    print("Total number of digits present", count_digit)
    print("Total number of special symbol present", count_ssymbol)


'''
Write a function deleteChar() which takes two 
parameters one is a string and other is a character. 
The function should create a new string after 
deleting all occurrences of the character from the 
string and return the new string
'''

'''
Write a function that takes a sentence as an input 
parameter where each word in the sentence is 
separated by a space. The function should replace 
each blank with a hyphen and then return the 
modified sentence.



