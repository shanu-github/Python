# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 20:01:45 2022

@author: shanu
"""

string= "Hello World"
string[0]
string[5]
len(string)
string[len(string)]
string[len(string)-1]
string[-2]
string[-1]
string[2:5]
string[0]= 8
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






















def str_count(mystring):
    print( "Given string", mystring)
    print( "Total number of characters in string",len(mystring))
    print( "Total number of words in string", len(mystring.split(' ')))
    alpha_count=0
    digit_count=0
    ssymbol_count=0
    alphabets ='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for ch in mystring :
        if ch in alphabets:
            alpha_count= alpha_count+1
        elif ch in alphabets.lower():
            alpha_count=alpha_count+1
        elif ch in '0123456789':
            digit_count=digit_count+1
        else:
            ssymbol_count=ssymbol_count+1
    print( "Total number of alphabets in string",alpha_count)
    print( "Total number of digits in string", digit_count)
    print( "Total number of special symbol in string", ssymbol_count)
            
    
str_count('I started learning Python from 30-09-2022')
