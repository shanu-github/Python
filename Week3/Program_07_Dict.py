# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 05:16:06 2022

@author: shanu
"""
#Dictionary
#It allows values to store in Key Value Pairs
#Classical example: Telephone directory

#creating a Dictionary
phone_book= {"Shubham": 9965473450,"Manju":9988776547, "Rajesh":8897403456}
#Shubham :key, phone number: value
print(phone_book)

#To access the phone number, we use key
phone_book["Shubham"]
phone_book["Rajesh"]

phone_book.get("Shubham")

#To get all keys and values
phone_book.keys()
phone_book.values()

#Add new entry in dictionary
phone_book["Riya"]= 8876543900
print(phone_book)

#check if particular key present or not
"Shubham" in phone_book
"Sameer" in phone_book

#Deleting an entry
del phone_book["Manju"]
print(phone_book)

#delete all the entries frim dictionary
phone_book.clear()
print(phone_book)

#Traversing throgh dictionary
for key in phone_book:
    print(key,':',phone_book[key])
 
for key,value in phone_book.items():
    print(key,':',value)

'''
Write a function to convert a number 
entered by the user into its corresponding 
number in words. For example, if the 
input is 876 then the output should be 
‘Eight Seven Six’.
'''
for ch in str(789):
    print(ch)

def convert(num):
     #numberNames is a dictionary of digits and corresponding number 
     #names
     numberNames = {0:'Zero',1:'One',2:'Two',3:'Three',4:'Four',\
     5:'Five',6:'Six',7:'Seven',8:'Eight',9:'Nine'}
     
     result = ''
     for ch in num:
         key = int(ch) #converts character to integer
         value = numberNames[key]
         result = result + ' ' + value
     return result
num = '987' #number is stored as string
result = convert(num) 
print("The number is:",num)
print("The numberName is:",result)
      
'''
Write a Python program to create a dictionary from 
a string.
Note: Track the count of the letters from the string.
Sample string : 'w3resource'
Expected output : {'3': 1, 's': 1, 'r': 2, 'u': 1, 'w': 1, 'c': 1, 
'e': 2, 'o': 1}
'''
def char_count(my_string):
    count_dict= dict()
    for ch in my_string:
        if ch in count_dict.keys():
            count_dict[ch]=count_dict[ch]+1
        else:
            count_dict[ch]=  1
    return(count_dict)

