#coding=utf-8
'''
Created on 2016年9月27日
@author: dengdan
'''
import numpy as np

float32 = 'float32'
floatX = float32
int32 = 'int32'
uint8 = 'uint8'
string = 'str'

def cast(obj, dtype):
    if isinstance(obj, list):
        return np.asarray(obj, dtype = floatX)
    return np.cast[dtype](obj)

def int(obj):
    return cast(obj, 'int')
    
def double(obj):
    return cast(obj, 'double')
    
def is_number(obj):
	try:
		obj + 1
	except:
		return False
	return True
    
def is_str(s):
    return type(s) == str

def is_list(s):
    return type(s) == list

def is_tuple(s):
    return type(s) == tuple
