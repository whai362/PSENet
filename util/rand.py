#coding=utf-8
'''
Created on 2016年9月27日

@author: dengdan
'''
import numpy as np
import time
import random

rng = np.random.RandomState(int(time.time()))

rand = np.random.rand
"""
Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)
"""


def normal(shape, mu = 0, sigma_square = 1):
    return rng.normal(mu, np.sqrt(sigma_square), shape)

def randint(low = 2 ** 30, high = None, shape = None):
    """
    low: the higher bound except when high is not None.
    high: when it is not none, low must be smaller than it
    shape: if not provided, a scalar will be returned
    """
    return rng.randint(low = low, high = high, size = shape)
    
def shuffle(lst):
    random.shuffle(lst)

def sample(lst, n):
    return random.sample(lst, n)


    