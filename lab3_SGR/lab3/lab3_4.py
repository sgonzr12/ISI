#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:08:19 2023

@author: mines46
"""

import numpy as np
import timeit


setup = '''
import numpy as np
'''


def calculate_squaresA():
    return [i**2 for i in range(1, 10000+1)]
    
    
def calculate_squaresB():
    squares_np = np.zeros(10000)
    for i in range(10000):
        squares_np[i] = (i+1)**2
    return squares_np
    
def calculate_squaresC():
    return np.arange(1, 10000+1)**2
    

    
num_runs = 1000

time_elapsed_a = timeit.timeit("calculate_squaresA()", "from __main__ import calculate_squaresA", number = num_runs)
    
print(time_elapsed_a/num_runs)
    
time_elapsed_b = timeit.timeit("calculate_squaresB()", "from __main__ import calculate_squaresB", number = num_runs)
    
print(time_elapsed_b/num_runs)
    
time_elapsed_c = timeit.timeit("calculate_squaresC()", "from __main__ import calculate_squaresC", number = num_runs)
    
print(time_elapsed_c/num_runs)