from numba import jit_module, prange
import numpy as np
import scipy.stats as norm

#################
# BASIC OPERATORS
#################

def vector_sum(vector):
    return np.sum(vector)

def vector_cumsum(vector):
    return np.cumsum(vector)

def vector_diff(vector1, vector2):
    return vector1 - vector2

def vector_division(vector1, vector2):
    return vector1 / vector2





