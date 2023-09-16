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

##################################
# Classification Metrics Operations
##################################

def area(x,y) -> float: 
    """
    Riemann Sum:
    Value of integral under continuous function over specific interval
    Subdivides te interval in smaller parts and approximates those parts by
    simple shapes as rectangles or triangles 
    """
    dx = np.diff(x)
    h = (y[:-1] + y[1:])/ 2
    
    return np.sum(h*dx)

def argmax_vector(vector) -> int:

    return np.argmax(vector)

def shannon_entropy(vector_p1) -> np.array:
    """
    Quantifies the amount of uncertainty in and associated with a random value.
    Measures the average amount of information contained in an event or set of events
    Bigger entropy = greater uncertainty; max on equal probability vector
    log2 -> measurement in bits or iformation units 
    """
    entropy = []
    for p1 in vector_p1:
        p0 = 1 - p1
        if p0 == 0 or p1 == 0:
            entropy.append(0)
        else:
            entropy.append(-p0*np.log2(p0) - p1*np.log2(p1))
    return np.array(entropy)

def information_gain_curve(partial_entropy,
                           accum_quant,
                           partial_entropy_c, 
                           accum_quant_c, 
                           total_quant,
                           initial_entropy) -> np.array:
    """"
    Measure how much a particular feature reduces uncertainty (entropy)
    """
    entropy = (partial_entropy*accum_quant + partial_entropy_c*accum_quant_c)/total_quant
    entropy = np.append(entropy, initial_entropy)

    return (initial_entropy - entropy)/initial_entropy

def partial_entropy_vector(entropy_vector,
                           entropy_aux,
                           partial_entropy_c,
                           accum_quant_c,
                           partial_entropy_r,
                           remain_quant,
                           total_quant
                           ) -> np.array:
    entropy = entropy_aux + (partial_entropy_c*accum_quant_c + partial_entropy_r*remain_quant)/total_quant

    return np.append(entropy_vector, entropy)

def entropy_vector_normalizer(entropy_vector, initial_entropy) -> np.array:

    return (initial_entropy - entropy_vector)/initial_entropy

def logloss(y, y_h) -> float:

    return -1*np.mean(np.where(y==1, np.log(y_h), np.log(1-y_h)))

def mean_calc(y) -> float:
    return np.mean(y)

##############################
# Regression Metrics Operations
##############################

def mae_calc(diff):
    return np.mean(np.abs(diff))

def mse_calc(diff):
    return np.mean(np.power(diff,2))

def ordering_chec(index_pair, value_pair, i):
    ind_a = index_pair[i]
    ind_b = index_pair[i+1]
    a = value_pair[ind_a]
    b = value_pair[ind_b]

    return (a[0] > a[1] and b[0] > b[1]) or (a[0] < a[1] and b[0] < b[1]) or (a[0] == a[1] and b[0] == b[1])




