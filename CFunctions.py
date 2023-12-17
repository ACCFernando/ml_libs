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

def mae_calc(diff) -> float:
    
    return np.mean(np.abs(diff))

def mse_calc(diff) -> float:
    
    return np.mean(np.power(diff,2))

def ordering_check(index_pairs, value_pairs, i) -> bool:
    """
    Returns True if the given value_pairs of y predicted and y target are ordered in the same way
    """
    ind_a = index_pairs[i]
    ind_b = index_pairs[i+1]
    a = value_pairs[ind_a]
    b = value_pairs[ind_b]

    return (a[0] > a[1] and b[0] > b[1]) or (a[0] < a[1] and b[0] < b[1]) or (a[0] == a[1] and b[0] == b[1])

##################################
# Probability distribution metrics
##################################

def acum_probability_distribution_points(y_points,y) -> np.array:
    
    """
    y_points = target
    y = unique y_points    
    """

    return np.array([np.sum(y_points <= v) for v in y])/y_points.size

def ks_calc(y_acum1, y_acum2) -> float:
    """
    KS - Kolmogorov Smirnov
    Difference between 2 cumulative probability distributions
    Can be used with a theoretical distribution F(x), ex: KS = MAXx | F(x) - 1/n |
    0 -> no separation between distributions; 1 -> perfect separation
    """

    return np.max(np.abs(y_acum1,y_acum2))

def deviation_calc(vector) -> float:

    return np.std(vector)

def bin_probability_count(inf_bin:float, sup_bin:float, min_values:list, max_values:list, probs:np.array) -> float:
    """
    Calculate the cumulative probability of events within a specified bin range.

    This function computes the total probability of events falling within a given range (bin),
    defined by `inf_bin` and `sup_bin`. It accounts for partial overlaps of events with the bin range
    by proportionally adjusting the contribution of each event's probability.

    Parameters:
    inf_bin (float): The lower boundary of the bin range.
    sup_bin (float): The upper boundary of the bin range.
    min_values (array-like): An array of minimum values, each corresponding to the lower bound of an event's range.
    max_values (array-like): An array of maximum values, each corresponding to the upper bound of an event's range.
    probs (array-like): An array of probabilities, each associated with the corresponding event defined by `min_values` and `max_values`.

    Returns:
    float: The cumulative probability of events occurring within the specified bin range.

    The function iterates through each event, calculates the contribution of its probability
    based on the intersection with the bin range, and accumulates these contributions to compute
    the cumulative probability within the specified bin range.

    Example:
    >>> bin_probability_count(1.0, 2.0, [0.5, 1.5], [1.5, 2.5], [0.3, 0.4])
    0.35
    """
    prob_bin = sum(
        prob * (min(sup_bin, max_val) - max(inf_bin, min_val)) / (max_val - min_val)
        for min_val, max_val, prob in zip(min_values, max_values, probs)
        if max(inf_bin, min_val) < min(sup_bin, max_val)
    )
    return prob_bin

###############################
# Group and ordering operations
###############################

def unique_qt(vector) -> int:

    unique_qt = 0
    v = np.nan
    for u in np.sort(vector):
        if u != v:
            unique_qt = unique_qt + 1
            v = u
    
    return unique_qt



