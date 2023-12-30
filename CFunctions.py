from numba import jit_module, prange
import numpy as np
import scipy.stats as norm
from typing import List, Tuple

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

def shannon_entropy(probabilities: List[float]) -> np.ndarray:
    """
    Calculate the Shannon entropy for a list of probabilities.

    Shannon entropy quantifies the amount of uncertainty or unpredictability 
    in a set of events, measured in bits. It computes the average information 
    content for each event. Higher entropy indicates greater uncertainty. 
    The function returns an array of entropy values for each probability in 
    the input list.

    Parameters:
    probabilities (List[float]): A list of probabilities, where each probability 
                                 is a number between 0 and 1.

    Returns:
    np.ndarray: An array containing the Shannon entropy values for each probability.
    """
    entropy = [-p * np.log2(p) - (1 - p) * np.log2(1 - p) if p != 0 and p != 1 else 0 for p in probabilities]
    return np.array(entropy)

def information_gain_curve(partial_entropy: List[float],
                           accum_quant: List[float],
                           partial_entropy_c: List[float], 
                           accum_quant_c: List[float], 
                           total_quant: float,
                           initial_entropy: float) -> np.ndarray:
    """
    Calculate the information gain curve for a given feature.

    Parameters:
    partial_entropy (List[float]): A list of entropies for the positive class.
    accum_quant (List[float]): A list of cumulative quantities for the positive class.
    partial_entropy_c (List[float]): A list of entropies for the negative class.
    accum_quant_c (List[float]): A list of cumulative quantities for the negative class.
    total_quant (float): The total quantity of elements.
    initial_entropy (float): The initial entropy before any split.

    Returns:
    np.ndarray: An array representing the normalized information gain curve.
    """
    entropy = (np.array(partial_entropy) * np.array(accum_quant) +
               np.array(partial_entropy_c) * np.array(accum_quant_c)) / total_quant

    entropy = np.append(entropy, initial_entropy)

    return (initial_entropy - entropy) / initial_entropy

def partial_entropy_vector(entropy_vector, entropy_aux, partial_entropy_c, accum_quant_c, partial_entropy_r, remain_quant, total_quant) -> np.array:
    """
    Calculate the partial entropy and append it to the given entropy vector.

    Args:
    - entropy_vector (np.array): The vector to which the partial entropy will be appended.
    - entropy_aux (float): An auxiliary value for entropy calculation.
    - partial_entropy_c (float): Partial entropy component related to accumulated quantity.
    - accum_quant_c (float): Accumulated quantity contributing to the partial entropy.
    - partial_entropy_r (float): Partial entropy component related to the remaining quantity.
    - remain_quant (float): Remaining quantity contributing to the partial entropy.
    - total_quant (float): Total quantity used for normalizing the entropy calculation.

    Returns:
    - np.array: The updated entropy vector with the newly calculated entropy appended.

    Raises:
    - ValueError: If total_quant is zero, to prevent division by zero.
    """
    if total_quant == 0:
        raise ValueError("total_quant must not be zero to avoid division by zero.")

    entropy = entropy_aux + (partial_entropy_c * accum_quant_c + partial_entropy_r * remain_quant) / total_quant
    return np.append(entropy_vector, entropy)


def entropy_vector_normalizer(entropy_vector, initial_entropy) -> np.array:

    return (initial_entropy - entropy_vector)/initial_entropy

def logloss(y, y_h) -> float:
    """
    Calculates the logarithmic loss between true labels and predicted probabilities.

    Args:
    - y (np.array): True binary labels (1 or 0).
    - y_h (np.array): Predicted probabilities, corresponding to the probability of the label being 1.

    Returns:
    - float: The logarithmic loss value.

    Raises:
    - ValueError: If any predicted probability is outside the range [0, 1].
    """
    if np.any((y_h < 0) | (y_h > 1)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    return -1 * np.mean(np.where(y == 1, np.log(y_h), np.log(1 - y_h)))


def mean_calc(y) -> float:
   
    return np.mean(y)

##############################
# Regression Metrics Operations
##############################

def mae_calc(diff) -> float:
    
    return np.mean(np.abs(diff))

def mse_calc(diff) -> float:
    
    return np.mean(np.power(diff,2))

def ordering_check(index_pairs: List[Tuple[int, int]], 
                   value_pairs: List[Tuple[float, float]], 
                   i: int) -> bool:
    """
    Check if two pairs of values are ordered in the same way.

    This function determines whether two pairs of values (a and b) are ordered
    similarly. Both pairs are considered as (predicted, target) values. The
    function returns True if both pairs are in the same order (both ascending,
    both descending, or both equal).

    Parameters:
    index_pairs (list of tuples): A list of index pairs.
    value_pairs (list of tuples): A list of value pairs (predicted, target).
    i (int): The index in index_pairs to check.

    Returns:
    bool: True if the pairs at index i and i+1 in value_pairs are ordered in the
          same way, False otherwise.
    """
    index_a, index_b = index_pairs[i]
    predicted_a, target_a = value_pairs[index_a]
    predicted_b, target_b = value_pairs[index_b]

    return ((predicted_a > target_a and predicted_b > target_b) or
            (predicted_a < target_a and predicted_b < target_b) or
            (predicted_a == target_a and predicted_b == target_b))


##################################
# Probability distribution metrics
##################################

def acum_probability_distribution_points(y_points: List[float], y: List[float]) -> List[float]:
    """
    Calculate the cumulative probability distribution for a given set of points.

    This function computes the cumulative distribution of the `y_points` array 
    with respect to the unique values in `y`. It returns an array representing 
    the cumulative proportion of `y_points` that are less than or equal to each 
    value in `y`.

    Parameters:
    y_points (List[float]): The array of data points for which the cumulative 
                            distribution is to be calculated.
    y (List[float]): An array of unique values from `y_points` that serve as 
                     reference points for the cumulative calculation.

    Returns:
    List[float]: An array of cumulative probabilities corresponding to each value 
                 in `y`.
    """
    return np.array([np.sum([point <= v for point in y_points]) for v in y]) / len(y_points)

def ks_calc(y_acum1: List[float], y_acum2: List[float]) -> float:
    """
    Calculate the Kolmogorov-Smirnov statistic for two cumulative distributions.

    The Kolmogorov-Smirnov statistic is a measure of the maximum deviation 
    between two cumulative probability distributions. It is commonly used 
    to test the hypothesis that two samples are drawn from the same distribution,
    or to compare a sample distribution with a theoretical distribution.

    The KS statistic ranges from 0 (indicating no separation between distributions)
    to 1 (indicating perfect separation).

    Parameters:
    y_acum1 (List[float]): The first cumulative probability distribution.
    y_acum2 (List[float]): The second cumulative probability distribution.

    Returns:
    float: The maximum absolute difference between the two cumulative 
           probability distributions.
    """
    y_acum1_array = np.array(y_acum1)
    y_acum2_array = np.array(y_acum2)
    return np.max(np.abs(y_acum1_array - y_acum2_array))


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
# Grouping and ordering operations
###############################

def unique_qt(vector) -> int:

    unique_qt = 0
    v = np.nan
    for u in np.sort(vector):
        if u != v:
            unique_qt = unique_qt + 1
            v = u
    
    return unique_qt

def unique_values_and_counts(vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Calculate unique values and their counts in a given numpy array.

    This function sorts the input array, computes the unique values in the array,
    and counts the number of occurrences of each unique value. It returns the
    unique values, their counts, and the total number of unique values.

    Parameters:
    vector (np.ndarray): A numpy array of numeric values.

    Returns:
    Tuple[np.ndarray, np.ndarray, int]: A tuple containing the array of unique values,
                                        the array of counts for each unique value,
                                        and the total number of unique values.
    """
    unique_values, counts = np.unique(vector, return_counts=True)
    num_unique_values = len(unique_values)
    return unique_values, counts, num_unique_values

import numpy as np
from typing import Tuple, List

def indices_and_counts(vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Calculate the indices, counts, and cumulative counts of unique values in a numpy array.

    This function sorts the array, determines the indices of the sorted array, and counts
    the occurrences of each unique value. It returns the indices of the sorted array, 
    cumulative counts up to each unique value, counts of each unique value, and the 
    total number of unique values.

    Parameters:
    vector (np.ndarray): A numpy array of numeric values.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, int]: A tuple containing the sorted indices,
                                                    the cumulative counts up to each unique value,
                                                    the counts of each unique value,
                                                    and the total number of unique values.
    """
    if len(vector) == 0:
        return np.array([]), np.array([]), np.array([]), 0

    inds_sorted = np.argsort(vector)
    vector_sorted, counts = np.unique(vector[inds_sorted], return_counts=True)
    cum_counts = np.cumsum(counts[:-1])
    num_unique_values = len(vector_sorted)

    return inds_sorted, cum_counts, counts, num_unique_values


def indices_unique_values_and_counts(vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Calculate the sorted indices, unique values, their counts, and cumulative counts in a numpy array.

    This function sorts the array, determines the unique values and their counts,
    and returns the sorted indices, the unique values, their cumulative counts 
    (up to each unique value), their individual counts, and the total number of 
    unique values.

    Parameters:
    vector (np.ndarray): A numpy array of numeric values.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]: A tuple containing 
    the sorted indices, the array of unique values, the cumulative counts up to 
    each unique value, the counts of each unique value, and the total number of 
    unique values.
    """
    if len(vector) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), 0

    inds_sorted = np.argsort(vector)
    unique_values, counts = np.unique(vector[inds_sorted], return_counts=True)
    cum_counts = np.cumsum(counts[:-1])
    num_unique_values = len(unique_values)

    return inds_sorted, unique_values, cum_counts, counts, num_unique_values

###################################################
#Conditiona Metrics (classification and regression)
###################################################

##### conditional entropy block #####
def calculate_initial_entropy(quantities1, non_zero_count) -> float:
    """
    Calculate the initial entropy.

    Args:
    quantities1 (numpy.array): Array of quantities for the first condition.
    non_zero_count (int): The count of non-zero elements.

    Returns:
    float: The initial entropy.
    """
    p_ini = np.sum(quantities1)/non_zero_count
    pc_ini = 1 - p_ini
    if(p_ini == 0 or pc_ini == 0):
        initial_entropy = 0
    else:
        initial_entropy = -p_ini*np.log2(p_ini) - pc_ini*np.log2(pc_ini)
    
    return initial_entropy

def calculate_partial_entropies(probabilities) -> np.array:
    """
    Calculate partial entropies.

    Args:
    probabilities (numpy.array): Array of probabilities.

    Returns:
    numpy.array: Array of calculated partial entropies.
    """
    entropies = []
    for p in probabilities:
        pc = 1 - p
        entropies.append(0 if p == 0 or pc == 0 else -p * np.log2(p) - pc * np.log2(pc))
    return np.array(entropies)

def calculate_relative_gain(ig, quantities, non_zero_count) -> float:
    """
    Calculate the relative gain.

    Args:
    ig (float): The information gain.
    quantities (numpy.array): Array of total quantities.
    non_zero_count (int): The count of non-zero elements.

    Returns:
    float: The relative gain.
    """
    fractions = quantities / non_zero_count
    entropy_division = -np.sum(fractions * np.log2(fractions))
    return ig / entropy_division if entropy_division != 0 else 0

def calculate_conditional_ig_rg(quantities1, quantities, probabilities1, non_zero_count) -> Tuple[float, float]:
    """
    Calculate the conditional information gain (IG) and relative gain (RG).

    Args:
    quantities1 (numpy.array): Array of quantities for the first condition.
    quantities (numpy.array): Array of total quantities.
    probabilities1 (numpy.array): Array of probabilities for the first condition.
    non_zero_count (int): The count of non-zero elements.

    Returns:
    Tuple[float, float]: A tuple containing the information gain (IG) and relative gain (RG).
    """
    initial_entropy = calculate_initial_entropy(quantities1, non_zero_count)
    partial_entropies = calculate_partial_entropies(probabilities1)
    entropy = np.sum(partial_entropies * quantities) / non_zero_count

    ig = (initial_entropy - entropy) / initial_entropy if initial_entropy != 0 else 0

    rg = calculate_relative_gain(ig, quantities, non_zero_count) if quantities.size > 1 else 0

    return ig, rg
##### /conditional entropy block #####

