import numpy as np

def normalize(X, mean, std):
    """
    Normalizes the input array X by subtracting the mean and dividing by the standard deviation.

    Parameters
    ----------
    X : numpy.ndarray
        The input data to be normalized. It should be a NumPy array of any shape, typically used for 
        features in machine learning models. Each element in X is modified in place.
    
    mean : numpy.ndarray or float
        The mean value(s) to subtract from X. This can either be:
        - A scalar (float) representing the mean to subtract from all elements of X.
        - A NumPy array of the same shape as X, or broadcastable to the shape of X, where each
          element in X will have the corresponding mean subtracted.
    
    std : numpy.ndarray or float
        The standard deviation value(s) to divide X by. This can either be:
        - A scalar (float) representing the standard deviation for all elements of X.
        - A NumPy array of the same shape as X, or broadcastable to the shape of X, where each
          element in X will be divided by the corresponding standard deviation.

    Returns
    -------
    None
        This function modifies the input array `X` in place and does not return a value.

    Example
    -------
    >>> import numpy as np
    >>> X = np.array([1.0, 2.0, 3.0, 4.0])
    >>> mean = np.mean(X)
    >>> std = np.std(X)
    >>> normalize(X, mean, std)
    >>> print(X)
    array([-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079])

    """
    X -= mean
    X /= std

def vectorize_sequences(sequences, dimension=10000):
    """
    Converts a list of integer sequences into a 2D NumPy array of binary vectors.

    This function transforms each sequence of integers (e.g., word indices) into a fixed-size binary vector 
    representation. Each sequence corresponds to a row in the resulting 2D array, where the columns represent 
    the possible indices (up to the specified `dimension`). If an integer `j` appears in a sequence, the 
    corresponding position in the binary vector is set to 1, otherwise 0. 

    This is commonly used for preparing textual data for machine learning models (e.g., bag-of-words or 
    one-hot encoding of text).

    Parameters
    ----------
    sequences : list of lists of int
        A list of sequences, where each sequence is a list of integers. Each integer corresponds to a specific 
        word or token index.

    dimension : int, optional
        The size of the binary vector for each sequence (default is 10,000). This defines the number of 
        possible features or word indices to represent. All sequences will be converted to binary vectors 
        of this length.

    Returns
    -------
    results : numpy.ndarray
        A 2D NumPy array of shape `(len(sequences), dimension)`. Each row corresponds to a sequence, and 
        each column corresponds to a specific feature (or token index). If an index appears in the sequence, 
        the corresponding value in the array is 1, otherwise 0.

    Raises
    ------
    IndexError
        If any integer in a sequence is greater than or equal to `dimension`, as it will be out of bounds for 
        the binary vector.

    Notes
    -----
    - This function creates a binary vector for each sequence, where the presence of an integer `j` sets 
      the `j`-th position in the vector to 1.
    - It assumes that the integers in the sequences are non-negative and less than `dimension`. 
    - This type of encoding is useful in natural language processing tasks where input sequences are tokenized 
      as word or character indices, such as in sentiment analysis or text classification.

    Example
    -------
    Suppose you have a dataset of sequences representing tokenized sentences, where each token is mapped to an integer index.

    >>> sequences = [[1, 3, 5], [2, 3, 4, 5]]
    >>> vectorized = vectorize_sequences(sequences, dimension=6)
    >>> print(vectorized)
    [[0. 1. 0. 1. 0. 1.]
     [0. 0. 1. 1. 1. 1.]]

    In this example, each sequence is converted into a binary vector of length 6, where the presence of an 
    index in the sequence sets the corresponding position in the vector to 1.

    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1
    return results

def split_vector(vector, *percentages):
    """
    Splits a vector into multiple sub-vectors based on provided percentages.

    This function takes a 1D array or list `vector` and splits it into multiple sub-arrays or sub-lists. 
    The split points are defined by percentages provided as arguments. 

    WARNING: The percentages should be strictly increasing.

    Example
    -------
    Suppose you have a vector of length 10 and you want to split it into three parts: 
    the first 30%, the next 20%, and the last 50%.

    >>> vector = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> split_vector(vector, 0.3, 0.5)
    ([1, 2, 3], [4, 5], [6, 7, 8, 9, 10])

    Parameters
    ----------
    vector : list or 1D numpy.ndarray
        The vector (array or list) to be split. It is assumed to be a 1D structure.
    
    *percentages : float
        Variable number of percentage values that determine where the splits should occur. These values should 
        be in the range [0, 1] and represent the proportion of the vector to include in each segment. The percentages 
        should sum to less than or equal to 1. The remaining portion of the vector will form the last split.

    Returns
    -------
    tuple
        A tuple of sub-vectors (list or numpy.ndarray, depending on the input type). Each sub-vector is a portion 
        of the original `vector` as defined by the provided `percentages`.

    Raises
    ------
    ValueError
        - If any percentage results in an index that exceeds the length of the vector.
        - If percentages are not strictly increasing.

    Notes
    -----
    - The function divides the vector by calculating the corresponding indices for each percentage. 
      The first split contains the first `percentage * len(vector)` elements, the second split the next portion, 
      and so on, until the end of the vector is reached.
    - The percentage sequence should be increasing
    
    Edge Cases
    ----------
    1. If no percentages are provided, the entire vector is returned as a single element tuple.
    
    2. If the percentages do not sum to 1, the remaining part of the vector (i.e., `1 - sum(percentages)`) 
       is returned as the final split.
    
    3. If a percentage results in an index beyond the vector length, a `ValueError` is raised.
    """
    vector_length = len(vector)
    splits = []
    start = 0
    for boundary in percentages:
        index = int(boundary * vector_length)
        if index > vector_length:
            raise ValueError("split_vector encountered out of range index.")
        splits.append(vector[start:index])
        start = index
    
    splits.append(vector[start:])

    return tuple(splits)