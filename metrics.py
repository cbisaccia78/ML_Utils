import numpy as np

def coefficient_of_variation(tr_X, tr_Y, create_compile_fit, val_split_percentage, num_tests):
    """
    Computes the coefficient of variation (CV) of validation loss over multiple training iterations 
    with random data shuffling.

    The coefficient of variation is defined as the ratio of the standard deviation to the mean of 
    the validation loss across different runs. This function trains a model multiple times using 
    different data shuffles and calculates the CV to assess the consistency of the model's validation 
    performance.

    Parameters
    ----------
    tr_X : numpy.ndarray
        The input training data. This should be a NumPy array where each row represents an example, 
        and each column represents a feature. This data will be shuffled during the function execution.
    
    tr_Y : numpy.ndarray
        The training labels corresponding to `tr_X`. This should be a NumPy array where each element 
        is the label for the corresponding row in `tr_X`. It will be shuffled in tandem with `tr_X`.
    
    create_compile_fit : callable
        A function that, when called, compiles a model, fits it to the training data, and returns 
        a history object containing the validation loss (`history['val_loss']`). This function is 
        expected to run the entire model training and validation process, including any callbacks or 
        epochs defined within it.
    
    val_split_percentage : float
        The percentage of the data to be used as validation data. This should be a float between 0 and 1.
        For example, `val_split_percentage = 0.8` means 20% of the data will be used for validation, and 
        80% will be used for training in each test run.
    
    num_tests : int
        The number of times to repeat the training process with different random shuffles of the data. 
        The validation loss from each iteration will be used to compute the coefficient of variation.

    Returns
    -------
    coeff_of_variation : float
        The coefficient of variation (CV) of the validation loss across the `num_tests` iterations.
        CV is calculated as the ratio of the standard deviation of validation losses to the mean of 
        validation losses. A higher CV indicates greater variability in the model's validation 
        performance, while a lower CV suggests more consistent performance.

    Notes
    -----
    - In each iteration, the function splits the training data into a training set and a validation set 
      based on `val_split_percentage`.
    - The data is shuffled before each iteration to ensure random splits between training and validation 
      sets, which helps test the robustness of the model across different data arrangements.
    - After each test, the validation loss is collected and used to calculate the mean and standard deviation.
    - The coefficient of variation can be useful in assessing model stability, particularly in scenarios 
      where validation performance varies significantly depending on the split of training data.

    Example
    -------
    Suppose `create_compile_fit` is a function that compiles and trains a machine learning model, 
    and returns a history dictionary with `val_loss`. Here's how you could use the function:

    >>> import numpy as np
    >>> def create_compile_fit(tr_X, tr_Y, v_X, v_Y):
    ...     # Dummy example of returning a history dictionary with 'val_loss'
    ...     return {'val_loss': [0.4, 0.35, 0.32, 0.3]}  # Example of a typical training history

    >>> tr_X = np.random.rand(1000, 10)  # 1000 examples, 10 features
    >>> tr_Y = np.random.randint(0, 2, size=1000)  # 1000 binary labels

    >>> cv = coefficient_of_variation(tr_X, tr_Y, create_compile_fit, 0.2, 5)
    >>> print(cv)
    0.0456  # Example output

    """
    num_examples = tr_X.shape[0]
    val_split = int(val_split_percentage*num_examples)
    val_losses = []
    for i in range(num_tests):
        random_indicies = np.random.permutation(num_examples)
        tr_X = tr_X[random_indicies]
        tr_Y = tr_Y[random_indicies]

        v_X = tr_X[val_split:]
        v_Y = tr_Y[val_split:]

        tr_X = tr_X[:val_split]
        tr_Y = tr_Y[:val_split]

        history = create_compile_fit(tr_X, tr_Y, v_X, v_Y)

        val_losses.append(history['val_loss'][-1])

        tr_X = np.vstack((tr_X, v_X))
        tr_Y = np.concatenate((tr_Y, v_Y))

    mean_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)

    coeff_of_variation = std_val_loss / mean_val_loss

    return coeff_of_variation