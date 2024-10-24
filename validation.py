import numpy as np

def kfold(tr_X, tr_Y, k, create_compile_fit, metric='val_loss'):
    """
    Performs K-Fold cross-validation on a dataset and returns the validation metric for each epoch for each fold.

    This function splits the dataset into `k` subsets (folds) for cross-validation. In each iteration, 
    one fold is used as the validation set while the remaining `k-1` folds are used for training. The 
    model is trained and validated `k` times, and the specified validation metric (e.g., validation loss) 
    from each fold is collected.

    Parameters
    ----------
    tr_X : numpy.ndarray
        The input training data. It should be a NumPy array where each row is an example, and each column 
        is a feature. The dataset is split into `k` folds.
    
    tr_Y : numpy.ndarray
        The training labels corresponding to `tr_X`. It should be a NumPy array where each element 
        represents the label for the corresponding row in `tr_X`.
    
    k : int
        The number of folds for cross-validation. This determines how many subsets the dataset is divided 
        into for training and validation.

    create_compile_fit : callable
        A function that, when called, creates, compiles, and fits a model on the training folds, validates it on 
        the validation fold, and returns a history object that includes the specified validation metric 
        (e.g., `val_loss`). This function should accept four parameters: training data (`trk_X`), 
        training labels (`trk_Y`), validation data (`v_X`), and validation labels (`v_Y`).
    
    metric : str, optional
        The name of the validation metric to retrieve from the history object returned by 
        `create_compile_fit`. By default, this is set to `'val_loss'`, but it can be any other metric 
        (e.g., `'val_accuracy'`) depending on the model's history.

    Returns
    -------
    val_metrics : list of lists
        A list containing the value of the specified validation metric for each epoch on each fold. 
        The length of the list is `k`, and each element (list) corresponds to the values of the `metric` 
        (e.g., validation loss or validation accuracy) for one of the `k` folds.

    Notes
    -----
    - The function handles cases where the number of examples in `tr_X` is not perfectly divisible by `k`. 
      In such cases, the remainder examples are distributed evenly among the last folds.
    - In each fold, the function splits the data into training and validation sets, trains the model, 
      and retrieves the final value of the specified validation metric.
    - The `create_compile_fit` function should return a history object (e.g., a dictionary containing 
      training and validation losses/metrics) from which the specified `metric` is extracted.

    Example
    -------
    Suppose `create_compile_fit` is a function that compiles, trains, and validates a machine learning model, 
    and returns a history dictionary. Here's how to use the function:

    >>> import numpy as np
    >>> def create_compile_fit(trk_X, trk_Y, v_X, v_Y):
    ...     # Example of returning a history dictionary with 'val_loss'
    ...     return {'val_loss': [0.5, 0.4, 0.35, 0.3]}  # Simulated training history for example purposes

    >>> tr_X = np.random.rand(100, 10)  # 100 examples, 10 features
    >>> tr_Y = np.random.randint(0, 2, size=100)  # 100 binary labels

    >>> val_metrics = strat_kfold(tr_X, tr_Y, 10, 5, create_compile_fit, metric='val_loss')
    >>> print(val_metrics)
    [[0.5, 0.4, 0.35, 0.3], [0.5, 0.4, 0.35, 0.3], [0.5, 0.4, 0.35, 0.3], [0.5, 0.4, 0.35, 0.3],[0.5, 0.4, 0.35, 0.3]]  # Example output

    """

    num_examples = tr_X.shape[0]
    val_examples_per_fold = num_examples // k
    remainder = num_examples % k
    val_metrics = []
    val_start = 0
    val_end = val_examples_per_fold
    for i in range(k-remainder):
        v_X = tr_X[val_start:val_end]
        v_Y = tr_Y[val_start:val_end]

        trk_X = np.vstack((tr_X[:val_start], tr_X[val_end:]))
        trk_Y = np.concatenate((tr_Y[:val_start], tr_Y[val_end:]))

        history = create_compile_fit(trk_X, trk_Y, v_X, v_Y)

        val_metrics.append(history[metric])

        val_start = val_end
        val_end = val_end + val_examples_per_fold


    val_end = val_end + 1
    for i in range(k-remainder, k): # if we have extra examples (n//k is not an integer), just distribute them evenly amongst last folds
        v_X = tr_X[val_start:val_end]
        v_Y = tr_Y[val_start:val_end]

        trk_X = np.vstack((tr_X[:val_start], tr_X[val_end:]))
        trk_Y = np.concatenate((tr_Y[:val_start], tr_Y[val_end:]))

        history = create_compile_fit(trk_X, trk_Y, v_X, v_Y)

        val_metrics.append(history[metric])

        val_start = val_end
        val_end = val_end + val_examples_per_fold + 1
    
    return val_metrics

def strat_kfold(tr_X, tr_Y, n, k, create_compile_fit, metric='val_loss'):
    """
    Performs multiple (n) k-fold cross-validations on a dataset and returns n*k validation metrics
    for each fold.

    In each iteration, the data is shuffled before starting the folds

    Parameters
    ----------
    tr_X : numpy.ndarray
        The input training data. It should be a NumPy array where each row is an example, and each column 
        is a feature. The dataset is split into `k` folds.
    
    tr_Y : numpy.ndarray
        The training labels corresponding to `tr_X`. It should be a NumPy array where each element 
        represents the label for the corresponding row in `tr_X`.
    
    n : int
        The number of times to run cross-validation.   

    k : int
        The number of folds for cross-validation. This determines how many subsets the dataset is divided 
        into for training and validation.

    create_compile_fit : callable
        A function that, when called, creates, compiles, and fits a model on the training folds, validates it on 
        the validation fold, and returns a history object that includes the specified validation metric 
        (e.g., `val_loss`). This function should accept four parameters: training data (`trk_X`), 
        training labels (`trk_Y`), validation data (`v_X`), and validation labels (`v_Y`).
    
    metric : str, optional
        The name of the validation metric to retrieve from the history object returned by 
        `create_compile_fit`. By default, this is set to `'val_loss'`, but it can be any other metric 
        (e.g., `'val_accuracy'`) depending on the model's history.

    Returns
    -------
    val_metrics : list of lists
        A list containing the value of the specified validation metric for each epoch on each fold. 
        The length of the list is `k`, and each element (list) corresponds to the values of the `metric` 
        (e.g., validation loss or validation accuracy) for one of the `k` folds.

    Notes
    -----
    - The function handles cases where the number of examples in `tr_X` is not perfectly divisible by `k`. 
      In such cases, the remainder examples are distributed evenly among the last folds.
    - In each fold, the function splits the data into training and validation sets, trains the model, 
      and retrieves the final value of the specified validation metric.
    - The `create_compile_fit` function should return a history object (e.g., a dictionary containing 
      training and validation losses/metrics) from which the specified `metric` is extracted.

    Example
    -------
    Suppose `create_compile_fit` is a function that compiles, trains, and validates a machine learning model, 
    and returns a history dictionary. Here's how to use the function:

    >>> import numpy as np
    >>> def create_compile_fit(trk_X, trk_Y, v_X, v_Y):
    ...     # Example of returning a history dictionary with 'val_loss'
    ...     return {'val_loss': [0.5, 0.4, 0.35, 0.3]}  # Simulated training history for example purposes

    >>> tr_X = np.random.rand(100, 10)  # 100 examples, 10 features
    >>> tr_Y = np.random.randint(0, 2, size=100)  # 100 binary labels

    >>> val_metrics = strat_kfold(tr_X, tr_Y, 2, 2, create_compile_fit, metric='val_loss')
    >>> print(val_metrics)
    [[0.5, 0.4, 0.35, 0.3], [0.5, 0.4, 0.35, 0.3], [0.5, 0.4, 0.35, 0.3], [0.5, 0.4, 0.35, 0.3]]  # Example output

    """
    val_metrics = []
    num_examples = tr_X.shape[0]
    for i in range(n):
        perm_index = np.random.permutation(num_examples)
        tr_X = tr_X[perm_index]
        tr_Y = tr_Y[perm_index]
        val_metrics.extend(kfold(tr_X, tr_Y, k, create_compile_fit, metric))
    return val_metrics
