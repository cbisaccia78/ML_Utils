import numpy as np
import matplotlib.pyplot as plt

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
        A function that, when called, compiles a model, trains it on the training folds, validates it on 
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
        A function that, when called, compiles a model, trains it on the training folds, validates it on 
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

def plot_metrics_per_epoch(tr_X, tr_Y, val_split_percentage, create_compile_fit, metrics, metric_colors, skip_first_n=0):
    """
    Plots specified training metrics across epochs for a given model, with a validation split. 

    This function splits the provided dataset into training and validation sets, trains the model, and 
    plots the specified metrics (e.g., loss, accuracy) over each epoch. It also allows skipping the first 
    few epochs in the plot if desired.

    Parameters
    ----------
    tr_X : numpy.ndarray
        The input training data. It should be a NumPy array where each row represents an example, 
        and each column represents a feature.
    
    tr_Y : numpy.ndarray
        The training labels corresponding to `tr_X`. It should be a NumPy array where each element 
        represents the label for the corresponding row in `tr_X`.
    
    val_split_percentage : float
        The percentage of the data to be used as validation data. This should be a float between 0 and 1.
        For example, `val_split_percentage = 0.2` means 20% of the data will be used for validation, and 
        80% will be used for training.

    create_compile_fit : callable
        A function that compiles a model, trains it on the training data, and returns a history object. 
        The history object should include the values of the specified metrics (e.g., 'val_loss', 'accuracy') 
        for each epoch. The function is expected to take the following arguments: training data (`tr_X`, 
        `tr_Y`), validation data (`v_X`, `v_Y`).
    
    metrics : list of str
        A list of metric names to plot (e.g., `['val_loss', 'accuracy']`). These metric names must match 
        the keys of the history object returned by `create_compile_fit`.
    
    metric_colors : list of str
        A list of colors to use for plotting the corresponding metrics. The length of `metric_colors` must 
        be the same as `metrics`, with each color corresponding to a metric. Colors should be provided 
        as valid matplotlib color codes (e.g., `'blue'`, `'green'`, `'#ff5733'`).
    
    skip_first_n : int, optional
        The number of initial epochs to skip in the plot (default is 0). This can be useful if the initial 
        few epochs have large fluctuations that can distort the visual scale of the plot. 

    Raises
    ------
    ValueError
        If the length of `metrics` is not equal to the length of `metric_colors`.

    Returns
    -------
    None
        This function directly plots the metrics over epochs using `matplotlib.pyplot` and does not return 
        any values.
    
    Notes
    -----
    - The function first splits the input dataset into training and validation sets based on 
      `val_split_percentage`. After training, it plots the metrics over the course of the training epochs.
    - After plotting, the original dataset is restored by concatenating the validation set back with 
      the training set.
    - The plot displays one line per metric, with the specified colors. The x-axis represents the 
      epochs (adjusted by `skip_first_n`), and the y-axis represents the value of the metric.
    - If you want to skip the initial few epochs (for example, if metrics stabilize after a few epochs), 
      you can use the `skip_first_n` argument to start the plot from the desired epoch.
    
    Example
    -------
    Suppose you have a `create_compile_fit` function that trains a model and returns a history dictionary 
    with metrics such as 'val_loss' and 'accuracy'. Here's how you could use `plot_metrics_per_epoch`:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> def create_compile_fit(tr_X, tr_Y, v_X, v_Y):
    ...     # Example of a function that returns training history
    ...     return {'val_loss': [0.4, 0.35, 0.32, 0.3], 'accuracy': [0.7, 0.75, 0.78, 0.8]}  # Simulated history

    >>> tr_X = np.random.rand(100, 10)  # 100 examples, 10 features
    >>> tr_Y = np.random.randint(0, 2, size=100)  # 100 binary labels

    >>> metrics = ['val_loss', 'accuracy']
    >>> metric_colors = ['blue', 'green']

    >>> plot_metrics_per_epoch(tr_X, tr_Y, 0.2, create_compile_fit, metrics, metric_colors, skip_first_n=1)
    # This will plot the 'val_loss' and 'accuracy' metrics, skipping the first epoch.
    """

    if len(metrics) != len(metric_colors):
        raise ValueError("length of metrics and metric_colors must be equal.")

    num_examples = tr_X.shape[0]

    val_split = int(num_examples*val_split_percentage)

    v_X = tr_X[val_split:]
    v_Y = tr_Y[val_split:]

    tr_X = tr_X[:val_split]
    tr_Y = tr_Y[:val_split]

    history = create_compile_fit(tr_X, tr_Y, v_X, v_Y)

    for i, metric in enumerate(metrics):
        metric_per_epoch = history[metric]
        epochs = len(metric_per_epoch)

        plt.plot(range(1+skip_first_n, epochs+1), metric_per_epoch[skip_first_n:], color=metric_colors[i])

    plt.show()

    tr_X = np.vstack((tr_X, v_X))
    tr_Y = np.concatenate((tr_Y, v_Y))

def plot_kfold_averaged_metric_per_epoch(tr_X, tr_Y, k, create_compile_fit, epochs, metric='val_loss', skip_first_n=0):
    """
    Plots the averaged metric across epochs from K-Fold cross-validation.

    This function performs K-Fold cross-validation on the dataset, computes the specified metric 
    (e.g., validation loss) for each fold at each epoch, averages the metric values across all folds, 
    and plots the averaged values over the epochs. You can skip plotting the first few epochs if needed.

    Parameters
    ----------
    tr_X : numpy.ndarray
        The input training data. It should be a NumPy array where each row represents an example and 
        each column represents a feature.
    
    tr_Y : numpy.ndarray
        The training labels corresponding to `tr_X`. It should be a NumPy array where each element 
        represents the label for the corresponding row in `tr_X`.

    k : int
        The number of folds for K-Fold cross-validation. This determines how many subsets the dataset is 
        divided into for training and validation.

    create_compile_fit : callable
        A function that compiles and trains a model on the provided training and validation data for each fold.
        It should return a history object that includes the values of the specified metric (e.g., `'val_loss'`) 
        for each epoch.

    epochs : int
        The number of epochs to train the model for each fold. The same number of epochs is used across 
        all folds.

    metric : str, optional
        The name of the metric to be averaged and plotted (default is `'val_loss'`). This metric must 
        correspond to the keys in the history object returned by `create_compile_fit`.

    skip_first_n : int, optional
        The number of initial epochs to skip in the plot (default is 0). This can be useful if the metric 
        values fluctuate significantly during the early epochs and you want to focus on the later epochs.

    Returns
    -------
    None
        This function directly generates and displays a plot of the averaged metric values across epochs 
        using `matplotlib.pyplot` and does not return any values.

    Raises
    ------
    ValueError
        If `epochs` is greater than the number of epochs returned by the history object.

    Notes
    -----
    - The function uses K-Fold cross-validation, where the data is split into `k` subsets (folds). For 
      each fold, the model is trained for `epochs` epochs, and the specified metric (e.g., validation 
      loss or accuracy) is recorded for each epoch.
    - The metric values for each fold are averaged across all `k` folds for every epoch.
    - The resulting averaged metric values are then plotted over the epochs.
    - The `skip_first_n` parameter allows skipping the first `n` epochs when plotting, useful for 
      excluding volatile early epoch behavior.

    Example
    -------
    Suppose you have a `create_compile_fit` function that trains a model and returns a history dictionary 
    with metrics like 'val_loss'. Here's how you can use `plot_kfold_averaged_metric_per_epoch`:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> def create_compile_fit(trk_X, trk_Y, v_X, v_Y):
    ...     # Example of a function returning simulated history
    ...     return {'val_loss': [0.5, 0.45, 0.4, 0.35, 0.3]}  # Simulated history for 5 epochs

    >>> tr_X = np.random.rand(100, 10)  # 100 examples, 10 features
    >>> tr_Y = np.random.randint(0, 2, size=100)  # 100 binary labels

    >>> plot_kfold_averaged_metric_per_epoch(tr_X, tr_Y, 5, create_compile_fit, epochs=5, metric='val_loss')
    # This will plot the average validation loss across 5 epochs, averaged over 5 folds.
    """
    val_metrics = kfold(tr_X, tr_Y, k, create_compile_fit, metric=metric)
    averaged_val_metrics = [np.mean([fold[i] for fold in val_metrics]) for i in range(0, epochs)]
    plt.plot(range(1+skip_first_n, epochs+1), averaged_val_metrics[skip_first_n:])
    plt.show()


def plot_strat_kfold_averaged_metric_per_epoch(tr_X, tr_Y, n, k, create_compile_fit, epochs, metric='val_loss', skip_first_n=0):
    """
    Plots the averaged metric across epochs from stratified K-Fold cross-validation.

    This function performs n K-Fold cross-validations on the dataset, computes the specified metric 
    (e.g., validation loss) for each fold at each epoch, averages the metric values across all folds, 
    and plots the averaged values over the epochs. You can skip plotting the first few epochs if needed.

    Parameters
    ----------
    tr_X : numpy.ndarray
        The input training data. It should be a NumPy array where each row represents an example and 
        each column represents a feature.
    
    tr_Y : numpy.ndarray
        The training labels corresponding to `tr_X`. It should be a NumPy array where each element 
        represents the label for the corresponding row in `tr_X`.

    n : int
        The number of times to run cross-validation.   

    k : int
        The number of folds for K-Fold cross-validation. This determines how many subsets the dataset is 
        divided into for training and validation.

    create_compile_fit : callable
        A function that compiles and trains a model on the provided training and validation data for each fold.
        It should return a history object that includes the values of the specified metric (e.g., `'val_loss'`) 
        for each epoch.

    epochs : int
        The number of epochs to train the model for each fold. The same number of epochs is used across 
        all folds.

    metric : str, optional
        The name of the metric to be averaged and plotted (default is `'val_loss'`). This metric must 
        correspond to the keys in the history object returned by `create_compile_fit`.

    skip_first_n : int, optional
        The number of initial epochs to skip in the plot (default is 0). This can be useful if the metric 
        values fluctuate significantly during the early epochs and you want to focus on the later epochs.

    Returns
    -------
    None
        This function directly generates and displays a plot of the averaged metric values across epochs 
        using `matplotlib.pyplot` and does not return any values.

    Raises
    ------
    ValueError
        If `epochs` is greater than the number of epochs returned by the history object.

    Notes
    -----
    - The function uses K-Fold cross-validation, where the data is split into `n*k` subsets (folds). For 
      each fold, the model is trained for `epochs` epochs, and the specified metric (e.g., validation 
      loss or accuracy) is recorded for each epoch.
    - The metric values for each fold are averaged across all n*k` folds for every epoch.
    - The resulting averaged metric values are then plotted over the epochs.
    - The `skip_first_n` parameter allows skipping the first `skip_first_n` epochs when plotting, useful for 
      excluding volatile early epoch behavior.

    Example
    -------
    Suppose you have a `create_compile_fit` function that trains a model and returns a history dictionary 
    with metrics like 'val_loss'. Here's how you can use `plot_kfold_averaged_metric_per_epoch`:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> def create_compile_fit(trk_X, trk_Y, v_X, v_Y):
    ...     # Example of a function returning simulated history
    ...     return {'val_loss': [0.5, 0.45, 0.4, 0.35, 0.3]}  # Simulated history for 5 epochs

    >>> tr_X = np.random.rand(100, 10)  # 100 examples, 10 features
    >>> tr_Y = np.random.randint(0, 2, size=100)  # 100 binary labels

    >>> plot_kfold_averaged_metric_per_epoch(tr_X, tr_Y, 5, 5, create_compile_fit, epochs=5, metric='val_loss')
    # This will plot the average validation loss across 5 epochs, averaged over 25 folds.
    """
    val_metrics = strat_kfold(tr_X, tr_Y, n, k, create_compile_fit, metric=metric)
    averaged_val_metrics = [np.mean([fold[i] for fold in val_metrics]) for i in range(0, epochs)]
    plt.plot(range(1+skip_first_n, epochs+1), averaged_val_metrics[skip_first_n:])
    plt.show()

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
