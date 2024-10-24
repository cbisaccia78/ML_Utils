import numpy as np
import matplotlib.pyplot as plt

import validation

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
    val_metrics = validation.kfold(tr_X, tr_Y, k, create_compile_fit, metric=metric)
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
    val_metrics = validation.strat_kfold(tr_X, tr_Y, n, k, create_compile_fit, metric=metric)
    averaged_val_metrics = [np.mean([fold[i] for fold in val_metrics]) for i in range(0, epochs)]
    plt.plot(range(1+skip_first_n, epochs+1), averaged_val_metrics[skip_first_n:])
    plt.show()