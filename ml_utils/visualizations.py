import numpy as np
import matplotlib.pyplot as plt

def plot_metrics_per_epoch(history, metrics, metric_colors, skip_first_n=0):
    """
    Plots specified training metrics across epochs for a given model, with a validation split. 

    This function splits the provided dataset into training and validation sets, trains the model, and 
    plots the specified metrics (e.g., loss, accuracy) over each epoch. It also allows skipping the first 
    few epochs in the plot if desired.

    Parameters
    ----------
    history : tf.keras.callbacks.History - result of a previously fitted model
    
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
    - The function plots the metrics over the course of the training epochs.
    - The plot displays one line per metric, with the specified colors. The x-axis represents the 
      epochs (adjusted by `skip_first_n`), and the y-axis represents the value of the metric.
    - If you want to skip the initial few epochs (for example, if metrics stabilize after a few epochs), 
      you can use the `skip_first_n` argument to start the plot from the desired epoch.
    
    Example
    -------

    ... 
    ... history = model.fit(...)

    >>> metrics = ['val_loss', 'accuracy']
    >>> metric_colors = ['blue', 'green']

    >>> plot_metrics_per_epoch(history, metrics, metric_colors, skip_first_n=1)
    # This will plot the 'val_loss' and 'accuracy' metrics, skipping the first epoch.
    """

    if len(metrics) != len(metric_colors):
        raise ValueError("length of metrics and metric_colors must be equal.")

    for i, metric in enumerate(metrics):
        metric_per_epoch = history[metric]
        epochs = len(metric_per_epoch)

        plt.plot(range(1+skip_first_n, epochs+1), metric_per_epoch[skip_first_n:], color=metric_colors[i])

    plt.show()

def plot_averaged_metric_per_epoch(val_metrics, epochs, skip_first_n=0):
    """
    Plots the averaged metric across epochs from K-Fold or strat-Kfold cross-validation.

    Given the specified metric (e.g., validation loss) for each fold at each epoch, this function
    averages the metric values across all folds, and plots the averaged values over the epochs. 
    You can skip plotting the first few epochs if needed.

    Parameters
    ----------
    val_metrics : list[list[nums]]
        The result of calling metrics.kfold()

    epochs : int
        The number of epochs to train the model for each fold. The same number of epochs is used across 
        all folds.

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

    ... val_metrics = metrics.kfold(...)

    >>> plot_kfold_averaged_metric_per_epoch(val_metrics, epochs=5, metric='val_loss')
    # This will plot the average validation loss across 5 epochs, averaged over 25 folds.
    """
    averaged_val_metrics = [np.mean([fold[i] for fold in val_metrics]) for i in range(0, epochs)]
    plt.plot(range(1+skip_first_n, epochs+1), averaged_val_metrics[skip_first_n:])
    plt.show()