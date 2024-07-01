
import pandas as pd
import numpy as np

def load_data(filepath, column_name, num_samples):
    return pd.read_csv(filepath)[column_name][0:num_samples].values


def sliding_window_view(arr, window_size, step):
    """
    Generate a sliding window view of an input array.

    Args:
        arr (np.ndarray): The input array.
        window_size (int): The size of the sliding window.
        step (int): The step size for the sliding window.

    Returns:
        np.ndarray: The sliding window view of the input array.
    """
    shape = ((arr.shape[0] - window_size) // step + 1, window_size)
    strides = (step * arr.strides[0], arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def prepare_datasets_multi_class(data, window_size, delay, feature_func2, train_samples):
    """
    Prepare datasets for multi-class classification.

    Args:
        data (List[np.ndarray]): List of time series data.
        window_size (int): Size of the sliding window.
        delay (int): Step size for the sliding window.
        feature_func2 (Callable[[np.ndarray], np.ndarray]): Function to extract features from time series.
        train_samples (int): Number of samples to use for training.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Training features and labels.
    """
    train_features = []
    train_labels = []
    
    for i, series in enumerate(data):
        # use data starting from train_samples
        train_series = series[:train_samples]
        windows = sliding_window_view(train_series, window_size, delay)
        features = np.apply_along_axis(feature_func2, 1, windows)
        
        train_features.append(features)
        train_labels.extend([i] * len(features))
    
    X_train = np.vstack(train_features)
    y_train = np.array(train_labels)
    
    # Shuffle the training data
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    return X_train, y_train

