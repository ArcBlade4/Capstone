import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import iirnotch, butter, lfilter, sosfilt, welch

# File loading
def load_data(file_path):
    """
    Load data from a CSV file.

    Takes:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# Split time and target columns from the data
def split_time_target(data):
    """
    Split the time and target columns from the data.

    Takes:
    data (pd.DataFrame): The input data.

    Returns:
    tuple: A tuple containing the data without time and target columns, the time column, and the target column.
    """
    if 'Time' in data.columns:
        time_column = data['Time']
        data = data.drop(columns=['Time'])
    else:
        time_column = None

    if 'Gesture' in data.columns:
        target_column = data['Gesture']
        data = data.drop(columns=['Gesture'])
    else:
        target_column = None

    return data, time_column, target_column
    

# Outlier removal
def remove_outliers(data, time, target):
    print(data.shape)

    data_copy = data.copy()
    numeric_cols = data_copy.select_dtypes(include=np.number).columns

    median = data_copy[numeric_cols].median()
    std = data_copy[numeric_cols].std()


    numeric_cols = data.select_dtypes(include=np.number).columns
    median = data[numeric_cols].median()
    std = data[numeric_cols].std()

    mask = ((data[numeric_cols] - median).abs() <= 3 * std).all(axis=1)

    print(f"Removed {(~mask).sum()} outliers ({(~mask).mean() * 100:.2f}%).")
    return data[mask], time[mask], target[mask]

# Zero centering
def zero_center(data):
    """
    Zero center the data by subtracting the mean.

    Takes:
    data (pd.DataFrame): The input data.

    Returns:
    pd.DataFrame: The zero-centered data.
    """
    centered_data = data - data.mean()
    return centered_data

# Noise reduction
def notch_filter(data, fs, freq=50.0, quality_factor=30.0):
    """
    Apply a notch filter to remove power line noise.

    Takes:
    data (pd.DataFrame): The input signal.
    fs (float): The sampling frequency of the signal.
    freq (float): The frequency to be removed (default is 50 Hz).
    quality_factor (float): The quality factor of the notch filter (default is 30).

    Returns:
    pd.DataFrame: The filtered signal.
    """
    if freq >= fs / 2:
        print(f"Skipping notch filter: {freq} Hz is above Nyquist ({fs/2} Hz).")
        return data
    b, a = iirnotch(freq, quality_factor, fs)
    filtered_data = lfilter(b, a, data, axis=0)
    return pd.DataFrame(filtered_data, columns=data.columns, index=data.index)


# Bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to the data.

    Takes:
    data (pd.DataFrame): The input signal.
    lowcut (float): The lower frequency bound of the filter.
    highcut (float): The upper frequency bound of the filter.
    fs (float): The sampling frequency of the signal.
    order (int): The order of the filter (default is 5).

    Returns:
    pd.DataFrame: The filtered signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered_data = sosfilt(sos, data)
    filtered_data = pd.DataFrame(filtered_data, columns=data.columns)
    return filtered_data




#scaling
def scale_data(data):
    """
    Scale the data using StandardScaler.

    Takes:
    data (pd.DataFrame): The input data.

    Returns:
    pd.DataFrame: The scaled data.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)

# pipline function
def preprocess_pipeline(data, fs, lowcut, highcut):
    """
    Preprocess the data through a series of steps: zero centering, filtering, outlier removal, and scaling.

    Takes:
    data (pd.DataFrame): The input data.
    fs (float): The sampling frequency of the signal.
    lowcut (float): The lower frequency bound for the bandpass filter.
    highcut (float): The upper frequency bound for the bandpass filter.
    """
    #split time and target columns from the data
    data, time, target = split_time_target(data)

    # Zero center the data
    data = zero_center(data)

    # Apply notch filter
    data = notch_filter(data, fs)

    # Apply bandpass filter
    data = bandpass_filter(data, lowcut, highcut, fs)

    # Remove outliers
    data, time, target = remove_outliers(data, time, target)

    #reset index of data, time and target
    data, time, target = (x.reset_index(drop=True) for x in (data, time, target))

    # Scale data
    data = scale_data(data)

    #re add target column to the preprocessed data
    data.insert(8, 'Gesture', target)

    #re add time column to the preprocessed data
    data.insert(0, 'Time', time)    

    return data



data = load_data('01.csv')


data = preprocess_pipeline(data, fs=200, lowcut=1, highcut=9)




print(data.head())



#save preprocessed data to csv
data.to_csv('preprocessed_data.csv', index=False)
