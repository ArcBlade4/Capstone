import numpy as np
import pandas as pd
from scipy.signal import iirnotch, butter, lfilter, sosfilt
from sklearn.preprocessing import StandardScaler


# File loading
def load_data(file_path):
    """Load a CSV into a DataFrame (returns None on failure)."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# Split time and target columns from the data
def split_time_target(data):
    """
    Returns (signal_data, time_column_or_None, target_column_or_None).
    """
    time_column = None
    target_column = None

    if 'Time' in data.columns:
        time_column = data['Time']
        data = data.drop(columns=['Time'])

    if 'Gesture' in data.columns:
        target_column = data['Gesture']
        data = data.drop(columns=['Gesture'])

    return data, time_column, target_column


# Outlier removal
def remove_outliers(data, time, target):
    """Drop rows where any channel is more than 3 std from its median."""
    numeric_cols = data.select_dtypes(include=np.number).columns
    median = data[numeric_cols].median()
    std = data[numeric_cols].std()

    mask = ((data[numeric_cols] - median).abs() <= 3 * std).all(axis=1)

    print(f"Removed {(~mask).sum()} outliers ({(~mask).mean() * 100:.2f}%).")
    data = data[mask]
    time = time[mask] if time is not None else None
    target = target[mask] if target is not None else None
    return data, time, target


# Zero centering
def zero_center(data):
    """Subtract each channel's mean."""
    return data - data.mean()


# Noise reduction
def notch_filter(data, fs, freq=50.0, quality_factor=30.0):
    """
    Remove power line noise. Skipped (with a message) if freq is at/above Nyquist.
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
    Butterworth bandpass applied along time (axis 0) for every channel.
    """
    nyq = 0.5 * fs
    if not (0 < lowcut < highcut < nyq):
        raise ValueError(
            f"Bandpass needs 0 < lowcut < highcut < Nyquist ({nyq:g} Hz); "
            f"got {lowcut:g}-{highcut:g} Hz at fs={fs:g} Hz."
        )
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    # NOTE: axis=0 is time. The default (axis=-1) would filter across channels.
    filtered_data = sosfilt(sos, data, axis=0)
    return pd.DataFrame(filtered_data, columns=data.columns, index=data.index)


# Scaling
def scale_data(data):
    """Standardise each channel (zero mean, unit variance)."""
    scaled_data = StandardScaler().fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)


# Pipeline
def preprocess_pipeline(data, fs, lowcut, highcut, notch_freq=50.0):
    """
    zero-centre -> notch -> bandpass -> outlier removal -> scaling.
    Time / Gesture columns are split off first and re-attached at the end
    (only if they were present in the input).
    """
    data, time, target = split_time_target(data)

    data = zero_center(data)
    data = notch_filter(data, fs, notch_freq)
    data = bandpass_filter(data, lowcut, highcut, fs)
    data, time, target = remove_outliers(data, time, target)

    data = data.reset_index(drop=True)
    if time is not None:
        time = time.reset_index(drop=True)
    if target is not None:
        target = target.reset_index(drop=True)

    data = scale_data(data)

    if target is not None:
        data.insert(len(data.columns), 'Gesture', target)
    if time is not None:
        data.insert(0, 'Time', time)

    return data


if __name__ == "__main__":
    # Command-line use (the GUI imports this module and does not run this block)
    df = load_data('01.csv')
    df = preprocess_pipeline(df, fs=200, lowcut=1, highcut=9)
    print(df.head())
    df.to_csv('preprocessed_data.csv', index=False)
