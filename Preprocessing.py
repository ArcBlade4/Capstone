import numpy as np
import pandas as pd
from scipy.signal import iirnotch, butter, lfilter, sosfilt, sosfilt_zi, tf2sos
from sklearn.preprocessing import StandardScaler
import os
import pickle

# Windowing settings (adjust to your data)
SAMPLE_RATE = 40 
WINDOW_MS = 300                                 
WINDOW_SIZE = SAMPLE_RATE * WINDOW_MS // 1000  
STEP = 4                  
GESTURES = {"fist": 1, "flexion": 2, "extension": 3, "radial": 4, "ulnar": 5, "rest": 6, "table": 7}
GESTURE_NAMES = list(GESTURES.values())   

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
def scale_data(data, scaler=None):
    """
    Standardise each channel (zero mean, unit variance).
    Fits a new StandardScaler unless one is given. Returns (scaled_df, scaler).
    """
    values = data.to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(values)
    return pd.DataFrame(scaler.transform(values), columns=data.columns), scaler
 
 
# Scaler file (pickle) shared between the offline pipeline and the live preprocessor
def save_scaler_if_missing(path, scaler, settings):
    """
    Pickle {"scaler": scaler, "settings": settings} to path, but only if the file
    does not exist yet (an existing scaler is never overwritten).
    Returns True if the file was written.
    """
    if os.path.exists(path):
        return False
    with open(path, 'wb') as f:
        pickle.dump({"scaler": scaler, "settings": dict(settings)}, f)
    return True
 
 
def load_scaler(path):
    """
    Load a scaler pickle. Returns (scaler, settings_dict).
    Raises ValueError with a readable message if it is missing or invalid.
    Only load pickle files you created yourself.
    """
    if not path or not os.path.exists(path):
        raise ValueError(f"Scaler file not found: {path or '(no path given)'}\n"
                         "Run the offline preprocessing once to create it.")
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Could not read scaler file: {e}")
    if isinstance(obj, dict) and "scaler" in obj:
        scaler, settings = obj["scaler"], obj.get("settings", {})
    else:
        scaler, settings = obj, {}      # a bare pickled StandardScaler also works
    if not (hasattr(scaler, "mean_") and hasattr(scaler, "scale_")):
        raise ValueError("That pickle does not contain a fitted StandardScaler.")
    return scaler, settings




# Pipeline
def preprocess_pipeline(data, fs, lowcut, highcut, notch_freq=50.0, window=False):
    """
    zero-centre -> notch -> bandpass -> outlier removal -> scaling
    (-> windowing, if window=True).
    Time / Gesture columns are split off first and re-attached at the end
    (only if they were present in the input).

    Returns the preprocessed DataFrame. With window=True it returns
    (df, windows, window_labels, starts) instead, and the input must
    contain a 'Gesture' column.
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

    data, scaler = load_data(data)

    # Windowing uses the scaled signal and its labels (before they are re-attached)
    if window:
        if target is None:
            raise ValueError("window=True requires a 'Gesture' column in the input.")
        windows, window_labels, starts = make_windows(data.to_numpy(), target.to_numpy())

    if target is not None:
        data.insert(len(data.columns), 'Gesture', target)
    if time is not None:
        data.insert(0, 'Time', time)

    if window:
        return data, windows, window_labels, starts
    return data


# Windowing
def make_windows(x, labels):
    """Slide a window along the signal.

    x       : signal, shape [samples, 8] (all recordings stored end to end)
    labels  : gesture label of every sample, same length as x

    Returns the windows, their labels, and each window's start position.
    """
    windows, window_labels, starts = [], [], []    # results are collected here

    # Try a window at every STEP-th position. The last start is
    # len(x) - WINDOW_SIZE, so the final window still fits inside the signal.
    for start in range(0, len(x) - WINDOW_SIZE + 1, STEP):
        end = start + WINDOW_SIZE                  # one past the window's last sample
        chunk_labels = labels[start:end]           # the label of every sample in this window
        gesture = chunk_labels[0]                  # the label the window starts with

        # Rule 1: the window must start on a gesture we classify
        if gesture not in GESTURES:                # skip pauses (0) and unused gestures
            continue

        # Rule 2: every sample must have that same label. This also drops
        # windows that run from one file into the next, because neighbouring
        # files hold different gestures.
        if np.any(chunk_labels != gesture):        # skip windows that cross into another label
            continue

        # The window passed both rules: keep it
        windows.append(x[start:end])               # the signal itself, shape [WINDOW_SIZE, 8]
        window_labels.append(gesture)              # its gesture
        starts.append(start)                       # where it begins, needed later for the time-based split
    return windows, window_labels, starts


# Live / streaming version of the pipeline
class StreamingPreprocessor:
    """
    Real-time counterpart of preprocess_pipeline. Feed it samples as they
    arrive (any chunk size) and it returns the preprocessed samples.

    Differences from the offline pipeline (a live stream has no "whole file"):
      * notch + bandpass are the same filters, but their state is carried
        between calls, and started from the first sample to avoid a transient
      * zero-centring is dropped (the bandpass removes DC anyway)
      * outliers are clipped to +/- clip_sigma instead of removed, so the
        stream keeps its timing
      * scaling uses running statistics (cumulative at first, then an
        exponential moving average over ~stats_window_s seconds)
      * the first warmup_s seconds are output as 0 while the statistics settle

    Takes:
    fs: float - sampling rate in Hz
    lowcut, highcut: float - bandpass edges in Hz
    notch_freq: float - mains frequency; skipped if at/above Nyquist (or 0)
    n_channels: int - number of signal channels

    process(samples) takes an array of shape (n, n_channels) and returns the
    same shape. reset() clears all state (call it after a gap in the data).
    """

    def __init__(self, fs, lowcut, highcut, notch_freq=50.0, n_channels=8,
                 order=5, quality_factor=30.0, stats_window_s=10.0,
                 warmup_s=2.0, clip_sigma=3.0):
        nyq = fs / 2
        if not (0 < lowcut < highcut < nyq):
            raise ValueError(
                f"Bandpass needs 0 < lowcut < highcut < Nyquist ({nyq:g} Hz); "
                f"got {lowcut:g}-{highcut:g} Hz at fs={fs:g} Hz."
            )

        sections = []
        self.notch_active = bool(notch_freq) and 0 < notch_freq < nyq
        if self.notch_active:
            b, a = iirnotch(notch_freq, quality_factor, fs)
            sections.append(tf2sos(b, a))
        sections.append(butter(order, [lowcut / nyq, highcut / nyq],
                               btype='band', output='sos'))
        self.sos = np.vstack(sections)
        self._zi_unit = sosfilt_zi(self.sos)   # state for a unit step input

        self.n_channels = n_channels
        self.alpha = 1.0 - np.exp(-1.0 / (stats_window_s * fs))
        self.warmup = int(warmup_s * fs)
        self.clip_sigma = clip_sigma
        self.reset()

    def reset(self):
        self.zi = None
        self.count = 0
        self.mean = np.zeros(self.n_channels)
        self.var = np.zeros(self.n_channels)

    def process(self, samples):
        x = np.atleast_2d(np.asarray(samples, dtype=float))
        if x.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {x.shape[1]}.")
        if len(x) == 0:
            return x

        if self.zi is None:
            # start the filters as if the signal had always been at its first value
            self.zi = self._zi_unit[:, :, None] * x[0][None, None, :]   # (sections, 2, channels)
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)

        out = np.zeros_like(y)
        for k in range(len(y)):
            self.count += 1
            a = max(1.0 / self.count, self.alpha)

            if self.count > self.warmup:
                std = np.maximum(np.sqrt(self.var), 1e-8)
                z = np.clip((y[k] - self.mean) / std, -self.clip_sigma, self.clip_sigma)
                out[k] = z
                v = self.mean + z * std        # clipped value feeds the statistics
            else:
                v = y[k]

            d = v - self.mean
            self.mean = self.mean + a * d
            self.var = (1.0 - a) * (self.var + a * d * d)
        return out


if __name__ == "__main__":
    # Command-line use (the GUI imports this module and does not run this block)
    df = load_data('01.csv')
    print(df)
    df, windows, window_labels, starts = preprocess_pipeline(
        df, fs=200, lowcut=1, highcut=9, window=True)
    print(df.head())
    print(f"{len(windows)} windows of shape {windows[0].shape}" if windows else "No windows found.")
    df.to_csv('preprocessed_data.csv', index=False)