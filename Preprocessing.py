"""
Preprocessing.py - single-file / live preprocessing that uses the SAME methods as PREPRO.py.

PREPRO.preprocess order, mirrored here:
    outliers (z-score from the MEAN, on the raw readings)
    -> resample onto an even time grid (uses the Time column)
    -> zero-centre -> notch (optional) -> Butterworth band-pass (order 4, causal)
PREPRO.make_datasets, mirrored here:
    windows only where the label is a known gesture and constant, optionally skipping real gaps
    channel mean/std come from the TRAINING part only (first 1 - val_frac - test_frac of the file)

Not mirrored (they need many recordings): class balancing and noise augmentation.
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.signal import (iirnotch, butter, lfilter, filtfilt, sosfilt, sosfiltfilt,
                          sosfilt_zi, tf2sos)

# Settings: these must match what the model was trained with (PREPRO defaults shown)
SAMPLE_RATE = 60                                # PREPRO: sample_rate=60
WINDOW_MS = 300
WINDOW_SIZE = SAMPLE_RATE * WINDOW_MS // 1000
STEP = 4
GESTURES = {"fist": 1, "flexion": 2, "extension": 3, "radial": 4, "ulnar": 5, "rest": 6, "table": 7}
GESTURE_NAMES = list(GESTURES.values())


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_data(file_path):
    """Load a CSV into a DataFrame (returns None on failure). Same cleanup as PREPRO."""
    try:
        t = pd.read_csv(file_path).dropna().reset_index(drop=True)   # drop an empty last row
        t.columns = [c.strip() for c in t.columns]                   # " Gesture" -> "Gesture"
        if "Gesture" in t.columns:
            t["Gesture"] = t["Gesture"].astype(str).str.strip()
        return t
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def split_time_target(data):
    """Returns (signal_data, time_column_or_None, target_column_or_None)."""
    time_column = None
    target_column = None
    if 'Time' in data.columns:
        time_column = data['Time']
        data = data.drop(columns=['Time'])
    if 'Gesture' in data.columns:
        target_column = data['Gesture'].astype(str).str.strip()
        data = data.drop(columns=['Gesture'])
    return data, time_column, target_column


# --------------------------------------------------------------------------------------
# Cleaning steps (PREPRO.preprocess)
# --------------------------------------------------------------------------------------
def remove_outliers(data, time, target, z=3.0):
    """Drop rows where any channel is more than z std from its MEAN (PREPRO._within_z)."""
    mean, std = data.mean(), data.std()
    mask = ((data - mean).abs() <= z * std).all(axis=1).to_numpy()
    print(f"Removed {(~mask).sum()} outliers ({(~mask).mean() * 100:.2f}%).")
    data = data[mask]
    time = time[mask] if time is not None else None
    target = target[mask] if target is not None else None
    return data, time, target


def resample_to_grid(data, time, target, fs, method="interp"):
    """
    Put the recording on an even grid of 1000/fs ms using the Time column (ms)
    (PREPRO._resample). Returns (data, time, target, age_ms).
    method "interp": linear interpolation; "last": copy the most recent real reading.
    The label is always the most recent real reading's label.
    age_ms = how long ago that real reading was (large = a real gap).
    """
    channels = list(data.columns)
    rec = data.copy()
    rec["__t"] = time.to_numpy(dtype=float)
    if target is not None:
        rec["__y"] = target.to_numpy()
    rec = rec.sort_values("__t", kind="stable").drop_duplicates(subset="__t", keep="last")

    times = rec["__t"].to_numpy()
    grid = np.arange(times[0], times[-1] + 1e-9, 1000.0 / fs)
    last = np.searchsorted(times, grid, side="right") - 1
    out = {}
    for ch in channels:
        values = rec[ch].to_numpy(dtype=float)
        out[ch] = np.interp(grid, times, values) if method == "interp" else values[last]
    new_data = pd.DataFrame(out, columns=channels)
    new_time = pd.Series(grid, name=time.name)
    new_target = pd.Series(rec["__y"].to_numpy()[last], name=target.name) if target is not None else None
    return new_data, new_time, new_target, grid - times[last]


def zero_center(data):
    """Subtract each channel's mean."""
    return data - data.mean()


def notch_filter(data, fs, freq=None, quality_factor=30.0, causal=True):
    """Notch filter (PREPRO: off by default). Skipped if freq is None/0 or at/above Nyquist."""
    if not freq:
        return data
    if freq >= fs / 2:
        print(f"Skipping notch filter: {freq} Hz is above Nyquist ({fs/2} Hz).")
        return data
    b, a = iirnotch(freq, quality_factor, fs)
    x = data.to_numpy(dtype=float)
    y = lfilter(b, a, x, axis=0) if causal else filtfilt(b, a, x, axis=0)
    return pd.DataFrame(y, columns=data.columns, index=data.index)


def _band_sos(fs, lowcut, highcut, order):
    """Butterworth band/high/low-pass as SOS (None if both edges are None)."""
    nyq = fs / 2
    for name, f in (("lowcut", lowcut), ("highcut", highcut)):
        if f and not 0 < f < nyq:
            raise ValueError(f"{name}={f} must be between 0 and {nyq:g} Hz (half of fs={fs:g})")
    if lowcut and highcut:
        if lowcut >= highcut:
            raise ValueError(f"lowcut ({lowcut}) must be below highcut ({highcut})")
        return butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    if lowcut:
        return butter(order, lowcut, btype="highpass", fs=fs, output="sos")
    if highcut:
        return butter(order, highcut, btype="lowpass", fs=fs, output="sos")
    return None


def bandpass_filter(data, lowcut, highcut, fs, order=4, causal=True):
    """Butterworth band-pass along time (axis 0). causal=True looks only at the past."""
    sos = _band_sos(fs, lowcut, highcut, order)
    if sos is None:
        return data
    x = data.to_numpy(dtype=float)
    y = sosfilt(sos, x, axis=0) if causal else sosfiltfilt(sos, x, axis=0)
    return pd.DataFrame(y, columns=data.columns, index=data.index)


# --------------------------------------------------------------------------------------
# Scaling: fixed per-channel mean/std from training data (PREPRO.make_datasets)
# --------------------------------------------------------------------------------------
class ChannelScaler:
    """(x - mean) / std per channel. Same job as PREPRO's channel_mean / channel_std."""

    def __init__(self, mean, std):
        self.mean_ = np.asarray(mean, dtype=float)
        self.scale_ = np.asarray(std, dtype=float)

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.mean_) / self.scale_


def save_scaler_if_missing(path, scaler, settings):
    """
    Pickle the channel mean/std (+ settings) to path, only if the file does not exist yet.
    Returns True if the file was written.
    """
    if os.path.exists(path):
        return False
    with open(path, 'wb') as f:
        pickle.dump({"mean": scaler.mean_, "std": scaler.scale_, "settings": dict(settings)}, f)
    return True


def load_scaler(path):
    """
    Load a scaler pickle. Returns (ChannelScaler, settings_dict).
    Raises ValueError with a readable message if it is missing or invalid.
    Also accepts the old format (a pickled StandardScaler). Only load pickles you created.
    """
    if not path or not os.path.exists(path):
        raise ValueError(f"Scaler file not found: {path or '(no path given)'}\n"
                         "Run the offline preprocessing once to create it.")
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Could not read scaler file: {e}")

    settings = {}
    if isinstance(obj, dict) and "mean" in obj and "std" in obj:
        scaler, settings = ChannelScaler(obj["mean"], obj["std"]), obj.get("settings", {})
    else:                                           # old format
        old = obj["scaler"] if isinstance(obj, dict) and "scaler" in obj else obj
        settings = obj.get("settings", {}) if isinstance(obj, dict) else {}
        if not (hasattr(old, "mean_") and hasattr(old, "scale_")):
            raise ValueError("That pickle does not contain channel mean/std.")
        scaler = ChannelScaler(old.mean_, old.scale_)
    return scaler, settings


# --------------------------------------------------------------------------------------
# Windowing (PREPRO.make_datasets, window cutting part)
# --------------------------------------------------------------------------------------
def make_windows(x, labels, window_size=WINDOW_SIZE, step=STEP, stale=None):
    """Slide a window along ONE recording.

    x       : signal, shape [samples, channels]
    labels  : gesture label (string) of every sample
    stale   : optional bool array, True = sample sits in/after a real gap; such windows are skipped

    Returns the windows, their labels, and each window's start position.
    """
    labels = np.asarray(labels)
    windows, window_labels, starts = [], [], []
    for start in range(0, len(x) - window_size + 1, step):
        end = start + window_size
        gesture = labels[start]
        if gesture not in GESTURES or np.any(labels[start:end] != gesture):
            continue                                # not a gesture, or the label changes
        if stale is not None and stale[start:end].any():
            continue                                # window spans a real gap
        windows.append(x[start:end])
        window_labels.append(gesture)
        starts.append(start)
    return windows, window_labels, starts


# --------------------------------------------------------------------------------------
# Pipeline (offline)
# --------------------------------------------------------------------------------------
def preprocess_pipeline(data, fs, lowcut, highcut, notch_freq=None, window=False,
                        resample=True, resample_method="interp", outlier_z=3.0,
                        order=4, causal=True, scale=True, scaler=None,
                        val_frac=0.2, test_frac=0.2, max_gap_ms=None,
                        window_size=WINDOW_SIZE, step=STEP):
    """
    outliers -> resample -> zero-centre -> notch -> bandpass -> scaling (-> windowing)
    Time / Gesture columns are split off first and re-attached at the end.

    resample    : needs a 'Time' column (ms); skipped (rows assumed evenly spaced) without one
    outlier_z   : None keeps every row
    scale       : standardise with a ChannelScaler. If `scaler` is None, mean/std are fitted on the
                  training part only: windows ending inside the first (1 - val_frac - test_frac)
                  of the recording (rows, if there are no labels). The fitted scaler is in df.attrs["scaler"].
    max_gap_ms  : skip windows containing a real gap longer than this (None = no check)

    Returns the DataFrame. With window=True returns (df, windows, window_labels, starts);
    the input must contain a 'Gesture' column. Windows are scaled the same way as df.
    """
    data, time, target = split_time_target(data)
    data = data.astype(float)
    if window and target is None:
        raise ValueError("window=True requires a 'Gesture' column in the input.")

    # 1. outliers on the raw readings
    if outlier_z is not None:
        data, time, target = remove_outliers(data, time, target, outlier_z)
    data = data.reset_index(drop=True)
    time = time.reset_index(drop=True) if time is not None else None
    target = target.reset_index(drop=True) if target is not None else None

    # 2. even time grid
    age = None
    if resample and time is not None:
        data, time, target, age = resample_to_grid(data, time, target, fs, resample_method)
    elif resample:
        print("No 'Time' column: not resampling, rows are assumed evenly spaced.")

    stale = None
    if max_gap_ms is not None:
        if age is not None:
            stale = age > max_gap_ms
        elif time is not None:
            stale = np.zeros(len(data), dtype=bool)
            stale[1:] = np.diff(time.to_numpy(dtype=float)) > max_gap_ms

    # 3. filters
    data = zero_center(data)
    data = notch_filter(data, fs, notch_freq, causal=causal)
    data = bandpass_filter(data, lowcut, highcut, fs, order=order, causal=causal)

    # 4. windows (unscaled) and scaling statistics from the training part only
    x = data.to_numpy()
    windows, window_labels, starts = [], [], []
    if target is not None and (window or (scale and scaler is None)):
        windows, window_labels, starts = make_windows(x, target.to_numpy(), window_size, step, stale)

    if scale:
        if scaler is None:
            cut1 = int(len(x) * (1 - val_frac - test_frac))
            train = [w for w, s in zip(windows, starts) if s + window_size <= cut1]
            if train:
                arr = np.stack(train)
            else:
                print("No training windows found: fitting the scaler on the training rows instead.")
                arr = x[:cut1] if cut1 > 1 else x
            scaler = ChannelScaler(arr.mean(axis=tuple(range(arr.ndim - 1))),
                                   arr.std(axis=tuple(range(arr.ndim - 1))) + 1e-8)
        data = pd.DataFrame(scaler.transform(x), columns=data.columns)
        windows = [scaler.transform(w).astype(np.float32) for w in windows]
    else:
        data = data.reset_index(drop=True)
    data.attrs["scaler"] = scaler

    if target is not None:
        data.insert(len(data.columns), 'Gesture', target.to_numpy())
    if time is not None:
        data.insert(0, 'Time', time.to_numpy())

    if window:
        return data, windows, window_labels, starts
    return data


# --------------------------------------------------------------------------------------
# Several recordings at once (PREPRO.preprocess + the statistics part of make_datasets)
# --------------------------------------------------------------------------------------
def preprocess_folder(paths, fs, lowcut, highcut, notch_freq=None, scaler=None, balance=True,
                      val_frac=0.2, test_frac=0.2, seed=0, window_size=WINDOW_SIZE, step=STEP,
                      max_gap_ms=None, progress=None, **kw):
    """
    Clean every file separately (outliers, resample, filters: nothing leaks across files), then
    standardise all of them with ONE scaler, like PREPRO: mean/std come from the training windows
    of all files (first 1 - val_frac - test_frac of each recording), with every gesture capped at
    the median window count when balance=True. A given `scaler` is used as is.

    paths    : list of CSV/TXT paths
    progress : optional function(str) called with status messages
    **kw     : passed to preprocess_pipeline (resample, resample_method, outlier_z, order, causal)

    Returns ([(path, DataFrame), ...], scaler).
    """
    cleaned, train_w, train_y, train_rows = [], [], [], []
    for i, path in enumerate(paths):
        if progress:
            progress(f"Cleaning file {i + 1}/{len(paths)}: {os.path.basename(path)}")
        raw = load_data(path)
        if raw is None or raw.empty:
            raise ValueError(f"Could not read any data from {path}")
        has_y = "Gesture" in raw.columns
        out = preprocess_pipeline(raw, fs, lowcut, highcut, notch_freq, window=has_y, scale=False,
                                  val_frac=val_frac, test_frac=test_frac, window_size=window_size,
                                  step=step, max_gap_ms=max_gap_ms, **kw)
        windows, labels, starts = [], [], []
        if has_y:
            df, windows, labels, starts = out
        else:
            df = out
        chs = [c for c in df.columns if c not in ("Time", "Gesture")]
        cut1 = int(len(df) * (1 - val_frac - test_frac))
        for w, y, st in zip(windows, labels, starts):
            if st + window_size <= cut1:                 # training part only
                train_w.append(w)
                train_y.append(y)
        train_rows.append(df[chs].to_numpy()[:cut1])
        cleaned.append((path, df, chs))

    if scaler is None:
        if progress:
            progress("Fitting the scaler on the training windows...")
        if train_w:
            idx = np.arange(len(train_y))
            if balance:
                y = np.asarray(train_y)
                labs, counts = np.unique(y, return_counts=True)
                cap = int(np.median(counts))
                rng = np.random.default_rng(seed)
                keep = []
                for lab in labs:
                    ii = np.flatnonzero(y == lab)
                    if len(ii) > cap:
                        ii = rng.choice(ii, cap, replace=False)
                    keep.extend(ii)
                idx = np.sort(keep)
            arr = np.stack([train_w[i] for i in idx])
        else:
            print("No training windows found: fitting the scaler on the training rows instead.")
            arr = np.vstack(train_rows)
        ax = tuple(range(arr.ndim - 1))
        scaler = ChannelScaler(arr.mean(axis=ax), arr.std(axis=ax) + 1e-8)

    results = []
    for path, df, chs in cleaned:
        df[chs] = scaler.transform(df[chs].to_numpy())
        df.attrs["scaler"] = scaler
        results.append((path, df))
    return results, scaler
