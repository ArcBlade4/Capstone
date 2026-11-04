"""
PREPRO.py - preprocessing for the sEMG gesture models.

Two functions do all the work:

    data = preprocess(...)        load every recording, remove outliers, clean the signal
                                  -> one table, with the Time column kept

    d = make_datasets(data, ...)  cut windows, split each recording along time,
                                  balance the training classes, normalise
                                  -> X_train, y_train, X_val, ... ready for PyTorch

Every number you might want to tune is an argument with a sensible default.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import iirnotch, butter, lfilter, filtfilt, sosfilt, sosfiltfilt


DEFAULT_CHANNELS = [f"V{i + 1}" for i in range(8)]
DEFAULT_GESTURES = {"fist": 1, "flexion": 2, "extension": 3, "radial": 4,
                    "ulnar": 5, "rest": 6, "table": 7}


# --------------------------------------------------------------------------------------
# 1. Cleaning
# --------------------------------------------------------------------------------------

def preprocess(data_folder="Data",
               pattern="*/*.txt",
               channels=DEFAULT_CHANNELS,
               time_col="Time",
               label_col="Gesture",
               sample_rate=60,
               resample=True,
               resample_method="interp",
               outlier_z=3.0,
               zero_centre=True,
               notch_hz=None,
               notch_q=30,
               low_hz=1,
               high_hz=8,
               filter_order=4,
               causal=True,
               per_file=True,
               verbose=True):
    """Load every recording and clean it. Returns one DataFrame for all files.

    Output columns: file, file_idx, Time, V1..V8, Gesture  (+ age_ms when resample=True)
    (rows in recording order, files one after another)

    Order of operations: load -> remove outliers -> resample onto an even time grid -> filter.

    Parameters
    ----------
    data_folder, pattern : where the recordings are, e.g. Data/01/01.txt
    channels             : signal columns to keep and clean
    time_col, label_col  : names of the time (ms) and gesture columns
    sample_rate          : readings per second per channel (Hz) after resampling. Used to design
                           the filters. Set it close to the measured rate printed below.
    resample             : True  = use the recorded Time: put every recording on an even grid of
                                   1000/sample_rate ms, so one row always means the same real
                                   time step (recommended; do the same to live data).
                           False = use rows as recorded and assume they are evenly spaced.
    resample_method      : "interp" = linear interpolation between the real readings
                           "last"   = copy the most recent real reading (no new values invented)
    outlier_z            : drop rows where any channel is more than this many standard
                           deviations from its mean. None = keep every row.
    zero_centre          : subtract each channel's mean.
    notch_hz, notch_q    : notch filter frequency (Hz) and sharpness. None or 0 = no notch.
    low_hz, high_hz      : band-pass edges (Hz). Set one to None for a high-pass or low-pass
                           only, or both to None for no band-pass.
    filter_order         : order of the Butterworth band-pass filter.
    causal               : True  = filters only look at the past (what a live system can do)
                           False = zero-phase filtering (looks both ways, offline only)
    per_file             : True  = statistics and filters are applied to each recording
                           separately, so nothing leaks across file boundaries.
                           False = everything is applied to all files joined together.
    """
    paths = sorted(Path(data_folder).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {Path(data_folder) / pattern}")

    # ---- load ------------------------------------------------------------------------
    tables = []
    for idx, path in enumerate(paths):
        t = pd.read_csv(path).dropna().reset_index(drop=True)      # drop an empty last row
        t.columns = [c.strip() for c in t.columns]                 # " Gesture" -> "Gesture"
        missing = [c for c in [time_col, label_col, *channels] if c not in t.columns]
        if missing:
            raise KeyError(f"{path}: missing columns {missing}; found {list(t.columns)}")
        t[label_col] = t[label_col].astype(str).str.strip()
        t.insert(0, "file", path.parent.name)                       # "01" ... "07"
        t.insert(1, "file_idx", idx)
        tables.append(t[["file", "file_idx", time_col, *channels, label_col]])
    data = pd.concat(tables, ignore_index=True)
    n_loaded = len(data)
    data[channels] = data[channels].astype(np.float64)
    timing = timing_report(data, time_col)                         # recorded timing, before any change

    # ---- outliers (on the real readings, before resampling) --------------------------
    if outlier_z is not None:
        keep = _within_z(data, channels, outlier_z, per_file)
        data = data.loc[keep].reset_index(drop=True)
    n_kept = len(data)

    # ---- even time grid --------------------------------------------------------------
    if resample:
        parts = []
        for f_idx, rec in data.groupby("file_idx", sort=True):
            r = _resample(rec, time_col, channels, label_col, 1000 / sample_rate, resample_method)
            r.insert(0, "file", rec["file"].iloc[0])
            r.insert(1, "file_idx", f_idx)
            parts.append(r)
        data = pd.concat(parts, ignore_index=True)

    # ---- filters ---------------------------------------------------------------------
    nyquist = sample_rate / 2
    notch = None
    if notch_hz:
        if not 0 < notch_hz < nyquist:
            raise ValueError(f"notch_hz={notch_hz} must be between 0 and {nyquist} Hz "
                             f"(half of sample_rate={sample_rate})")
        notch = iirnotch(notch_hz, Q=notch_q, fs=sample_rate)

    band = None
    if low_hz or high_hz:
        for name, f in (("low_hz", low_hz), ("high_hz", high_hz)):
            if f and not 0 < f < nyquist:
                raise ValueError(f"{name}={f} must be between 0 and {nyquist} Hz "
                                 f"(half of sample_rate={sample_rate})")
        if low_hz and high_hz:
            band = butter(filter_order, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
        elif low_hz:
            band = butter(filter_order, low_hz, btype="highpass", fs=sample_rate, output="sos")
        else:
            band = butter(filter_order, high_hz, btype="lowpass", fs=sample_rate, output="sos")

    def clean(x):
        if zero_centre:
            x = x - x.mean(axis=0)
        if notch is not None:
            x = lfilter(*notch, x, axis=0) if causal else filtfilt(*notch, x, axis=0)
        if band is not None:
            x = sosfilt(band, x, axis=0) if causal else sosfiltfilt(band, x, axis=0)
        return x

    if per_file:
        for _, rows in data.groupby("file_idx").groups.items():
            data.loc[rows, channels] = clean(data.loc[rows, channels].to_numpy())
    else:
        data[channels] = clean(data[channels].to_numpy())
    data[channels] = data[channels].astype(np.float32)

    if verbose:
        print(f"loaded {len(paths)} files, {n_loaded} rows")
        print("recorded timing per file (gaps between rows, ms):")
        print(timing.to_string(), "\n")
        if outlier_z is not None:
            print(f"outliers (|z| > {outlier_z}): removed {n_loaded - n_kept} rows, kept {n_kept}")
        if resample:
            print(f"resampled to an even {1000 / sample_rate:.1f} ms grid ({sample_rate} Hz, "
                  f"{resample_method}): {len(data)} rows")
        else:
            print(f"NOT resampled: rows are assumed to be evenly spaced at {sample_rate} Hz")
        print(f"filters at {sample_rate} Hz: notch {notch_hz or 'off'}, "
              f"band {low_hz or '-'}..{high_hz or '-'} Hz, {'causal' if causal else 'zero-phase'}, "
              f"{'per file' if per_file else 'all files together'}")
    return data


def _resample(rec, time_col, channels, label_col, slot_ms, method="interp"):
    """Put one recording on an even time grid of slot_ms.

    Channels are interpolated between the real readings ("interp") or copied from the most
    recent real reading ("last"). The label is always the most recent real reading's label.
    age_ms = how long ago the most recent real reading was (large values = a real gap).
    """
    rec = rec.sort_values(time_col).drop_duplicates(subset=time_col, keep="last")
    times = rec[time_col].to_numpy(dtype=float)
    grid = np.arange(times[0], times[-1] + 1e-9, slot_ms)
    last = np.searchsorted(times, grid, side="right") - 1          # most recent real reading
    out = pd.DataFrame({time_col: grid})
    for ch in channels:
        values = rec[ch].to_numpy(dtype=float)
        out[ch] = np.interp(grid, times, values) if method == "interp" else values[last]
    out[label_col] = rec[label_col].to_numpy()[last]
    out["age_ms"] = grid - times[last]
    return out


def timing_report(data, time_col="Time", pause_ms=100):
    """Per file, in plain terms: how fast and how evenly the readings arrived.

    rate (Hz)     : real readings per second, ignoring pauses (use this for SAMPLE_RATE)
    short <=10 ms : share of readings that came almost on top of the previous one (bursts)
    pauses        : gaps longer than pause_ms, and the longest one in seconds
    duplicates    : readings with exactly the same time stamp as the previous one
    """
    rows = []
    for f_idx, rec in data.groupby("file_idx", sort=True):
        gaps = rec[time_col].diff().dropna()
        normal = gaps[gaps <= pause_ms]
        rows.append({"file": rec["file"].iloc[0], "gesture": rec.iloc[0, -1], "rows": len(rec),
                     "length (s)": (rec[time_col].iloc[-1] - rec[time_col].iloc[0]) / 1000,
                     "rate (Hz)": 1000 / normal.mean() if normal.mean() > 0 else np.nan,
                     "short <=10 ms (%)": 100 * (gaps <= 10).mean(),
                     "pauses": int((gaps > pause_ms).sum()),
                     "longest pause (s)": gaps.max() / 1000,
                     "duplicates": int((gaps == 0).sum())})
    return pd.DataFrame(rows).set_index("file").round(1)


def _within_z(data, channels, z, per_file):
    """True for rows where every channel is within z standard deviations of its mean."""
    x = data[channels]
    if per_file:
        g = data.groupby("file_idx")[channels]
        mean, std = g.transform("mean"), g.transform("std")
    else:
        mean, std = x.mean(), x.std()
    return ((x - mean).abs() <= z * std).all(axis=1).to_numpy()


def estimate_rate(data, time_col="Time"):
    """Readings per second per channel, from the median gap in the time column (ms)."""
    gaps = data.groupby("file_idx")[time_col].diff().dropna()
    gaps = gaps[gaps > 0]
    return 1000 / gaps.median() if len(gaps) else float("nan")


def live_grid(times_ms, values, sample_rate, n_rows, method="interp"):
    """Real-time version of the resampling: use it on the live stream so it matches training.

    times_ms : 1-D array of the recent real time stamps (ms), oldest first
    values   : array [len(times_ms), n_channels] of the matching readings
    Returns the last n_rows rows on the same even grid as preprocess(resample=True),
    ending at the newest reading, plus age_ms for each row (how stale it is).
    Then apply the same filters (keep the filter state between calls, e.g. sosfilt with zi)
    and the same channel_mean / channel_std before passing the window to the model.
    """
    times_ms = np.asarray(times_ms, dtype=float)
    values = np.asarray(values, dtype=float)
    slot_ms = 1000 / sample_rate
    grid = times_ms[-1] - slot_ms * np.arange(n_rows)[::-1]
    if grid[0] < times_ms[0]:
        raise ValueError("not enough history in the buffer for this many rows")
    last = np.searchsorted(times_ms, grid, side="right") - 1
    if method == "interp":
        out = np.column_stack([np.interp(grid, times_ms, values[:, c]) for c in range(values.shape[1])])
    else:
        out = values[last]
    return out, grid - times_ms[last]


# --------------------------------------------------------------------------------------
# 2. Windows, split, balance, normalise
# --------------------------------------------------------------------------------------

def make_datasets(data,
                  window_size,
                  step,
                  channels=DEFAULT_CHANNELS,
                  gestures=DEFAULT_GESTURES,
                  time_col="Time",
                  label_col="Gesture",
                  val_frac=0.2,
                  test_frac=0.2,
                  max_gap_ms=None,
                  balance_train=True,
                  augment_factor=2,
                  noise_std=0.05,
                  data_seed=0,
                  verbose=True):
    """Cut windows, split every recording along time, balance training, normalise.

    Parameters
    ----------
    window_size        : samples per window
    step               : samples between the starts of neighbouring windows
    val_frac, test_frac: last (val_frac + test_frac) of each recording is held out:
                         first part train, then validation, then test
    max_gap_ms         : skip windows that contain a real gap in the recording longer than this
                         (uses age_ms after resampling, otherwise the Time differences).
                         None = no check.
    balance_train      : cap every training gesture at the median count
    augment_factor     : training-set multiplier. 2 = keep every training window and add one noisy
                         copy (doubles the data); 3 = two noisy copies; 1 = no augmentation.
                         Only the training set is augmented; val and test are never touched.
    noise_std          : standard deviation of the Gaussian noise added to the copies, in units of
                         each channel's training std (the data is already normalised by then, so
                         0.05 means noise at 5% of the signal's typical spread).
    data_seed          : fixed seed for the capping and the noise (keeps the data identical across
                         model seeds)

    Returns a dict with X_train, y_train, X_val, y_val, X_test, y_test, test_file,
    gesture_names, n_classes, channel_mean, channel_std, summary.
    """
    X, labels, file_of_window, local_start, global_start, file_len = [], [], [], [], [], {}
    offset = 0
    for f_idx, rec in data.groupby("file_idx", sort=True):
        sig = rec[channels].to_numpy(dtype=np.float32)
        lab = rec[label_col].to_numpy()
        n = len(rec)
        file_len[f_idx] = n
        stale = np.zeros(n, dtype=bool)                          # True = row sits in/after a real gap
        if max_gap_ms is not None:
            if "age_ms" in rec:
                stale = rec["age_ms"].to_numpy() > max_gap_ms
            else:
                stale[1:] = np.diff(rec[time_col].to_numpy()) > max_gap_ms
        for s in range(0, n - window_size + 1, step):
            e = s + window_size
            g = lab[s]
            if g not in gestures or np.any(lab[s:e] != g):       # not a gesture, or label changes
                continue
            if stale[s:e].any():
                continue                                         # window spans a real gap
            X.append(sig[s:e]); labels.append(g)
            file_of_window.append(f_idx); local_start.append(s); global_start.append(offset + s)
        offset += n
    if not X:
        raise ValueError("no windows were made - check window_size, step and the labels")

    X = np.stack(X)
    file_of_window = np.array(file_of_window)
    start = np.array(local_start)
    end = start + window_size
    global_start = np.array(global_start)
    file_names = data.groupby("file_idx")["file"].first()

    # labels 0 .. n_classes-1 with no gaps
    by_number = {num - 1: name for name, num in gestures.items()}
    y_raw = np.array([gestures[g] for g in labels]) - 1
    present = np.unique(y_raw)
    y = np.searchsorted(present, y_raw)
    gesture_names = [by_number[p] for p in present]

    # chronological split inside every recording; windows crossing a cut are dropped
    n_rec = np.array([file_len[f] for f in file_of_window])
    cut1 = (n_rec * (1 - val_frac - test_frac)).astype(int)
    cut2 = (n_rec * (1 - test_frac)).astype(int)
    is_train = end <= cut1
    is_val = (start >= cut1) & (end <= cut2)
    is_test = start >= cut2

    # leak check: no raw sample may be used by two splits
    used = np.zeros((3, offset), dtype=np.int8)
    for k, mask in enumerate((is_train, is_val, is_test)):
        for s in global_start[mask]:
            used[k, s:s + window_size] = 1
    shared = int((used.sum(axis=0) > 1).sum())
    assert shared == 0, f"split leak: {shared} raw samples are shared between splits"
    for name, ys in (("train", y[is_train]), ("val", y[is_val]), ("test", y[is_test])):
        missing = [gesture_names[c] for c in range(len(present)) if not (ys == c).any()]
        assert not missing, f"no {name} windows for: {missing}"

    summary = (pd.DataFrame({"file": file_names.loc[file_of_window].to_numpy(),
                             "gesture": np.array(gesture_names)[y],
                             "train": is_train, "val": is_val, "test": is_test})
               .groupby(["file", "gesture"])
               .agg(windows=("train", "size"), train=("train", "sum"),
                    val=("val", "sum"), test=("test", "sum")))
    summary["dropped"] = summary["windows"] - summary[["train", "val", "test"]].sum(axis=1)

    X_train, y_train = X[is_train], y[is_train]
    X_val, y_val = X[is_val], y[is_val]
    X_test, y_test = X[is_test], y[is_test]
    test_file = file_names.loc[file_of_window[is_test]].to_numpy()

    # cap every training gesture at the median count
    if balance_train:
        counts = np.bincount(y_train, minlength=len(present))
        cap = int(np.median(counts[counts > 0]))
        rng = np.random.default_rng(data_seed)
        keep = []
        for c in range(len(present)):
            idx = np.flatnonzero(y_train == c)
            if len(idx) > cap:
                idx = rng.choice(idx, cap, replace=False)
            keep.extend(idx)
        keep = np.sort(keep)
        before = counts
        X_train, y_train = X_train[keep], y_train[keep]

    # normalise each channel with training statistics only
    channel_mean = X_train.mean(axis=(0, 1))
    channel_std = X_train.std(axis=(0, 1)) + 1e-8
    X_train = ((X_train - channel_mean) / channel_std).astype(np.float32)
    X_val = ((X_val - channel_mean) / channel_std).astype(np.float32)
    X_test = ((X_test - channel_mean) / channel_std).astype(np.float32)

    # augment the training set only: originals + noisy copies (stats above come from the originals)
    n_train_original = len(X_train)
    if augment_factor > 1:
        noise_rng = np.random.default_rng(data_seed + 1)
        X_orig, y_orig = X_train, y_train
        X_parts, y_parts = [X_orig], [y_orig]
        for _ in range(augment_factor - 1):
            noise = noise_rng.normal(0.0, noise_std, X_orig.shape).astype(np.float32)
            X_parts.append(X_orig + noise)
            y_parts.append(y_orig)
        X_train = np.concatenate(X_parts)
        y_train = np.concatenate(y_parts)

    if verbose:
        print(summary, "\n")
        print(f"windows: {window_size} samples, step {step}; "
              f"train {len(X_train)}, val {len(X_val)}, test {len(X_test)}; "
              f"samples shared between splits: {shared}")
        if augment_factor > 1:
            print(f"augmented training set: {n_train_original} -> {len(X_train)} windows "
                  f"(x{augment_factor}, Gaussian noise std {noise_std} of channel std)")
        if balance_train:
            after = np.bincount(y_train[:n_train_original], minlength=len(present))
            print(f"capped each training gesture at {cap} windows:")
            for name, b, a in zip(gesture_names, before, after):
                print(f"  {name:12s} {b:5d} -> {a:5d}")

    return {"X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "test_file": test_file,
            "gesture_names": gesture_names, "n_classes": len(present),
            "channel_mean": channel_mean, "channel_std": channel_std,
            "summary": summary}