"""
classifier.py - load an ML model and classify preprocessed EMG data.

Supported model files:
    .keras / .h5          Keras / TensorFlow model
    .pt / .pth            PyTorch: TorchScript (torch.jit.save) or a whole pickled model (torch.save(model))
    .onnx                 ONNX model (needs onnxruntime)
    .pkl / .joblib        scikit-learn style model (predict / predict_proba). Only load files you made.

Data goes in as a DataFrame of PREPROCESSED rows (Time?, V1..V8, Gesture?), is cut into
windows exactly like Preprocessing.make_windows (WINDOW_SIZE samples, STEP apart), reshaped to
whatever the model expects, and every window gets a gesture name + confidence.

Assumption: a neural net outputs one probability per gesture in the order of
Preprocessing.GESTURES (fist=1 ... table=7), i.e. output index i -> gesture id i + 1.
"""
import os
import pickle

import numpy as np
import pandas as pd

import Preprocessing as pp

ID_TO_NAME = {v: k for k, v in pp.GESTURES.items()}


def make_windows(x, window_size=pp.WINDOW_SIZE, step=pp.STEP):
    """x: [samples, channels] -> (array [n, window_size, channels], start indices)."""
    starts = list(range(0, len(x) - window_size + 1, step))
    if not starts:
        return np.empty((0, window_size, x.shape[1]), dtype=np.float32), starts
    return np.stack([x[s:s + window_size] for s in starts]).astype(np.float32), starts


def _fit_shape(w, target):
    """Reshape windows [n, W, C] to the model's input shape (batch dim ignored)."""
    if target is None:
        return w
    target = [d for d in target]
    rank = len(target)
    n, W, C = w.shape
    if rank == 2:                                    # [batch, features]
        return w.reshape(n, -1)
    if rank == 3:                                    # [batch, time, ch] or [batch, ch, time]
        if target[1] == C and target[2] == W and W != C:
            return np.transpose(w, (0, 2, 1))
        return w
    if rank == 4:                                    # add a singleton channel axis
        if target[-1] == 1:
            return w[..., None]
        if target[1] == 1:
            return w[:, None, :, :]
    return w


def _name(label):
    try:
        return ID_TO_NAME.get(int(label), str(label))
    except (TypeError, ValueError):
        return str(label)


class ModelWrapper:
    def __init__(self, path):
        if not path or not os.path.exists(path):
            raise ValueError(f"Model file not found: {path or '(no path given)'}")
        ext = os.path.splitext(path)[1].lower()
        self.path = path
        if ext in (".keras", ".h5"):
            try:
                import tensorflow as tf
            except ImportError:
                raise ValueError("Loading a Keras model needs tensorflow (pip install tensorflow).")
            self.kind = "keras"
            self.model = tf.keras.models.load_model(path)
            self.input_shape = self.model.input_shape
        elif ext in (".pt", ".pth"):
            try:
                import torch
            except ImportError:
                raise ValueError("Loading a PyTorch model needs torch (pip install torch).")
            self.kind = "torch"
            self.torch = torch
            try:
                self.model = torch.jit.load(path, map_location="cpu")          # TorchScript
            except Exception:
                try:
                    self.model = torch.load(path, map_location="cpu", weights_only=False)  # whole model
                except Exception as e:
                    raise ValueError(f"Could not load the .pt file: {e}")
            if isinstance(self.model, dict) or not callable(self.model):
                raise ValueError("This .pt file only holds weights (a state_dict). Save the whole model "
                                 "with torch.save(model, path) or export TorchScript with "
                                 "torch.jit.script/trace(model).save(path).")
            self.model.eval()
            self.input_shape = None
            self._layout = None                     # which input layout worked (found on first call)
        elif ext == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError:
                raise ValueError("Loading an ONNX model needs onnxruntime (pip install onnxruntime).")
            self.kind = "onnx"
            self.model = ort.InferenceSession(path)
            meta = self.model.get_inputs()[0]
            self.input_name = meta.name
            self.input_shape = [d if isinstance(d, int) else None for d in meta.shape]
        elif ext in (".pkl", ".joblib"):
            self.kind = "sklearn"
            try:
                import joblib
                self.model = joblib.load(path)
            except Exception:
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
            self.input_shape = None
            if not hasattr(self.model, "predict"):
                raise ValueError("That pickle has no predict() method.")
        else:
            raise ValueError(f"Unsupported model type '{ext}'. Use .keras, .h5, .onnx, .pkl or .joblib.")

    def predict(self, windows):
        """windows [n, W, C] -> (list of gesture names, np.array of confidences)."""
        if self.kind == "sklearn":
            x = windows.reshape(len(windows), -1)
            if hasattr(self.model, "predict_proba"):
                p = self.model.predict_proba(x)
                idx = p.argmax(1)
                labels = [_name(self.model.classes_[i]) for i in idx]
                return labels, p.max(1)
            return [_name(v) for v in self.model.predict(x)], np.full(len(x), np.nan)

        if self.kind == "torch":
            p = self._predict_torch(windows)
            idx = p.argmax(1)
            return [_name(i + 1) for i in idx], p.max(1)

        x = _fit_shape(windows, self.input_shape)
        if self.kind == "keras":
            p = np.asarray(self.model.predict(x, verbose=0))
        else:
            p = np.asarray(self.model.run(None, {self.input_name: x.astype(np.float32)})[0])
        if p.ndim == 1 or p.shape[1] == 1:           # single sigmoid output: not a gesture classifier
            raise ValueError("The model has a single output; expected one score per gesture.")
        idx = p.argmax(1)
        return [_name(i + 1) for i in idx], p.max(1)


def _predict_torch(self, windows, batch=1024):
    torch = self.torch
    layouts = _torch_layouts()
    if self._layout is None:                          # find the layout the model accepts
        errors = []
        for name, fn in layouts:
            try:
                with torch.no_grad():
                    self.model(torch.from_numpy(np.ascontiguousarray(fn(windows[:2]))).float())
                self._layout = name
                break
            except Exception as e:
                errors.append(f"{name}: {str(e).splitlines()[0][:100]}")
        if self._layout is None:
            raise ValueError("The model rejected every input layout I tried:\n  " + "\n  ".join(errors))
    fn = dict(layouts)[self._layout]
    outs = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            x = torch.from_numpy(np.ascontiguousarray(fn(windows[i:i + batch]))).float()
            o = self.model(x)
            if isinstance(o, (tuple, list)):
                o = o[0]
            outs.append(o.detach().cpu().numpy())
    p = _to_probs(np.concatenate(outs))
    if p.shape[1] == 1:
        raise ValueError("The model has a single output; expected one score per gesture.")
    return p


ModelWrapper._predict_torch = _predict_torch

def _torch_layouts():
    """Input layouts tried in turn for a PyTorch model: [n,W,C], [n,C,W], [n,1,C,W], [n,1,W,C], [n,W*C]."""
    return [
        ("time_channels", lambda w: w),
        ("channels_time", lambda w: np.transpose(w, (0, 2, 1))),
        ("1_channels_time", lambda w: np.transpose(w, (0, 2, 1))[:, None]),
        ("1_time_channels", lambda w: w[:, None]),
        ("flat", lambda w: w.reshape(len(w), -1)),
    ]


def _to_probs(out):
    """Turn model output into probabilities (softmax if it looks like logits)."""
    out = np.asarray(out, dtype=float)
    if out.ndim == 1:
        out = out[None]
    if out.min() < 0 or not np.allclose(out.sum(1), 1.0, atol=1e-3):
        e = np.exp(out - out.max(1, keepdims=True))
        out = e / e.sum(1, keepdims=True)
    return out


def classify_dataframe(df, model, window_size=pp.WINDOW_SIZE, step=pp.STEP):
    """Classify every window of a preprocessed DataFrame. Returns a DataFrame of predictions."""
    chs = [c for c in df.columns if c not in ("Time", "Gesture")]
    if not chs:
        raise ValueError("No signal columns found in the data.")
    x = df[chs].to_numpy(dtype=float)
    if len(x) < window_size:
        raise ValueError(f"Need at least {window_size} rows, got {len(x)}.")
    windows, starts = make_windows(x, window_size, step)
    labels, conf = model.predict(windows)

    out = pd.DataFrame({"start": starts, "prediction": labels, "confidence": conf})
    if "Time" in df.columns:
        t = df["Time"].to_numpy(dtype=float)
        out.insert(1, "end_time_ms", [t[s + window_size - 1] for s in starts])
    if "Gesture" in df.columns:
        truth = df["Gesture"].astype(str).str.strip().to_numpy()
        out["truth"] = [truth[s + window_size - 1] for s in starts]
    return out
