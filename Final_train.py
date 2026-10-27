"""
Final_train.py - train and save the sEMG gesture model (script version of Exp3_Model_complex.ipynb).

Pipeline (training and saving only: no tuning, no testing, no charts):
    Step 1   load and clean the recordings          (PREPRO.preprocess)
    Step 2   windows, split, balance, normalise     (PREPRO.make_datasets)
    Step 3   PyTorch datasets
    Step 4   build the CNN-LSTM from PARAMS and check its shapes
    Step 5   train with every seed (early stopping on validation, best validation epoch kept)
    Step 6   save the seed with the best validation accuracy: full checkpoint, weights, pickle,
             TorchScript, scaler

The test split is still held out by make_datasets (TEST_FRAC) but is not used here, so it stays
unseen for a separate evaluation.

Run from the folder that contains PREPRO.py and the Data folder:
    python Final_train.py

The model classes can also be imported by other code (e.g. the live program) without running the
pipeline:  from Final_train import CNNLSTMModel, build_model
"""

import copy
import pickle
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import PREPRO as pp


# ======================================================================================
# Settings
# ======================================================================================

# ---------- data ----------
DATA_FOLDER = "Data"
CHANNELS = [f"V{i+1}" for i in range(8)]
GESTURES = {"fist": 1, "flexion": 2, "extension": 3, "radial": 4, "ulnar": 5, "rest": 6, "table": 7}

# ---------- signal (passed to pp.preprocess) ----------
SAMPLE_RATE = 60
PREP = dict(
    sample_rate=SAMPLE_RATE,
    resample=False,
    outlier_z=3.0,
    zero_centre=True,
    notch_hz=None,
    low_hz=1,
    high_hz=8,
    filter_order=4,
    causal=True,
    per_file=True,
)

# ---------- windows and split (passed to pp.make_datasets) ----------
WINDOW_MS = 250
WINDOW_SIZE = SAMPLE_RATE * WINDOW_MS // 1000
STEP = 2
SPLIT_BY = "chrono"
VAL_FRAC, TEST_FRAC = 0.2, 0.1
MAX_GAP_MS = None
BALANCE_TRAIN = True
DATA_SEED = 0

# ---------- training ----------
BATCH_SIZE = 64
EPOCHS = 60                                        # upper limit; training stops early
PATIENCE = 8                                       # stop after this many epochs without a better val accuracy
LEARNING_RATE = 0.0040619880891515974
SEEDS = [0, 1, 2]                                  # model seeds: starting weights, batch order, dropout
WEIGHT_DECAY = 8.504021823437331e-06               # L2 regularisation; 0 disables

# ---------- hardware ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # GPU if available


# ======================================================================================
# Models setup
# ======================================================================================

def pad_left(x, kernel_size, dilation=1, stride=1):
    """
    What it does:
        Add zeros on the left (the past) of the time axis so a convolution keeps the sequence length
        and never uses a future sample. Return the padded tensor.

    Steps:
        1. Compute the causal padding: (kernel_size - 1) * dilation zeros.
        2. With stride > 1, add extra zeros so the last filter position ends on the newest sample.
        3. Pad the last dimension (time) on the left only.

    Input:  x [batch, channels, time]
    Output: [batch, channels, time + padding]
    """
    pad = (kernel_size - 1) * dilation + (-(x.shape[-1] - 1)) % stride
    return F.pad(x, (pad, 0))


def hand_made_features_torch(x):
    """
    What it does:
        Compute 4 classic sEMG features per channel over the whole window (torch version of
        hand_made_features, used inside the model so live windows get the same features).

    Steps:
        1. RMS: square, average over time, square root (signal strength).
        2. MAV: average absolute value (average size).
        3. WL: sum of absolute changes between neighbouring samples (how much it wiggles).
        4. ZC: number of sign changes (zero crossings, rough frequency).
        5. Join the four [batch, 8] blocks into [batch, 32].

    Input:  x [batch, time, 8]
    Output: [batch, 32]  (8 RMS, 8 MAV, 8 WL, 8 ZC)
    """
    rms = torch.sqrt(torch.mean(x ** 2, dim=1))                             # signal strength
    mav = torch.mean(torch.abs(x), dim=1)                                   # average size
    wl  = torch.sum(torch.abs(torch.diff(x, dim=1)), dim=1)                 # how much it wiggles
    zc  = torch.sum(torch.diff(torch.sign(x), dim=1) != 0, dim=1).float()   # zero crossings
    return torch.cat([rms, mav, wl, zc], dim=1)


class CNNFrontEnd(nn.Module):
    """
    What it does:
        Turn the raw window into a shorter sequence of pattern features for the LSTM.

    Steps:
        1. Flip the window to channels-first for Conv1d.
        2. Each layer: pad on the left, convolve, batch norm, ReLU.
        3. Only the last layer uses the stride (2 = half as many time steps).
        4. Flip back to time-first for the LSTM.

    Settings: width = filters in the first layer (later layers 2 x width), kernel_size = samples per
    filter, stride = time-step reduction on the last layer, n_layers = number of conv layers.
    out_channels = number of features per time step passed to the LSTM.
    """
    def __init__(self, n_channels=8, width=32, kernel_size=5, stride=2, n_layers=2):
        super().__init__()
        self.kernel_size = kernel_size
        widths = [width] + [2 * width] * (n_layers - 1)          # e.g. 32, 64
        self.strides = [1] * (n_layers - 1) + [stride]           # e.g. 1, 2  (stride on last layer only)

        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        n_in = n_channels
        for w, s in zip(widths, self.strides):
            self.convs.append(nn.Conv1d(n_in, w, kernel_size, stride=s))
            self.bns.append(nn.BatchNorm1d(w))
            n_in = w
        self.out_channels = n_in                                 # last layer's filters -> LSTM input size

    def forward(self, x):                                        # x: [batch, time, 8]
        """Pattern features: [batch, time / stride, out_channels]."""
        x = x.transpose(1, 2)                                    # -> [batch, 8, time]
        for conv, bn, s in zip(self.convs, self.bns, self.strides):
            x = torch.relu(bn(conv(pad_left(x, self.kernel_size, stride=s))))
        return x.transpose(1, 2)                                 # -> [batch, time / stride, out_channels]


class CNNLSTMModel(nn.Module):
    """
    What it does:
        Classify one window with a CNN front-end (patterns) followed by an LSTM (order over time),
        optionally joined with 32 hand-made features before the final layer.

    Steps:
        1. CNNFrontEnd turns the window into a sequence of pattern features.
        2. LSTM reads that sequence step by step.
        3. Keep the LSTM output at the last time step.
        4. If use_features: compute RMS, MAV, WL, ZC from the raw window, rescale with batch norm,
           and join them with the LSTM output (hidden + 32 numbers).
        5. Dropout, then a linear layer gives one score per gesture.

    Settings: width, kernel_size, stride, cnn_layers = front-end; hidden, lstm_layers = LSTM;
    dropout = share of values switched off in training; use_features = add hand-made features.
    """
    def __init__(self, n_channels=8, n_classes=7, width=32, kernel_size=5, stride=2,
                 hidden=64, cnn_layers=2, lstm_layers=1, dropout=0.3, use_features=False):
        super().__init__()
        self.cnn = CNNFrontEnd(n_channels, width, kernel_size, stride, cnn_layers)
        self.lstm = nn.LSTM(input_size=self.cnn.out_channels, hidden_size=hidden, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

        # Optional hand-made features: computed from the raw window, they skip the CNN and LSTM
        # and are joined with the LSTM's last output just before the final linear layer.
        self.use_features = use_features
        n_features = 4 * n_channels if use_features else 0
        if use_features:
            self.feature_bn = nn.BatchNorm1d(n_features)
        self.fc = nn.Linear(hidden + n_features, n_classes)

    def all_steps(self, x):
        """LSTM output at every (strided) time step: [batch, time / stride, hidden]."""
        out, _ = self.lstm(self.cnn(x))
        return out

    def forward(self, x):
        """Gesture scores for each window: [batch, n_classes]."""
        last_step = self.all_steps(x)[:, -1, :]
        if self.use_features:
            features = self.feature_bn(hand_made_features_torch(x))
            last_step = torch.cat([last_step, features], dim=1)
        return self.fc(self.dropout(last_step))


MODEL_CLASSES = {
    "CNN-LSTM": CNNLSTMModel,
}

# every model setting in one place (tuned values)
PARAMS = {
    "CNN-LSTM": dict(width=32, kernel_size=5, stride=2, hidden=128, cnn_layers=1, lstm_layers=1, dropout=0.025240962295266033, use_features=True)
    }



for p in PARAMS.values():                                   # training settings
    p.update(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE)

TRAIN_KEYS = ("lr", "weight_decay", "batch_size")           # these go to the optimiser, not the model


def build_model(name, seed=0, params=None, n_classes=None):
    """
    What it does:
        Create one untrained model of the given type with a fixed seed, on DEVICE.

    Steps:
        1. Use the given params, otherwise PARAMS[name].
        2. Use the given n_classes, otherwise N_CLASSES from the data.
        3. Set the seed, so the same seed always gives the same starting weights.
        4. Drop the training-only settings (lr, weight_decay, batch_size).
        5. Build the model class from MODEL_CLASSES and move it to DEVICE.
    """
    params = params or PARAMS[name]
    n_classes = n_classes if n_classes is not None else N_CLASSES
    torch.manual_seed(seed)                                 # same seed -> same starting weights
    model_args = {k: v for k, v in params.items() if k not in TRAIN_KEYS}
    return MODEL_CLASSES[name](n_classes=n_classes, **model_args).to(DEVICE)   # model on GPU/CPU


# ======================================================================================
# Training loop
# ======================================================================================

loss_fn = nn.CrossEntropyLoss()        # Loss function: compares gesture scores with true labels.


def train_one_epoch(model, loader, optimizer):
    """
    What it does:
        Train the model for one pass over all training windows. Return average loss and accuracy.

    Steps:
        1. Switch the model to training mode (dropout on, batch norm updates its statistics).
        2. Take one batch of windows and labels.
        3. Compute gesture scores (forward pass).
        4. Compute the loss between scores and labels.
        5. Clear old gradients.
        6. Compute new gradients (backward pass).
        7. Update the weights.
        8. Add the batch loss and correct predictions to running totals.
        9. Repeat steps 2-8 for every batch.
        10. Return average loss and accuracy.
    """
    model.train()                                              # 1. Training mode.
    total_loss, correct, count = 0.0, 0, 0
    for windows, labels in loader:                             # 2. One batch.
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)    #    Move batch to the model's device.
        scores = model(windows)                                # 3. Forward pass.
        loss = loss_fn(scores, labels)                         # 4. Loss.
        optimizer.zero_grad()                                  # 5. Clear gradients.
        loss.backward()                                        # 6. Backward pass.
        optimizer.step()                                       # 7. Update weights.
        total_loss += loss.item() * len(labels)                # 8. Sum loss over windows.
        correct += (scores.argmax(dim=1) == labels).sum().item()   # 8. Count correct predictions.
        count += len(labels)
    return total_loss / count, correct / count                 # 10. Average loss, accuracy.


def evaluate(model, loader):
    """
    What it does:
        Score the model on a dataset without changing any weights. Return average loss and accuracy.

    Steps:
        1. Switch the model to evaluation mode (dropout off, batch norm uses stored statistics).
        2. Turn off gradient tracking.
        3. Take one batch of windows and labels.
        4. Compute gesture scores.
        5. Add the batch loss and correct predictions to running totals.
        6. Repeat steps 3-5 for every batch.
        7. Return average loss and accuracy.
    """
    model.eval()                                               # 1. Evaluation mode.
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():                                      # 2. No gradients.
        for windows, labels in loader:                         # 3. One batch.
            windows, labels = windows.to(DEVICE), labels.to(DEVICE)   # Move batch to the model's device.
            scores = model(windows)                            # 4. Forward pass only.
            total_loss += loss_fn(scores, labels).item() * len(labels)   # 5. Sum loss.
            correct += (scores.argmax(dim=1) == labels).sum().item()     # 5. Count correct.
            count += len(labels)
    return total_loss / count, correct / count                 # 7. Average loss, accuracy.


def train_model(name, seed=0, params=None):
    """
    What it does:
        Train one model from scratch with early stopping. Keep the weights from the epoch with the
        best validation accuracy, save them to file and return the training history.

    Steps:
        1. Load settings: given params, otherwise PARAMS[name].
        2. Build the model with the given seed.
        3. Build the training loader (batch size from settings, shuffle order fixed by seed).
        4. Build the optimiser (AdamW: learning rate and weight decay from settings).
        5. For each epoch:
           5a. Train for one epoch.
           5b. Score on the validation set.
           5c. Record losses and accuracies.
           5d. If validation accuracy is a new best: store a copy of the weights, reset wait counter.
               Otherwise: increase wait counter.
           5e. Print epoch results.
           5f. Stop if wait counter reaches PATIENCE.
        6. Save best weights to "best_{name}_seed{seed}.pt".
        7. Return the history.
    """
    params = params or PARAMS[name]                            # 1. Settings.
    model = build_model(name, seed, params)                    # 2. Model, seeded weights.
    loader = torch.utils.data.DataLoader(train_data, batch_size=params["batch_size"], shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))   # 3. Seeded shuffle.
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                                  weight_decay=params["weight_decay"])                  # 4. Optimiser.
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_state, wait = 0.0, None, 0

    for epoch in range(1, EPOCHS + 1):                         # 5. Epoch loop (EPOCHS = upper limit).
        train_loss, train_acc = train_one_epoch(model, loader, optimizer)    # 5a. Train.
        val_loss, val_acc = evaluate(model, val_loader)                      # 5b. Validate.
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)   # 5c. Record.
        history["train_acc"].append(train_acc);   history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:                             # 5d. New best epoch.
            best_val_acc, wait = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}  # Copy weights in memory.
        else:
            wait += 1                                          # 5d. No improvement.

        print(f"  epoch {epoch:2d}  train acc {train_acc:.1%}  val acc {val_acc:.1%}"
              f"{'  <- best' if wait == 0 else ''}")          # 5e. Progress.
        if wait >= PATIENCE:                                   # 5f. Early stop.
            break

    torch.save(best_state, f"best_{name}_seed{seed}.pt")       # 6. One file write per model and seed.
    return history                                             # 7. History.


# ======================================================================================
# Saving helpers
# ======================================================================================

def save_model(name, model, channel_mean, channel_std, val_accuracy):
    """
    What it does:
        Save a full checkpoint: the weights plus everything needed to rebuild the model and to
        prepare live windows exactly as in training. File: "{name}_semg.pt".

    Steps:
        1. Collect the weights, normalisation (channel mean and std), preprocessing settings,
           window size, step, gesture names, split type, best validation accuracy and model settings.
        2. Save them together in one file with torch.save.
    """
    torch.save({                                   # rebuild with build_model(name, params=...)
        "model_name": name,
        "model_weights": model.state_dict(),
        "channel_mean": channel_mean.tolist(),     # plain lists, so torch.load works without extra flags
        "channel_std": channel_std.tolist(),
        "preprocessing": PREP,                     # every pp.preprocess setting
        "window_size": WINDOW_SIZE,
        "step": STEP,
        "gestures": GESTURE_NAMES,
        "split": SPLIT_BY,
        "val_accuracy": float(val_accuracy),
        "params": PARAMS[name]}, f"{name}_semg.pt")
    print(f"saved {name}_semg.pt")


# ======================================================================================
# Pipeline
# ======================================================================================



if __name__ == "__main__":
    print("training on", DEVICE, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    print(f"window: {WINDOW_MS} ms = {WINDOW_SIZE} samples at {SAMPLE_RATE} Hz, step {STEP}")

    # ---------- Step 1: load and clean the recordings ----------
    data = pp.preprocess(DATA_FOLDER, channels=CHANNELS, **PREP)



    # ---------- Step 2: windows, split, balance and normalise ----------
    d = pp.make_datasets(data, WINDOW_SIZE, STEP, channels=CHANNELS, gestures=GESTURES,
                         val_frac=VAL_FRAC, test_frac=TEST_FRAC, max_gap_ms=MAX_GAP_MS,
                         balance_train=BALANCE_TRAIN, data_seed=DATA_SEED)

    X_train, y_train = d["X_train"], d["y_train"]
    X_val,   y_val   = d["X_val"],   d["y_val"]
    GESTURE_NAMES    = d["gesture_names"]
    N_CLASSES        = d["n_classes"]
    channel_mean, channel_std = d["channel_mean"], d["channel_std"]




    # ---------- Step 3: put the data into PyTorch ----------
    train_data = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_data   = torch.utils.data.TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)   # for early stopping
    # the training loader is built inside train_model (seeded shuffle, batch size from PARAMS)

    batch_x = torch.tensor(X_train[:BATCH_SIZE]).to(DEVICE)
    print("one batch of windows:", tuple(batch_x.shape), "(batch, time steps, channels)")





    # ---------- Step 4: check the model builds and gives the right shapes ----------
    for name in MODEL_CLASSES:
        model = build_model(name)
        model.eval()
        with torch.no_grad():
            steps = model.all_steps(batch_x)
            scores = model(batch_x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:9s} input {tuple(batch_x.shape)} -> every step {tuple(steps.shape)} "
              f"-> scores {tuple(scores.shape)}   parameters: {n_params:,}")
        



    # ---------- Step 5: train with every seed ----------
    histories, training_seconds, chosen_seed, best_val = {}, {}, {}, {}
    for name in MODEL_CLASSES:
        run_histories, times = {}, []
        for seed in SEEDS:
            print(f"Training {name}, seed {seed}")
            start_time = time.time()
            run_histories[seed] = train_model(name, seed)
            times.append(time.time() - start_time)

        # the seed with the best validation accuracy is kept for saving
        chosen_seed[name] = max(SEEDS, key=lambda s: max(run_histories[s]["val_acc"]))
        histories[name] = run_histories[chosen_seed[name]]
        training_seconds[name] = float(np.mean(times))
        vals = [max(run_histories[s]["val_acc"]) for s in SEEDS]
        best_val[name] = max(run_histories[chosen_seed[name]]["val_acc"])
        print(f"  {name}: validation accuracy {np.mean(vals):.1%} ± {np.std(vals):.1%} "
              f"over {len(SEEDS)} seeds, using seed {chosen_seed[name]} ({best_val[name]:.1%}), "
              f"{training_seconds[name]:.0f} s per seed\n")



    # ---------- Step 6: save everything needed to use the model later ----------
    for name in MODEL_CLASSES:
        model = build_model(name)                                                   # best seed's weights
        model.load_state_dict(torch.load(f"best_{name}_seed{chosen_seed[name]}.pt", map_location=DEVICE))
        model.eval()

        save_model(name, model, channel_mean, channel_std, best_val[name])          # full checkpoint
        torch.save(model.state_dict(), f"{name}_model.pt")                          # weights only
        print(f"saved {name}_model.pt")
        with open(f"{name}_semg_model.pkl", "wb") as f:                             # whole model object
            pickle.dump(model, f)
        print(f"saved {name}_semg_model.pkl")


        # TorchScript: runs without these class definitions (CPU copy, eval mode).
        # Export check only: the traced copy must give the same output as the original.
        m = copy.deepcopy(model).cpu().eval()
        with torch.no_grad():
            traced = torch.jit.trace(m, torch.tensor(X_val[:2]))     # [batch, window, 8]
            x = torch.tensor(X_val[:50])                             # check another batch size
            assert torch.allclose(traced(x), m(x), atol=1e-5)
        traced.save(f"{name}_ts.pt")
        print(f"saved {name}_ts.pt")





    # scaler in the format Preprocessing.load_scaler expects
    settings = {"fs": SAMPLE_RATE, "lowcut": PREP["low_hz"], "highcut": PREP["high_hz"], "notch": 0}
    with open("scaler.pkl", "wb") as f:
        pickle.dump({"mean": np.asarray(channel_mean),
                     "std": np.asarray(channel_std),
                     "settings": settings}, f)
    print("saved scaler.pkl")