# sEMG Gesture Classification
 
Real-time-capable classification of hand and wrist gestures from 8-channel surface EMG (sEMG).
A CNN-LSTM model is trained on recorded data, and a desktop GUI records, preprocesses, visualises and classifies new recordings.
 
The project has two parts:
 
1. **ML trainer** (`Final_train.py`): turns a folder of labelled recordings into a trained model plus a matching scaler.
2. **GUI and prediction app** (`main.py`): records from the serial device, preprocesses data, plots it and runs the trained model.

## Gestures
 
| ID | Gesture   |
|----|-----------|
| 1  | fist      |
| 2  | flexion   |
| 3  | extension |
| 4  | radial    |
| 5  | ulnar     |
| 6  | rest      |
| 7  | table     |


The model outputs one score per gesture. Output index `i` maps to gesture ID `i + 1`.

---

## Repository Structure

| File | Role |
|------|------|
| `main.py` | Tkinter GUI with three tabs (Record, Preprocess, Classify) and a live/recorded/preprocessed graph |
| `collector.py` | Serial acquisition process. Reads every line from the device, timestamps it and writes **raw** values to CSV |
| `Preprocessing.py` | Single-file / live preprocessing. Mirrors `PREPRO.py` so GUI data is treated like training data |
| `classifier.py` | Loads a model (`.pt` TorchScript, `.keras`, `.onnx`, `.pkl`) and classifies preprocessed windows |
| `PREPRO.py` | Training-side preprocessing: cleaning, windowing, chronological split, balancing, normalisation, augmentation |
| `Final_train.py` | Trains the CNN-LSTM with several seeds and saves the best one |



Expected layout for training:
 
```
project/
├── main.py
├── collector.py
├── Preprocessing.py
├── classifier.py
├── training/
│   ├── Final_train.py
│   ├── PREPRO.py
│   └── Data/
│       ├── 01/01.txt
│       ├── 02/02.txt
│       └── ...
├── CNN-LSTM_ts.pt        # created by training
└── scaler.pkl            # created by training
```
 
`PREPRO.preprocess` loads recordings with the pattern `Data/*/*.txt`, so each recording lives in its own numbered subfolder.
 
---

## Installation

Recommended Python version: 3.11+

```bash
pip install numpy pandas scipy pyserial torch
```
Training uses a GPU automatically if CUDA is available, otherwise the CPU.
 
---

## Data format
 
Raw recordings are CSV files with a header row:
 
```
Time,V1,V2,V3,V4,V5,V6,V7,V8
```
 
* `Time`: milliseconds since recording started (pauses are excluded).
* `V1`–`V8`: integer ADC (Analog-to-Digital Converter) readings, one column per sensor channel.
* **Training files also need a `Gesture` column** with one of the gesture names above. Files recorded by `main.py` have the none label by default, so label must be chosen when creating training set.
The device is expected to send one sample per line over serial: eight integers separated by spaces or commas, at 115200 baud by default. Malformed lines are dropped.
 
---

# How to use

## Training 

1. Delete the old `CNN-LSTM_ts.pt` and `scaler.pkl`.
2. Record data with `main.py` (Record tab), while changing labels per recording.
3. Replace the contents of the `Data` folder with your recordings (one numbered subfolder per recording).
4. Run the trainer from the `training` folder:
```bash
   python Final_train.py
```
5. Wait for training to finish. Three seeds are trained, each with early stopping (patience 8, max 60 epochs), and the seed with the best validation accuracy is kept.
6. Check that `CNN-LSTM_ts.pt` and `scaler.pkl` exist. Copy them next to `main.py`.

Files written by training:
 
| File | Contents |
|------|----------|
| `best_CNN-LSTM_seed{N}.pt` | Best-epoch weights for each seed |
| `CNN-LSTM_semg.pt` | Full checkpoint: weights, normalisation, preprocessing settings, window size, params |
| `CNN-LSTM_model.pt` | Weights only |
| `CNN-LSTM_semg_model.pkl` | Pickled model object (needs the class definitions) |
| `CNN-LSTM_ts.pt` | **TorchScript model. This is the one the GUI uses** |
| `scaler.pkl` | Per-channel mean/std and filter settings |
 
The test split (last 10% of each recording) is held out but never used during training, so it is available for a separate evaluation.

### Using the GUI
 
```bash
python main.py
```
 
1. **Record**: in the *Record data* tab choose the serial port and baud rate, choose where to save, choose label, and press **Start**. **Pause**/**Resume** discards incoming data while paused. **Stop** saves the file. The live graph shows raw values.
2. **Preprocess**: in the *Preprocess* tab pick a file or folder, check the settings (they must match training), and press **Run preprocessing**. Output is saved next to each source as `*_preprocessed.csv`.
3. **Classify**: in the *Classify* tab select `CNN-LSTM_ts.pt` as the model, then choose either:
   * the current preprocessed data (what is on the graph), or
   * a CSV file. Tick *File is raw* if it still needs preprocessing, which uses the Preprocess tab settings and the scaler.
4. Press **Classify**. Results (window count per gesture, confidence, accuracy if a `Gesture` column exists) appear in the tab and are saved as `*_predictions.csv`.
The graph can switch between **Live (raw)**, **Recorded (raw)** and **Preprocessed (file)**, with per-channel toggles for V1–V8.
 
---
## Pipeline
 
```
Inputs ─► Data collection ─► Visualisation ─► Preprocessing ─► Model ─► Outputs
```

### Inputs
* 8-channel sEMG from serial device (ADC integer values, 115200 baud)
* Previously recorded CSV files (raw or already preprocessed)

### Data collection (`collector.py`)
 
* Runs as a separate process so the GUI never blocks acquisition.
* Reads every line the device sends, with no "latest value" sampling, so no samples are dropped.
* Timestamps from arrival time. Samples that arrive together in one serial chunk are spread evenly between the previous and current chunk.
* Writes **raw** values to CSV and sends rows to the GUI for live plotting (the queue drops items if the GUI falls behind, but the CSV is never affected).

### Visualisation (`main.py`)
 
* Live scrolling view of the last 300 samples per channel.
* Static views for recorded raw data and preprocessed data (downsampled to 20,000 points for plotting only).

### Preprocessing (`Preprocessing.py`, mirrors `PREPRO.py`)
 
Applied per recording, in this order:
 
1. **Outlier removal**: drop rows where any channel is more than 3 standard deviations from its mean.
2. **Resampling** onto an even time grid using the `Time` column (linear interpolation).
3. **Zero-centre**: subtract each channel's mean.
4. **Notch filter** (optional, off by default).
5. **Butterworth band-pass**, order 4, causal (looks only at the past, as a live system must). Default 1–8 Hz.
6. **Standardisation** with the saved per-channel mean/std from `scaler.pkl` (z-score).
Windowing for training and classification uses a 250 ms window with a step of 2 samples. For training, only windows with a single constant known gesture label are kept.

### Model (`Final_train.py`)
 
A CNN-LSTM:
 
* **CNN front-end**: causal (left-padded) 1D convolutions + batch norm + ReLU, with stride on the last layer.
* **LSTM**: reads the CNN output over time; the last time step is kept.
* **Hand-made features** (optional, on in the tuned configuration): RMS, MAV, waveform length and zero crossings per channel (32 values), batch-normalised and joined with the LSTM output.
* **Dropout + linear layer** gives one score per gesture.
Tuned configuration: width 32, kernel size 5, stride 2, 1 CNN layer, 1 LSTM layer with 128 hidden units, dropout ≈ 0.025, hand-made features on. Training uses AdamW (lr ≈ 0.0041, weight decay ≈ 8.5e-6), batch size 64.
 
Training data handling:
 
* Chronological split inside every recording: first 70% train, next 20% validation, last 10% test. Windows crossing a cut are dropped, and a leak check asserts no raw sample is shared between splits.
* Training classes are capped at the median window count to balance them.
* Normalisation statistics come from the training set only.
* Training set is augmented with one noisy copy (Gaussian noise, 5% of channel std). Validation and test are never augmented.
### Outputs
 
* `*_preprocessed.csv`: filtered and standardised signals.
* `*_predictions.csv`: per-window prediction, confidence, end time and (if available) ground truth.
* On-screen counts per gesture and accuracy against the `Gesture` column when present.
---

## Key settings that must match between training and use
 
The model only works if live data is prepared the way training data was. These values are stored in `scaler.pkl`, and the GUI refuses to run if they differ from the Preprocess tab.
 
| Setting | Where set in training | Where set in the GUI |
|---------|----------------------|----------------------|
| Sample rate | `SAMPLE_RATE` in `Final_train.py` | `SAMPLE_RATE` in `Preprocessing.py`, *fs* field |
| Window length | `WINDOW_MS` (250 ms) | `WINDOW_MS` in `Preprocessing.py` |
| Step | `STEP` | `STEP` in `Preprocessing.py` |
| Band-pass | `low_hz`, `high_hz` | *Low cut*, *High cut* fields |
| Notch | `notch_hz` | *Notch* field |
| Scaling | `channel_mean`/`channel_std` | `scaler.pkl` |
 
---

## Findings

### Limitations
* **Small dataset:** The dataset used to train the model had a sample size of one and had 76000~ datapoints.
* **Low sample rate.** Effective rates of 40–60 Hz per channel limit the usable bandwidth to under 30 Hz, so most of the sEMG spectrum is not captured.
* **Irregular timing.** Serial samples arrive in bursts, and timestamps are assigned on arrival, so spacing is approximate.
* **Chronological split within the same sessions.** Validation and test data come from the end of the same recordings as the training data, which can overstate generalisation.
* **Gesture set is fixed** to seven classes, with no "unknown" or rejection class; the model always outputs one of them. Confidence is a softmax score, not a calibrated probability.
* **Classification is offline in the GUI.** It processes recordings after the fact rather than streaming predictions live.

### Future steps
* **Live Classification:** 
* **Confidence Thresholding:**
* **Increasing Gesture Amount:**



