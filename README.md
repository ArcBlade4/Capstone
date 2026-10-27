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

