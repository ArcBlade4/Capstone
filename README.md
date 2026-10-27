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
