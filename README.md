# sEMG Gesture Classification
 
Real-time-capable classification of hand and wrist gestures from 8-channel surface EMG (sEMG).
A CNN-LSTM model is trained on recorded data, and a desktop GUI records, preprocesses, visualises and classifies new recordings.
 
The project has two parts:
 
1. **ML trainer** (`Final_train.py`): turns a folder of labelled recordings into a trained model plus a matching scaler.
2. **GUI and prediction app** (`main.py`): records from the serial device, preprocesses data, plots it and runs the trained model.
