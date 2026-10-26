"""
Offline Visualisation of sEMG Data using a Line Graph

Read from a CSV file only

Requires the following libraries: pandas, matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt

#Replace the file path with whichever CSV file of the position to visualise
df = pd.read_csv("franticData.csv")

#Set by the sEMG device
sampling_rate = 500 

#Separate the time axis and the EMG channels into separate variables
emg_channels = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
num_channels = len(emg_channels)
time_axis = df['Time']

plt.figure(figsize=(14, 6))

#Overlay the lines on the graph for each channel, with a legend
for channel in emg_channels:
    plt.plot(time_axis, df[channel], label=channel, alpha=0.7, linewidth=1)

plt.title('sEMG Data Visualisation (Position #)', fontsize=14, fontweight='bold')
plt.xlabel('Time (Miliseconds)', fontsize=12)
plt.ylabel('Millivots (mV)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1), title="Channels")
plt.tight_layout()

plt.show()
