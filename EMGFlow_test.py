#offline data visualisation - segmentation for the 8 channels on the sEMG

import pandas as pd
import matplotlib.pyplot as plt

file_path = 'franticData.csv'
df = pd.read_csv(file_path)

sampling_rate = 500

emg_channels = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
num_channels = len(emg_channels)
time_axis = df['Time']

fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels), sharex=True)

for i, channel in enumerate(emg_channels):
    axes[i].plot(time_axis, df[channel], color='crimson', linewidth=0.7)
    axes[i].set_ylabel(f"{channel}", fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.5)
    axes[i].tick_params(axis='y', labelsize=9)

axes[-1].set_xlabel('Time (Miliseconds)', fontsize=12)
plt.suptitle('EMG Raw Signals', fontsize=14, fontweight='bold')

plt.tight_layout()

plt.show()