#offline data visualisation - all 8 channels on the sEMG on the same graph denoted with separate lines

import pandas as pd
import matplotlib.pyplot as plt

file_path = 'emgflow_test1.csv'
df = pd.read_csv(file_path)

sampling_rate = 500 

emg_channels = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
num_channels = len(emg_channels)
time_axis = df['Time']

plt.figure(figsize=(14, 6))

for channel in emg_channels:
    plt.plot(time_axis, df[channel], label=channel, alpha=0.7, linewidth=1)

plt.title('EMG Raw Signals (Overlapping)', fontsize=14, fontweight='bold')
plt.xlabel('Time (Miliseconds)', fontsize=12)
plt.ylabel('TBD', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1), title="Channels")
plt.tight_layout()

plt.show()
