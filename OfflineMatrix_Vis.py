import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("franticData.csv")

emg_channels = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
num_channels = len(emg_channels)
time_axis = df['Time']

emg_matrix = df[emg_channels].to_numpy().T 

plt.figure(figsize=(14, 5))
#sns.heatmap(emg_matrix, cmap='magma', cbar_kws={'label': 'TBD'})
sns.heatmap(emg_matrix[:, ::10], cmap='magma', cbar_kws={'label': 'TBD'})
plt.title('EMG Heatmap Matrix')
plt.ylabel('Channels')
plt.xlabel('Time')
plt.yticks(ticks=np.arange(8) + 0.5, labels=emg_channels, rotation=0)
plt.tight_layout()
plt.show()