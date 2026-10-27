"""
Offline Visualisation of sEMG Data using a Heatmap Matrix

Read from a CSV file only

Requires the following libraries: pandas, numpy, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Replace the file path with whichever CSV file of the position to visualise
df = pd.read_csv("csv_data/01.csv")

#Separate the time axis and the EMG channels into separate variables
emg_channels = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
num_channels = len(emg_channels)
time_axis = df['Time']

#Covert the pandas dataframe into a numpy array and transpose it to have the channels as rows the time as columns
emg_matrix = df[emg_channels].to_numpy().T 

#Creates the tick positions for the x axis based on the length of the time column of the specific CSV file
tick_positions = range(len(time_axis))

plt.figure(figsize=(14, 5))
#Seaborn heatmap
sns.heatmap(emg_matrix,                #Data
            cmap='mako_r',                  #Colour palette (Reversed)
            cbar_kws={'label': 'Millivots (mV)'})          # Sets the label for the colour bar
plt.title('sEMG Heatmap Matrix (Position #)', fontsize=14, fontweight='bold')
plt.ylabel('Channels')
plt.xlabel('Time (Milliseconds)')
plt.xticks(ticks=tick_positions[::30], labels=time_axis[::30])           #only show every 30th tick and tick labels on the x axis to avoid cluttering the x-axis
plt.yticks(ticks=np.arange(8) + 0.5, labels=emg_channels, rotation=0)    #Position the y tick labels in the middle of each row for each channel
plt.tight_layout()
plt.show()
