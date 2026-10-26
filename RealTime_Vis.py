"""
Real Time Visualisation of sEMG Data

Read directly from serial port

Requires the following libraries: collections, serial, matplotlib
"""

import collections
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation


PORT = input("Enter the serial port: ")
BAUDRATE = 115200
WINDOW_SIZE = 300

emg_channels = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
num_channels = len(emg_channels)

#Creates a queue for each channel to store the data, once the queue is full, the oldest data will be removed automatically
data_buffers = [collections.deque(maxlen=WINDOW_SIZE) for _ in range(num_channels)]
for buf in data_buffers:
    buf.extend([0.0] * WINDOW_SIZE)             #Prevent empty buffer errors at the start

#If there is no data received in a maximum of one second, the program will continue to run without freezing 
ser = serial.Serial(PORT, BAUDRATE, timeout=1)

#Window setup
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title('Real-Time sEMG Data Visualisation')
ax.grid(True)

#Add lines to the graph for each channel and a legend to identify them
lines = [ax.plot(list(data_buffers[i]), label=f"V{i+1}", linewidth=1)[0] for i in range(num_channels)]
ax.legend(loc="upper left")

#Axis scaling range limits
ax.set_xlim(0, WINDOW_SIZE - 1)
ax.set_ylim(-50, 4500)

def animate():
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()     #Skip any corrupted or missing data lines
            tokens = line.replace(',', ' ').split()
            if len(tokens) == num_channels:
                for i in range(num_channels):
                    data_buffers[i].append(float(tokens[i]))         #converts the string to a float and adds it to the buffer for each channel
        except ValueError:
            continue

    for i in range(num_channels):
        lines[i].set_ydata(list(data_buffers[i]))              #only updates the data points, rather than the full graph and convert the buffer to a list for plotting
        
    return lines


ani = animation.FuncAnimation(fig, animate, interval=20,    #The delay between frames of the 'animation' - the real-time graph is 20 milliseconds, roughly 50 frames per second.
                              blit=True,                    #Only redraws the lines of the graph that have changed, rather that the full graph
                              cache_frame_data=False)       #Disable frame caching to reduce high memory consumption
plt.show()
