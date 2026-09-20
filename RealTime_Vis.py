import collections
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PORT = input("Enter the serial port: ")
BAUDRATE = 115200
WINDOW_SIZE = 300

emg_channels = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
num_channels = len(emg_channels)

data_buffers = [collections.deque(maxlen=WINDOW_SIZE) for _ in range(num_channels)]
for buf in data_buffers:
    buf.extend([0.0] * WINDOW_SIZE)

ser = serial.Serial(PORT, BAUDRATE, timeout=1)

#window setup
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title('Real-Time EMG Data Visualisation')
ax.grid(True)

#lines for each channel
lines = [ax.plot(list(data_buffers[i]), label=f"V{i+1}", linewidth=1)[0] for i in range(num_channels)]
ax.legend(loc="upper left")

#axis scaling range limits
ax.set_xlim(0, WINDOW_SIZE - 1)
ax.set_ylim(-50, 4500)

def animate(frame):
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            tokens = line.replace(',', ' ').split()
            if len(tokens) == num_channels:
                for i in range(num_channels):
                    data_buffers[i].append(float(tokens[i]))
        except ValueError:
            continue

    for i in range(num_channels):
        lines[i].set_ydata(list(data_buffers[i]))
        
    return lines

ani = animation.FuncAnimation(fig, animate, interval=20, blit=True, cache_frame_data=False)
plt.show()
