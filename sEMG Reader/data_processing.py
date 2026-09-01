import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import EMGFlow

running = True

# Define a function to read the sEMG data from a Dat file
def read_semg_data(file_path, mode):
    with open(file_path, 'rb') as f:
        raw = f.read()

    PACKET_SIZE = 98

    num_packets = len(raw) // PACKET_SIZE

    packets = np.frombuffer(
        raw[:num_packets * PACKET_SIZE],
        dtype=np.uint8
    ).reshape(num_packets, PACKET_SIZE)

    print("Packets:", num_packets)
    print("Shape:", packets.shape)

    for i in range(10):
        print(
            f"Packet {i}: "
            f"header={packets[i, 0:2].tolist()} "
            f"bytes2-7={packets[i, 2:8].tolist()} "
            f"end={packets[i, 96:98].tolist()}"
        )
    


# Define main function to read the sEMG data
def init(file_path, mode):
    file_path = file_path
    mode = mode
    read_semg_data(file_path, mode)

def monitor_file_size(file_path, last_size, index):

    rint("Thread started")

    while running:
        size = os.path.getsize(file_path)
        last_size = size
        index += 1
        print(f"File size: {size} bytes, Index: {index}")
   

    return last_size, index