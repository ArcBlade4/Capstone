import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import EMGFlow

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

def keep_last_100_packets(file_path):
    PACKET_SIZE = 98

    with open(file_path, "rb") as file:
        data = file.read()

    # Get complete packets
    packets = [
        data[i:i + PACKET_SIZE]
        for i in range(0, len(data) - PACKET_SIZE + 1, PACKET_SIZE)
    ]

    # Keep only the newest 100
    packets = packets[-100:]

    with open(file_path, "wb") as file:
        file.write(b"".join(packets))

# Define main function to read the sEMG data
def main(file_path, mode):
    last_size = 0
    index = 0


    while True:
        size = os.path.getsize(file_path)

        if size != last_size:
            print(f"{index}: File size: {size} bytes (+{size - last_size})")
            last_size = size
            index += 1
        keep_last_100_packets(file_path)


        time.sleep(0.1) # Sleep for 100 milliseconds to avoid busy waiting