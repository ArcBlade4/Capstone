import serial
import time
from pynput import keyboard
import pandas as pd
import asyncio
from multiprocessing import process

queue = []
PORT = input("Enter the serial port: ")
FILE_PATH = input("Enter the path to save the CSV file: ")
BAUDRATE = 115200
START_TIME = time.perf_counter()

paused = False
pause_started_at = None
total_paused = 0.0


def create_csv_file(file_path):
    """
    create_csv_file creates a file at a given location

    Takes:
    file_path: str - location of file

    Returns:
    None - creates file
    """
    headers = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
    df = pd.DataFrame(columns=headers)
    df.to_csv(file_path, index=False)

#append the data to the csv file
def save_to_csv(file_path, data):
    """
    save_to_csv saves given data to a provied csv file
    
    Takes:
    file_path: str - location of csv
    data: array-like - data to be saved

    Returns
    None - appends data to csv
    """
    df = pd.DataFrame([data], columns=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8'])
    df.to_csv(file_path, mode='a', header=False, index=False)

def time_since_start():
    """
    time_since_start records time elasped since beginning of program

    Takes:
    None

    Returns:
    elasped: int - time elasped - paused times
    """
    elapsed = time.perf_counter() - START_TIME - total_paused
    if paused and pause_started_at is not None:
        elapsed -= (time.perf_counter() - pause_started_at)
    return elapsed

def on_press(key):
    """
    on_press checks when key is pressed and handles pause/resume

    Takes:
    key: keyboardInput - the pressed key

    Returns:
    None - toggles pause/resume
    """
    global paused, pause_started_at, total_paused
    try:
        if key.char == 'p':
            paused = not paused
            if paused:
                pause_started_at = time.perf_counter()
                print(f"\n--- Serial reading PAUSED ---\n")
            else:
                total_paused += time.perf_counter() - pause_started_at
                pause_started_at = None
                print(f"\n--- Serial reading RESUMED ---\n")
    except AttributeError:
        pass  # ignore special keys

def on_release(key):
    """
    on_release checks when key is pressed and handles progam end

    Takes:
    key: keyboardInput - the pressed key

    Returns:
    False
    """
    # Stop the listener (and the program) if Escape is pressed
    if key == keyboard.Key.esc:
        print(f"\n---Stopping program---\n")
        return False
    


    
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


async def main(ms):
    interval = ms/1000
    next_time = time.monotonic()
    while listener.running:
        await read_serial(ser)
        remaining_time = next_time - time.monotonic()
        if remaining_time > 0 :
            #print(remaining_time)
            await asyncio.sleep(remaining_time)
        else:
            next_time = time.monotonic()
            if len(queue) > 0:
                next_item = queue.pop(0)
                print(next_item)
                if len(next_item) != 9:
                    #print("dropped")
                    pass
                else:
                    save_to_csv(FILE_PATH, next_item)
            next_time += interval




async def read_serial(ser):
    if paused:
        ser.reset_input_buffer()
        time.sleep(0.1)
    elif ser.in_waiting > 0:
        raw_data = ser.readline()
        try:
            decoded_data = raw_data.decode('utf-8').strip()
            split_data = decoded_data.split(' ')
            for i in range(len(split_data)):
                split_data[i] = int(split_data[i])
            data_time = time_since_start()
            data_time *= 1000
            data_time = round(data_time,0)
            data = [data_time] + split_data
            if len(queue) < 6:
                queue.append(data)
            else:
                pass
            ser.reset_input_buffer()
                
        except UnicodeDecodeError:
            print("Received data could not be decoded.")









try:
    with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
        print(f"Connected to {PORT} at {BAUDRATE} baud.")
        print("Press 'p' to pause/resume reading, 'Esc' to quit.")
        ser.reset_input_buffer()
        create_csv_file(FILE_PATH)            
        asyncio.run(main(50))
        
except serial.SerialException as e:
    print(f"Error opening or using serial port: {e}")
except KeyboardInterrupt:
    print("\nProgram stopped by user.")
finally:
    listener.stop()





