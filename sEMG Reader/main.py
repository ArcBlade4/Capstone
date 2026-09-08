import serial
import time
from pynput import keyboard
import pandas as pd
import asyncio



PORT = input("Enter the serial port: ")
FILE_PATH = input("Enter the path to save the CSV file: ")
BAUDRATE = 115200
START_TIME = time.perf_counter()

paused = False
pause_started_at = None
total_paused = 0.0



#create csv file with headers
def create_csv_file(file_path):
    headers = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
    df = pd.DataFrame(columns=headers)
    df.to_csv(file_path, index=False)

#append the data to the csv file
def save_to_csv(file_path, data):
    df = pd.DataFrame([data], columns=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8'])
    df.to_csv(file_path, mode='a', header=False, index=False)

def time_since_start():
    elapsed = time.perf_counter() - START_TIME - total_paused
    if paused and pause_started_at is not None:
        elapsed -= (time.perf_counter() - pause_started_at)
    return elapsed

def on_press(key):
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
    # Stop the listener (and the program) if Escape is pressed
    if key == keyboard.Key.esc:
        return False
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()




async def main(ms, task):
    loop = asyncio.get_running_loop()
    interval = ms/1000
    next_time = loop.time() + interval
    while listener.running:
        delay = next_time - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            await read_serial(ser)
        except Exception as e:
            print(f"Error in task: {e}")
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
            if len(split_data) != 8:
                print(f"Unexpected data length: {decoded_data}")
            else:
                for i in range(len(split_data)):
                    split_data[i] = int(split_data[i])
                time.sleep(0.1)
                ser.reset_input_buffer()
                data_time = round(time_since_start(),2)
                data_time *= 1000
                print(f"Received: {split_data} at {data_time} ms since start.")
                save_to_csv(FILE_PATH, [data_time] + split_data)
        except UnicodeDecodeError:
            print("Received data could not be decoded.")
                

try:
    with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
        print(f"Connected to {PORT} at {BAUDRATE} baud.")
        print("Press 'p' to pause/resume reading, 'Esc' to quit.")
        ser.reset_input_buffer()
        create_csv_file(FILE_PATH)            
        asyncio.run(main(50, read_serial(ser)))
except serial.SerialException as e:
    print(f"Error opening or using serial port: {e}")
except KeyboardInterrupt:
    print("\nProgram stopped by user.")
finally:
    listener.stop()





