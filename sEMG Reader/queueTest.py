import serial
import time
import multiprocessing as mp
import csv
from pynput import keyboard

BAUDRATE = 115200

INTERVAL_MS = 50         
INTERVAL_S = INTERVAL_MS
FLUSH_EVERY_N_ROWS = 100     

HEADERS = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']


def serial_reader(port, baudrate, shared_values, new_data_flag, paused_flag,
                   stop_event, ready_event, error_queue):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        error_queue.put(str(e))
        stop_event.set()
        return

    ready_event.set()

    while not stop_event.is_set():
        if paused_flag.value:
            ser.reset_input_buffer()
            time.sleep(0.05)
            continue
        try:
            if ser.in_waiting > 0:
                raw_data = ser.readline()
                decoded_data = raw_data.decode('utf-8').strip()
                split_data = decoded_data.split(' ')
                if len(split_data) == 8:
                    values = [int(v) for v in split_data]
                    with shared_values.get_lock():
                        shared_values[:] = values
                        new_data_flag.value = 1
            else:
                time.sleep(0)
        except UnicodeDecodeError:
            print("Received data could not be decoded.")
        except ValueError:
            pass  # malformed sample, skip it

    ser.close()


def writer_loop(file_path, shared_values, new_data_flag, paused_flag, stop_event,
                 start_time, pause_state):

    def time_since_start():
        paused, pause_started_at, total_paused = pause_state
        elapsed = time.perf_counter() - start_time - total_paused.value
        if paused.value and pause_started_at.value != 0.0:
            elapsed -= (time.perf_counter() - pause_started_at.value)
        return elapsed

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)

        next_time = time.perf_counter()
        rows_since_flush = 0

        while not stop_event.is_set():
            while time.perf_counter() < next_time:
                time.sleep((INTERVAL_S/1000)-10)
                pass 

            if not paused_flag.value and new_data_flag.value:
                with shared_values.get_lock():
                    values = list(shared_values)
                t_ms = round(time_since_start() * 1000, 0)
                writer.writerow([t_ms] + values)
                rows_since_flush += 1
                if rows_since_flush >= FLUSH_EVERY_N_ROWS:
                    f.flush()
                    rows_since_flush = 0

            next_time += INTERVAL_S


def main():
    port = input("Enter the serial port: ")
    file_path = input("Enter the path to save the CSV file: ")

    start_time = time.perf_counter()

    paused = mp.Value('b', 0)
    pause_started_at = mp.Value('d', 0.0)
    total_paused = mp.Value('d', 0.0)
    pause_state = (paused, pause_started_at, total_paused)

    shared_values = mp.Array('i', 8)
    new_data_flag = mp.Value('b', 0)
    stop_event = mp.Event()
    ready_event = mp.Event()
    error_queue = mp.Queue()

    def on_press(key):
        try:
            if key.char == 'p':
                if not paused.value:
                    paused.value = 1
                    pause_started_at.value = time.perf_counter()
                    print("\n--- Serial reading PAUSED ---\n")
                else:
                    total_paused.value += time.perf_counter() - pause_started_at.value
                    pause_started_at.value = 0.0
                    paused.value = 0
                    print("\n--- Serial reading RESUMED ---\n")
        except AttributeError:
            pass  # ignore special keys

    def on_release(key):
        if key == keyboard.Key.esc:
            print("\n--- Stopping program ---\n")
            stop_event.set()
            return False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    reader_process = mp.Process(
        target=serial_reader,
        args=(port, BAUDRATE, shared_values, new_data_flag, paused, stop_event, ready_event, error_queue),
        daemon=True,
    )
    reader_process.start()

    try:
        while not ready_event.is_set() and not stop_event.is_set():
            time.sleep(0.01)

        if stop_event.is_set():
            print(f"Error opening or using serial port: {error_queue.get()}")
        else:
            print(f"Connected to {port} at {BAUDRATE} baud.")
            print("Press 'p' to pause/resume reading, 'Esc' to quit.")
            writer_loop(file_path, shared_values, new_data_flag, paused, stop_event,
                        start_time, pause_state)

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        stop_event.set()
        listener.stop()
        reader_process.join(timeout=1)
        if reader_process.is_alive():
            reader_process.terminate()


if __name__ == "__main__":
    main()