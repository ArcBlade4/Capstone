"""
collector.py - serial acquisition + CSV writing (refactored from main.py).

The two worker functions are meant to be run as separate processes by the GUI:
    serial_reader -> reads the serial port and keeps the latest sample in shared memory
    writer_loop   -> every INTERVAL_S writes the latest sample to CSV and also pushes
                     (time_ms, values) onto plot_queue so the GUI can draw it live
"""
import csv
import queue
import time

import serial

BAUDRATE = 115200

INTERVAL_MS = 50
INTERVAL_S = INTERVAL_MS / 1000.0
FLUSH_EVERY_N_ROWS = 5
SPIN_MARGIN_S = 0.0005

HEADERS = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']


def serial_reader(port, baudrate, shared_values, new_data_flag, paused_flag,
                  stop_event, ready_event, error_queue):
    """
    Continuously reads lines from the serial port and stores the latest parsed
    sample into shared memory for the writer process.

    Takes:
    port: str - the COM port to read from
    baudrate: int - serial baud rate
    shared_values: mp.Array('i', 8) - shared array updated with the latest sample
    new_data_flag: mp.Value('b') - set to 1 whenever a new sample is written
    paused_flag: mp.Value('b') - when true, reading is skipped and the input buffer is drained
    stop_event: mp.Event - signals the process to stop
    ready_event: mp.Event - set once the serial port has been opened
    error_queue: mp.Queue - used to pass a serial error message back to the GUI

    Malformed lines (wrong token count / non-integer values) are silently skipped.
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        error_queue.put(str(e))
        stop_event.set()
        return

    ready_event.set()

    try:
        while not stop_event.is_set():
            if paused_flag.value:
                ser.reset_input_buffer()
                time.sleep(0.05)
                continue
            try:
                if ser.in_waiting > 0:
                    raw_data = ser.readline()
                    decoded = raw_data.decode('utf-8', errors='ignore').strip()
                    tokens = decoded.replace(',', ' ').split()
                    if len(tokens) == 8:
                        values = [int(t) for t in tokens]
                        with shared_values.get_lock():
                            shared_values[:] = values
                            new_data_flag.value = 1
                else:
                    time.sleep(0)  # yield without adding real delay
            except serial.SerialException as e:
                error_queue.put(str(e))
                stop_event.set()
                return
            except ValueError:
                pass  # malformed sample, skip it
    finally:
        ser.close()


def writer_loop(file_path, shared_values, new_data_flag, paused_flag, stop_event,
                start_time, pause_state, plot_queue):
    """
    Runs on a fixed INTERVAL_S tick. Whenever unpaused and a new sample is
    available, writes one row (elapsed ms + latest values) to the CSV and
    pushes the same row to plot_queue for the live graph.

    Takes:
    file_path: str - where the CSV is saved
    shared_values / new_data_flag / paused_flag / stop_event: shared state (see serial_reader)
    start_time: float - time.perf_counter() captured when recording began
    pause_state: tuple(mp.Value, mp.Value, mp.Value) - (paused, pause_started_at, total_paused)
    plot_queue: mp.Queue - receives (t_ms, [8 ints]); dropped silently if the GUI falls behind
    """

    def time_since_start():
        """Seconds since start, net of any time spent paused."""
        paused, pause_started_at, total_paused = pause_state
        elapsed = time.perf_counter() - start_time - total_paused.value
        if paused.value and pause_started_at.value != 0.0:
            elapsed -= (time.perf_counter() - pause_started_at.value)
        return elapsed

    try:
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)

            next_time = time.perf_counter()
            rows_since_flush = 0

            while not stop_event.is_set():
                remaining = next_time - time.perf_counter()
                if remaining > 0:
                    if remaining > SPIN_MARGIN_S:
                        time.sleep(remaining - SPIN_MARGIN_S)
                    while time.perf_counter() < next_time:
                        pass  # short busy-spin closes the gap left by sleep()

                if not paused_flag.value and new_data_flag.value:
                    with shared_values.get_lock():
                        values = list(shared_values)
                        new_data_flag.value = 0  # mark this sample as consumed

                    t_ms = int(round(time_since_start() * 1000))
                    writer.writerow([t_ms] + values)

                    try:
                        plot_queue.put_nowait((t_ms, values))
                    except queue.Full:
                        pass

                    rows_since_flush += 1
                    if rows_since_flush >= FLUSH_EVERY_N_ROWS:
                        f.flush()
                        rows_since_flush = 0

                next_time += INTERVAL_S
    finally:
        # don't let unread queue items block this process from exiting
        plot_queue.cancel_join_thread()
