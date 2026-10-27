"""
collector.py - serial acquisition and CSV writing.

`recorder` is meant to be run as a separate process by the GUI. It:
  * reads every line the device sends (no 16 ms "latest value" sampling, so no dropped samples)
  * timestamps each sample from when it arrived (samples that arrive together in one
    serial chunk are spread evenly between the previous chunk and this one)
  * writes the RAW values to the CSV and sends (time, raw) rows to the GUI for the live graph

Filtering / scaling is done afterwards with the Preprocess tab (Preprocessing.py).
"""
import csv
import queue
import time

import serial

BAUDRATE = 115200
HEADERS = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
N_CH = len(HEADERS) - 1


def recorder(port, baudrate, file_path, paused_flag, stop_event,
             ready_event, error_queue, plot_queue):
    """
    port, baudrate: serial settings
    file_path: str - where the raw CSV is saved
    paused_flag: mp.Value('b') - while true, incoming data is discarded
    stop_event: mp.Event - signals the process to stop
    ready_event: mp.Event - set once the port and file are open
    error_queue: mp.Queue - a serial error message for the GUI
    plot_queue: mp.Queue - receives ("rows", [(t_ms, raw[8]), ...]); dropped silently if the GUI falls behind
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        error_queue.put(str(e))
        stop_event.set()
        return

    def send(item):
        try:
            plot_queue.put_nowait(item)
        except queue.Full:
            pass

    try:
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
            f.flush()

            ready_event.set()
            t0 = time.perf_counter()
            total_paused = 0.0
            pause_started = None
            last_t = 0.0
            buf = b''

            def now_t():
                return time.perf_counter() - t0 - total_paused

            while not stop_event.is_set():
                try:
                    if paused_flag.value:
                        if pause_started is None:
                            pause_started = time.perf_counter()
                        ser.reset_input_buffer()
                        buf = b''
                        time.sleep(0.05)
                        continue
                    if pause_started is not None:           # just resumed
                        total_paused += time.perf_counter() - pause_started
                        pause_started = None
                        buf = b''
                        last_t = now_t()

                    first = ser.read(1)                     # blocks until data (or 0.1 s timeout)
                    if not first:
                        continue
                    data = buf + first + ser.read(ser.in_waiting)
                except serial.SerialException as e:
                    error_queue.put(str(e))
                    stop_event.set()
                    return

                lines = data.split(b'\n')
                buf = lines.pop()                           # incomplete last line
                if len(buf) > 4096:
                    buf = b''
                vals = []
                for ln in lines:
                    tokens = ln.decode('utf-8', errors='ignore').replace(',', ' ').split()
                    if len(tokens) == N_CH:
                        try:
                            vals.append([int(t) for t in tokens])
                        except ValueError:
                            pass                            # malformed sample
                if not vals:
                    continue

                # spread samples that arrived together evenly since the last chunk
                now = now_t()
                n = len(vals)
                ts = [last_t + (now - last_t) * (k + 1) / n for k in range(n)]
                last_t = now

                for t, v in zip(ts, vals):
                    writer.writerow([int(round(t * 1000))] + v)
                f.flush()
                send(("rows", [(int(round(t * 1000)), v) for t, v in zip(ts, vals)]))
    finally:
        ser.close()
        # don't let unread queue items block this process from exiting
        plot_queue.cancel_join_thread()