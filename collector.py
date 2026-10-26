"""
collector.py - serial acquisition, live preprocessing and CSV writing.

`recorder` is meant to be run as a separate process by the GUI. It:
  * reads every line the device sends (no 16 ms "latest value" sampling, so no dropped samples)
  * timestamps each sample from when it arrived (samples that arrive together in one
    serial chunk are spread evenly between the previous chunk and this one)
  * runs every sample through the streaming preprocessor, writes the PREPROCESSED
    values to the CSV, and sends (raw, preprocessed) rows to the GUI for the live graph

If the sampling rate is not given, it is measured from the first CALIB_S seconds.
"""
import csv
import queue
import time

import numpy as np
import serial

import Preprocessing as pp

BAUDRATE = 115200
HEADERS = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
N_CH = len(HEADERS) - 1
CALIB_S = 2.0          # seconds of data used to measure the sampling rate (auto mode)


def recorder(port, baudrate, file_path, pp_params, paused_flag, stop_event,
             ready_event, error_queue, plot_queue):
    """
    Takes:
    port, baudrate: serial settings
    file_path: str - where the preprocessed CSV is saved
    pp_params: tuple (fs, lowcut, highcut, notch_freq, envelope_hz)
               fs=None -> measure it from the first CALIB_S seconds
               notch_freq=0 -> no notch; envelope_hz=None -> no rectify/smooth stage
    paused_flag: mp.Value('b') - while true, incoming data is discarded
    stop_event: mp.Event - signals the process to stop
    ready_event: mp.Event - set once the port and file are open
    error_queue: mp.Queue - a serial / settings error message for the GUI
    plot_queue: mp.Queue - receives ("rows", [(t_ms, raw[8], pre[8]), ...]) and
                ("fs", sampling_rate_hz); dropped silently if the GUI falls behind
    """
    fs_fixed, low, high, notch, env = pp_params

    def make_proc(fs):
        return pp.StreamingPreprocessor(fs, low, high, notch, n_channels=N_CH,
                                        envelope_hz=env)

    proc = None
    if fs_fixed:
        try:
            proc = make_proc(fs_fixed)
        except ValueError as e:
            error_queue.put(str(e))
            stop_event.set()
            return

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
            calib_ts, calib_vals = [], []

            def now_t():
                return time.perf_counter() - t0 - total_paused

            def write_rows(ts, pre):
                for t, p in zip(ts, pre):
                    writer.writerow([int(round(t * 1000))] + [f"{v:.5f}" for v in p])
                f.flush()

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
                        calib_ts.clear()
                        calib_vals.clear()
                        if proc is not None:
                            proc.reset()                    # gap in the data

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
                    if len(tokens) == 8:
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

                if proc is None:
                    # still measuring the sampling rate: show raw only
                    calib_ts += ts
                    calib_vals += vals
                    send(("rows", [(int(round(t * 1000)), v, [0.0] * N_CH)
                                   for t, v in zip(ts, vals)]))
                    span = calib_ts[-1] - calib_ts[0]
                    if span >= CALIB_S and len(calib_ts) >= 10:
                        fs_est = (len(calib_ts) - 1) / span
                        try:
                            proc = make_proc(fs_est)
                        except ValueError as e:
                            error_queue.put(str(e))
                            stop_event.set()
                            return
                        pre = proc.process(np.asarray(calib_vals, dtype=float))
                        write_rows(calib_ts, pre)
                        send(("fs", fs_est))
                        calib_ts, calib_vals = [], []
                    continue

                pre = proc.process(np.asarray(vals, dtype=float))
                write_rows(ts, pre)
                send(("rows", [(int(round(t * 1000)), v, p.tolist())
                               for t, v, p in zip(ts, vals, pre)]))
    finally:
        ser.close()
        # don't let unread queue items block this process from exiting
        plot_queue.cancel_join_thread()