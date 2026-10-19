"""
gui.py - EMG Data Studio

One window that:
  1. collects data from the serial port into a CSV (with pause / resume / stop)
  2. preprocesses a CSV with Preprocessing.py
  3. shows a graph you can toggle: view (Live / Recorded raw / Preprocessed),
     individual channels V1-V8, and the whole graph on/off

Run:  python gui.py
Needs: pyserial pandas numpy scipy scikit-learn matplotlib   (tkinter ships with Python)
"""
import collections
import multiprocessing as mp
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import serial.tools.list_ports
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import collector
import Preprocessing as pp

WINDOW_SIZE = 300            # samples shown in the live view
MAX_STATIC_POINTS = 20000    # downsample long recordings for plotting only
LIVE_YLIM = (-50, 4500)
NUM_CH = 8
UI_POLL_MS = 40
VIEWS = ("Live", "Recorded (raw)", "Preprocessed")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EMG Data Studio")
        self.geometry("1300x780")
        self.minsize(1000, 620)

        self.sess = None             # dict of shared state while recording
        self.state = "idle"          # idle | connecting | recording
        self.rows_seen = 0
        self.raw_df = None
        self.pre_df = None
        self._pp_result = None
        self.live_buffers = [
            collections.deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE) for _ in range(NUM_CH)
        ]

        self._build_ui()
        self.refresh_ports()
        self._set_buttons()
        self.apply_view()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        side = ttk.Frame(self, padding=8)
        side.pack(side="left", fill="y")
        main = ttk.Frame(self, padding=(0, 8, 8, 8))
        main.pack(side="right", fill="both", expand=True)

        # ---- 1. Collect
        box = ttk.LabelFrame(side, text="1. Collect data", padding=8)
        box.pack(fill="x", pady=(0, 8))
        box.columnconfigure(1, weight=1)

        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value=str(collector.BAUDRATE))
        self.out_var = tk.StringVar(value=os.path.join(os.getcwd(), "recording.csv"))

        ttk.Label(box, text="Port").grid(row=0, column=0, sticky="w", pady=2)
        self.port_box = ttk.Combobox(box, textvariable=self.port_var, width=16)
        self.port_box.grid(row=0, column=1, sticky="ew", pady=2)
        self.refresh_btn = ttk.Button(box, text="Refresh", width=8, command=self.refresh_ports)
        self.refresh_btn.grid(row=0, column=2, padx=(4, 0))

        ttk.Label(box, text="Baud").grid(row=1, column=0, sticky="w", pady=2)
        self.baud_entry = ttk.Entry(box, textvariable=self.baud_var, width=10)
        self.baud_entry.grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(box, text="Save to").grid(row=2, column=0, sticky="w", pady=2)
        self.out_entry = ttk.Entry(box, textvariable=self.out_var, width=22)
        self.out_entry.grid(row=2, column=1, sticky="ew", pady=2)
        self.out_browse = ttk.Button(box, text="Browse", width=8, command=self.browse_out)
        self.out_browse.grid(row=2, column=2, padx=(4, 0))

        btns = ttk.Frame(box)
        btns.grid(row=3, column=0, columnspan=3, pady=(8, 2), sticky="ew")
        self.start_btn = ttk.Button(btns, text="Start", command=self.start_recording)
        self.pause_btn = ttk.Button(btns, text="Pause", command=self.toggle_pause)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.finish_recording)
        for b in (self.start_btn, self.pause_btn, self.stop_btn):
            b.pack(side="left", expand=True, fill="x", padx=2)

        self.rec_status = ttk.Label(box, text="Idle", wraplength=300, foreground="#555")
        self.rec_status.grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # ---- 2. Preprocess
        pbox = ttk.LabelFrame(side, text="2. Preprocess", padding=8)
        pbox.pack(fill="x", pady=(0, 8))
        pbox.columnconfigure(1, weight=1)

        self.pp_in_var = tk.StringVar()
        self.auto_fs = tk.BooleanVar(value=True)
        self.fs_var = tk.StringVar(value="20")
        self.low_var = tk.StringVar(value="1")
        self.high_var = tk.StringVar(value="9")
        self.notch_var = tk.StringVar(value="50")

        ttk.Label(pbox, text="Input CSV").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.pp_in_var, width=22).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(pbox, text="Browse", width=8, command=self.browse_pp_in).grid(row=0, column=2, padx=(4, 0))

        ttk.Checkbutton(pbox, text="Sampling rate from Time column", variable=self.auto_fs,
                        command=self._sync_fs_entry).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Label(pbox, text="fs (Hz)").grid(row=2, column=0, sticky="w", pady=2)
        self.fs_entry = ttk.Entry(pbox, textvariable=self.fs_var, width=10)
        self.fs_entry.grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(pbox, text="Low cut (Hz)").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.low_var, width=10).grid(row=3, column=1, sticky="w", pady=2)
        ttk.Label(pbox, text="High cut (Hz)").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.high_var, width=10).grid(row=4, column=1, sticky="w", pady=2)
        ttk.Label(pbox, text="Notch (Hz)").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.notch_var, width=10).grid(row=5, column=1, sticky="w", pady=2)

        self.pp_btn = ttk.Button(pbox, text="Run preprocessing", command=self.run_preprocess)
        self.pp_btn.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        self.pp_status = ttk.Label(pbox, text="", wraplength=300, foreground="#555")
        self.pp_status.grid(row=7, column=0, columnspan=3, sticky="w")
        self._sync_fs_entry()

        # ---- 3. Graph controls
        gbox = ttk.LabelFrame(side, text="3. Graph", padding=8)
        gbox.pack(fill="x")

        self.view_var = tk.StringVar(value=VIEWS[0])
        for v in VIEWS:
            ttk.Radiobutton(gbox, text=v, value=v, variable=self.view_var,
                            command=self.apply_view).pack(anchor="w")

        self.show_graph = tk.BooleanVar(value=True)
        ttk.Checkbutton(gbox, text="Show graph", variable=self.show_graph,
                        command=self.toggle_graph).pack(anchor="w", pady=(6, 2))

        ttk.Label(gbox, text="Channels").pack(anchor="w", pady=(4, 0))
        chf = ttk.Frame(gbox)
        chf.pack(fill="x")
        self.ch_vars = []
        for i in range(NUM_CH):
            var = tk.BooleanVar(value=True)
            self.ch_vars.append(var)
            ttk.Checkbutton(chf, text=f"V{i+1}", variable=var,
                            command=self.update_visibility).grid(row=i // 4, column=i % 4, sticky="w", padx=2)
        allnone = ttk.Frame(gbox)
        allnone.pack(fill="x", pady=(4, 0))
        ttk.Button(allnone, text="All", width=6, command=lambda: self.set_all_channels(True)).pack(side="left")
        ttk.Button(allnone, text="None", width=6, command=lambda: self.set_all_channels(False)).pack(side="left", padx=4)

        # ---- Plot area
        self.plot_holder = ttk.Frame(main)
        self.plot_holder.pack(fill="both", expand=True)
        self.placeholder = ttk.Label(main, text="Graph hidden", foreground="#777")

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.lines = [self.ax.plot([], [], label=f"V{i+1}", linewidth=1)[0] for i in range(NUM_CH)]
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_holder)
        tb_frame = ttk.Frame(self.plot_holder)
        tb_frame.pack(side="bottom", fill="x")
        NavigationToolbar2Tk(self.canvas, tb_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    # ------------------------------------------------------------- helpers
    def _sync_fs_entry(self):
        self.fs_entry.config(state="disabled" if self.auto_fs.get() else "normal")

    def refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_box["values"] = ports
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])

    def browse_out(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                         initialfile=os.path.basename(self.out_var.get()))
        if p:
            self.out_var.set(p)

    def browse_pp_in(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if p:
            self.pp_in_var.set(p)

    def _set_buttons(self):
        idle = self.state == "idle"
        rec = self.state == "recording"
        self.start_btn.config(state="normal" if idle else "disabled")
        self.pause_btn.config(state="normal" if rec else "disabled")
        self.stop_btn.config(state="normal" if self.state in ("connecting", "recording") else "disabled")
        for w in (self.port_box, self.baud_entry, self.out_entry, self.out_browse, self.refresh_btn):
            w.config(state="normal" if idle else "disabled")

    def set_all_channels(self, value):
        for v in self.ch_vars:
            v.set(value)
        self.update_visibility()

    # ------------------------------------------------------------ recording
    def start_recording(self):
        port = self.port_var.get().strip()
        path = self.out_var.get().strip()
        try:
            baud = int(self.baud_var.get())
        except ValueError:
            messagebox.showerror("Invalid baud rate", "Baud rate must be a whole number.")
            return
        if not port:
            messagebox.showerror("No port", "Choose or type a serial port first.")
            return
        if not path:
            messagebox.showerror("No file", "Choose where to save the CSV.")
            return
        if os.path.exists(path) and not messagebox.askyesno(
                "Overwrite?", f"{os.path.basename(path)} already exists. Overwrite it?"):
            return

        s = dict(
            path=path,
            paused=mp.Value('b', 0),
            pause_started_at=mp.Value('d', 0.0),
            total_paused=mp.Value('d', 0.0),
            shared=mp.Array('i', 8),
            new_flag=mp.Value('b', 0),
            stop=mp.Event(),
            ready=mp.Event(),
            errors=mp.Queue(),
            plot_q=mp.Queue(maxsize=5000),
            reader=None,
            writer=None,
        )
        s['reader'] = mp.Process(
            target=collector.serial_reader,
            args=(port, baud, s['shared'], s['new_flag'], s['paused'],
                  s['stop'], s['ready'], s['errors']),
            daemon=True,
        )
        s['reader'].start()

        self.sess = s
        self.state = "connecting"
        self.rows_seen = 0
        self.rec_status.config(text=f"Connecting to {port}...")
        self._set_buttons()
        self.after(50, self._poll_connect, s)

    def _poll_connect(self, s):
        if self.sess is not s:
            return
        if s['ready'].is_set():
            start_time = time.perf_counter()
            s['writer'] = mp.Process(
                target=collector.writer_loop,
                args=(s['path'], s['shared'], s['new_flag'], s['paused'], s['stop'],
                      start_time, (s['paused'], s['pause_started_at'], s['total_paused']),
                      s['plot_q']),
                daemon=True,
            )
            s['writer'].start()

            for buf in self.live_buffers:
                buf.extend([0.0] * WINDOW_SIZE)
            self.state = "recording"
            self.rec_status.config(text="Recording...")
            self.view_var.set(VIEWS[0])
            self.apply_view()
            self._set_buttons()
            self.after(UI_POLL_MS, self._poll_live, s)
        elif s['stop'].is_set():
            try:
                msg = s['errors'].get(timeout=0.5)
            except queue.Empty:
                msg = "Could not open the serial port."
            self.finish_recording(error=msg)
        else:
            self.after(50, self._poll_connect, s)

    def _poll_live(self, s):
        if self.sess is not s:
            return

        got = False
        for _ in range(2000):
            try:
                _, values = s['plot_q'].get_nowait()
            except queue.Empty:
                break
            for i in range(NUM_CH):
                self.live_buffers[i].append(float(values[i]))
            self.rows_seen += 1
            got = True

        if s['stop'].is_set():  # reader hit a serial error
            try:
                msg = s['errors'].get(timeout=0.5)
            except queue.Empty:
                msg = "Recording stopped unexpectedly."
            self.finish_recording(error=msg)
            return

        if got:
            label = "PAUSED" if s['paused'].value else "Recording"
            self.rec_status.config(text=f"{label} - {self.rows_seen} samples")
            if self.view_var.get() == VIEWS[0] and self.show_graph.get():
                for i, ln in enumerate(self.lines):
                    ln.set_ydata(list(self.live_buffers[i]))
                self.canvas.draw_idle()

        self.after(UI_POLL_MS, self._poll_live, s)

    def toggle_pause(self):
        s = self.sess
        if not s or self.state != "recording":
            return
        if not s['paused'].value:
            s['paused'].value = 1
            s['pause_started_at'].value = time.perf_counter()
            self.pause_btn.config(text="Resume")
            self.rec_status.config(text=f"PAUSED - {self.rows_seen} samples")
        else:
            s['total_paused'].value += time.perf_counter() - s['pause_started_at'].value
            s['pause_started_at'].value = 0.0
            s['paused'].value = 0
            self.pause_btn.config(text="Pause")

    def finish_recording(self, error=None):
        s = self.sess
        if s is None:
            return
        self.sess = None          # ends the polling loops
        s['stop'].set()
        wrote_file = s['writer'] is not None
        for key in ("writer", "reader"):
            p = s[key]
            if p is not None:
                p.join(timeout=2)
                if p.is_alive():
                    p.terminate()

        self.state = "idle"
        self.pause_btn.config(text="Pause")
        self._set_buttons()

        if error:
            self.rec_status.config(text=f"Error: {error}")
            messagebox.showerror("Serial error", error)

        if wrote_file and os.path.exists(s['path']):
            df = pp.load_data(s['path'])
            if df is not None and len(df):
                self.raw_df = df
                self.pp_in_var.set(s['path'])
                self.rec_status.config(text=f"Saved {len(df)} rows to {s['path']}")
                self.view_var.set(VIEWS[1])
                self.apply_view()
                return
        if not error:
            self.rec_status.config(text="Stopped (no data recorded).")

    # --------------------------------------------------------- preprocessing
    def run_preprocess(self):
        in_path = self.pp_in_var.get().strip()
        if not in_path or not os.path.exists(in_path):
            messagebox.showerror("No input", "Choose an existing CSV to preprocess.")
            return
        try:
            low = float(self.low_var.get())
            high = float(self.high_var.get())
            notch = float(self.notch_var.get())
            fs = None if self.auto_fs.get() else float(self.fs_var.get())
        except ValueError:
            messagebox.showerror("Invalid value", "fs, cutoffs and notch must be numbers.")
            return

        self.pp_btn.config(state="disabled")
        self.pp_status.config(text="Working...")
        self._pp_result = None
        threading.Thread(target=self._pp_worker, args=(in_path, fs, low, high, notch),
                         daemon=True).start()
        self.after(100, self._poll_pp)

    def _pp_worker(self, in_path, fs, low, high, notch):
        try:
            raw = pp.load_data(in_path)
            if raw is None or raw.empty:
                raise ValueError("Could not read any data from that CSV.")
            if fs is None:
                if 'Time' not in raw.columns:
                    raise ValueError("No 'Time' column - untick the auto option and enter fs.")
                dt = np.median(np.diff(raw['Time'].to_numpy(dtype=float)))
                if not dt > 0:
                    raise ValueError("Could not estimate fs from the Time column.")
                fs = 1000.0 / dt   # recorder writes Time in milliseconds
            out = pp.preprocess_pipeline(raw, fs, low, high, notch)
            out_path = os.path.splitext(in_path)[0] + "_preprocessed.csv"
            out.to_csv(out_path, index=False)
            self._pp_result = ("ok", raw, out, fs, out_path)
        except Exception as e:  # shown to the user in the GUI
            self._pp_result = ("error", str(e))

    def _poll_pp(self):
        if self._pp_result is None:
            self.after(100, self._poll_pp)
            return
        self.pp_btn.config(state="normal")
        res, self._pp_result = self._pp_result, None
        if res[0] == "error":
            self.pp_status.config(text=f"Error: {res[1]}")
            messagebox.showerror("Preprocessing failed", res[1])
            return
        _, raw, out, fs, out_path = res
        self.raw_df, self.pre_df = raw, out
        self.pp_status.config(text=f"Done (fs={fs:.2f} Hz). {len(out)} rows saved to {out_path}")
        self.view_var.set(VIEWS[2])
        self.apply_view()

    # ------------------------------------------------------------------ plot
    def apply_view(self):
        mode = self.view_var.get()
        ax = self.ax

        if mode == VIEWS[0]:
            x = np.arange(WINDOW_SIZE)
            for i, ln in enumerate(self.lines):
                ln.set_data(x, list(self.live_buffers[i]))
                ln.set_label(f"V{i+1}")
            ax.set_xlim(0, WINDOW_SIZE - 1)
            ax.set_ylim(*LIVE_YLIM)
            ax.set_xlabel("Latest samples")
            ax.set_ylabel("ADC value")
            ax.set_title("Live EMG")
        else:
            df = self.raw_df if mode == VIEWS[1] else self.pre_df
            cols = []
            if df is not None and len(df):
                cols = [c for c in df.columns if c not in ("Time", "Gesture")][:NUM_CH]
                step = max(1, len(df) // MAX_STATIC_POINTS)
                sub = df.iloc[::step]
                x = sub["Time"].to_numpy(dtype=float) if "Time" in sub.columns else sub.index.to_numpy()
            for i, ln in enumerate(self.lines):
                if i < len(cols):
                    ln.set_data(x, sub[cols[i]].to_numpy(dtype=float))
                    ln.set_label(str(cols[i]))
                else:
                    ln.set_data([], [])
            ax.set_xlabel("Time (ms)")
            if mode == VIEWS[1]:
                ax.set_ylabel("ADC value")
                ax.set_title("Recorded (raw)" if cols else "Recorded (raw) - nothing loaded yet")
            else:
                ax.set_ylabel("Standardised value (z-score)")
                ax.set_title("Preprocessed" if cols else "Preprocessed - run step 2 first")

        self.update_visibility()

    def update_visibility(self):
        visible = []
        for ln, var in zip(self.lines, self.ch_vars):
            ln.set_visible(var.get())
            if var.get() and len(ln.get_xdata()) > 0:
                visible.append(ln)

        if visible:
            self.ax.legend(handles=visible, loc="upper left", ncol=4, fontsize=8)
        else:
            leg = self.ax.get_legend()
            if leg:
                leg.remove()

        if self.view_var.get() != VIEWS[0]:
            self.ax.relim(visible_only=True)
            self.ax.autoscale_view()
        self.canvas.draw_idle()

    def toggle_graph(self):
        if self.show_graph.get():
            self.placeholder.pack_forget()
            self.plot_holder.pack(fill="both", expand=True)
            self.update_visibility()
        else:
            self.plot_holder.pack_forget()
            self.placeholder.pack(expand=True)

    # ------------------------------------------------------------------ exit
    def on_close(self):
        if self.sess is not None:
            self.finish_recording()
        self.destroy()


if __name__ == "__main__":
    mp.freeze_support()   # needed on Windows / frozen builds
    App().mainloop()
