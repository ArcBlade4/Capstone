"""
gui.py - EMG Data Studio

One window that:
  1. collects data from the serial port into a CSV (with pause / resume / stop).
     The values saved to the CSV are the live-PREPROCESSED values.
  2. preprocesses a (raw) CSV with Preprocessing.py
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
LIVE_RAW = "Live (raw)"
LIVE_PRE = "Live (preprocessed)"
REC_RAW = "Recorded (raw)"
PRE_FILE = "Preprocessed (file)"
VIEWS = (LIVE_RAW, LIVE_PRE, REC_RAW, PRE_FILE)
LIVE_VIEWS = (LIVE_RAW, LIVE_PRE)
LIVE_PRE_YLIM = (-4, 4)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EMG Data Studio")
        self.geometry("1300x800")
        self.minsize(1000, 640)

        self.sess = None             # dict of shared state while recording
        self.state = "idle"          # idle | connecting | recording
        self.rows_seen = 0
        self.fs_used = None          # sampling rate used by the live preprocessor
        self.raw_df = None
        self.pre_df = None
        self._pp_result = None
        self.live_buffers = [
            collections.deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE) for _ in range(NUM_CH)
        ]
        self.live_pre_buffers = [
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

        # ---- 2. Preprocess (settings used for BOTH live recording and the offline button)
        pbox = ttk.LabelFrame(side, text="2. Preprocess", padding=8)
        pbox.pack(fill="x", pady=(0, 8))
        pbox.columnconfigure(1, weight=1)

        self.pp_in_var = tk.StringVar()
        self.auto_fs = tk.BooleanVar(value=True)
        self.fs_var = tk.StringVar(value="38")
        self.low_var = tk.StringVar(value="1")
        self.high_var = tk.StringVar(value="9")
        self.notch_var = tk.StringVar(value="0")
        self.env_var = tk.StringVar(value="4")

        ttk.Label(pbox, text="Input CSV").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.pp_in_var, width=22).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(pbox, text="Browse", width=8, command=self.browse_pp_in).grid(row=0, column=2, padx=(4, 0))

        ttk.Checkbutton(pbox, text="Measure sampling rate automatically", variable=self.auto_fs,
                        command=self._sync_fs_entry).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Label(pbox, text="fs (Hz)").grid(row=2, column=0, sticky="w", pady=2)
        self.fs_entry = ttk.Entry(pbox, textvariable=self.fs_var, width=10)
        self.fs_entry.grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(pbox, text="Low cut (Hz)").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.low_var, width=10).grid(row=3, column=1, sticky="w", pady=2)
        ttk.Label(pbox, text="High cut (Hz)").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.high_var, width=10).grid(row=4, column=1, sticky="w", pady=2)
        ttk.Label(pbox, text="Notch (Hz, 0=off)").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.notch_var, width=10).grid(row=5, column=1, sticky="w", pady=2)
        ttk.Label(pbox, text="Envelope (Hz, 0=off)").grid(row=6, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.env_var, width=10).grid(row=6, column=1, sticky="w", pady=2)

        self.pp_btn = ttk.Button(pbox, text="Run preprocessing on file", command=self.run_preprocess)
        self.pp_btn.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        self.pp_status = ttk.Label(pbox, text="", wraplength=300, foreground="#555")
        self.pp_status.grid(row=8, column=0, columnspan=3, sticky="w")
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

    def _read_pp_settings(self):
        """Parse the step-2 fields -> (fs or None, low, high, notch, envelope or None)."""
        try:
            low = float(self.low_var.get())
            high = float(self.high_var.get())
            notch = float(self.notch_var.get())
            env = float(self.env_var.get())
            fs = None if self.auto_fs.get() else float(self.fs_var.get())
        except ValueError:
            raise ValueError("fs, cutoffs, notch and envelope must be numbers.")
        return fs, low, high, notch, (env if env > 0 else None)

    # ------------------------------------------------------------ recording
    def _live_pp_params(self):
        """Parse + validate the settings for the recording process."""
        fs, low, high, notch, env = self._read_pp_settings()
        if fs is not None:   # with auto fs the rate isn't known until recording has started
            pp.StreamingPreprocessor(fs, low, high, notch, n_channels=NUM_CH,
                                     envelope_hz=env)   # raises ValueError if invalid
        return (fs, low, high, notch, env)

    def _rec_text(self, label):
        if self.fs_used:
            return f"{label} - {self.rows_seen} samples (fs {self.fs_used:.1f} Hz)"
        return f"{label} - {self.rows_seen} samples (measuring sampling rate...)"

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

        try:
            pp_params = self._live_pp_params()
        except ValueError as e:
            messagebox.showerror("Preprocessing settings", str(e))
            return

        s = dict(
            path=path,
            paused=mp.Value('b', 0),
            stop=mp.Event(),
            ready=mp.Event(),
            errors=mp.Queue(),
            plot_q=mp.Queue(maxsize=5000),
            proc=None,
        )
        s['proc'] = mp.Process(
            target=collector.recorder,
            args=(port, baud, path, pp_params, s['paused'], s['stop'], s['ready'],
                  s['errors'], s['plot_q']),
            daemon=True,
        )
        s['proc'].start()

        self.sess = s
        self.state = "connecting"
        self.rows_seen = 0
        self.fs_used = pp_params[0]          # None until measured in auto mode
        self.rec_status.config(text=f"Connecting to {port}...")
        self._set_buttons()
        self.after(50, self._poll_connect, s)

    def _poll_connect(self, s):
        if self.sess is not s:
            return
        if s['ready'].is_set():
            for buf in self.live_buffers + self.live_pre_buffers:
                buf.extend([0.0] * WINDOW_SIZE)
            self.state = "recording"
            self.rec_status.config(text="Recording...")
            if self.view_var.get() not in LIVE_VIEWS:
                self.view_var.set(LIVE_RAW)
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

        # the recorder process does the preprocessing; here we only display it
        new_raw, new_pre = [], []
        for _ in range(500):
            try:
                kind, payload = s['plot_q'].get_nowait()
            except queue.Empty:
                break
            if kind == "fs":
                self.fs_used = payload
            else:
                for _t, raw, pre in payload:
                    new_raw.append(raw)
                    new_pre.append(pre)
        got = bool(new_raw)
        if got:
            self.rows_seen += len(new_raw)
            raw_arr = np.asarray(new_raw, dtype=float)
            pre_arr = np.asarray(new_pre, dtype=float)
            for i in range(NUM_CH):
                self.live_buffers[i].extend(raw_arr[:, i])
                self.live_pre_buffers[i].extend(pre_arr[:, i])

        if s['stop'].is_set():  # recorder hit an error
            try:
                msg = s['errors'].get(timeout=0.5)
            except queue.Empty:
                msg = "Recording stopped unexpectedly."
            self.finish_recording(error=msg)
            return

        if got:
            self.rec_status.config(text=self._rec_text("PAUSED" if s['paused'].value else "Recording"))
            if self.view_var.get() in LIVE_VIEWS and self.show_graph.get():
                self._draw_live()

        self.after(UI_POLL_MS, self._poll_live, s)

    def _draw_live(self):
        bufs = self.live_pre_buffers if self.view_var.get() == LIVE_PRE else self.live_buffers
        for i, ln in enumerate(self.lines):
            ln.set_ydata(list(bufs[i]))
        self.canvas.draw_idle()

    def toggle_pause(self):
        s = self.sess
        if not s or self.state != "recording":
            return
        # the recorder handles the time bookkeeping and filter reset itself
        if not s['paused'].value:
            s['paused'].value = 1
            self.pause_btn.config(text="Resume")
            self.rec_status.config(text=self._rec_text("PAUSED"))
        else:
            s['paused'].value = 0
            self.pause_btn.config(text="Pause")

    def finish_recording(self, error=None):
        s = self.sess
        if s is None:
            return
        self.sess = None          # ends the polling loops
        s['stop'].set()
        was_started = s['ready'].is_set()
        p = s['proc']
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()

        self.state = "idle"
        self.pause_btn.config(text="Pause")
        self._set_buttons()

        if error:
            self.rec_status.config(text=f"Error: {error}")
            messagebox.showerror("Recording error", error)

        if was_started and os.path.exists(s['path']):
            df = pp.load_data(s['path'])
            if df is not None and len(df):
                # the saved file is already preprocessed
                self.pre_df = df
                self.rec_status.config(text=f"Saved {len(df)} preprocessed rows to {s['path']}")
                self.view_var.set(PRE_FILE)
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
            fs, low, high, notch, env = self._read_pp_settings()
        except ValueError as e:
            messagebox.showerror("Invalid value", str(e))
            return

        self.pp_btn.config(state="disabled")
        self.pp_status.config(text="Working...")
        self._pp_result = None
        threading.Thread(target=self._pp_worker, args=(in_path, fs, low, high, notch, env),
                         daemon=True).start()
        self.after(100, self._poll_pp)

    def _pp_worker(self, in_path, fs, low, high, notch, env):
        try:
            raw = pp.load_data(in_path)
            if raw is None or raw.empty:
                raise ValueError("Could not read any data from that CSV.")
            if fs is None:
                if 'Time' not in raw.columns:
                    raise ValueError("No 'Time' column - untick the auto option and enter fs.")
                t = raw['Time'].to_numpy(dtype=float)
                span = t[-1] - t[0]
                if len(t) < 2 or not span > 0:
                    raise ValueError("Could not estimate fs from the Time column.")
                fs = 1000.0 * (len(t) - 1) / span   # Time is in ms; mean rate over the whole file
            out = pp.preprocess_pipeline(raw, fs, low, high, notch, envelope_hz=env)
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
        self.view_var.set(PRE_FILE)
        self.apply_view()

    # ------------------------------------------------------------------ plot
    def apply_view(self):
        mode = self.view_var.get()
        ax = self.ax

        if mode in LIVE_VIEWS:
            pre = mode == LIVE_PRE
            bufs = self.live_pre_buffers if pre else self.live_buffers
            x = np.arange(WINDOW_SIZE)
            for i, ln in enumerate(self.lines):
                ln.set_data(x, list(bufs[i]))
                ln.set_label(f"V{i+1}")
            ax.set_xlim(0, WINDOW_SIZE - 1)
            ax.set_ylim(*(LIVE_PRE_YLIM if pre else LIVE_YLIM))
            ax.set_xlabel("Latest samples")
            ax.set_ylabel("Standardised value (clipped to +/-3)" if pre else "ADC value")
            ax.set_title("Live EMG (preprocessed)" if pre else "Live EMG (raw)")
        else:
            df = self.raw_df if mode == REC_RAW else self.pre_df
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
            if mode == REC_RAW:
                ax.set_ylabel("ADC value")
                ax.set_title("Recorded (raw)" if cols else "Recorded (raw) - nothing loaded yet")
            else:
                ax.set_ylabel("Standardised value (z-score)")
                ax.set_title("Preprocessed" if cols else "Preprocessed - nothing loaded yet")

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

        if self.view_var.get() not in LIVE_VIEWS:
            # set_xlim/set_ylim (used by the Live view) switch autoscaling off,
            # so it has to be turned back on before rescaling to the data
            self.ax.set_autoscale_on(True)
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