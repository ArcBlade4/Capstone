"""
main.py

One window that:
  1. collects data from the serial port into a CSV (with pause / resume / stop).
     The values saved to the CSV are the RAW readings (preprocess them in the Preprocess tab).
  2. preprocesses a (raw) CSV with Preprocessing.py
  3. shows a graph you can toggle: view (Live / Recorded raw / Preprocessed),
     individual channels V1-V8, and the whole graph on/off

Run:  python gui.py
Needs: pyserial pandas numpy scipy matplotlib   (tkinter ships with Python)
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

import classifier
import collector
import Preprocessing as pp

WINDOW_SIZE = 250            # samples shown in the live view
MAX_STATIC_POINTS = 20000    # downsample long recordings for plotting only
LIVE_YLIM = (-50, 4500)
NUM_CH = 8
UI_POLL_MS = 60
LIVE_RAW = "Live (raw)"
REC_RAW = "Recorded (raw)"
PRE_FILE = "Preprocessed (file)"
VIEWS = (LIVE_RAW, REC_RAW, PRE_FILE)
LIVE_VIEWS = (LIVE_RAW,)


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
        self._pp_progress = ""
        self.model = None            # loaded classifier (classifier.ModelWrapper)
        self._model_path = None
        self._cls_result = None
        self._cls_progress = ""
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

        # ---- Record / Preprocess: one section, swapped with tabs
        self.tabs = ttk.Notebook(side)
        self.tabs.pack(fill="x", pady=(0, 8))

        # ---- Record tab
        box = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(box, text="Record data")
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
    
        # ---- Preprocess tab (settings used for BOTH live recording and the offline button)
        pbox = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(pbox, text="Preprocess")
        pbox.columnconfigure(1, weight=1)

        self.pp_in_var = tk.StringVar()
        self.auto_fs = tk.BooleanVar(value=False)
        self.fs_var = tk.StringVar(value=str(pp.SAMPLE_RATE))   # rate the model was trained at
        self.low_var = tk.StringVar(value="1")
        self.high_var = tk.StringVar(value="8")
        self.notch_var = tk.StringVar(value="0")
        self.scaler_var = tk.StringVar(value=os.path.join(os.getcwd(), "scaler.pkl"))

        ttk.Label(pbox, text="Input").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.pp_in_var, width=22).grid(row=0, column=1, sticky="ew", pady=2)
        inb = ttk.Frame(pbox)
        inb.grid(row=0, column=2, padx=(4, 0))
        ttk.Button(inb, text="File", width=6, command=self.browse_pp_in).pack(side="left")
        ttk.Button(inb, text="Folder", width=6, command=self.browse_pp_folder).pack(side="left", padx=(2, 0))

        ttk.Checkbutton(pbox, text="Measure sampling rate from file (offline only)", variable=self.auto_fs,
                        command=self._sync_fs_entry).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Label(pbox, text="fs (Hz, resample to)").grid(row=2, column=0, sticky="w", pady=2)
        self.fs_entry = ttk.Entry(pbox, textvariable=self.fs_var, width=10)
        self.fs_entry.grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(pbox, text="Low cut (Hz)").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.low_var, width=10).grid(row=3, column=1, sticky="w", pady=2)
        ttk.Label(pbox, text="High cut (Hz)").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.high_var, width=10).grid(row=4, column=1, sticky="w", pady=2)
        ttk.Label(pbox, text="Notch (Hz, 0=off)").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.notch_var, width=10).grid(row=5, column=1, sticky="w", pady=2)
        ttk.Label(pbox, text="Scaler file").grid(row=6, column=0, sticky="w", pady=2)
        ttk.Entry(pbox, textvariable=self.scaler_var, width=22).grid(row=6, column=1, sticky="ew", pady=2)
        ttk.Button(pbox, text="Browse", width=8, command=self.browse_scaler).grid(row=6, column=2, padx=(4, 0))

        self.pp_btn = ttk.Button(pbox, text="Run preprocessing", command=self.run_preprocess)
        self.pp_btn.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        self.pp_status = ttk.Label(pbox, text="", wraplength=300, foreground="#555")
        self.pp_status.grid(row=8, column=0, columnspan=3, sticky="w")
        self._sync_fs_entry()

        # ---- Classify tab: feed preprocessed data to an ML model
        cbox = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(cbox, text="Classify")
        cbox.columnconfigure(1, weight=1)

        self.model_var = tk.StringVar()
        self.cls_src = tk.StringVar(value="current")
        self.cls_in_var = tk.StringVar()
        self.cls_raw = tk.BooleanVar(value=False)

        ttk.Label(cbox, text="Model").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(cbox, textvariable=self.model_var, width=22).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(cbox, text="Browse", width=8, command=self.browse_model).grid(row=0, column=2, padx=(4, 0))

        ttk.Label(cbox, text="Data").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Radiobutton(cbox, text="Current preprocessed data (graph)", value="current",
                        variable=self.cls_src).grid(row=1, column=1, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Radiobutton(cbox, text="CSV file:", value="file",
                        variable=self.cls_src).grid(row=2, column=1, sticky="w")
        ttk.Entry(cbox, textvariable=self.cls_in_var, width=22).grid(row=3, column=1, sticky="ew", pady=2)
        ttk.Button(cbox, text="Browse", width=8, command=self.browse_cls_in).grid(row=3, column=2, padx=(4, 0))
        ttk.Checkbutton(cbox, text="File is raw: preprocess it first (uses Preprocess tab settings + scaler)",
                        variable=self.cls_raw).grid(row=4, column=0, columnspan=3, sticky="w")

        self.cls_btn = ttk.Button(cbox, text="Classify", command=self.run_classify)
        self.cls_btn.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        self.cls_status = ttk.Label(cbox, text="", wraplength=300, foreground="#555")
        self.cls_status.grid(row=6, column=0, columnspan=3, sticky="w")

        cols = ("time", "prediction", "conf", "truth")
        self.cls_tree = ttk.Treeview(cbox, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (60, 80, 50, 70)):
            self.cls_tree.heading(c, text=c)
            self.cls_tree.column(c, width=w, anchor="center")
        self.cls_tree.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        # ---- 3. Graph controls
        gbox = ttk.LabelFrame(side, text="Graph", padding=8)
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

    def browse_pp_folder(self):
        p = filedialog.askdirectory(title="Folder with the recordings (subfolders are included)")
        if p:
            self.pp_in_var.set(p)

    @staticmethod
    def _find_inputs(path):
        """A file -> [file]; a folder -> every .csv/.txt inside it (subfolders too), sorted."""
        if not os.path.isdir(path):
            return [path]
        found = []
        for root, _dirs, files in os.walk(path):
            for f in files:
                low = f.lower()
                if low.endswith((".csv", ".txt")) and not low.endswith("_preprocessed.csv"):
                    found.append(os.path.join(root, f))
        return sorted(found)

    def _set_progress(self, msg):
        self._pp_progress = msg

    def browse_scaler(self):
        p = filedialog.asksaveasfilename(defaultextension=".pkl", confirmoverwrite=False,
                                         filetypes=[("Scaler pickle", "*.pkl"), ("All files", "*.*")],
                                         initialfile=os.path.basename(self.scaler_var.get()))
        if p:
            self.scaler_var.set(p)

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
        """Parse the Preprocess-tab fields -> (fs or None, low, high, notch, scaler_path)."""
        try:
            low = float(self.low_var.get())
            high = float(self.high_var.get())
            notch = float(self.notch_var.get())
            fs = None if self.auto_fs.get() else float(self.fs_var.get())
        except ValueError:
            raise ValueError("fs, cutoffs and notch must be numbers.")
        return fs, low, high, notch, self.scaler_var.get().strip()

    @staticmethod
    def _scaler_settings(fs, low, high, notch):
        return {"fs": fs, "lowcut": low, "highcut": high, "notch": notch}

    @staticmethod
    def _settings_diff(saved, now):
        """Names of the settings that differ between a scaler file and the current fields."""
        diff = []
        for k, v in now.items():
            if k in saved and v is not None and saved[k] is not None and abs(float(saved[k]) - float(v)) > 0.5:
                diff.append(f"{k}: file {saved[k]:g}, now {v:g}")
        return diff

    # ------------------------------------------------------------ recording
    def _rec_text(self, label):
        return f"{label} - {self.rows_seen} samples"

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
            stop=mp.Event(),
            ready=mp.Event(),
            errors=mp.Queue(),
            plot_q=mp.Queue(maxsize=5000),
            proc=None,
        )
        s['proc'] = mp.Process(
            target=collector.recorder,
            args=(port, baud, path, s['paused'], s['stop'], s['ready'],
                  s['errors'], s['plot_q']),
            daemon=True,
        )
        s['proc'].start()

        self.sess = s
        self.state = "connecting"
        self.rows_seen = 0
        self.fs_used = None
        self.rec_status.config(text=f"Connecting to {port}...")
        self._set_buttons()
        self.after(50, self._poll_connect, s)

    def _poll_connect(self, s):
        if self.sess is not s:
            return
        if s['ready'].is_set():
            for buf in self.live_buffers:
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

        # the recorder process does the saving; here we only display the raw signal
        new_raw = []
        for _ in range(500):
            try:
                kind, payload = s['plot_q'].get_nowait()
            except queue.Empty:
                break
            if kind == "fs":
                self.fs_used = payload
            else:
                for _t, raw in payload:
                    new_raw.append(raw)
        got = bool(new_raw)
        if got:
            self.rows_seen += len(new_raw)
            raw_arr = np.asarray(new_raw, dtype=float)
            for i in range(NUM_CH):
                self.live_buffers[i].extend(raw_arr[:, i])

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
        for i, ln in enumerate(self.lines):
            ln.set_ydata(list(self.live_buffers[i]))
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
                # the saved file is raw: show it and queue it up for the Preprocess tab
                self.raw_df = df
                self.pp_in_var.set(s['path'])
                self.rec_status.config(text=f"Saved {len(df)} raw rows to {s['path']}")
                self.view_var.set(REC_RAW)
                self.apply_view()
                return
        if not error:
            self.rec_status.config(text="Stopped (no data recorded).")

    # --------------------------------------------------------- preprocessing
    def run_preprocess(self):
        in_path = self.pp_in_var.get().strip()
        if not in_path or not os.path.exists(in_path):
            messagebox.showerror("No input", "Choose an existing CSV file or a folder to preprocess.")
            return
        try:
            fs, low, high, notch, scaler_path = self._read_pp_settings()
        except ValueError as e:
            messagebox.showerror("Invalid value", str(e))
            return

        self.pp_btn.config(state="disabled")
        self.pp_status.config(text="Working...")
        self._pp_progress = ""
        self._pp_result = None
        threading.Thread(target=self._pp_worker, args=(in_path, fs, low, high, notch, scaler_path),
                         daemon=True).start()
        self.after(100, self._poll_pp)

    def _pp_worker(self, in_path, fs, low, high, notch, scaler_path):
        try:
            paths = self._find_inputs(in_path)
            if not paths:
                raise ValueError("No .csv or .txt files found in that folder.")

            if fs is None:                       # pool the gaps of all files, median as PREPRO.estimate_rate
                gaps = []
                for p in paths:
                    raw = pp.load_data(p)
                    if raw is None or 'Time' not in raw.columns:
                        raise ValueError(f"No 'Time' column in {os.path.basename(p)} - "
                                         "untick the auto option and enter fs.")
                    g = np.diff(raw['Time'].to_numpy(dtype=float))
                    gaps.append(g[g > 0])
                gaps = np.concatenate(gaps)
                if len(gaps) == 0:
                    raise ValueError("Could not estimate fs from the Time column.")
                fs = 1000.0 / np.median(gaps)    # Time is in ms
            now = self._scaler_settings(fs, low, high, notch)

            # use the shared scaler if it exists (so offline and live agree), else fit and save one
            scaler, note = None, ""
            if scaler_path and os.path.exists(scaler_path):
                scaler, saved = pp.load_scaler(scaler_path)
                diff = self._settings_diff(saved, now)
                if diff:
                    raise ValueError("The scaler file was made with different settings:\n  "
                                     + "\n  ".join(diff) + "\nChange the fields or choose another scaler file.")
                note = " Used existing scaler."
            fitted = scaler is None

            results, scaler = pp.preprocess_folder(paths, fs, low, high, notch or None, scaler=scaler,
                                                   progress=self._set_progress)
            if fitted and scaler_path and pp.save_scaler_if_missing(scaler_path, scaler, now):
                note = f" Saved new scaler to {scaler_path}."
            for p, df in results:
                df.to_csv(os.path.splitext(p)[0] + "_preprocessed.csv", index=False)

            first_path, first_df = results[0]
            if len(results) == 1:
                summary = f"{len(first_df)} rows saved to {os.path.splitext(first_path)[0]}_preprocessed.csv"
            else:
                total = sum(len(d) for _, d in results)
                summary = (f"{len(results)} files, {total} rows saved next to each source as "
                           f"*_preprocessed.csv (graph shows {os.path.basename(first_path)})")
            self._pp_result = ("ok", pp.load_data(first_path), first_df, fs, summary, note)
        except Exception as e:  # shown to the user in the GUI
            self._pp_result = ("error", str(e))

    def _poll_pp(self):
        if self._pp_result is None:
            if self._pp_progress:
                self.pp_status.config(text=self._pp_progress)
            self.after(100, self._poll_pp)
            return
        self.pp_btn.config(state="normal")
        res, self._pp_result = self._pp_result, None
        if res[0] == "error":
            self.pp_status.config(text=f"Error: {res[1]}")
            messagebox.showerror("Preprocessing failed", res[1])
            return
        _, raw, out, fs, summary, note = res
        self.raw_df, self.pre_df = raw, out
        self.pp_status.config(text=f"Done (fs={fs:.2f} Hz). {summary}.{note}")
        self.view_var.set(PRE_FILE)
        self.apply_view()

    # --------------------------------------------------------- classification
    def browse_model(self):
        p = filedialog.askopenfilename(filetypes=[
            ("Models", "*.pt *.pth *.keras *.h5 *.onnx *.pkl *.joblib"), ("All files", "*.*")])
        if p:
            self.model_var.set(p)

    def browse_cls_in(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if p:
            self.cls_in_var.set(p)
            self.cls_src.set("file")

    def run_classify(self):
        model_path = self.model_var.get().strip()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("No model", "Choose an existing model file.")
            return
        source = self.cls_src.get()
        in_path, df, ppset = None, None, None
        if source == "current":
            if self.pre_df is None or not len(self.pre_df):
                messagebox.showerror("No data", "There is no preprocessed data loaded yet. "
                                     "Record or preprocess something first, or pick a CSV file.")
                return
            df = self.pre_df.copy()
        else:
            in_path = self.cls_in_var.get().strip()
            if not in_path or not os.path.isfile(in_path):
                messagebox.showerror("No input", "Choose an existing CSV file.")
                return
            if self.cls_raw.get():
                try:
                    ppset = self._read_pp_settings()
                except ValueError as e:
                    messagebox.showerror("Invalid value", str(e))
                    return

        self.cls_btn.config(state="disabled")
        self.cls_status.config(text="Working...")
        self._cls_progress = ""
        self._cls_result = None
        threading.Thread(target=self._cls_worker, args=(model_path, in_path, df, ppset),
                         daemon=True).start()
        self.after(100, self._poll_cls)

    def _cls_worker(self, model_path, in_path, df, ppset):
        try:
            if self.model is None or self._model_path != model_path:
                self._cls_progress = "Loading model..."
                self.model = classifier.ModelWrapper(model_path)
                self._model_path = model_path

            if in_path is not None:
                if ppset is None:                                 # already preprocessed
                    df = pp.load_data(in_path)
                else:                                             # raw: same steps as the Preprocess tab
                    self._cls_progress = "Preprocessing..."
                    fs, low, high, notch, scaler_path = ppset
                    if not scaler_path or not os.path.exists(scaler_path):
                        raise ValueError("A scaler file is needed to preprocess raw data "
                                         "(run the Preprocess tab on training data first).")
                    scaler, saved = pp.load_scaler(scaler_path)
                    raw = pp.load_data(in_path)
                    if fs is None:
                        if raw is None or "Time" not in raw.columns:
                            raise ValueError("No 'Time' column: untick auto fs and enter fs.")
                        g = np.diff(raw["Time"].to_numpy(dtype=float))
                        fs = 1000.0 / np.median(g[g > 0])
                    diff = self._settings_diff(saved, self._scaler_settings(fs, low, high, notch))
                    if diff:
                        raise ValueError("The scaler file was made with different settings:\n  "
                                         + "\n  ".join(diff))
                    results, _ = pp.preprocess_folder([in_path], fs, low, high, notch or None, scaler=scaler)
                    df = results[0][1]
            if df is None or df.empty:
                raise ValueError("Could not read any data.")

            self._cls_progress = "Classifying..."
            res = classifier.classify_dataframe(df, self.model)
            out_path = (os.path.splitext(in_path)[0] + "_predictions.csv" if in_path
                        else os.path.join(os.getcwd(), "predictions.csv"))
            res.to_csv(out_path, index=False)
            self._cls_result = ("ok", res, out_path)
        except Exception as e:  # shown to the user in the GUI
            self._cls_result = ("error", str(e))

    def _poll_cls(self):
        if self._cls_result is None:
            if self._cls_progress:
                self.cls_status.config(text=self._cls_progress)
            self.after(100, self._poll_cls)
            return
        self.cls_btn.config(state="normal")
        res, self._cls_result = self._cls_result, None
        if res[0] == "error":
            self.cls_status.config(text=f"Error: {res[1]}")
            messagebox.showerror("Classification failed", res[1])
            return
        _, out, out_path = res

        counts = out["prediction"].value_counts()
        text = f"{len(out)} windows. " + ", ".join(f"{k}: {v}" for k, v in counts.items())
        if "truth" in out.columns:
            known = out[out["truth"].isin(pp.GESTURES)]
            if len(known):
                text += f"\nAccuracy vs. Gesture column: {(known['prediction'] == known['truth']).mean() * 100:.1f}%"
        self.cls_status.config(text=text + f"\nSaved to {out_path}")

        self.cls_tree.delete(*self.cls_tree.get_children())
        step = max(1, len(out) // 500)                            # keep the table light
        for _, r in out.iloc[::step].iterrows():
            t = f"{r['end_time_ms']:.0f}" if "end_time_ms" in out.columns else int(r["start"])
            conf = "" if np.isnan(r["confidence"]) else f"{r['confidence']:.2f}"
            self.cls_tree.insert("", "end", values=(t, r["prediction"], conf, r.get("truth", "")))

    # ------------------------------------------------------------------ plot
    def apply_view(self):
        mode = self.view_var.get()
        ax = self.ax

        if mode in LIVE_VIEWS:
            x = np.arange(WINDOW_SIZE)
            for i, ln in enumerate(self.lines):
                ln.set_data(x, list(self.live_buffers[i]))
                ln.set_label(f"V{i+1}")
            ax.set_xlim(0, WINDOW_SIZE - 1)
            ax.set_ylim(*LIVE_YLIM)
            ax.set_xlabel("Latest samples")
            ax.set_ylabel("ADC value")
            ax.set_title("Live EMG (raw)")
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
