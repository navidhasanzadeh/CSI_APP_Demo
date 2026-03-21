"""
D455 RGB + Depth viewer with:
- Tkinter UI (video on left, status panel + controls on right)
- One-shot NTP sync at startup (server UTC time), with Re-sync button and server selection
- Threading (separate threads as requested):
    1) Capture thread: wait_for_frames + align depth->color, dispatch to queues
    2) RGB thread: converts RGB frame -> numpy, updates UI latest, contributes to recording pairing
    3) Depth thread: converts depth -> numpy, makes depth visualization for UI, contributes to recording pairing
    4) Time thread: updates time values for UI
    5) RecordWorker thread: writes videos/CSVs/optional NPZ bundles asynchronously
NEW (performance):
- UI only converts/updates images when a NEW frame arrives (avoids repeating expensive conversions).
- UI refresh rate is decoupled from camera FPS (default 30 UI updates/sec) to reduce CPU load.
- Recording path reduced copying: we no longer copy/store depth_vis for recording; the writer thread
  generates depth visualization from raw depth for the depth video. This reduces memory bandwidth.
- Record pairing cache is bounded/purged to avoid growth if frames drop under load.
- NPZ saving uses np.savez (uncompressed) for speed. If you need smaller files, switch back to np.savez_compressed.
Dependencies:
    pip install numpy opencv-python pillow pyrealsense2
"""
import threading
import time
from collections import OrderedDict
from queue import Queue

import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import depth_camera as dc

# UI update rate: keep UI lower than camera FPS to avoid CPU bottleneck
ui_update_ms = 100  # ~30 Hz UI


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("D455 Viewer + Server UTC + Recording")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # RealSense
        self.pipeline, self.align, self.chosen_color, self.chosen_depth = dc.start_pipeline_with_fallback(
            dc.preferred_color, dc.preferred_depth
        )

        # NTP sync
        self.sync_state = dc.SyncState()
        dc.start_time_sync_thread(
            state=self.sync_state,
            preferred_server=dc.preferred_time_server,
            fallback_servers=dc.fallback_servers,
            enable_fallback=dc.enable_fallback_servers,
            attempts_per_server=dc.ntp_attempts_per_server,
            timeout_s=dc.ntp_timeout_s,
        )

        # Recorder + writer
        self.recorder = dc.Recorder()
        self.record_queue: Queue = Queue(maxsize=dc.record_queue_maxsize)
        self.record_worker = dc.RecordWorker(self.recorder, self.record_queue)
        self.record_worker.start()

        # Pairing map (OrderedDict for bounded LRU-ish behavior)
        self.record_pair_lock = threading.Lock()
        self.record_pairs: "OrderedDict[int, dc.PartialForRecord]" = OrderedDict()

        # Latest for UI
        self.latest_lock = threading.Lock()
        self.latest = dc.LatestFrame()

        # Queues for RGB and Depth
        self.rgb_q: Queue = Queue(maxsize=dc.rgb_queue_maxsize)
        self.depth_q: Queue = Queue(maxsize=dc.depth_queue_maxsize)

        # Workers
        self.capture_worker = dc.CaptureWorker(
            pipeline=self.pipeline,
            align=self.align,
            sync_state=self.sync_state,
            rgb_q=self.rgb_q,
            depth_q=self.depth_q,
            latest_lock=self.latest_lock,
            latest=self.latest,
        )
        self.rgb_worker = dc.RGBWorker(
            rgb_q=self.rgb_q,
            latest_lock=self.latest_lock,
            latest=self.latest,
            recorder=self.recorder,
            record_pair_lock=self.record_pair_lock,
            record_pairs=self.record_pairs,
            record_q=self.record_queue,
        )
        self.depth_worker = dc.DepthWorker(
            depth_q=self.depth_q,
            latest_lock=self.latest_lock,
            latest=self.latest,
            recorder=self.recorder,
            record_pair_lock=self.record_pair_lock,
            record_pairs=self.record_pairs,
            record_q=self.record_queue,
        )
        self.capture_worker.start()
        self.rgb_worker.start()
        self.depth_worker.start()

        # Time thread
        self.time_lock = threading.Lock()
        self.time_state = dc.TimeState()
        self.time_stop_event = threading.Event()
        self.time_worker = dc.TimeWorker(self.sync_state, self.time_lock, self.time_state, self.time_stop_event)
        self.time_worker.start()

        # UI state
        self.running = True
        self.rgb_photo = None
        self.depth_photo = None
        self._last_seq_rendered = -1  # only re-render on new frames
        self._build_ui()
        self._ui_tick()

    def _build_ui(self):
        self.root.geometry("1280x680")
        self.main = ttk.Frame(self.root, padding=8)
        self.main.pack(fill="both", expand=True)
        self.main.columnconfigure(0, weight=1)
        self.main.columnconfigure(1, weight=0)
        self.main.rowconfigure(0, weight=1)

        # Left: video
        self.video_frame = ttk.Frame(self.main)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.video_frame.columnconfigure(0, weight=1)
        self.video_frame.columnconfigure(1, weight=1)
        self.video_frame.rowconfigure(0, weight=1)

        self.rgb_label = ttk.Label(self.video_frame, text="RGB")
        self.rgb_label.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.depth_label = ttk.Label(self.video_frame, text="Depth")
        self.depth_label.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        # Right: panel
        self.side = ttk.Frame(self.main, width=360)
        self.side.grid(row=0, column=1, sticky="nsew")
        self.side.columnconfigure(0, weight=1)
        ttk.Label(self.side, text="Status", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))

        # REC indicator
        self.rec_row = ttk.Frame(self.side)
        self.rec_row.grid(row=1, column=0, sticky="w", pady=(0, 6))
        self.rec_canvas = tk.Canvas(self.rec_row, width=14, height=14, highlightthickness=0)
        self.rec_canvas.grid(row=0, column=0, sticky="w")
        self.rec_canvas.create_oval(2, 2, 12, 12, fill="red", outline="red")
        self.rec_text = ttk.Label(self.rec_row, text="REC", font=("Segoe UI", 10, "bold"))
        self.rec_text.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.rec_row.grid_remove()

        self.var_fps = tk.StringVar(value="FPS: -")
        self.var_unix = tk.StringVar(value="Unix time: -")
        self.var_utc = tk.StringVar(value="UTC (server): syncing…")
        self.var_ntp = tk.StringVar(value="NTP: syncing…")
        self.var_streams = tk.StringVar(value="Streams: -")
        self.var_recording = tk.StringVar(value="Recording: OFF")
        self.var_session = tk.StringVar(value="Session: -")
        self.var_queue = tk.StringVar(value="Record queue: -")
        labels = [
            self.var_fps,
            self.var_unix,
            self.var_utc,
            self.var_ntp,
            self.var_streams,
            self.var_recording,
            self.var_session,
            self.var_queue,
        ]
        for i, v in enumerate(labels, start=2):
            ttk.Label(self.side, textvariable=v, wraplength=340, justify="left").grid(row=i, column=0, sticky="w", pady=2)

        ttk.Separator(self.side).grid(row=10, column=0, sticky="ew", pady=12)

        # Experiment / Participant
        ttk.Label(self.side, text="Experiment ID:", font=("Segoe UI", 10, "bold")).grid(row=11, column=0, sticky="w")
        self.exp_var = tk.StringVar(value="")
        ttk.Entry(self.side, textvariable=self.exp_var).grid(row=12, column=0, sticky="ew", pady=(2, 6))
        ttk.Label(self.side, text="Participant ID:", font=("Segoe UI", 10, "bold")).grid(row=13, column=0, sticky="w")
        self.part_var = tk.StringVar(value="")
        ttk.Entry(self.side, textvariable=self.part_var).grid(row=14, column=0, sticky="ew", pady=(2, 8))
        ttk.Separator(self.side).grid(row=15, column=0, sticky="ew", pady=8)

        # Time server
        ttk.Label(self.side, text="Time server:", font=("Segoe UI", 10, "bold")).grid(row=16, column=0, sticky="w")
        self.server_var = tk.StringVar(value=dc.preferred_time_server or dc.fallback_servers[0])
        ttk.Entry(self.side, textvariable=self.server_var).grid(row=17, column=0, sticky="ew", pady=(2, 6))
        self.fallback_var = tk.BooleanVar(value=dc.enable_fallback_servers)
        ttk.Checkbutton(self.side, text="Fallback if fails", variable=self.fallback_var).grid(row=18, column=0, sticky="w", pady=(0, 6))
        ttk.Separator(self.side).grid(row=19, column=0, sticky="ew", pady=8)

        # Raw NPZ checkbox
        self.save_raw_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.side,
            text="Save raw NPZ bundles",
            variable=self.save_raw_var,
            command=self._on_toggle_save_raw,
        ).grid(row=20, column=0, sticky="w", pady=(0, 4))

        # Buffer mode checkbox
        self.buffer_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.side,
            text="Buffer mode (save after stop)",
            variable=self.buffer_mode_var,
        ).grid(row=21, column=0, sticky="w", pady=(0, 10))

        # Buttons
        self.btn_record = ttk.Button(self.side, text="Record", command=self.toggle_record)
        self.btn_record.grid(row=22, column=0, sticky="ew", pady=(0, 6))
        self.btn_resync = ttk.Button(self.side, text="Re-sync time", command=self.resync_time)
        self.btn_resync.grid(row=23, column=0, sticky="ew", pady=(0, 6))
        self.btn_close = ttk.Button(self.side, text="Close", command=self.on_close)
        self.btn_close.grid(row=24, column=0, sticky="ew", pady=(0, 6))

        c = self.chosen_color
        d = self.chosen_depth
        self.var_streams.set(f"Streams: COLOR {c[0]}x{c[1]}@{c[2]} | DEPTH {d[0]}x{d[1]}@{d[2]}")

    def _set_rec_indicator(self, is_on: bool):
        if is_on:
            self.rec_row.grid()
        else:
            self.rec_row.grid_remove()

    def _on_toggle_save_raw(self):
        self.recorder.set_save_raw_npz(self.save_raw_var.get())

    def resync_time(self):
        dc.start_time_sync_thread(
            state=self.sync_state,
            preferred_server=self.server_var.get().strip(),
            fallback_servers=dc.fallback_servers,
            enable_fallback=self.fallback_var.get(),
            attempts_per_server=dc.ntp_attempts_per_server,
            timeout_s=dc.ntp_timeout_s,
        )

    def toggle_record(self):
        if self.recorder.is_recording:
            self.recorder.stop()
            self._set_rec_indicator(False)
            self.var_recording.set("Recording: OFF")
            self.var_session.set("Session: -")
            self.btn_record.configure(text="Record")
            with self.record_pair_lock:
                self.record_pairs.clear()
            return
        utc_ns = dc.server_time_utc_ns(self.sync_state) or int(time.time_ns())
        nominal_fps = float(self.chosen_color[2])
        rgb_size = (self.chosen_color[0], self.chosen_color[1])
        depth_vis_size = (self.chosen_depth[0], self.chosen_depth[1])
        exp_id = dc.sanitize_id(self.exp_var.get(), "exp")
        part_id = dc.sanitize_id(self.part_var.get(), "p")
        self.recorder.set_save_raw_npz(self.save_raw_var.get())
        self.recorder.set_buffer_mode(self.buffer_mode_var.get())
        self.recorder.start(
            rgb_size=rgb_size,
            depth_vis_size=depth_vis_size,
            fps=nominal_fps,
            session_start_utc_ns=utc_ns,
            experiment_id=exp_id,
            participant_id=part_id,
        )
        self._set_rec_indicator(True)
        self.var_recording.set("Recording: ON")
        self.var_session.set(f"Session: {self.recorder.session_dir}")
        self.btn_record.configure(text="Stop recording")

    def _ui_tick(self):
        if not self.running:
            return

        # Read shared time state
        with self.time_lock:
            unix_ns = self.time_state.unix_ns
            utc_ns = self.time_state.utc_ns
            ntp_line = self.time_state.ntp_line

        # Read shared latest frame state
        with self.latest_lock:
            rgb_bgr = self.latest.rgb_bgr
            depth_vis_bgr = self.latest.depth_vis_bgr
            fps = self.latest.fps
            seq = self.latest.seq

        # Update labels (cheap)
        self.var_fps.set(f"FPS: {fps:.2f}")
        self.var_unix.set(f"Unix time: {unix_ns} ns ({unix_ns/1e9:.6f} s)")
        self.var_utc.set(f"UTC (server): {utc_ns} ns ({dc.utc_iso_from_ns(utc_ns)})")
        self.var_ntp.set(ntp_line)
        self.var_queue.set(f"Record queue: {self.record_queue.qsize()}/{dc.record_queue_maxsize}")

        # Only re-render images when NEW frames arrived
        if (seq != self._last_seq_rendered) and (rgb_bgr is not None) and (depth_vis_bgr is not None):
            self._last_seq_rendered = seq
            # Convert for Tk display
            rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            depth_rgb = cv2.cvtColor(depth_vis_bgr, cv2.COLOR_BGR2RGB)
            # Creating PhotoImage is relatively expensive; do it only on new frames
            self.rgb_photo = ImageTk.PhotoImage(Image.fromarray(rgb_rgb))
            self.depth_photo = ImageTk.PhotoImage(Image.fromarray(depth_rgb))
            self.rgb_label.configure(image=self.rgb_photo)
            self.depth_label.configure(image=self.depth_photo)

        self.root.after(ui_update_ms, self._ui_tick)

    def on_close(self):
        self.running = False
        try:
            self.recorder.stop()
        except Exception:
            pass
        try:
            self.time_worker.stop()
        except Exception:
            pass
        try:
            self.capture_worker.stop()
        except Exception:
            pass
        try:
            self.rgb_worker.stop()
        except Exception:
            pass
        try:
            self.depth_worker.stop()
        except Exception:
            pass
        try:
            self.record_worker.stop()
        except Exception:
            pass
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
