"""
Shared depth camera functionality for RealSense D455 capture, recording, and timing.
"""
import os
import csv
import re
import socket
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

# ================== CONFIGURATION FLAGS ==================
use_custom_colormap = False  # custom depth colormap
min_depth = 0.1  # meters
max_depth = 4.0  # meters

# ---- RealSense stream preferences (auto-fallback if unsupported) ----
preferred_color = (640, 480, 30)  # (w,h,fps)
preferred_depth = (640, 480, 30)  # (w,h,fps)

# ---- NTP / Time Sync Options ----
preferred_time_server = "time.cloudflare.com"  # e.g., "192.168.1.10" for a local NTP server
enable_fallback_servers = True
fallback_servers = ["time.cloudflare.com", "time.google.com", "pool.ntp.org", "time.windows.com"]
ntp_attempts_per_server = 3
ntp_timeout_s = 1.0

# ---- Recording Options ----
record_root_dir = "recordings"
video_codec_fourcc = "MJPG"  # try "MJPG" for faster encoding on some systems

# ---- Threading / Performance ----
rgb_queue_maxsize = 8
depth_queue_maxsize = 8
record_queue_maxsize = 16  # bounded queue; frames may drop if disk is slow

time_update_hz = 500  # time label updates; higher is rarely useful

# Pairing cache: purge old unmatched entries
pair_cache_max_items = 32
pair_cache_purge_keep_last = 16


# =============================================================================
# Helper utilities
# =============================================================================
def sanitize_id(s: str, default: str) -> str:
    s = (s or "").strip()
    if not s:
        return default
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else default


# =============================================================================
# SNTP / NTP one-shot sync
# =============================================================================
NTP_PORT = 123
NTP_EPOCH_DELTA = 2208988800  # seconds between 1900-01-01 and 1970-01-01


def _unix_ns_to_ntp_parts(unix_ns: int) -> Tuple[int, int]:
    unix_s = unix_ns // 1_000_000_000
    ns_rem = unix_ns - unix_s * 1_000_000_000
    ntp_s = unix_s + NTP_EPOCH_DELTA
    ntp_frac = int((ns_rem * (1 << 32)) / 1_000_000_000) & 0xFFFFFFFF
    return ntp_s, ntp_frac


def _ntp_parts_to_unix_ns(ntp_s: int, ntp_frac: int) -> int:
    unix_s = ntp_s - NTP_EPOCH_DELTA
    ns = int((ntp_frac * 1_000_000_000) / (1 << 32))
    return unix_s * 1_000_000_000 + ns


def sntp_one_shot(server: str, timeout_s: float = 1.0) -> Tuple[int, int, int]:
    """
    Returns (offset_ns, delay_ns, t4_unix_ns).
      offset_ns: estimate of (server_time - local_time) at receive time.
      delay_ns : round-trip delay estimate.
      t4_unix_ns: local UNIX time when response was received.
    """
    packet = bytearray(48)
    packet[0] = 0x23  # LI=0, VN=4, Mode=3
    t1_unix_ns = time.time_ns()
    t1_ntp_s, t1_ntp_frac = _unix_ns_to_ntp_parts(t1_unix_ns)
    struct.pack_into("!II", packet, 40, t1_ntp_s, t1_ntp_frac)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout_s)
        s.sendto(packet, (server, NTP_PORT))
        data, _ = s.recvfrom(512)
        t4_unix_ns = time.time_ns()
    if len(data) < 48:
        raise RuntimeError(f"Short NTP response from {server}: {len(data)} bytes")
    t2_ntp_s, t2_ntp_frac = struct.unpack_from("!II", data, 32)
    t3_ntp_s, t3_ntp_frac = struct.unpack_from("!II", data, 40)
    t2_unix_ns = _ntp_parts_to_unix_ns(t2_ntp_s, t2_ntp_frac)
    t3_unix_ns = _ntp_parts_to_unix_ns(t3_ntp_s, t3_ntp_frac)
    offset_ns = ((t2_unix_ns - t1_unix_ns) + (t3_unix_ns - t4_unix_ns)) // 2
    delay_ns = (t4_unix_ns - t1_unix_ns) - (t3_unix_ns - t2_unix_ns)
    return offset_ns, delay_ns, t4_unix_ns


def best_startup_sync(servers: List[str], attempts_per_server: int, timeout_s: float):
    best = None
    last_err = None
    for server in servers:
        for _ in range(attempts_per_server):
            try:
                offset_ns, delay_ns, t4_unix_ns = sntp_one_shot(server, timeout_s=timeout_s)
                sample = {
                    "server": server,
                    "offset_ns": offset_ns,
                    "delay_ns": delay_ns,
                    "t4_unix_ns": t4_unix_ns,
                    "mono_at_t4_ns": time.monotonic_ns(),
                }
                if (best is None) or (sample["delay_ns"] < best["delay_ns"]):
                    best = sample
            except Exception as e:
                last_err = e
    if best is None:
        raise RuntimeError(f"All SNTP attempts failed. Last error: {last_err}")
    return best


@dataclass
class SyncState:
    ok: bool = False
    server: Optional[str] = None
    offset_ns: int = 0
    delay_ns: int = 0
    mono_ref_ns: int = 0
    ref_utc_ns_at_mono_ref: int = 0
    error: Optional[str] = None


def start_time_sync_thread(
    state: SyncState,
    preferred_server: Optional[str],
    fallback_servers: List[str],
    enable_fallback: bool,
    attempts_per_server: int,
    timeout_s: float,
):
    def worker():
        try:
            servers = []
            if preferred_server and preferred_server.strip():
                servers.append(preferred_server.strip())
            if enable_fallback:
                for s in fallback_servers:
                    if not servers or s != servers[0]:
                        servers.append(s)
            if not servers:
                raise RuntimeError("No time servers provided.")
            best = best_startup_sync(servers, attempts_per_server=attempts_per_server, timeout_s=timeout_s)
            ref_utc_ns = best["t4_unix_ns"] + best["offset_ns"]
            state.ok = True
            state.server = best["server"]
            state.offset_ns = int(best["offset_ns"])
            state.delay_ns = max(0, int(best["delay_ns"]))
            state.mono_ref_ns = int(best["mono_at_t4_ns"])
            state.ref_utc_ns_at_mono_ref = int(ref_utc_ns)
            state.error = None
        except Exception as e:
            state.ok = False
            state.server = None
            state.error = str(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def server_time_utc_ns(state: SyncState) -> Optional[int]:
    if not state.ok:
        return None
    mono_now = time.monotonic_ns()
    return state.ref_utc_ns_at_mono_ref + (mono_now - state.mono_ref_ns)


def utc_iso_from_ns(unix_ns_utc: int) -> str:
    dt = datetime.fromtimestamp(unix_ns_utc / 1_000_000_000, tz=timezone.utc)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def safe_utc_tag_from_ns(unix_ns_utc: int) -> str:
    dt = datetime.fromtimestamp(unix_ns_utc / 1_000_000_000, tz=timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%S_%fZ")


# =============================================================================
# RealSense pipeline start with fallback
# =============================================================================

def _device_present_or_raise():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device detected. Check USB connection/power and try again.")


def start_pipeline_with_fallback(
    preferred_color_cfg: Tuple[int, int, int],
    preferred_depth_cfg: Tuple[int, int, int],
):
    _device_present_or_raise()
    pipeline = rs.pipeline()
    color_candidates = [
        preferred_color_cfg,
        (640, 480, 30),
        (848, 480, 30),
        (1280, 720, 30),
        (640, 480, 15),
    ]
    depth_candidates = [
        preferred_depth_cfg,
        (640, 480, 60),
        (640, 480, 30),
        (848, 480, 30),
        (1280, 720, 30),
    ]
    last_err = None
    for cw, ch, cfps in color_candidates:
        for dw, dh, dfps in depth_candidates:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cfps)
            cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
            try:
                pipeline.start(cfg)
                align = rs.align(rs.stream.color)
                return pipeline, align, (cw, ch, cfps), (dw, dh, dfps)
            except RuntimeError as e:
                last_err = e
                try:
                    pipeline.stop()
                except Exception:
                    pass
                continue
    raise RuntimeError(
        "Failed to start RealSense pipeline with any tried configuration. "
        f"Last error: {last_err}\n"
        "Common causes: another app is using the camera, unsupported FPS/resolution, USB bandwidth/power issues."
    )


# =============================================================================
# Depth visualization helper
# =============================================================================

def make_depth_vis(depth_raw_u16: np.ndarray) -> np.ndarray:
    if not use_custom_colormap:
        return cv2.applyColorMap(cv2.convertScaleAbs(depth_raw_u16, alpha=0.03), cv2.COLORMAP_JET)
    depth_m = depth_raw_u16.astype(np.float32) / 1000.0
    normalized = (max_depth - depth_m) / (max_depth - min_depth)
    normalized = np.clip(normalized, 0, 1)
    norm_8bit = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(norm_8bit, cv2.COLORMAP_JET)


# =============================================================================
# Recording (async writer)
# =============================================================================


@dataclass
class RecordItem:
    frame_index: int
    unix_ns: int
    utc_ns: int
    rgb_bgr: np.ndarray
    depth_u16: np.ndarray  # raw depth (uint16)


class Recorder:
    def __init__(self):
        self.is_recording = False
        self.save_raw_npz = True
        self.buffer_mode = False
        self.buffered_items: List[RecordItem] = []
        self._lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self.session_dir: Optional[str] = None
        self.bundle_dir: Optional[str] = None
        self.rgb_writer: Optional[cv2.VideoWriter] = None
        self.depth_writer: Optional[cv2.VideoWriter] = None
        self.rgb_csv_fh = None
        self.depth_csv_fh = None
        self.rgb_csv_writer = None
        self.depth_csv_writer = None
        self.session_start_utc_ns: Optional[int] = None
        self.session_start_tag: Optional[str] = None
        self.experiment_id: str = "exp"
        self.participant_id: str = "p"
        self._next_frame_index = 0

    def set_save_raw_npz(self, enabled: bool):
        with self._lock:
            self.save_raw_npz = bool(enabled)

    def get_save_raw_npz(self) -> bool:
        with self._lock:
            return self.save_raw_npz

    def set_buffer_mode(self, enabled: bool):
        self.buffer_mode = enabled

    def start(
        self,
        rgb_size: Tuple[int, int],
        depth_vis_size: Tuple[int, int],
        fps: float,
        session_start_utc_ns: int,
        experiment_id: str,
        participant_id: str,
    ):
        if self.is_recording:
            return
        os.makedirs(record_root_dir, exist_ok=True)
        self.session_start_utc_ns = int(session_start_utc_ns)
        self.session_start_tag = safe_utc_tag_from_ns(self.session_start_utc_ns)
        self.experiment_id = sanitize_id(experiment_id, "exp")
        self.participant_id = sanitize_id(participant_id, "p")
        session_name = f"session_exp{self.experiment_id}_pt{self.participant_id}_{self.session_start_tag}"
        self.session_dir = os.path.join(record_root_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)
        self.bundle_dir = os.path.join(
            self.session_dir,
            f"frame_bundles_exp{self.experiment_id}_pt{self.participant_id}_{self.session_start_tag}",
        )
        os.makedirs(self.bundle_dir, exist_ok=True)
        rgb_path = os.path.join(
            self.session_dir,
            f"rgb_video_exp{self.experiment_id}_pt{self.participant_id}_{self.session_start_tag}.avi",
        )
        depth_path = os.path.join(
            self.session_dir,
            f"depth_video_exp{self.experiment_id}_pt{self.participant_id}_{self.session_start_tag}.avi",
        )
        fourcc = cv2.VideoWriter_fourcc(*video_codec_fourcc)
        fps_use = max(1.0, float(fps))
        self.rgb_writer = cv2.VideoWriter(rgb_path, fourcc, fps_use, rgb_size, True)
        self.depth_writer = cv2.VideoWriter(depth_path, fourcc, fps_use, depth_vis_size, True)
        if not self.rgb_writer.isOpened():
            raise RuntimeError(f"Failed to open RGB VideoWriter at: {rgb_path}")
        if not self.depth_writer.isOpened():
            raise RuntimeError(f"Failed to open Depth VideoWriter at: {depth_path}")
        rgb_csv_path = os.path.join(
            self.session_dir,
            f"rgb_timestamps_exp{self.experiment_id}_pt{self.participant_id}_{self.session_start_tag}.csv",
        )
        depth_csv_path = os.path.join(
            self.session_dir,
            f"depth_timestamps_exp{self.experiment_id}_pt{self.participant_id}_{self.session_start_tag}.csv",
        )
        self.rgb_csv_fh = open(rgb_csv_path, "w", newline="", encoding="utf-8")
        self.depth_csv_fh = open(depth_csv_path, "w", newline="", encoding="utf-8")
        self.rgb_csv_writer = csv.writer(self.rgb_csv_fh)
        self.depth_csv_writer = csv.writer(self.depth_csv_fh)
        header = ["frame_index", "unix_ns", "unix_s", "utc_ns", "utc_iso_utc"]
        self.rgb_csv_writer.writerow(header)
        self.depth_csv_writer.writerow(header)
        self._next_frame_index = 0
        self.buffered_items = []
        self.is_recording = True

    def _close_writers(self):
        try:
            if self.rgb_writer is not None:
                self.rgb_writer.release()
            if self.depth_writer is not None:
                self.depth_writer.release()
        finally:
            self.rgb_writer = None
            self.depth_writer = None
        try:
            if self.rgb_csv_fh is not None:
                self.rgb_csv_fh.flush()
                self.rgb_csv_fh.close()
            if self.depth_csv_fh is not None:
                self.depth_csv_fh.flush()
                self.depth_csv_fh.close()
        finally:
            self.rgb_csv_fh = None
            self.depth_csv_fh = None
            self.rgb_csv_writer = None
            self.depth_csv_writer = None
        self.is_recording = False

    def stop(self):
        if not self.is_recording:
            return

        if self.buffer_mode:
            def write_and_close():
                with self._buffer_lock:
                    for item in self.buffered_items:
                        self.write_item(item)
                    self.buffered_items = []
                self._close_writers()

            threading.Thread(target=write_and_close, daemon=True).start()
        else:
            self._close_writers()

    def next_frame_index(self) -> int:
        idx = self._next_frame_index
        self._next_frame_index += 1
        return idx

    def write_item(self, item: RecordItem):
        if not self.is_recording:
            return
        if self.rgb_writer is None or self.depth_writer is None:
            return
        self.rgb_writer.write(item.rgb_bgr)
        depth_vis_bgr = make_depth_vis(item.depth_u16)
        self.depth_writer.write(depth_vis_bgr)
        unix_s = item.unix_ns / 1_000_000_000
        utc_iso = utc_iso_from_ns(item.utc_ns)
        row = [item.frame_index, item.unix_ns, f"{unix_s:.9f}", item.utc_ns, utc_iso]
        self.rgb_csv_writer.writerow(row)
        self.depth_csv_writer.writerow(row)
        if self.get_save_raw_npz():
            frame_tag = f"frame{item.frame_index:06d}_utc{safe_utc_tag_from_ns(item.utc_ns)}_unixns{item.unix_ns}"
            npz_path = os.path.join(
                self.bundle_dir,
                f"bundle_exp{self.experiment_id}_pt{self.participant_id}_{frame_tag}.npz",
            )
            np.savez(
                npz_path,
                frame_index=np.int64(item.frame_index),
                unix_ns=np.int64(item.unix_ns),
                utc_ns=np.int64(item.utc_ns),
                depth_u16=item.depth_u16.astype(np.uint16, copy=False),
                rgb_bgr=item.rgb_bgr.astype(np.uint8, copy=False),
            )


class RecordWorker:
    def __init__(self, recorder: Recorder, q: Queue):
        self.recorder = recorder
        self.q = q
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            while True:
                self.q.get_nowait()
        except Empty:
            pass
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.1)
            except Empty:
                continue
            try:
                self.recorder.write_item(item)
            except Exception:
                pass
            finally:
                self.q.task_done()


# =============================================================================
# Separate threads: Capture, RGB, Depth
# =============================================================================


@dataclass
class RGBTask:
    src_frame_number: int
    unix_ns: int
    utc_ns: int
    color_frame: rs.video_frame


@dataclass
class DepthTask:
    src_frame_number: int
    unix_ns: int
    utc_ns: int
    depth_frame: rs.depth_frame


@dataclass
class LatestFrame:
    rgb_bgr: Optional[np.ndarray] = None
    depth_vis_bgr: Optional[np.ndarray] = None
    fps: float = 0.0
    seq: int = 0  # increments when either stream updates (for UI to detect changes)


@dataclass
class TimeState:
    unix_ns: int = 0
    utc_ns: int = 0
    ntp_line: str = "NTP: syncing…"


@dataclass
class PartialForRecord:
    unix_ns: int
    utc_ns: int
    rgb_bgr: Optional[np.ndarray] = None
    depth_u16: Optional[np.ndarray] = None


class CaptureWorker:
    def __init__(
        self,
        pipeline: rs.pipeline,
        align: rs.align,
        sync_state: SyncState,
        rgb_q: Queue,
        depth_q: Queue,
        latest_lock: threading.Lock,
        latest: LatestFrame,
    ):
        self.pipeline = pipeline
        self.align = align
        self.sync_state = sync_state
        self.rgb_q = rgb_q
        self.depth_q = depth_q
        self.latest_lock = latest_lock
        self.latest = latest
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._prev_mon_ns = time.monotonic_ns()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                c = aligned.get_color_frame()
                d = aligned.get_depth_frame()
                if not c or not d:
                    continue
                unix_ns = time.time_ns()
                utc_ns = server_time_utc_ns(self.sync_state) or unix_ns
                fn = int(c.get_frame_number())
                now_mon = time.monotonic_ns()
                dt = (now_mon - self._prev_mon_ns) / 1_000_000_000
                fps = (1.0 / dt) if dt > 0 else 0.0
                self._prev_mon_ns = now_mon
                with self.latest_lock:
                    self.latest.fps = fps
                try:
                    self.rgb_q.put_nowait(RGBTask(fn, unix_ns, utc_ns, c))
                except Full:
                    pass
                try:
                    self.depth_q.put_nowait(DepthTask(fn, unix_ns, utc_ns, d))
                except Full:
                    pass
            except Exception:
                continue


class RGBWorker:
    def __init__(
        self,
        rgb_q: Queue,
        latest_lock: threading.Lock,
        latest: LatestFrame,
        recorder: Recorder,
        record_pair_lock: threading.Lock,
        record_pairs: "OrderedDict[int, PartialForRecord]",
        record_q: Queue,
    ):
        self.rgb_q = rgb_q
        self.latest_lock = latest_lock
        self.latest = latest
        self.recorder = recorder
        self.record_pair_lock = record_pair_lock
        self.record_pairs = record_pairs
        self.record_q = record_q
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _purge_if_needed(self):
        if len(self.record_pairs) <= pair_cache_max_items:
            return
        while len(self.record_pairs) > pair_cache_purge_keep_last:
            self.record_pairs.popitem(last=False)

    def _try_emit_record(self, src_fn: int):
        with self.record_pair_lock:
            p = self.record_pairs.get(src_fn)
            if not p:
                return
            if (p.rgb_bgr is None) or (p.depth_u16 is None):
                return
            if not self.recorder.is_recording:
                self.record_pairs.pop(src_fn, None)
                return
            frame_index = self.recorder.next_frame_index()
            item = RecordItem(
                frame_index=frame_index,
                unix_ns=p.unix_ns,
                utc_ns=p.utc_ns,
                rgb_bgr=p.rgb_bgr,
                depth_u16=p.depth_u16,
            )
            self.record_pairs.pop(src_fn, None)
        if self.recorder.buffer_mode:
            with self.recorder._buffer_lock:
                self.recorder.buffered_items.append(item)
        else:
            try:
                self.record_q.put_nowait(item)
            except Full:
                pass

    def _run(self):
        while not self._stop.is_set():
            try:
                task: RGBTask = self.rgb_q.get(timeout=0.1)
            except Empty:
                continue
            try:
                rgb_bgr = np.asanyarray(task.color_frame.get_data())
                with self.latest_lock:
                    self.latest.rgb_bgr = rgb_bgr
                    self.latest.seq += 1
                if self.recorder.is_recording:
                    rgb_copy = rgb_bgr.copy()
                    with self.record_pair_lock:
                        p = self.record_pairs.get(task.src_frame_number)
                        if p is None:
                            p = PartialForRecord(unix_ns=task.unix_ns, utc_ns=task.utc_ns)
                            self.record_pairs[task.src_frame_number] = p
                        else:
                            p.unix_ns = task.unix_ns
                            p.utc_ns = task.utc_ns
                        p.rgb_bgr = rgb_copy
                        self.record_pairs.move_to_end(task.src_frame_number, last=True)
                        self._purge_if_needed()
                    self._try_emit_record(task.src_frame_number)
            except Exception:
                pass
            finally:
                self.rgb_q.task_done()


class DepthWorker:
    def __init__(
        self,
        depth_q: Queue,
        latest_lock: threading.Lock,
        latest: LatestFrame,
        recorder: Recorder,
        record_pair_lock: threading.Lock,
        record_pairs: "OrderedDict[int, PartialForRecord]",
        record_q: Queue,
    ):
        self.depth_q = depth_q
        self.latest_lock = latest_lock
        self.latest = latest
        self.recorder = recorder
        self.record_pair_lock = record_pair_lock
        self.record_pairs = record_pairs
        self.record_q = record_q
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _purge_if_needed(self):
        if len(self.record_pairs) <= pair_cache_max_items:
            return
        while len(self.record_pairs) > pair_cache_purge_keep_last:
            self.record_pairs.popitem(last=False)

    def _try_emit_record(self, src_fn: int):
        with self.record_pair_lock:
            p = self.record_pairs.get(src_fn)
            if not p:
                return
            if (p.rgb_bgr is None) or (p.depth_u16 is None):
                return
            if not self.recorder.is_recording:
                self.record_pairs.pop(src_fn, None)
                return
            frame_index = self.recorder.next_frame_index()
            item = RecordItem(
                frame_index=frame_index,
                unix_ns=p.unix_ns,
                utc_ns=p.utc_ns,
                rgb_bgr=p.rgb_bgr,
                depth_u16=p.depth_u16,
            )
            self.record_pairs.pop(src_fn, None)
        if self.recorder.buffer_mode:
            with self.recorder._buffer_lock:
                self.recorder.buffered_items.append(item)
        else:
            try:
                self.record_q.put_nowait(item)
            except Full:
                pass

    def _run(self):
        while not self._stop.is_set():
            try:
                task: DepthTask = self.depth_q.get(timeout=0.1)
            except Empty:
                continue
            try:
                depth_u16 = np.asanyarray(task.depth_frame.get_data())
                depth_vis = make_depth_vis(depth_u16)
                with self.latest_lock:
                    self.latest.depth_vis_bgr = depth_vis
                    self.latest.seq += 1
                if self.recorder.is_recording:
                    depth_copy = depth_u16.copy()
                    with self.record_pair_lock:
                        p = self.record_pairs.get(task.src_frame_number)
                        if p is None:
                            p = PartialForRecord(unix_ns=task.unix_ns, utc_ns=task.utc_ns)
                            self.record_pairs[task.src_frame_number] = p
                        else:
                            p.unix_ns = task.unix_ns
                            p.utc_ns = task.utc_ns
                        p.depth_u16 = depth_copy
                        self.record_pairs.move_to_end(task.src_frame_number, last=True)
                        self._purge_if_needed()
                    self._try_emit_record(task.src_frame_number)
            except Exception:
                pass
            finally:
                self.depth_q.task_done()


# =============================================================================
# Time thread
# =============================================================================


class TimeWorker:
    def __init__(self, sync_state: SyncState, time_lock: threading.Lock, time_state: TimeState, stop_event: threading.Event):
        self.sync_state = sync_state
        self.time_lock = time_lock
        self.time_state = time_state
        self.stop_event = stop_event
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self.stop_event.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        period = 1.0 / max(1, int(time_update_hz))
        while not self.stop_event.is_set():
            unix_ns = time.time_ns()
            utc_ns = server_time_utc_ns(self.sync_state) or unix_ns
            if self.sync_state.ok:
                off_ms = self.sync_state.offset_ns / 1_000_000
                rtt_ms = self.sync_state.delay_ns / 1_000_000
                ntp_line = f"NTP: {self.sync_state.server} | Offset: {off_ms:+.3f} ms | RTT: {rtt_ms:.3f} ms"
            else:
                ntp_line = f"NTP: {self.sync_state.error or 'syncing…'}"
            with self.time_lock:
                self.time_state.unix_ns = unix_ns
                self.time_state.utc_ns = utc_ns
                self.time_state.ntp_line = ntp_line
            time.sleep(period)
