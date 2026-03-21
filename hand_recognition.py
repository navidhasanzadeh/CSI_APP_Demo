"""Asynchronous MediaPipe hand tracking utilities."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:  # pragma: no cover - optional dependency
    import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency
    mp = None

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

HandSample = Tuple[float, float, float, float, float, float, float]


class HandRecognitionError(RuntimeError):
    """Raised when the hand recognition backend is unavailable."""


class HandRecognitionEngine:
    """Runs MediaPipe hand tracking in a dedicated worker thread."""

    def __init__(
        self,
        *,
        max_num_hands: int = 2,
        model_complexity: str = "light",
        use_light_model: bool | None = None,
        left_wrist_color: str | Sequence[int] | None = None,
        right_wrist_color: str | Sequence[int] | None = None,
        wrist_circle_radius: int | None = None,
    ) -> None:
        if cv2 is None:
            raise HandRecognitionError("OpenCV is required for hand recognition.")
        if mp is None:
            raise HandRecognitionError("Install 'mediapipe' to enable hand recognition.")

        if use_light_model is not None:
            model_complexity = "light" if use_light_model else "full"
        model_complexity = (model_complexity or "light").strip().lower()
        if model_complexity not in {"light", "full"}:
            model_complexity = "light"

        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=0 if model_complexity == "light" else 1,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._drawing = mp.solutions.drawing_utils
        self._connections = mp.solutions.hands.HAND_CONNECTIONS
        self._left_spec = self._drawing.DrawingSpec(
            color=(0, 200, 255), thickness=4, circle_radius=5
        )
        self._right_spec = self._drawing.DrawingSpec(
            color=(255, 120, 0), thickness=4, circle_radius=5
        )
        self._connection_spec = self._drawing.DrawingSpec(
            color=(255, 255, 255), thickness=3, circle_radius=3
        )
        self._left_wrist_bgr = self._coerce_color(left_wrist_color, (255, 0, 0))
        self._right_wrist_bgr = self._coerce_color(right_wrist_color, (0, 0, 255))
        self._wrist_circle_radius = max(1, int(wrist_circle_radius or 18))

        self._frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._latest_frame = None
        self._samples: List[HandSample] = []
        self._is_processing = False
        self._saved = False
        self._mirror_view = True

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._worker_thread is not None:
            return
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="HandRecognition", daemon=True
        )
        self._worker_thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:  # pragma: no cover - cleanup best effort
                pass
            self._hands = None

    @staticmethod
    def _coerce_color(color: str | Sequence[int] | None, default_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Return a BGR tuple from a hex string or BGR/RGB sequence."""

        def _bgr_from_rgb(rgb: Sequence[int]) -> Tuple[int, int, int]:
            try:
                r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            except Exception:
                return default_bgr
            if any(v < 0 or v > 255 for v in (r, g, b)):
                return default_bgr
            return (b, g, r)

        if isinstance(color, str):
            value = color.strip().lstrip("#")
            if len(value) == 6:
                try:
                    r = int(value[0:2], 16)
                    g = int(value[2:4], 16)
                    b = int(value[4:6], 16)
                    return (b, g, r)
                except ValueError:
                    return default_bgr
            return default_bgr

        if color is not None:
            return _bgr_from_rgb(color)
        return default_bgr

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """Submit a frame for processing and return the latest annotated frame."""

        if self._hands is None or cv2 is None:
            return frame

        self.start()
        try:
            self._frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

        with self._result_lock:
            latest = self._latest_frame
        return latest if latest is not None else frame

    def _worker_loop(self):  # pragma: no cover - realtime loop
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._set_processing(True)
            annotated, samples = self._process_hand_landmarks(frame)
            self._set_processing(False)

            with self._result_lock:
                if annotated is not None:
                    self._latest_frame = annotated
                if samples:
                    self._samples.extend(samples)

    def _set_processing(self, value: bool) -> None:
        with self._result_lock:
            self._is_processing = value

    def set_mirror_view(self, value: bool) -> None:
        """Control whether the handedness labels are mirrored."""

        with self._result_lock:
            self._mirror_view = bool(value)

    def _process_hand_landmarks(self, frame, *, sample_timestamp_ns: float | None = None):
        if self._hands is None or cv2 is None:
            return frame, []

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            return frame, []

        rgb_frame.flags.writeable = False
        try:
            results = self._hands.process(rgb_frame)
        except Exception:
            return frame, []
        finally:
            rgb_frame.flags.writeable = True

        samples: List[HandSample] = []
        timestamp_ns = float(sample_timestamp_ns) if sample_timestamp_ns else time.time_ns()
        hand_presence = {0.0: 0.0, 1.0: 0.0}

        if results and results.multi_hand_landmarks:
            handedness_list = getattr(results, "multi_handedness", None)
            if not handedness_list:
                handedness_list = [None] * len(results.multi_hand_landmarks)
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, handedness_list
            ):
                label_value = 1.0
                label_text = "right"
                if handedness and handedness.classification:
                    label_text = handedness.classification[0].label or "right"

                is_left = label_text.lower().startswith("left")
                label_value = 0.0 if is_left else 1.0
                if self._mirror_view:
                    # The webcam preview is mirrored, so swap handedness labels
                    label_value = 1.0 - label_value

                x_coords = [float(lm.x) for lm in hand_landmarks.landmark]
                y_coords = [float(lm.y) for lm in hand_landmarks.landmark]
                z_coords = [float(lm.z) for lm in hand_landmarks.landmark]

                median_x = float(np.median(x_coords)) if x_coords else float("nan")
                median_y = float(np.median(y_coords)) if y_coords else float("nan")
                median_z = float(np.median(z_coords)) if z_coords else float("nan")

                hand_presence[label_value] = 1.0
                samples.append(
                    (
                        float(timestamp_ns),
                        label_value,
                        0.0,
                        median_x,
                        median_y,
                        median_z,
                        1.0,
                    )
                )

                if self._drawing is not None and self._connections is not None:
                    spec = self._left_spec if label_value == 0.0 else self._right_spec
                    self._drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self._connections,
                        spec,
                        self._connection_spec,
                    )

                    wrist_idx = getattr(mp.solutions.hands.HandLandmark, "WRIST", 0)
                    if 0 <= wrist_idx < len(hand_landmarks.landmark):
                        wrist = hand_landmarks.landmark[wrist_idx]
                        h, w = frame.shape[:2]
                        center = (int(wrist.x * w), int(wrist.y * h))
                        color = (
                            self._left_wrist_bgr
                            if label_value == 0.0
                            else self._right_wrist_bgr
                        )
                        cv2.circle(
                            frame,
                            center,
                            self._wrist_circle_radius,
                            color,
                            thickness=-1,
                            lineType=cv2.LINE_AA,
                        )

        for label_value, presence in hand_presence.items():
            samples.append(
                (
                    float(timestamp_ns),
                    label_value,
                    -1.0,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    presence,
                )
            )
        return frame, samples

    def process_video_file(
        self,
        video_path: Path,
        *,
        mirror_view: bool | None = None,
        progress_callback=None,
        base_timestamp_ns: float | int | None = None,
    ) -> bool:
        """Run hand landmark detection over a recorded video."""

        if cv2 is None or not video_path:
            return False
        video = cv2.VideoCapture(str(video_path))
        if not video or not video.isOpened():
            return False

        if mirror_view is not None:
            self.set_mirror_view(mirror_view)

        fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        frame_period_ns = int(1e9 / fps)

        start_timestamp_ns = (
            float(base_timestamp_ns) if base_timestamp_ns is not None else time.time_ns()
        )

        self._saved = False
        self._samples = []
        self._latest_frame = None

        total_frames = 0
        try:
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            total_frames = 0

        processed_frames = 0
        if progress_callback is not None:
            try:
                progress_callback(processed_frames, total_frames)
            except Exception:
                pass

        try:
            frame_index = 0
            while True:
                ret, frame = video.read()
                if not ret or frame is None:
                    break
                frame_timestamp_ns = start_timestamp_ns + frame_index * frame_period_ns
                annotated, samples = self._process_hand_landmarks(
                    frame, sample_timestamp_ns=frame_timestamp_ns
                )
                if annotated is not None:
                    self._latest_frame = annotated
                if samples:
                    self._samples.extend(samples)

                processed_frames += 1
                frame_index += 1
                if progress_callback is not None:
                    try:
                        progress_callback(processed_frames, total_frames)
                    except Exception:
                        pass
        finally:
            video.release()

        return bool(self._samples)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def drain(self, timeout: float = 1.0) -> None:
        """Wait for the worker queue to empty and processing to finish."""

        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._result_lock:
                busy = self._is_processing
            if not busy and self._frame_queue.empty():
                break
            time.sleep(0.01)

    def save_results(self, results_dir: Path) -> bool:
        if self._saved:
            return True
        if results_dir is None:
            return False

        self.drain()

        with self._result_lock:
            if not self._samples:
                self._saved = True
                return True
            data = np.array(self._samples, dtype=np.float64)

        try:
            landmarks_path = results_dir / "hand_landmarks.npy"
            np.save(landmarks_path, data)

            if plt is not None:
                self._save_plot(results_dir, data)
        except Exception:
            pass
        finally:
            self._saved = True
        return True

    def _save_plot(self, results_dir: Path, data: np.ndarray) -> None:
        """Create a per-hand summary plot for the wrist landmark only."""

        # Action-mode samples are persisted by the main window before this method
        # runs. Reuse those timestamps so the plots share a common horizontal axis
        # with the rest of the experiment artefacts.
        action_signal = self._load_action_signal(results_dir)
        shared_timestamps = action_signal[:, 0] if action_signal.size else None

        base_timestamp = 0.0
        if shared_timestamps is not None:
            base_timestamp = float(shared_timestamps[0])
        elif data.size:
            base_timestamp = float(data[0, 0])

        def _normalize_ts(timestamps: np.ndarray) -> np.ndarray:
            if timestamps.size == 0:
                return timestamps
            return (timestamps.astype(np.float64) - base_timestamp) / 1e9

        ordered_action = np.empty((0, 2), dtype=np.float64)
        normalized_action_ts = np.empty((0,), dtype=np.float64)
        if action_signal.size:
            ordered_action = action_signal[np.argsort(action_signal[:, 0])]
            normalized_action_ts = _normalize_ts(ordered_action[:, 0])

        def _action_segments() -> List[Tuple[float, float]]:
            if ordered_action.size == 0:
                return []

            segments: List[Tuple[float, float]] = []
            values = ordered_action[:, 1] > 0.5
            start_ts: float | None = None

            for ts, active in zip(normalized_action_ts, values):
                if active and start_ts is None:
                    start_ts = float(ts)
                elif not active and start_ts is not None:
                    segments.append((start_ts, float(ts)))
                    start_ts = None

            if start_ts is not None:
                segments.append((start_ts, float(normalized_action_ts[-1])))

            return segments

        action_segments = _action_segments()

        def _shade_action_windows(ax):
            for start, end in action_segments:
                ax.axvspan(start, end, color="red", alpha=0.08, linewidth=0)

        measurements: Sequence[Tuple[str, int | None]] = (
            ("Action mode", None),
            ("x (normalized)", 3),
            ("y (normalized)", 4),
            ("z (normalized)", 5),
            ("Hand in frame", 6),
        )
        fig, axes = plt.subplots(
            len(measurements),
            2,
            figsize=(12, 12),
            sharex="col",
            dpi=300,
        )
        if len(measurements) == 1:
            axes = np.array([axes])

        hands = (
            (0.0, "Left hand", "tab:blue"),
            (1.0, "Right hand", "tab:orange"),
        )
        wrist_index = 0.0
        presence_samples = data[data[:, 2] < 0]

        def _presence_segments(
            presence_rows: np.ndarray, label_value: float
        ) -> List[Tuple[float, float]]:
            """Return [start, end] timestamp pairs where a hand was detected."""

            if presence_rows.size == 0:
                return []

            filtered = presence_rows[np.isclose(presence_rows[:, 1], label_value)]
            if filtered.size == 0:
                return []

            ordered = filtered[np.argsort(filtered[:, 0])]
            timestamps = ordered[:, 0]
            flags = ordered[:, 6] > 0.5

            segments: List[Tuple[float, float]] = []
            current_start: float | None = None

            for ts, present in zip(timestamps, flags):
                if present and current_start is None:
                    current_start = float(ts)
                elif not present and current_start is not None:
                    segments.append((current_start, float(ts)))
                    current_start = None

            if current_start is not None:
                segments.append((current_start, float(timestamps[-1])))

            return segments

        for col_index, (label_value, title, color) in enumerate(hands):
            subset = data[
                (data[:, 1] == label_value) & (np.isclose(data[:, 2], wrist_index))
            ]
            presence_segments = _presence_segments(presence_samples, label_value)
            for row_index, (axis_label, value_index) in enumerate(measurements):
                ax = axes[row_index, col_index]
                _shade_action_windows(ax)
                if row_index == 0:
                    # Action mode subplot
                    if action_signal.size:
                        ax.step(
                            normalized_action_ts,
                            ordered_action[:, 1],
                            where="post",
                            linewidth=2.2,
                            color="black",
                        )
                        ax.set_ylim(-0.1, 1.1)
                        ax.set_yticks([0, 1])
                        ax.set_yticklabels(["Off", "On"])
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "Action signal unavailable",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=10,
                        )
                    ax.set_ylabel(axis_label)
                else:
                    if value_index == 6:
                        if presence_samples.size:
                            ordered = presence_samples[np.argsort(presence_samples[:, 0])]
                            ordered = ordered[np.isclose(ordered[:, 1], label_value)]
                            if ordered.size:
                                timestamps = _normalize_ts(ordered[:, 0])
                                values = ordered[:, value_index]
                                ax.step(
                                    timestamps,
                                    values,
                                    where="post",
                                    linewidth=1.6,
                                    color="black",
                                )
                                ax.set_ylim(-0.1, 1.1)
                                ax.set_yticks([0, 1])
                                ax.set_yticklabels(["Out", "In"])
                            else:
                                ax.text(
                                    0.5,
                                    0.5,
                                    "No hand detections recorded",
                                    ha="center",
                                    va="center",
                                    transform=ax.transAxes,
                                    fontsize=10,
                                )
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                "No hand detections recorded",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                                fontsize=10,
                            )
                    elif subset.size and presence_segments:
                        for start_ts, end_ts in presence_segments:
                            window = subset[
                                (subset[:, 0] >= start_ts) & (subset[:, 0] <= end_ts)
                            ]
                            if not window.size:
                                continue
                            timestamps = _normalize_ts(window[:, 0])
                            values = window[:, value_index]
                            valid = ~np.isnan(values)
                            if np.any(valid):
                                ax.plot(
                                    timestamps[valid],
                                    values[valid],
                                    linewidth=1.5,
                                    color=color,
                                    alpha=0.95,
                                )
                    ax.set_ylabel(axis_label)
                ax.grid(True, linestyle="--", alpha=0.4)
                if row_index == 0:
                    ax.set_title(title)
            axes[-1, col_index].set_xlabel("Time since start (s)")

        fig.tight_layout()
        plot_path = results_dir / "hand_landmarks.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)

    def _load_action_signal(self, results_dir: Path) -> np.ndarray:
        signal_path = results_dir / "action_signal.npy"
        if not signal_path.exists():
            return np.empty((0, 2), dtype=np.float64)
        try:
            data = np.load(signal_path)
            if data.ndim != 2 or data.shape[1] < 2:
                return np.empty((0, 2), dtype=np.float64)
            return data[:, :2].astype(np.float64, copy=False)
        except Exception:
            return np.empty((0, 2), dtype=np.float64)

    def has_samples(self) -> bool:
        with self._result_lock:
            return bool(self._samples)

