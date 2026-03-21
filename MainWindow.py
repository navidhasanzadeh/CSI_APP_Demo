# -*- coding: utf-8 -*-
"""
Refactored MainWindow for HAR data collection.

- Removes most global state and commented-out code.
- Uses QTimer for all time-based logic in the GUI thread.
- Uses a dedicated worker thread only for playing blocking beep sounds.
- Keeps experiment behaviour (baselines, action/stop cycles, repetitions, videos).
"""

import csv
import hashlib
import queue
import sys
import time
import random
import re
import subprocess
import threading
import wave
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import matplotlib.pyplot as plt
import requests
try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from PyQt5 import QtCore, uic
from PyQt5.QtCore import Qt, QTimer, QUrl, pyqtSignal, QObject, QSize
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QInputDialog,
    QLineEdit,
    QSizePolicy,
)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtMultimedia import (
    QMediaPlayer,
    QMediaContent,
)
from PyQt5.QtMultimediaWidgets import QVideoWidget

from voice_assistant import GTTSVoiceAssistant
from hand_recognition import HandRecognitionEngine, HandRecognitionError
from wifi_csi_manager import WiFiCSIManager, WiFiRouter
from packet_counter import count_packets_for_macs
from transferring_files_dialog import TransferringFilesDialog
from password_manager import is_password_required, verify_password
from demo_csi_plot_dialog import DemoCSIPlotDialog, load_csi_for_framework
import time_reference as time_reference_module

try:  # pragma: no cover - optional dependency
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional dependency
    sd = None


def generate_participant_id(name: str, age_group: str, gender: str) -> str:
    seed = f"{(name or '').lower()}|{(age_group or '').lower()}|{(gender or '').lower()}"
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:12]


def generate_experiment_id() -> str:
    return f"{random.randint(100000, 999999)}"


class BeepWorker(QObject):
    """
    Worker object that plays a short beep using SoX `play` command.
    Runs in its own QThread so the GUI never blocks.
    """

    @QtCore.pyqtSlot()
    def do_beep(self):
        duration = 0.2
        frequency = 880
        try:
            import sys as _sys

            if _sys.platform.startswith("darwin"):
                # macOS uses coreaudio
                cmd = [
                    "play",
                    "-nq",
                    "-t",
                    "coreaudio",
                    "synth",
                    str(duration),
                    "sine",
                    str(frequency),
                ]
            elif _sys.platform.startswith("linux"):
                # Linux usually uses ALSA
                cmd = [
                    "play",
                    "-nq",
                    "-t",
                    "alsa",
                    "synth",
                    str(duration),
                    "sine",
                    str(frequency),
                ]
            else:
                # Unsupported OS: silently skip
                return

            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            # Beep failure should never crash the app
            pass


class MicrophoneRecorder:
    """Lightweight microphone recorder that streams audio to a WAV file."""

    def __init__(self, output_path: Path, samplerate: int = 44100, channels: int = 1):
        self.output_path = Path(output_path)
        self.samplerate = samplerate
        self.channels = channels
        self.stream = None
        self.wave_file = None
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._writer_thread = None

    def start(self):
        if sd is None:
            raise RuntimeError("sounddevice is required for microphone capture")
        if self.stream is not None:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.wave_file = wave.open(str(self.output_path), "wb")
        self.wave_file.setnchannels(self.channels)
        self.wave_file.setsampwidth(2)  # 16-bit samples
        self.wave_file.setframerate(self.samplerate)

        def _callback(indata, frames, time_info, status):  # pragma: no cover - realtime
            if not self._stop_event.is_set():
                self._queue.put(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
            callback=_callback,
        )
        self.stream.start()
        self._stop_event.clear()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="MicrophoneWriter", daemon=True
        )
        self._writer_thread.start()

    def _writer_loop(self):  # pragma: no cover - realtime loop
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                data = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.wave_file is not None:
                self.wave_file.writeframes(data.tobytes())

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        self._stop_event.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2.0)
            self._writer_thread = None

        # flush remaining data
        while not self._queue.empty():
            try:
                data = self._queue.get_nowait()
            except queue.Empty:
                break
            if self.wave_file is not None:
                self.wave_file.writeframes(data.tobytes())

        if self.wave_file is not None:
            try:
                self.wave_file.close()
            except Exception:
                pass
            self.wave_file = None
        self._queue = queue.Queue()


class MainWindow(QMainWindow):
    """
    Main GUI window controlling the experiment flow.

    Constructor signature must match the original:
        MainWindow(experiment_dict, subject_dict, actions_dict)
    """

    # simple state enumeration
    MODE_BEGIN_BASELINE = 0
    MODE_ACTIONS = 1
    MODE_BETWEEN_BASELINE = 2
    MODE_END_BASELINE = 3
    MODE_FINISHED = 4
    UI_LABEL_CONFIGS = [
        ("label_11", "name_title", "Name:", 12),
        ("label_participant_title", "participant_title", "Participant ID:", 12),
        ("label_12", "experiment_title", "Experiment ID:", 12),
        ("label_second_name_title", "second_name_title", "Second Name:", 12),
        ("label_second_participant_title", "second_participant_title", "Second Participant ID:", 12),
        ("label_5", "remaining_time_title", "Remaining Time:", 12),
        ("label_2", "elapsed_time_title", "Elapsed Time:", 12),
        ("label_7", "count_title", "Count:", 12),
        ("label_9", "current_action_title", "Current action:", 12),
        ("label", "brand_header", "Wirlab", 26),
    ]
    UI_VALUE_LABEL_CONFIGS = [
        ("label_name", "name_value", 12),
        ("label_participant_id", "participant_value", 12),
        ("label_id", "experiment_value", 12),
        ("label_second_name", "second_name_value", 12),
        ("label_second_participant_id", "second_participant_value", 12),
    ]

    # signal to request asynchronous beep
    request_beep = pyqtSignal()
    transfer_started = pyqtSignal(str, int)
    transfer_progress = pyqtSignal(str, int, int)
    transfer_finished = pyqtSignal(str, bool)
    demo_plot_requested = pyqtSignal(dict)

    def __init__(
        self,
        experiment_dict,
        subject_dict,
        actions_dict,
        voice_profile=None,
        wifi_profile=None,
        camera_profile=None,
        depth_camera_profile=None,
        ui_profile=None,
        environment_profile=None,
        parent=None,
        *,
        prestarted_wifi=None,
        start_wifi_capture: bool = True,
        results_dir=None,
        time_reference=None,
    ):
        super().__init__(parent)

        # Keep the original UI layout
        uic.loadUi("mainwindow.ui", self)

        # store configs
        self.experiment_dict = experiment_dict
        self.subject_dict = dict(subject_dict or {})
        self._normalize_subject_metadata()
        self.actions_dict = actions_dict
        self.voice_profile = voice_profile if isinstance(voice_profile, dict) else {}
        self.camera_profile = camera_profile if isinstance(camera_profile, dict) else {}
        self.depth_camera_profile = (
            depth_camera_profile if isinstance(depth_camera_profile, dict) else {}
        )
        self.ui_profile = ui_profile if isinstance(ui_profile, dict) else {}
        self.environment_info = (
            environment_profile if isinstance(environment_profile, dict) else {}
        )
        self.time_reference = (
            time_reference
            if isinstance(time_reference, time_reference_module.TimeReference)
            else time_reference_module.get_global_time_reference()
        )
        if isinstance(wifi_profile, list):
            self.wifi_profile = {"access_points": wifi_profile}
        elif isinstance(wifi_profile, dict):
            self.wifi_profile = wifi_profile
        else:
            self.wifi_profile = {}
        self.wifi_profile.setdefault("count_packets", False)
        self.wifi_profile.setdefault("reboot_after_summary", False)
        self.wifi_manager = WiFiCSIManager(self.wifi_profile)
        self.use_webcam = self._should_use_webcam()
        self.hand_recognition_enabled = self._should_use_hand_recognition()
        self.hand_recognition_mode = self._hand_recognition_mode()
        self.camera_device_source = self._camera_device_source()
        self.voice_assistant_enabled = self._should_use_voice_assistant()
        self.webcam_capture = None
        self.webcam_writer = None
        self.webcam_timer = None
        self.webcam_preview_label = None
        self._webcam_output_path = None
        self._webcam_frame_log = None
        self._webcam_frame_index = 0
        self.voice_assistant = None
        self.microphone_recorder = None
        self.preview_widget = None
        self.preview_player = None
        self._base_webcam_size: QSize | None = None
        self._base_preview_size: QSize | None = None
        self._base_action_frame_size: QSize | None = None
        self._base_log_height: int | None = None
        self._ui_defaults = None
        self.log_widget = getattr(self, "logTextEdit", None)
        if self.log_widget is not None:
            try:
                self.log_widget.setReadOnly(True)
                self.log_widget.clear()
            except Exception:
                pass
        self._ui_log_entries = []
        self._text_log_handle = None
        self._text_log_path = None
        self.results_dir = Path(results_dir) if results_dir else None

        self._apply_window_mode()

        # transcript overlay
        self.transcript_label = None
        self._current_transcript_text = ""

        # parsed experiment parameters
        self.stop_time = float(self.experiment_dict.get("stop_time", 6))
        self.action_time = float(self.experiment_dict.get("action_time", 4))
        self.repetitions_per_action = int(
            self.experiment_dict.get("each_action_repitition_times", 20)
        )

        self.beginning_baseline_duration = float(
            self.experiment_dict.get("beginning_baseline_recording", 15)
        )
        self.between_actions_baseline_duration = float(
            self.experiment_dict.get("between_actions_baseline_recording", 10)
        )
        self.ending_baseline_duration = float(
            self.experiment_dict.get("ending_baseline_recording", 20)
        )

        # action order is the key order of actions_dict
        self.action_names = list(self.actions_dict.keys())
        self.action_video_paths = [str(self.actions_dict[name]) for name in self.action_names]
        self._expected_experiment_duration = self._compute_expected_duration()

        # experiment state
        self.mode = self.MODE_BEGIN_BASELINE
        self.current_action_index = 0
        self.current_repetition = 0
        self.action_window = 0  # 0 = stop/baseline, 1 = action window

        # timing references
        self.experiment_start_time = None
        self.phase_start_time = None

        # cached status to avoid redundant repaints
        self._current_status_text = ""
        self._current_status_color = ""

        # logging
        self.log_file_handle = None
        self._init_logging()

        # storage for binary action signal (time in ns, action flag)
        self._action_signal_samples = []
        # storage for CSI capture on/off intervals (start_ns, end_ns)
        self._csi_capture_intervals = []
        self._signal_saved = False
        self._csi_signal_saved = False
        self._combined_signal_saved = False
        self._hand_data_saved = False
        self.hand_recognition_engine = None
        self._summary_window_shown = False
        self._summary_window_pending = False
        self._experiment_end_time = None
        self._transfer_dialog = TransferringFilesDialog(self)
        self._reboot_dialog: QMessageBox | None = None
        self._reboot_thread: threading.Thread | None = None

        # fill static labels
        self._init_labels()

        # video + media player
        self._init_video_player()
        self._init_webcam()
        self._init_hand_recognition()
        self._init_voice_assistant()

        # wi-fi CSI capture controls
        self._wifi_capture_threads = []
        self._wifi_routers = {}
        self._wifi_capture_runs = {}
        self._wifi_download_results = []
        self._ap_indices: dict[str, int] = {}
        self._ap_capture_counters: dict[str, int] = {}
        self._wifi_shutdown_event = threading.Event()
        self._wifi_capture_finalized = False
        self._wifi_scenario = str(
            self.wifi_profile.get("csi_capture_scenario", "scenario_2")
        ).lower()
        self._demo_capture_active = False
        self.pre_action_csi_duration = float(
            self.wifi_profile.get("pre_action_capture_duration", 2.0)
        )
        self.post_action_csi_duration = float(
            self.wifi_profile.get("post_action_capture_duration", 2.0)
        )
        self._pre_action_capture_started = False
        self._post_action_capture_started = False
        self._depth_capture_tokens_started: set[tuple[str, int, int]] = set()
        self._depth_capture_threads: list[threading.Thread] = []
        self.transfer_started.connect(self._transfer_dialog.handle_transfer_started)
        self.transfer_progress.connect(self._transfer_dialog.handle_transfer_progress)
        self.transfer_finished.connect(self._transfer_dialog.handle_transfer_finished)
        self.demo_plot_requested.connect(self._show_demo_plot_from_metadata)
        self._prestarted_wifi = prestarted_wifi or []
        self._init_ap_status_ui()
        self._init_demo_mode_controls()
        if self._prestarted_wifi:
            self._adopt_prestarted_wifi(self._prestarted_wifi)
            if self._wifi_scenario == "scenario_1":
                QtCore.QTimer.singleShot(0, self._start_prestarted_wifi_captures)
        elif start_wifi_capture:
            if self._wifi_scenario == "scenario_1":
                self._start_wifi_capture_workers()
            elif self._wifi_scenario == "demo":
                self._prepare_demo_wifi_routers()

        # ensure the beginning baseline timing starts as soon as the phase begins
        self._start_beginning_baseline(self._now())

        # timers for GUI updates
        self._init_timers()

        # separate thread only for beeping
        self._init_beep_thread()
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_beep_thread)

        # menu / actions connections
        self._make_connections()

        QtCore.QTimer.singleShot(0, self._apply_ui_profile_settings)

        # announce the initial baseline instructions
        self._announce_beginning_baseline()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _normalize_subject_metadata(self):
        subject = self.subject_dict
        subject.setdefault("name", "participant")
        subject.setdefault("age_group", "Blank")
        subject.setdefault("gender", "")
        if not subject.get("experiment_id"):
            subject["experiment_id"] = generate_experiment_id()
        if subject.get("age_group") == "Blank":
            subject.setdefault("age_value", 0)
        subject["participant_id"] = generate_participant_id(
            subject.get("name", ""),
            subject.get("age_group", ""),
            subject.get("gender", ""),
        )
        subject.setdefault("has_second_participant", False)
        subject.setdefault("second_name", "")
        subject.setdefault("second_gender", "")
        subject.setdefault("second_age_group", "Blank")
        subject["second_age_value"] = (
            0 if subject.get("second_age_group") == "Blank" else subject.get("second_age_value", 0)
        )
        subject["second_participant_id"] = generate_participant_id(
            subject.get("second_name", ""),
            subject.get("second_age_group", ""),
            subject.get("second_gender", ""),
        )
        subject.setdefault("second_dominant_hand", "")
        subject.setdefault("second_height_cm", 0.0)
        subject.setdefault("second_weight_kg", 0.0)
        subject.setdefault("second_description", "")

    def _init_labels(self):
        profile = self._ui_profile_with_defaults()
        # subject info
        if hasattr(self, "label_name"):
            self.label_name.setText(self.subject_dict.get("name", ""))
        if hasattr(self, "label_participant_id"):
            self.label_participant_id.setText(
                self.subject_dict.get("participant_id", "")
            )
        has_second_participant = self.subject_dict.get("has_second_participant", False)
        if hasattr(self, "label_second_name_title"):
            self.label_second_name_title.setVisible(has_second_participant)
        if hasattr(self, "label_second_name"):
            self.label_second_name.setText(self.subject_dict.get("second_name", ""))
            self.label_second_name.setVisible(has_second_participant)
        if hasattr(self, "label_second_participant_title"):
            self.label_second_participant_title.setVisible(has_second_participant)
        if hasattr(self, "label_second_participant_id"):
            self.label_second_participant_id.setText(
                self.subject_dict.get("second_participant_id", "")
            )
            self.label_second_participant_id.setVisible(has_second_participant)
        if hasattr(self, "label_id"):
            self.label_id.setText(self.subject_dict.get("experiment_id", ""))

        # initial status
        self.set_status(
            str(profile.get("status_baseline_text", "Baseline mode - Please do not move!")),
            "yellow",
        )
        if hasattr(self, "label_action"):
            try:
                self.label_action.setRange(0, 100)
                self.label_action.setValue(0)
                self.label_action.setTextVisible(True)
            except Exception:
                pass
        if hasattr(self, "label_curaction"):
            self.label_curaction.setText("Baseline")
        if hasattr(self, "label_count"):
            self.label_count.setText("--")
        if hasattr(self, "label_5"):
            self.label_5.setText(str(profile.get("remaining_time_title_text", "Remaining Time:")))
        if hasattr(self, "label_remain_time"):
            remaining = self._expected_experiment_duration
            self.label_remain_time.setText(self._format_time_remaining(remaining))
        if hasattr(self, "label_11"):
            self.label_11.setText(str(profile.get("name_title_text", "Name:")))
        if hasattr(self, "label_participant_title"):
            self.label_participant_title.setText(
                str(profile.get("participant_title_text", "Participant ID:"))
            )
        if hasattr(self, "label_12"):
            self.label_12.setText(str(profile.get("experiment_title_text", "Experiment ID:")))
        if hasattr(self, "label_second_name_title"):
            self.label_second_name_title.setText(
                str(profile.get("second_name_title_text", "Second Name:"))
            )
        if hasattr(self, "label_second_participant_title"):
            self.label_second_participant_title.setText(
                str(
                    profile.get(
                        "second_participant_title_text", "Second Participant ID:"
                    )
                )
            )
        if hasattr(self, "label_2"):
            self.label_2.setText(str(profile.get("elapsed_time_title_text", "Elapsed Time:")))
        if hasattr(self, "label_7"):
            self.label_7.setText(str(profile.get("count_title_text", "Count:")))
        if hasattr(self, "label_9"):
            self.label_9.setText(str(profile.get("current_action_title_text", "Current action:")))
        if hasattr(self, "label"):
            self.label.setText(str(profile.get("brand_header_text", "Wirlab")))

    def _init_video_player(self):
        """
        Embed a QVideoWidget inside the placeholder widget named 'videoWidget'.
        """
        container = self.findChild(QWidget, "videoWidget")
        if container is None:
            # fallback: create a floating video widget
            self.video_widget = QVideoWidget(self)
        else:
            layout = container.layout()
            if layout is None:
                layout = QVBoxLayout(container)
                layout.setContentsMargins(0, 0, 0, 0)
                container.setLayout(layout)
            self.video_widget = QVideoWidget(container)
            layout.addWidget(self.video_widget)

        self.media_player = QMediaPlayer(self)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.stateChanged.connect(self._on_media_state_changed)

        # Prepare webcam preview overlay placeholder
        self.video_widget.installEventFilter(self)
        self.webcam_preview_label = QLabel(self.video_widget)
        self.webcam_preview_label.setObjectName("webcamPreview")
        self.webcam_preview_label.setFixedSize(240, 180)
        self.webcam_preview_label.setAlignment(Qt.AlignCenter)
        profile = self._ui_profile_with_defaults()
        webcam_color = QColor(str(profile.get("webcam_font_color", "#ffffff")))
        if not webcam_color.isValid():
            webcam_color = QColor("#ffffff")
        self.webcam_preview_label.setStyleSheet(
            "background-color: black; border: 2px solid white; "
            f"color: {webcam_color.name()}; "
            f"margin-left: {int(profile.get('webcam_offset_x', 0))}px; "
            f"margin-top: {int(profile.get('webcam_offset_y', 0))}px;"
        )
        webcam_font = self.webcam_preview_label.font()
        webcam_font.setPointSize(int(profile.get("webcam_font_size", 12)))
        self.webcam_preview_label.setFont(webcam_font)
        self.webcam_preview_label.setText(
            str(profile.get("webcam_disabled_text", "Webcam disabled"))
        )
        self.webcam_preview_label.hide()
        self._position_webcam_preview()

        # Transcript overlay shown during idle/stop phases
        self.transcript_label = QLabel(self.video_widget)
        self.transcript_label.setObjectName("transcriptOverlay")
        self.transcript_label.setAlignment(Qt.AlignCenter)
        self.transcript_label.setWordWrap(True)
        self.transcript_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 180);"
            "color: white;"
            "font-size: 24px;"
            "padding: 16px;"
        )
        self.transcript_label.hide()
        self._layout_transcript_overlay()

        # Preview widget that loops the upcoming action video during baselines
        self.preview_widget = QVideoWidget(self.video_widget)
        self.preview_widget.setObjectName("nextActionPreview")
        preview_size = self.webcam_preview_label.size()
        if preview_size.isEmpty():
            self.preview_widget.setFixedSize(240, 180)
        else:
            self.preview_widget.setFixedSize(preview_size)
        self.preview_widget.setStyleSheet(
            "background-color: black; border: 2px solid white;"
        )
        self.preview_widget.hide()
        self.preview_player = QMediaPlayer(self)
        self.preview_player.setVideoOutput(self.preview_widget)
        self.preview_player.mediaStatusChanged.connect(
            self._on_preview_media_status_changed
        )
        self._layout_preview_widget()
        self._setup_about_menu()
        self._capture_base_frame_sizes()

    def _label_font_family(self, label: QLabel | None) -> str:
        if label is None:
            return ""
        return label.font().family()

    def _label_font_size(self, label: QLabel | None, fallback: int) -> int:
        if label is None:
            return fallback
        size = label.font().pointSize()
        return size if size > 0 else fallback

    def _label_color_hex(self, label: QLabel | None, fallback: str = "#000000") -> str:
        if label is None:
            return fallback
        palette = label.palette()
        color = palette.color(label.foregroundRole())
        if not color.isValid():
            color = QColor(fallback)
        return color.name()

    def _coerce_bool(self, value, default: bool = False) -> bool:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on", "y"}:
                return True
            if lowered in {"0", "false", "no", "off", "n"}:
                return False
            return default
        try:
            return bool(value)
        except Exception:
            return default

    def _ui_profile_text(self, key: str, fallback: str) -> str:
        profile = self._ui_profile_with_defaults()
        value = profile.get(key, fallback)
        return str(value) if value is not None else ""

    def _format_guidance_message(self, template: str, action: str | None = None) -> str:
        action_value = action or ""
        try:
            return template.format(action=action_value)
        except Exception:
            return template.replace("{action}", action_value)

    def _default_ui_settings(self) -> dict:
        if self._ui_defaults is None:
            self._ui_defaults = {
                "camera_frame_percent": 100.0,
                "preview_frame_percent": 100.0,
                "action_frame_percent": 100.0,
                "details_pane_percent": 30.0,
                "show_log": True,
                "log_frame_percent": 100.0,
                "start_fullscreen": False,
                "count_font_family": self._label_font_family(getattr(self, "label_count", None)),
                "count_font_size": self._label_font_size(getattr(self, "label_count", None), 12),
                "count_font_color": self._label_color_hex(getattr(self, "label_count", None)),
                "action_font_family": self._label_font_family(getattr(self, "label_curaction", None)),
                "action_font_size": self._label_font_size(getattr(self, "label_curaction", None), 12),
                "action_font_color": self._label_color_hex(getattr(self, "label_curaction", None)),
                "time_font_family": self._label_font_family(getattr(self, "label_time", None)),
                "time_font_size": self._label_font_size(getattr(self, "label_time", None), 20),
                "time_font_color": self._label_color_hex(getattr(self, "label_time", None)),
                "remaining_time_value_font_family": self._label_font_family(getattr(self, "label_remain_time", None)),
                "remaining_time_value_font_size": self._label_font_size(getattr(self, "label_remain_time", None), 12),
                "remaining_time_value_font_color": self._label_color_hex(getattr(self, "label_remain_time", None)),
                "elapsed_time_value_font_family": self._label_font_family(getattr(self, "label_elapsed_time", None)),
                "elapsed_time_value_font_size": self._label_font_size(getattr(self, "label_elapsed_time", None), 12),
                "elapsed_time_value_font_color": self._label_color_hex(getattr(self, "label_elapsed_time", None)),
                "transcript_position": "middle",
                "transcript_font_size": self._label_font_size(
                    getattr(self, "transcript_label", None), 24
                ),
                "transcript_font_color": "#ffffff",
                "transcript_offset_x": 0,
                "transcript_offset_y": 0,
                "status_font_size": 20,
                "status_font_color": "#000000",
                "status_offset_x": 0,
                "status_offset_y": 0,
                "status_baseline_text": "Baseline mode - Please do not move!",
                "status_stop_text": "Stop - Please do not move!",
                "status_action_text": "Action",
                "status_finished_text": "Experiment finished",
                "guide_next_action_with_name": "Next action: {action}. Perform the action only once after the beep.",
                "guide_next_action_none": "No action. Please remain still.",
                "guide_beginning_with_action": (
                    "The experiment starts soon. Get ready to perform the activity {action}. "
                    "This is a preview of the next action."
                ),
                "guide_beginning_no_action": "The experiment starts soon. Get ready to perform the activity.",
                "guide_between_with_action": (
                    "Wait for the next action to start soon. Next action: {action}. "
                    "This is a preview of the next action."
                ),
                "guide_between_no_action": "Wait for the next action to start soon.",
                "guide_end_baseline": "Remain seated and do not move until the experiment is finished.",
                "guide_experiment_finished": "The experiment is finished. Thank you for participating.",
                "log_placeholder_text": "Log output will appear here.",
                "log_font_size": 10,
                "log_font_color": "#000000",
                "log_offset_x": 0,
                "log_offset_y": 0,
                "webcam_font_size": 12,
                "webcam_font_color": "#ffffff",
                "webcam_offset_x": 0,
                "webcam_offset_y": 0,
                "webcam_disabled_text": "Webcam disabled",
                "webcam_starting_text": "Starting webcam...",
                "webcam_unavailable_text": "Webcam unavailable",
                "webcam_opencv_unavailable_text": "OpenCV unavailable",
            }
            for label_name, prefix, default_text, default_size in self.UI_LABEL_CONFIGS:
                label = getattr(self, label_name, None)
                self._ui_defaults.setdefault(
                    f"{prefix}_text", label.text() if label is not None else default_text
                )
                self._ui_defaults.setdefault(
                    f"{prefix}_font_size",
                    self._label_font_size(label, default_size),
                )
                self._ui_defaults.setdefault(
                    f"{prefix}_color", self._label_color_hex(label, "#000000")
                )
                self._ui_defaults.setdefault(f"{prefix}_offset_x", 0)
                self._ui_defaults.setdefault(f"{prefix}_offset_y", 0)
            for label_name, prefix, default_size in self.UI_VALUE_LABEL_CONFIGS:
                label = getattr(self, label_name, None)
                self._ui_defaults.setdefault(
                    f"{prefix}_font_size",
                    self._label_font_size(label, default_size),
                )
                self._ui_defaults.setdefault(
                    f"{prefix}_color", self._label_color_hex(label, "#000000")
                )
                self._ui_defaults.setdefault(f"{prefix}_offset_x", 0)
                self._ui_defaults.setdefault(f"{prefix}_offset_y", 0)
        return dict(self._ui_defaults)

    def _ui_profile_with_defaults(self) -> dict:
        merged = self._default_ui_settings()
        for key, value in (self.ui_profile or {}).items():
            merged[key] = value
        return merged

    def _apply_window_mode(self):
        profile = self._ui_profile_with_defaults()
        start_fullscreen = self._coerce_bool(profile.get("start_fullscreen", False))
        if start_fullscreen:
            self.setWindowState(Qt.WindowFullScreen)
        else:
            # ensure we leave fullscreen if previously set
            current = self.windowState() & ~Qt.WindowFullScreen
            self.setWindowState(current | Qt.WindowMaximized)

    def _capture_base_frame_sizes(self):
        if self.webcam_preview_label is not None:
            size = self.webcam_preview_label.size()
            if size.isEmpty():
                size = QSize(240, 180)
            self._base_webcam_size = size
        if self.preview_widget is not None:
            size = self.preview_widget.size()
            if size.isEmpty():
                size = self._base_webcam_size or QSize(240, 180)
            self._base_preview_size = size
        if hasattr(self, "video_widget") and self.video_widget is not None:
            size = self.video_widget.size()
            if size.isEmpty():
                size = self.video_widget.sizeHint()
            if size.isEmpty():
                size = QSize(640, 480)
            self._base_action_frame_size = size
        if self.log_widget is not None:
            height = self.log_widget.height() or self.log_widget.sizeHint().height() or 120
            self._base_log_height = height

    def _apply_label_style(
        self,
        label: QLabel | None,
        family: str,
        size: int,
        color_hex: str,
        offset_x: int = 0,
        offset_y: int = 0,
    ):
        if label is None:
            return
        font = label.font()
        if family:
            font.setFamily(str(family))
        if size > 0:
            font.setPointSize(int(size))
        label.setFont(font)
        color = QColor(color_hex)
        if not color.isValid():
            color = QColor("#000000")
        label.setStyleSheet(
            f"color: {color.name()}; margin-left: {int(offset_x)}px; margin-top: {int(offset_y)}px;"
        )

    def _apply_details_pane_size(self, profile: dict | None = None):
        splitter = getattr(self, "splitter", None)
        details_widget = getattr(self, "groupBox", None)
        if splitter is None or details_widget is None:
            return
        if splitter.count() < 2:
            return
        profile = profile or self._ui_profile_with_defaults()
        try:
            percent = float(profile.get("details_pane_percent", 30.0))
        except (TypeError, ValueError):
            percent = 30.0
        percent = max(10.0, min(90.0, percent))
        total_width = splitter.size().width()
        if total_width <= 0:
            total_width = splitter.sizeHint().width()
        if total_width <= 0:
            total_width = self.width()
        if total_width <= 0:
            return
        details_width = max(1, round(total_width * percent / 100.0))
        other_width = max(1, total_width - details_width)
        details_widget.setMinimumWidth(details_width)
        details_widget.setMaximumWidth(details_width)
        details_widget.setSizePolicy(QSizePolicy.Fixed, details_widget.sizePolicy().verticalPolicy())
        if splitter.indexOf(details_widget) == 0:
            sizes = [details_width, other_width]
        else:
            sizes = [other_width, details_width]
        splitter.setChildrenCollapsible(False)
        splitter.setSizes(sizes)

    def _apply_ui_profile_settings(self):
        self._capture_base_frame_sizes()
        profile = self._ui_profile_with_defaults()
        self._apply_window_mode()

        def _scaled_size(base: QSize | None, percent: float, fallback: QSize) -> QSize:
            base_size = base if base and not base.isEmpty() else fallback
            factor = max(percent, 10.0) / 100.0
            return QSize(
                max(1, round(base_size.width() * factor)),
                max(1, round(base_size.height() * factor)),
            )

        webcam_size = _scaled_size(self._base_webcam_size, float(profile.get("camera_frame_percent", 100.0)), QSize(240, 180))
        preview_size = _scaled_size(self._base_preview_size, float(profile.get("preview_frame_percent", 100.0)), webcam_size)
        action_size = _scaled_size(self._base_action_frame_size, float(profile.get("action_frame_percent", 100.0)), QSize(640, 480))

        if self.webcam_preview_label is not None:
            self.webcam_preview_label.setFixedSize(webcam_size)
            self._position_webcam_preview()
        if self.preview_widget is not None:
            self.preview_widget.setFixedSize(preview_size)
            self._layout_preview_widget()
        if hasattr(self, "video_widget") and self.video_widget is not None:
            self.video_widget.setMinimumSize(action_size)
            self.video_widget.setMaximumSize(action_size)
            self.video_widget.resize(action_size)
            self._layout_transcript_overlay()

        if self.log_widget is not None:
            self.log_widget.setVisible(bool(profile.get("show_log", True)))
            base_height = self._base_log_height or self.log_widget.sizeHint().height() or 120
            height = max(1, round(base_height * max(float(profile.get("log_frame_percent", 100.0)), 10.0) / 100.0))
            self.log_widget.setFixedHeight(height)
            log_font = self.log_widget.font()
            log_font.setPointSize(int(profile.get("log_font_size", 10)))
            self.log_widget.setFont(log_font)
            log_color = QColor(str(profile.get("log_font_color", "#000000")))
            if not log_color.isValid():
                log_color = QColor("#000000")
            self.log_widget.setStyleSheet(
                f"color: {log_color.name()}; padding-left: {int(profile.get('log_offset_x', 0))}px; "
                f"padding-top: {int(profile.get('log_offset_y', 0))}px;"
            )
            if not self._ui_log_entries:
                placeholder = str(profile.get("log_placeholder_text", "")).strip()
                if placeholder:
                    self.log_widget.setPlainText(placeholder)

        self._apply_details_pane_size(profile)

        if self.webcam_preview_label is not None:
            webcam_color = QColor(str(profile.get("webcam_font_color", "#ffffff")))
            if not webcam_color.isValid():
                webcam_color = QColor("#ffffff")
            webcam_font = self.webcam_preview_label.font()
            webcam_font.setPointSize(int(profile.get("webcam_font_size", 12)))
            self.webcam_preview_label.setFont(webcam_font)
            self.webcam_preview_label.setStyleSheet(
                "background-color: black; border: 2px solid white; "
                f"color: {webcam_color.name()}; "
                f"margin-left: {int(profile.get('webcam_offset_x', 0))}px; "
                f"margin-top: {int(profile.get('webcam_offset_y', 0))}px;"
            )

        self._apply_label_style(
            getattr(self, "label_count", None),
            profile.get("count_font_family", ""),
            int(profile.get("count_font_size", 12)),
            str(profile.get("count_font_color", "#000000")),
        )
        self._apply_label_style(
            getattr(self, "label_curaction", None),
            profile.get("action_font_family", ""),
            int(profile.get("action_font_size", 12)),
            str(profile.get("action_font_color", "#000000")),
        )
        self._apply_label_style(
            getattr(self, "label_time", None),
            profile.get("time_font_family", ""),
            int(profile.get("time_font_size", 20)),
            str(profile.get("time_font_color", "#000000")),
        )
        self._apply_label_style(
            getattr(self, "label_remain_time", None),
            profile.get("remaining_time_value_font_family", ""),
            int(profile.get("remaining_time_value_font_size", 12)),
            str(profile.get("remaining_time_value_font_color", "#000000")),
        )
        self._apply_label_style(
            getattr(self, "label_elapsed_time", None),
            profile.get("elapsed_time_value_font_family", ""),
            int(profile.get("elapsed_time_value_font_size", 12)),
            str(profile.get("elapsed_time_value_font_color", "#000000")),
        )

        for label_name, prefix, _default_text, _default_size in self.UI_LABEL_CONFIGS:
            label = getattr(self, label_name, None)
            if label is None:
                continue
            label.setText(str(profile.get(f"{prefix}_text", label.text())))
            self._apply_label_style(
                label,
                "",
                int(profile.get(f"{prefix}_font_size", label.font().pointSize())),
                str(profile.get(f"{prefix}_color", "#000000")),
                int(profile.get(f"{prefix}_offset_x", 0)),
                int(profile.get(f"{prefix}_offset_y", 0)),
            )
        for label_name, prefix, _default_size in self.UI_VALUE_LABEL_CONFIGS:
            label = getattr(self, label_name, None)
            if label is None:
                continue
            self._apply_label_style(
                label,
                "",
                int(profile.get(f"{prefix}_font_size", label.font().pointSize())),
                str(profile.get(f"{prefix}_color", "#000000")),
                int(profile.get(f"{prefix}_offset_x", 0)),
                int(profile.get(f"{prefix}_offset_y", 0)),
            )

        if hasattr(self, "label_action"):
            self._apply_status_color(str(self._current_status_color or "yellow"))

        if self.transcript_label is not None:
            transcript_size = max(6, int(profile.get("transcript_font_size", 24)))
            position = str(profile.get("transcript_position", "middle")).lower()
            alignment = Qt.AlignCenter
            if position == "top":
                alignment = Qt.AlignHCenter | Qt.AlignTop
            self.transcript_label.setAlignment(alignment)
            transcript_color = QColor(str(profile.get("transcript_font_color", "#ffffff")))
            if not transcript_color.isValid():
                transcript_color = QColor("#ffffff")
            self.transcript_label.setStyleSheet(
                "background-color: rgba(0, 0, 0, 180);"
                f"color: {transcript_color.name()};"
                f"font-size: {transcript_size}px;"
                "padding: 16px;"
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_details_pane_size()

    def _format_time_remaining(self, seconds: float) -> str:
        total_ms = int(max(0.0, seconds) * 1000)
        hours, remainder = divmod(total_ms, 3600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, milliseconds = divmod(remainder, 1000)
        return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"

    def _should_use_webcam(self):
        source = self.camera_profile if self.camera_profile else self.experiment_dict
        value = source.get("use_webcam", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "y"}
        return bool(value)

    def _should_use_hand_recognition(self):
        source = self.camera_profile if self.camera_profile else self.experiment_dict
        value = source.get("use_hand_recognition", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "y"}
        return bool(value)

    def _hand_recognition_mode(self) -> str:
        return "live"

    def _camera_device_source(self):
        source = self.camera_profile if self.camera_profile else self.experiment_dict
        device = source.get("camera_device", 0)
        if isinstance(device, str):
            device = device.strip()
            if not device:
                return 0
            try:
                return int(device)
            except ValueError:
                return device
        try:
            return int(device)
        except (TypeError, ValueError):
            return 0

    def _should_use_voice_assistant(self):
        if not isinstance(self.voice_profile, dict):
            return False
        value = self.voice_profile.get("use_voice_assistant", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "y"}
        return bool(value)

    def _init_voice_assistant(self):
        if not self.voice_assistant_enabled:
            return
        try:
            language = (self.voice_profile.get("language") or "en").strip() or "en"
            self.voice_assistant = GTTSVoiceAssistant(language=language, parent=self)
        except Exception:
            self.voice_assistant = None
            self.voice_assistant_enabled = False

    # ------------------------------------------------------------------
    # Wi-Fi CSI capture handling
    # ------------------------------------------------------------------
    def _init_ap_status_ui(self):
        # Access point indicators have been removed from the UI.
        self._ap_status_widgets = {}

    def _update_ap_status(self, ap_key: str, color: str, message: str):  # noqa: ARG002
        """Previously updated access point indicators; now a no-op."""
        return

    # ------------------------------------------------------------------
    # Wi-Fi session helpers
    # ------------------------------------------------------------------
    def _get_run_sessions(self, ap_key: str) -> list[dict]:
        sessions = self._wifi_capture_runs.get(ap_key)
        if sessions is None:
            return []
        if isinstance(sessions, dict):
            sessions = [sessions]
            self._wifi_capture_runs[ap_key] = sessions
        return sessions

    def _register_ap_index(self, ap_key: str) -> int:
        if ap_key not in self._ap_indices:
            self._ap_indices[ap_key] = len(self._ap_indices) + 1
        return self._ap_indices[ap_key]

    def _get_ap_directory_name(self, ap_key: str) -> str:
        self._register_ap_index(ap_key)
        return ap_key

    def _reserve_capture_index(self, ap_key: str) -> int:
        self._register_ap_index(ap_key)
        next_index = self._ap_capture_counters.get(ap_key, 1)
        self._ap_capture_counters[ap_key] = next_index + 1
        return next_index

    def _append_run_session(self, ap_key: str, run_info: dict) -> dict:
        sessions = self._get_run_sessions(ap_key)
        sessions.append(run_info)
        self._wifi_capture_runs[ap_key] = sessions
        return run_info

    def _get_base_run_info(self, ap_key: str) -> dict | None:
        sessions = self._get_run_sessions(ap_key)
        return sessions[0] if sessions else None

    def _adopt_prestarted_wifi(self, routers_info: list[dict]):
        if not routers_info:
            return

        for info in routers_info:
            ap_key = info.get("key")
            router = info.get("router")
            run_info = info.get("run_info", {})
            if not ap_key or router is None:
                continue
            self._register_ap_index(ap_key)
            self._wifi_routers[ap_key] = router
            run_info.setdefault("duration", self._expected_experiment_duration)
            run_info.setdefault("started", False)
            run_info.setdefault("delete_prev_pcap", self._should_delete_prev_pcap())
            self._append_run_session(ap_key, run_info)
            self._update_ap_status(ap_key, "yellow", "Ready for capture")

    def _start_wifi_capture_workers(self):
        if self._wifi_capture_runs:
            for ap_key in self._wifi_capture_runs.keys():
                sessions = self._get_run_sessions(ap_key)
                started = any(run.get("started", False) for run in sessions)
                color = "green" if started else "yellow"
                message = "Capturing" if started else "Ready for capture"
                self._update_ap_status(ap_key, color, message)
            return
        scenario = self._wifi_scenario
        if scenario != "scenario_1":
            return

        access_points = self.wifi_manager.prioritize_transmitters_first(
            self.wifi_profile.get("access_points", [])
        )
        if not access_points:
            return

        for idx, ap in enumerate(access_points):
            thread = threading.Thread(
                target=self._wifi_capture_worker,
                args=(idx, ap),
                name=f"WiFiCapture-{idx}",
                daemon=True,
            )
            thread.start()
            self._wifi_capture_threads.append(thread)

    def _init_demo_mode_controls(self):
        self.demo_controls_widget = None
        self.btn_demo_collect = None
        self.btn_demo_close = None
        if self._wifi_scenario != "demo":
            return

        details_group = getattr(self, "groupBox", None)
        details_layout = details_group.layout() if details_group is not None else None
        if details_layout is None:
            return

        self.demo_controls_widget = QWidget(self)
        controls_layout = QVBoxLayout(self.demo_controls_widget)
        controls_layout.setContentsMargins(0, 8, 0, 0)
        title_label = QLabel("Demo CSI Mode")
        title_label.setStyleSheet("font-weight: bold;")
        controls_layout.addWidget(title_label)

        buttons_layout = QHBoxLayout()
        self.btn_demo_collect = QPushButton("Collect + Plot CSI", self.demo_controls_widget)
        self.btn_demo_collect.clicked.connect(self._run_demo_capture)
        buttons_layout.addWidget(self.btn_demo_collect)

        self.btn_demo_close = QPushButton("Close Window", self.demo_controls_widget)
        self.btn_demo_close.clicked.connect(self.close)
        buttons_layout.addWidget(self.btn_demo_close)
        controls_layout.addLayout(buttons_layout)
        details_layout.addWidget(self.demo_controls_widget)

    def _prepare_demo_wifi_routers(self):
        access_points = self.wifi_manager.prioritize_transmitters_first(
            self.wifi_profile.get("access_points", [])
        )
        for idx, ap in enumerate(access_points):
            thread = threading.Thread(
                target=self._demo_wifi_prepare_worker,
                args=(idx, ap),
                name=f"WiFiDemoPrepare-{idx}",
                daemon=True,
            )
            thread.start()
            self._wifi_capture_threads.append(thread)

    def _demo_wifi_prepare_worker(self, idx: int, ap: dict):
        ap_name = ap.get("name") or ap.get("ssid") or f"Access Point {idx + 1}"
        ap_key = self._sanitize_ap_name(ap_name)
        self._register_ap_index(ap_key)
        remote_dir = self._get_capture_remote_dir(ap)
        channel_bw = self.wifi_manager.format_channel_bandwidth(
            ap.get("channel", ""), ap.get("bandwidth", "")
        )
        mac_addresses = self.wifi_manager.parse_mac_addresses(
            ap.get("transmitter_macs", "")
        )
        skip_router_setup = bool(
            ap.get("initialized_success")
            or ap.get("skip_router_setup")
            or self.wifi_profile.get("skip_router_setup")
        )
        self._update_ap_status(ap_key, "yellow", "Connecting…")
        try:
            router = self.wifi_manager.connect_router(ap, WiFiRouter)
            self.wifi_manager.ensure_router_ready(
                router,
                channel_bw=channel_bw,
                mac_addresses=mac_addresses,
                skip_setup=skip_router_setup,
                log=self._append_log_entry,
            )
            self._wifi_routers[ap_key] = router
            self._append_run_session(
                ap_key,
                {
                    "ap": ap,
                    "exp_name": self._build_capture_prefix(ap_name),
                    "remote_dir": remote_dir,
                    "macs": mac_addresses,
                    "channel_bw": channel_bw,
                    "duration": max(self.action_time, 1.0),
                    "delete_prev_pcap": False,
                    "started": False,
                },
            )
            self._update_ap_status(ap_key, "green", "Ready for demo capture")
        except Exception as exc:  # pragma: no cover - network dependency
            self._append_log_entry(f"Demo setup failed on {ap_name}: {exc}")
            self._update_ap_status(ap_key, "red", "Demo setup failed")

    def _run_demo_capture(self):
        if self._wifi_scenario != "demo":
            return
        if self._demo_capture_active:
            return
        if not self._wifi_routers:
            self._append_log_entry("Demo mode: no prepared routers are available.")
            return
        self._demo_capture_active = True
        if self.btn_demo_collect is not None:
            self.btn_demo_collect.setEnabled(False)
        self.set_status("Demo capture started - perform the action now.", "green")
        self._set_transcript_message("Demo capture running. Please perform the action now.")

        worker = threading.Thread(
            target=self._run_demo_capture_worker,
            name="WiFiDemoCaptureCoordinator",
            daemon=True,
        )
        worker.start()
        self._wifi_capture_threads.append(worker)

    def _run_demo_capture_worker(self):
        duration = max(
            float(self.pre_action_csi_duration) + float(self.action_time) + float(self.post_action_csi_duration),
            1.0,
        )
        repetition = max(1, self.current_repetition + 1)
        action_index = max(0, self.current_action_index)
        action_name = self.action_names[action_index] if self.action_names else "demo"
        timestamp_suffix = self._now_datetime().strftime("%Y%m%d_%H%M%S")

        for ap_key, router in list(self._wifi_routers.items()):
            base_info = self._get_base_run_info(ap_key) or {}
            ap = base_info.get("ap", {})
            ap_name = ap.get("name") or ap.get("ssid") or ap_key
            exp_name = f"{self._build_capture_prefix(ap_name)}_demo_r{repetition}_{timestamp_suffix}"
            run_info = {
                "ap": ap,
                "exp_name": exp_name,
                "remote_dir": base_info.get("remote_dir", self._get_capture_remote_dir(ap)),
                "macs": base_info.get("macs", []),
                "channel_bw": base_info.get("channel_bw", ""),
                "duration": duration,
                "delete_prev_pcap": False,
                "started": False,
                "activity": f"demo_{action_name}",
                "repetition_count": repetition,
                "action_index": action_index,
            }
            self._append_run_session(ap_key, run_info)
            self._start_capture_on_router(
                ap_key,
                router,
                run_info=run_info,
                override_duration=duration,
                override_exp_name=exp_name,
                override_remote_dir=run_info["remote_dir"],
            )
            self._download_capture_session(ap_key, router, run_info)
            run_info["downloaded"] = True

        self._demo_capture_active = False
        if self.btn_demo_collect is not None:
            QtCore.QTimer.singleShot(0, lambda: self.btn_demo_collect.setEnabled(True))
        QtCore.QTimer.singleShot(
            0, lambda: self.set_status("Demo capture finished. Press Collect to run again.", "yellow")
        )

    def _maybe_schedule_action_captures(self, t_in_cycle: float):
        if self._wifi_scenario != "scenario_2":
            return
        if not self._wifi_routers:
            return

        pre_trigger = max(self.stop_time - self.pre_action_csi_duration, 0.0)
        if (
            self.action_window == 0
            and not self._pre_action_capture_started
            and t_in_cycle >= pre_trigger
        ):
            repetition = max(1, self.current_repetition + 1)
            duration = (
                self.pre_action_csi_duration
                + self.action_time
                + self.post_action_csi_duration
            )
            self._start_action_adjacent_capture(
                "pre_action",
                duration,
                self.current_action_index,
                repetition,
            )
            self._pre_action_capture_started = True

    def _trigger_post_action_capture(self):
        if self._wifi_scenario != "scenario_2":
            return
        if self._pre_action_capture_started:
            # A single capture already spans pre, action, and post windows.
            self._post_action_capture_started = True
            return
        if self._post_action_capture_started:
            return
        if not self._wifi_routers:
            return

        repetition = max(1, self.current_repetition)
        self._start_action_adjacent_capture(
            "post_action",
            self.post_action_csi_duration,
            self.current_action_index,
            repetition,
        )
        self._post_action_capture_started = True

    def _wifi_capture_worker(self, idx: int, ap: dict):
        ap_name = ap.get("name") or ap.get("ssid") or f"Access Point {idx + 1}"
        ap_key = self._sanitize_ap_name(ap_name)
        self._register_ap_index(ap_key)
        remote_dir = self._get_capture_remote_dir(ap)
        duration = max(self._expected_experiment_duration, 1.0)
        exp_name = self._build_capture_prefix(ap_name)
        channel_bw = self.wifi_manager.format_channel_bandwidth(
            ap.get("channel", ""), ap.get("bandwidth", "")
        )
        mac_addresses = self.wifi_manager.parse_mac_addresses(
            ap.get("transmitter_macs", "")
        )
        delete_prev_pcap = self._should_delete_prev_pcap()
        skip_router_setup = bool(
            ap.get("initialized_success")
            or ap.get("skip_router_setup")
            or self.wifi_profile.get("skip_router_setup")
        )

        self._update_ap_status(ap_key, "yellow", "Connecting…")
        try:
            router = self.wifi_manager.connect_router(ap, WiFiRouter)
        except Exception as exc:  # pragma: no cover - network dependency
            self._append_log_entry(f"Failed to connect to {ap_name}: {exc}")
            self._update_ap_status(ap_key, "red", "Connection failed")
            return

        self._wifi_routers[ap_key] = router
        self._append_run_session(
            ap_key,
            {
                "ap": ap,
                "exp_name": exp_name,
                "remote_dir": remote_dir,
                "macs": mac_addresses,
                "channel_bw": channel_bw,
                "duration": duration,
                "delete_prev_pcap": delete_prev_pcap,
                "started": False,
            },
        )

        try:
            self.wifi_manager.ensure_router_ready(
                router,
                channel_bw=channel_bw,
                mac_addresses=mac_addresses,
                skip_setup=skip_router_setup,
                log=self._append_log_entry,
            )

            self._start_capture_on_router(
                ap_key,
                router,
                run_info=self._wifi_capture_runs[ap_key],
                override_duration=duration,
                override_exp_name=exp_name,
                override_remote_dir=remote_dir,
            )
        except Exception as exc:  # pragma: no cover - network dependency
            self._append_log_entry(f"CSI capture failed on {ap_name}: {exc}")
            self._update_ap_status(ap_key, "red", "Capture failed")

    def _start_action_adjacent_capture(
        self, label: str, duration: float, action_index: int, repetition: int
    ):
        if not self._wifi_routers:
            return

        action_name = ""
        if 0 <= action_index < len(self.action_names):
            action_name = self.action_names[action_index]

        timestamp_suffix = self._now_datetime().strftime("%Y%m%d_%H%M%S")
        router_items = [
            (ap_key, router, self._get_base_run_info(ap_key))
            for ap_key, router in list(self._wifi_routers.items())
        ]
        router_items = [item for item in router_items if item[2] is not None]
        router_items.sort(
            key=lambda item: self.wifi_manager.is_sniffer(item[2].get("ap", {}))
        )

        for ap_key, router, base_info in router_items:
            ap = base_info.get("ap", {})
            ap_name = ap.get("name") or ap.get("ssid") or ap_key
            base_prefix = self._build_capture_prefix(ap_name)
            exp_name = (
                f"{base_prefix}_{label}_a{action_index + 1}_r{repetition}_{timestamp_suffix}"
            )
            run_info = {
                "ap": ap,
                "exp_name": exp_name,
                "remote_dir": base_info.get("remote_dir", self._get_capture_remote_dir(ap)),
                "macs": base_info.get("macs", []),
                "channel_bw": base_info.get("channel_bw", ""),
                "duration": duration,
                "delete_prev_pcap": base_info.get(
                    "delete_prev_pcap", self._should_delete_prev_pcap()
                ),
                "started": False,
                "activity": f"{label}_{action_name}".strip("_"),
                "repetition_count": repetition,
                "action_index": action_index,
            }
            self._append_run_session(ap_key, run_info)
            thread = threading.Thread(
                target=self._start_capture_on_router,
                args=(ap_key, router),
                kwargs={
                    "run_info": run_info,
                    "override_duration": duration,
                    "override_exp_name": exp_name,
                    "override_remote_dir": run_info["remote_dir"],
                },
                name=f"WiFiCapture-{ap_key}-{label}",
                daemon=True,
            )
            thread.start()
            self._wifi_capture_threads.append(thread)

    def _get_capture_remote_dir(self, ap: dict) -> str:
        return self.wifi_manager.get_capture_remote_dir(ap)

    def _should_delete_prev_pcap(self) -> bool:
        if self._wifi_scenario == "scenario_2":
            return False
        return bool(self.wifi_profile.get("delete_prev_pcap", False))

    def _start_prestarted_wifi_captures(self):
        if not self._wifi_capture_runs:
            return

        router_items = list(self._wifi_routers.items())
        router_items.sort(
            key=lambda item: self.wifi_manager.is_sniffer(
                (self._get_base_run_info(item[0]) or {}).get("ap", {})
            )
        )

        for ap_key, router in router_items:
            sessions = [run for run in self._get_run_sessions(ap_key) if not run.get("started")]
            for run_info in sessions:
                if router is None:
                    continue
                thread = threading.Thread(
                    target=self._start_capture_on_router,
                    args=(ap_key, router),
                    kwargs={"run_info": run_info},
                    name=f"WiFiCaptureStart-{ap_key}",
                    daemon=True,
                )
                thread.start()
                self._wifi_capture_threads.append(thread)

    def _start_capture_on_router(
        self,
        ap_key: str,
        router,
        *,
        run_info: dict | None = None,
        override_duration: float | None = None,
        override_exp_name: str | None = None,
        override_remote_dir: str | None = None,
    ):
        sessions = self._get_run_sessions(ap_key)
        run_info = run_info or (sessions[0] if sessions else {})
        ap = run_info.get("ap", {})
        ap_name = ap.get("name") or ap.get("ssid") or ap_key
        capture_index = run_info.get("capture_index")
        if capture_index is None:
            capture_index = self._reserve_capture_index(ap_key)
            run_info["capture_index"] = capture_index
        duration = max(
            float(
                override_duration
                or run_info.get("duration")
                or self._expected_experiment_duration
            ),
            1.0,
        )
        remote_dir = override_remote_dir or run_info.get("remote_dir", "/mnt/CSI_USB/")
        exp_name = override_exp_name or run_info.get("exp_name") or self._build_capture_prefix(ap_name)
        mac_addresses = run_info.get("macs", [])
        delete_prev_pcap = self._should_delete_prev_pcap()
        if self._wifi_scenario != "scenario_2":
            delete_prev_pcap = bool(run_info.get("delete_prev_pcap", delete_prev_pcap))

        start_ns = self._now_ns()
        action_name = run_info.get("activity") or "full_experiment"
        repetition_count = int(run_info.get("repetition_count") or 1)
        exp_name = self._build_capture_prefix(
            ap_name,
            start_ns=start_ns,
            action_name=action_name,
            repetition_count=repetition_count,
        )
        run_info["exp_name"] = exp_name
        run_info["start_timestamp_ns"] = start_ns
        run_info["capture_index"] = capture_index

        try:
            self.wifi_manager.start_csi_capture(
                router,
                duration=duration,
                remote_directory=remote_dir,
                exp_name=exp_name,
                delete_prev_pcap=delete_prev_pcap,
            )
            depth_action_index = run_info.get("action_index", self.current_action_index)
            depth_repetition = int(run_info.get("repetition_count") or 1)
            self._start_depth_capture_if_enabled(
                label=action_name,
                action_index=depth_action_index,
                repetition=depth_repetition,
            )
            end_ns = self._now_ns()
            capture_label = self._format_capture_label(
                ap_name,
                start_ns=start_ns,
                end_ns=end_ns,
                action_name=action_name,
                count=repetition_count,
            )
            run_info["duration"] = duration
            run_info["remote_dir"] = remote_dir
            run_info["started"] = True
            run_info["end_timestamp_ns"] = end_ns
            run_info["capture_label"] = capture_label
            run_info["repetition_count"] = repetition_count
            if run_info not in sessions:
                self._append_run_session(ap_key, run_info)
            self._append_log_entry(
                f"Ended CSI capture on {ap_name}: expected {duration:.1f} seconds, obtained {(end_ns - start_ns) / 1_000_000_000:.1f} seconds."
            )
            self._update_ap_status(ap_key, "green", "Capturing")
            self._update_ap_status(ap_key, "yellow", "Ready for next capture")
            self._csi_capture_intervals.append((start_ns, end_ns))
        except Exception as exc:  # pragma: no cover - network dependency
            self._append_log_entry(f"CSI capture failed on {ap_name}: {exc}")
            self._update_ap_status(ap_key, "red", "Capture failed")

    def _format_capture_label(
        self,
        ap_name: str,
        *,
        start_ns: int,
        end_ns: int | None,
        action_name: str,
        count: int,
        include_end: bool = True,
    ) -> str:
        sanitized_ap = self._sanitize_ap_name(ap_name)
        date_clock = datetime.fromtimestamp(start_ns / 1_000_000_000).strftime(
            "%Y%m%d_%H%M%S"
        )

        parts = [
            f"D{sanitized_ap}",
            f"DC{date_clock}",
        ]
        return "_".join(parts)

    # ------------------------------------------------------------------
    # Depth camera capture
    # ------------------------------------------------------------------
    def _start_depth_capture_if_enabled(
        self,
        *,
        label: str,
        action_index: int,
        repetition: int,
        reuse_token: bool = False,
    ) -> None:
        profile = self.depth_camera_profile or {}
        if not profile or not profile.get("enabled"):
            return

        api_ip = str(profile.get("api_ip", "")).strip()
        if not api_ip:
            return

        token = (label, action_index, repetition)
        if reuse_token:
            # Do not start a second overlapping capture for the same window.
            if token not in self._depth_capture_tokens_started:
                return
        elif token in self._depth_capture_tokens_started:
            return

        url = self._build_depth_capture_url(api_ip)
        if not url:
            return

        payload = self._build_depth_capture_payload(
            label=label,
            action_index=action_index,
            repetition=repetition,
        )

        thread = threading.Thread(
            target=self._invoke_depth_capture,
            args=(url, payload, token),
            name=f"DepthCapture-{label}-{repetition}",
            daemon=True,
        )
        self._append_log_entry(
            f"Calling depth camera API at {url} for '{label}' "
            f"(action {action_index + 1}, repetition {repetition})."
        )
        thread.start()
        self._depth_capture_threads.append(thread)
        self._depth_capture_tokens_started.add(token)

    def _build_depth_capture_url(self, api_ip: str) -> str:
        base = api_ip
        if not base.startswith(("http://", "https://")):
            base = f"http://{base}"
        if ":" not in base.split("//", 1)[-1]:
            base = f"{base}:5000"
        base = base.rstrip("/")
        return f"{base}/api/capture/start"

    @staticmethod
    def _normalize_capture_resolution(
        value, fallback: tuple[int, int, int | None]
    ) -> dict:
        fallback_width, fallback_height, fallback_fps = fallback
        width = fallback_width
        height = fallback_height
        fps = fallback_fps

        def _update(current, candidate):
            return MainWindow._coerce_positive_int(candidate, current) or current

        if isinstance(value, dict):
            width = _update(width, value.get("width"))
            height = _update(height, value.get("height"))
            fps = _update(fps, value.get("fps"))
        elif isinstance(value, (list, tuple)):
            if len(value) >= 1:
                width = _update(width, value[0])
            if len(value) >= 2:
                height = _update(height, value[1])
            if len(value) >= 3:
                fps = _update(fps, value[2])
        elif isinstance(value, str):
            numbers = re.findall(r"\d+", value)
            if len(numbers) >= 2:
                width = _update(width, numbers[0])
                height = _update(height, numbers[1])
            if len(numbers) >= 3:
                fps = _update(fps, numbers[2])

        resolution = {"width": width, "height": height}
        if fps and fps > 0:
            resolution["fps"] = fps
        return resolution

    def _build_depth_capture_payload(
        self,
        *,
        label: str,
        action_index: int,
        repetition: int,
    ) -> dict:
        def build_save_location(profile: dict, repetition: int, action_index: int) -> str:
            base_location = (profile or {}).get("save_location") or "recordings"
            experiment_folder = None
            try:
                if self.results_dir:
                    experiment_folder = Path(self.results_dir).name
            except Exception:
                experiment_folder = None

            repetition_index = 1
            try:
                rep_count = max(int(repetition), 1)
                action_offset = max(int(action_index), 0)
                repetitions_per_action = max(int(self.repetitions_per_action), 0)
                if repetitions_per_action > 0:
                    repetition_index = action_offset * repetitions_per_action + rep_count
                else:
                    repetition_index = rep_count
            except Exception:
                repetition_index = max(rep_count if "rep_count" in locals() else 1, 1)

            repetition_index = str(repetition_index)
            path = PurePosixPath(base_location)
            if experiment_folder:
                path = path / experiment_folder
            path = path / repetition_index
            return path.as_posix()

        subject = self.subject_dict or {}
        experiment = self.experiment_dict or {}
        profile = self.depth_camera_profile or {}
        fps_value = self._coerce_positive_int(profile.get("fps"), default=None)
        fallback_fps = fps_value if fps_value is not None else 0
        rgb_res = self._normalize_capture_resolution(
            profile.get("rgb_resolution", {}), (640, 480, fallback_fps)
        )
        depth_res = self._normalize_capture_resolution(
            profile.get("depth_resolution", {}), (640, 480, fallback_fps)
        )
        duration = max(self._depth_capture_duration(), 0.0)
        exp_id = self._sanitize_token(subject.get("experiment_id", "exp"), "exp")
        experiment_name = experiment.get("name") or experiment.get("profile") or exp_id
        participant_id = self._combined_participant_id() or "p"
        client_now_ns = self._now_ns()
        payload: dict[str, object] = {
            "experiment_id": exp_id,
            "experiment_name": experiment_name,
            "experiment_title": "",
            # "experiment_title": experiment_name,
            "participant_id": "",
            # "participant_id": participant_id,
            "client_unix_ns": client_now_ns,
            "client_timestamp_ns": "",
            "client_datetime": "",
            # "client_timestamp_ns": client_now_ns,
            # "client_datetime": self._client_datetime_iso(),
            "duration_seconds": duration,
            "duration": duration,
            "duration_sec": duration,
            "rgb_resolution": rgb_res,
            "depth_resolution": depth_res,
            "save_raw_npz": bool(profile.get("save_raw_npz", True)),
            "save_location": build_save_location(profile, repetition, action_index),
            "fps": fps_value if fps_value else None,
        }
        payload.update(
            {
                "label": label,
                "action_index": int(action_index),
                "repetition": int(repetition),
            }
        )

        if payload.get("fps") is None:
            payload.pop("fps", None)
        return payload

    def _invoke_depth_capture(self, url: str, payload: dict, token: tuple[str, int]):
        try:
            response = requests.post(url, json=payload, timeout=5)
            if not response.ok:
                raise RuntimeError(f"Depth capture error: {response.status_code} {response.text}")
            self._append_log_entry(
                f"Depth capture started via API ({url}) for '{payload.get('label', '')}' "
                f"(repetition {payload.get('repetition', '')})."
            )
        except Exception as exc:  # pragma: no cover - network dependency
            self._append_log_entry(f"Depth capture API call to {url} failed: {exc}")
            try:
                self._depth_capture_tokens_started.discard(token)
            except Exception:
                pass

    def _depth_capture_duration(self) -> float:
        return (
            float(self.pre_action_csi_duration)
            + float(self.action_time)
            + float(self.post_action_csi_duration)
        )

    def _build_capture_prefix(
        self,
        ap_name: str,
        *,
        start_ns: int | None = None,
        action_name: str | None = None,
        repetition_count: int | None = None,
    ) -> str:
        start_ns = start_ns or self._now_ns()
        return self._format_capture_label(
            ap_name,
            start_ns=start_ns,
            end_ns=None,
            action_name=action_name or "full_experiment",
            count=repetition_count or 1,
            include_end=False,
        )

    def _finalize_wifi_captures(self):
        if self._wifi_capture_finalized:
            return

        self._wifi_capture_finalized = True
        self._wifi_shutdown_event.set()
        for thread in self._wifi_capture_threads:
            thread.join(timeout=5.0)

        download_threads = []
        total_downloads = 0
        for ap_key, router in list(self._wifi_routers.items()):
            run_sessions = [
                run
                for run in self._get_run_sessions(ap_key)
                if run.get("started") and not run.get("downloaded")
            ]
            if not run_sessions:
                continue
            total_downloads += sum(
                1
                for run in run_sessions
                if self.wifi_manager.is_sniffer(run.get("ap", {}))
            )
            thread = threading.Thread(
                target=self._download_capture_sessions_for_ap,
                args=(ap_key, router, run_sessions),
                name=f"WiFiDownload-{ap_key}",
                daemon=True,
            )
            thread.start()
            download_threads.append(thread)

        if download_threads:
            self._transfer_dialog.reset()
            self._transfer_dialog.set_total_files(total_downloads)
            self._transfer_dialog.show()
            watcher = threading.Thread(
                target=self._wait_for_downloads,
                args=(download_threads,),
                name="WiFiDownloadWatcher",
                daemon=True,
            )
            watcher.start()
        else:
            self._handle_downloads_complete()

    def _download_capture_sessions_for_ap(self, ap_key: str, router, run_sessions: list[dict]):
        for run_info in run_sessions:
            try:
                self._download_capture_session(ap_key, router, run_info)
            except Exception as exc:  # pragma: no cover - defensive
                self._append_log_entry(f"Unexpected download error for {ap_key}: {exc}")
                self._update_ap_status(ap_key, "red", "Download failed")

    def _download_capture_session(self, ap_key: str, router, run_info: dict):
        if run_info is None:
            return

        ap = run_info.get("ap", {})
        ap_name = ap.get("name") or ap.get("ssid") or ap_key
        remote_dir = run_info.get("remote_dir", "/mnt/CSI_USB/")
        exp_name = run_info.get("exp_name", "capture")
        mac_addresses = run_info.get("macs", [])
        channel_bw = run_info.get("channel_bw", "")

        if not self.wifi_manager.is_sniffer(ap):
            self._append_log_entry(
                f"{ap_name} is configured as a transmitter; skipping file download."
            )
            self._update_ap_status(ap_key, "green", "Transmission complete")
            return

        try:
            if getattr(router, "capture_thread", None):
                router.capture_thread.join(timeout=5.0)
        except Exception:
            pass

        try:
            remote_listing = router.send_command(f"ls {remote_dir}")
        except Exception as exc:  # pragma: no cover - network dependency
            self._append_log_entry(f"Failed to list captures for {ap_name}: {exc}")
            self._update_ap_status(ap_key, "red", "Download failed")
            return

        pcap_files = [line.strip() for line in remote_listing]
        matching_files = self.wifi_manager.filter_matching_pcaps(pcap_files, exp_name)
        target_file = self.wifi_manager.latest_pcap_filename(matching_files)
        if not target_file:
            self._append_log_entry(
                f"No CSI capture file found for {ap_name} in {remote_dir}."
            )
            self._update_ap_status(ap_key, "red", "No capture found")
            return

        captures_root = self.results_dir / "csi_captures"
        ap_dir = captures_root / self._get_ap_directory_name(ap_key)
        capture_label = run_info.get("capture_label", exp_name)
        capture_index = int(run_info.get("capture_index") or 1)
        capture_dir = ap_dir / str(capture_index)
        capture_dir.mkdir(parents=True, exist_ok=True)
        use_ftp = str(ap.get("download_mode", "SFTP")).strip().upper() != "SFTP"
        remote_path = f"{remote_dir}/{target_file}"
        file_size = self._get_remote_file_size(router, remote_path)

        self.transfer_started.emit(target_file, file_size or 0)

        self._update_ap_status(ap_key, "yellow", "Downloading…")
        try:
            progress_cb = lambda filename, size, sent: self.transfer_progress.emit(
                target_file, sent, size
            )
            local_path, remote_path = self.wifi_manager.download_capture(
                router,
                remote_dir=remote_dir,
                target_file=target_file,
                local_directory=capture_dir,
                use_ftp=use_ftp,
                progress_cb=progress_cb,
            )
        except Exception as exc:  # pragma: no cover - network dependency
            self._append_log_entry(f"Download failed for {ap_name}: {exc}")
            self._update_ap_status(ap_key, "red", "Download failed")
            self.transfer_finished.emit(target_file, False)
            return

        desired_filename = f"{capture_index}_{capture_label}.pcap"
        final_path = capture_dir / desired_filename
        if local_path.name != desired_filename:
            try:
                local_path.rename(final_path)
                local_path = final_path
            except Exception:
                local_path = local_path

        count_packets_enabled = bool(self.wifi_profile.get("count_packets"))
        packets = 0
        mac_counts = {mac: 0 for mac in self.wifi_manager.normalize_macs(mac_addresses)}
        if count_packets_enabled:
            try:
                packets, mac_counts = count_packets_for_macs(
                    str(local_path), mac_addresses
                )
            except Exception:
                pass
            self._show_packet_counts(ap_name, mac_counts, packets)

        framework = (ap.get("framework") or "nexmon").strip().lower() or "nexmon"

        metadata = {
            "ap_name": ap_name,
            "local_path": local_path,
            "remote_path": remote_path,
            "activity": run_info.get("activity", "full_experiment"),
            "repetition_count": run_info.get(
                "repetition_count", self.repetitions_per_action
            ),
            "capture_label": capture_label,
            "capture_index": capture_index,
            "start_timestamp_ns": run_info.get("start_timestamp_ns"),
            "end_timestamp_ns": run_info.get("end_timestamp_ns"),
            "timestamp": self._now_datetime().isoformat(),
            "participant": self.subject_dict.get("name", "participant"),
            "participant_id": self.subject_dict.get("participant_id", ""),
            "second_participant": self.subject_dict.get("second_name", ""),
            "second_participant_id": self.subject_dict.get("second_participant_id", ""),
            "experiment_id": self.subject_dict.get("experiment_id", ""),
            "packets": packets,
            "mac_counts": mac_counts,
            "channel": ap.get("channel", ""),
            "frequency": ap.get("frequency", ""),
            "bandwidth": ap.get("bandwidth", ""),
            "framework": framework,
            "transmitter_macs": mac_addresses,
            "capture_dir": capture_dir,
            "file_size_bytes": (local_path.stat().st_size if local_path.exists() else None),
        }
        self._wifi_download_results.append(metadata)
        if self._wifi_scenario == "demo":
            self.demo_plot_requested.emit(metadata)
        self._update_ap_status(ap_key, "green", "Downloaded")
        self.transfer_finished.emit(target_file, True)

    def _show_demo_plot_from_metadata(self, metadata: dict):
        local_path = metadata.get("local_path")
        if not local_path:
            return
        framework = metadata.get("framework", "nexmon")
        try:
            csi_data, time_vals = load_csi_for_framework(
                local_path,
                framework=framework,
                bandwidth=80,
            )
            if csi_data.ndim == 3:
                csi_data = csi_data[:, :, :, np.newaxis]
            if csi_data.ndim == 2:
                csi_data = csi_data[:, :, np.newaxis, np.newaxis]
            title = f"Demo CSI Plot - {Path(local_path).name}"
            dialog = DemoCSIPlotDialog(
                csi_data=csi_data,
                time_values=time_vals,
                title=title,
                parent=self,
            )
            dialog.exec_()
        except Exception as exc:
            self._append_log_entry(f"Demo plot failed for {local_path}: {exc}")

    def _show_packet_counts(
        self, ap_name: str, mac_counts: dict[str, int], total_packets: int
    ):
        header = f"Packet counts for {ap_name}:"
        total_line = f"Total packets: {total_packets}"
        if mac_counts:
            mac_lines = [
                f"{mac}: {count}" for mac, count in sorted(mac_counts.items())
            ]
        else:
            mac_lines = ["No MAC addresses detected."]

        message = "\n".join([header, total_line, "", *mac_lines])
        QMessageBox.information(self, "Packet Counts", message)

    def _get_remote_file_size(self, router, remote_path: str) -> int | None:
        try:
            output = router.send_command(f"stat -c %s {remote_path}")
            if output:
                return int(str(output[0]).strip())
        except Exception:
            return None
        return None

    def _wait_for_downloads(self, threads: list[threading.Thread]):
        for thread in threads:
            thread.join()
        QtCore.QTimer.singleShot(0, self._handle_downloads_complete)

    def _handle_downloads_complete(self):
        self._write_csi_manifest()
        if self._transfer_dialog is not None:
            self._transfer_dialog.finish_all()
            self._transfer_dialog.hide()
        self._maybe_show_summary_window()

    def _maybe_reboot_access_points(self):
        should_reboot = bool(self.wifi_profile.get("reboot_after_summary"))
        if not should_reboot:
            return

        if self._wifi_scenario not in {"scenario_1", "scenario_2"}:
            return

        access_points = self.wifi_manager.prioritize_transmitters_first(
            self.wifi_profile.get("access_points", [])
        )
        if not access_points:
            return

        self._reboot_dialog, self._reboot_thread = (
            self.wifi_manager.reboot_access_points_with_dialog(
                access_points,
                router_cls=WiFiRouter,
                parent=self,
                existing_routers=self._wifi_routers,
                log=self._append_log_entry,
                thread_name="AccessPointRebooter",
                on_finished=lambda: setattr(self, "_reboot_dialog", None),
            )
        )

    def _write_csi_manifest(self):
        if not self._wifi_download_results:
            return
        manifest_path = self.results_dir / "csi_captures" / "captures.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "file_path",
                    "activity",
                    "count",
                    "timestamp",
                    "participant",
                    "participant_id",
                    "second_participant",
                    "second_participant_id",
                    "ap_name",
                    "capture_label",
                    "capture_index",
                    "start_timestamp_ns",
                    "end_timestamp_ns",
                    "capture_directory",
                    "packets",
                    "transmitter_mac_counts",
                    "transmitter_macs",
                    "channel",
                    "frequency",
                    "bandwidth",
                    "framework",
                    "file_size_bytes",
                ]
            )
            for entry in self._wifi_download_results:
                writer.writerow(
                    [
                        str(entry.get("local_path")),
                        entry.get("activity", ""),
                        entry.get("repetition_count", ""),
                        entry.get("timestamp", ""),
                        entry.get("participant", ""),
                        entry.get("participant_id", ""),
                        entry.get("second_participant", ""),
                        entry.get("second_participant_id", ""),
                        entry.get("ap_name", ""),
                        entry.get("capture_label", ""),
                        entry.get("capture_index", ""),
                        entry.get("start_timestamp_ns", ""),
                        entry.get("end_timestamp_ns", ""),
                        str(entry.get("capture_dir", "")),
                        entry.get("packets", 0),
                        entry.get("mac_counts", {}),
                        ", ".join(entry.get("transmitter_macs", [])),
                        entry.get("channel", ""),
                        entry.get("frequency", ""),
                        entry.get("bandwidth", ""),
                        entry.get("framework", "nexmon"),
                        entry.get("file_size_bytes", ""),
                    ]
                )

    def _maybe_show_summary_window(self):
        if self._summary_window_pending and not self._summary_window_shown:
            self._summary_window_pending = False
            self._show_summary_window()

    def _sanitize_ap_name(self, name: str) -> str:
        sanitized = self.wifi_manager.sanitize_ap_name(name)
        return sanitized.strip("_") or "ap"

    def _init_hand_recognition(self):
        if not self.hand_recognition_enabled:
            return
        if not self.use_webcam:
            self.hand_recognition_enabled = False
            self._append_log_entry(
                "Hand recognition requires the webcam feed; disabling option."
            )
            return
        try:
            source = self.camera_profile if self.camera_profile else self.experiment_dict
            model_complexity = (source.get("hand_model_complexity") or "light").strip()
            model_complexity = model_complexity.lower()
            if model_complexity not in {"light", "full"}:
                model_complexity = "light"
            left_color = source.get("hand_left_wrist_color")
            right_color = source.get("hand_right_wrist_color")
            wrist_radius = source.get("hand_wrist_circle_radius")
            self.hand_recognition_engine = HandRecognitionEngine(
                max_num_hands=2,
                model_complexity=model_complexity,
                left_wrist_color=left_color,
                right_wrist_color=right_color,
                wrist_circle_radius=wrist_radius,
            )
            self.hand_recognition_engine.start()
            self._append_log_entry(
                "Hand recognition enabled. Joint coordinates will be recorded."
            )
        except HandRecognitionError as exc:
            self.hand_recognition_enabled = False
            self.hand_recognition_engine = None
            self._append_log_entry(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self.hand_recognition_enabled = False
            self.hand_recognition_engine = None
            self._append_log_entry(f"Failed to initialize hand recognition: {exc}")

    def _init_webcam(self):
        if not self.use_webcam:
            # keep placeholder hidden unless explicitly requested
            return

        if cv2 is None:
            self.use_webcam = False
            if self.webcam_preview_label is not None:
                self.webcam_preview_label.setText(
                    self._ui_profile_text(
                        "webcam_opencv_unavailable_text", "OpenCV unavailable"
                    )
                )
                self.webcam_preview_label.show()
                self._position_webcam_preview()
            return

        self.webcam_capture = cv2.VideoCapture(self.camera_device_source)
        if not self.webcam_capture or not self.webcam_capture.isOpened():
            self.use_webcam = False
            if self.webcam_preview_label is not None:
                self.webcam_preview_label.setText(
                    self._ui_profile_text(
                        "webcam_unavailable_text", "Webcam unavailable"
                    )
                )
                self.webcam_preview_label.show()
                self._position_webcam_preview()
            return

        preferred_fps = 30.0
        self.webcam_capture.set(cv2.CAP_PROP_FPS, preferred_fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            try:
                self.webcam_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                # Some backends do not allow adjusting the buffer size
                pass

        self._open_webcam_frame_log()
        self._webcam_frame_index = 0

        fps = self.webcam_capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = preferred_fps
        else:
            fps = max(fps, preferred_fps)
        width = int(self.webcam_capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(self.webcam_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self._webcam_frame_size = (width, height)

        output_dir = getattr(self, "results_dir", None)
        if not output_dir:
            output_dir = Path("results")
            output_dir.mkdir(parents=True, exist_ok=True)
        self._webcam_output_path = output_dir / "webcam.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.webcam_writer = cv2.VideoWriter(
            str(self._webcam_output_path),
            fourcc,
            float(fps),
            (width, height),
        )
        if self.webcam_writer is not None and not self.webcam_writer.isOpened():
            # fall back to no writer if initialization failed
            self.webcam_writer.release()
            self.webcam_writer = None

        if self.webcam_preview_label is not None:
            self.webcam_preview_label.setText(
                self._ui_profile_text("webcam_starting_text", "Starting webcam...")
            )
            self.webcam_preview_label.show()
            self._position_webcam_preview()

        interval_ms = max(15, int(1000 / min(max(fps, 1.0), 60.0)))
        self.webcam_timer = QTimer(self)
        self.webcam_timer.timeout.connect(self._capture_webcam_frame)
        self.webcam_timer.start(interval_ms)
        self._capture_webcam_frame()
        self._start_microphone_recording()

    def _capture_webcam_frame(self):
        if self.webcam_capture is None:
            return
        ret, frame = self.webcam_capture.read()
        if not ret or frame is None:
            return

        try:
            if self.webcam_writer is not None:
                self.webcam_writer.write(frame)
        except Exception:
            # ignore intermittent writer failures
            pass

        display_frame = frame
        if self.hand_recognition_enabled and self.hand_recognition_mode == "live":
            display_frame = self._process_hand_recognition(frame.copy())

        self._log_webcam_frame()
        self._update_webcam_preview(display_frame)

    def _process_hand_recognition(self, frame):
        if not self.hand_recognition_enabled:
            return frame
        if self.hand_recognition_engine is None:
            return frame

        return self.hand_recognition_engine.process_frame(frame)

    def _update_webcam_preview(self, frame):
        if self.webcam_preview_label is None or frame is None or cv2 is None:
            return

        try:
            mirrored = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
        except Exception:
            return

        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        scaled = image.scaled(
            self.webcam_preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.webcam_preview_label.setPixmap(QPixmap.fromImage(scaled))

    def _position_webcam_preview(self):
        if self.webcam_preview_label is None or self.video_widget is None:
            return
        margin = 16
        size = self.webcam_preview_label.size()
        parent = self.video_widget
        x = max(margin, parent.width() - size.width() - margin)
        y = max(margin, parent.height() - size.height() - margin)
        self.webcam_preview_label.move(x, y)
        self.webcam_preview_label.raise_()

    def _layout_transcript_overlay(self):
        if self.transcript_label is None or self.video_widget is None:
            return
        profile = self._ui_profile_with_defaults()
        offset_x = int(profile.get("transcript_offset_x", 0))
        offset_y = int(profile.get("transcript_offset_y", 0))
        rect = self.video_widget.rect()
        rect.translate(offset_x, offset_y)
        self.transcript_label.setGeometry(rect)
        if self.transcript_label.isVisible():
            self.transcript_label.raise_()
            if self.webcam_preview_label is not None and self.webcam_preview_label.isVisible():
                self.webcam_preview_label.raise_()
        if self.preview_widget is not None and self.preview_widget.isVisible():
            self.preview_widget.raise_()

    def _layout_preview_widget(self):
        if self.preview_widget is None or self.video_widget is None:
            return
        parent = self.video_widget
        size = self.preview_widget.size()
        margin = 24
        x = (parent.width() - size.width()) // 2
        y = parent.height() - size.height() - margin
        x = max(margin, min(x, parent.width() - size.width() - margin))
        y = max(margin, y)
        self.preview_widget.move(x, y)
        if self.preview_widget.isVisible():
            self.preview_widget.raise_()

    def _setup_about_menu(self):
        action = getattr(self, "actionAbout", None)
        if action is not None:
            action.triggered.connect(self._show_about_dialog)

    def _show_about_dialog(self):
        features = """
        <ul>
            <li>Guided experiment sequencing with baseline, action, and rest timers.</li>
            <li>Action video previews with synchronized countdowns and webcam capture.</li>
            <li>Automated logging of CSI captures, timestamps, and participant metadata.</li>
            <li>Optional spoken prompts powered by the gTTS engine.</li>
            <li>Profile manager for experiments, actions, participants, and voices.</li>
        </ul>
        """
        about_html = f"""
        <h2>Wi-Fi CSI Data Collection Software</h2>
        <p><b>Developer:</b> Navid Hasanzadeh</p>
        <p><b>Website:</b> <a href='https://navidhasanzadeh.com'>navidhasanzadeh.com</a></p>
        <p><b>Email:</b> <a href='mailto:navid.hasanzadeh@mail.utoronto.ca'>navid.hasanzadeh@mail.utoronto.ca</a></p>
        <p>Wireless and Internet Research Laboratory (WIRLab)<br/>
        University of Toronto</p>
        <h3>Key Features</h3>
        {features}
        """
        dlg = QMessageBox(self)
        dlg.setWindowTitle("WIRLab - About Wi-Fi CSI Data Collection")
        dlg.setTextFormat(Qt.RichText)
        dlg.setIcon(QMessageBox.Information)
        dlg.setText(about_html)
        dlg.exec_()

    def eventFilter(self, obj, event):
        if obj is self.video_widget:
            if event.type() == QtCore.QEvent.Resize:
                self._position_webcam_preview()
                self._layout_transcript_overlay()
                self._layout_preview_widget()
        return super().eventFilter(obj, event)

    def _shutdown_webcam(self):
        if self.webcam_timer is not None:
            try:
                self.webcam_timer.stop()
            except Exception:
                pass
            self.webcam_timer.deleteLater()
            self.webcam_timer = None
        if self.webcam_capture is not None:
            try:
                self.webcam_capture.release()
            except Exception:
                pass
            self.webcam_capture = None
        if self.webcam_writer is not None:
            try:
                self.webcam_writer.release()
            except Exception:
                pass
            self.webcam_writer = None
        self._close_webcam_frame_log()
        self._stop_microphone_recording()
        self._shutdown_hand_recognition()

    def _shutdown_hand_recognition(self):
        if self.hand_recognition_engine is None:
            return
        try:
            self.hand_recognition_engine.shutdown()
        except Exception:
            pass
        finally:
            self.hand_recognition_engine = None

    def _start_microphone_recording(self):
        if not getattr(self, "results_dir", None):
            return
        if self.microphone_recorder is not None:
            return
        if sd is None:
            self._append_log_entry(
                "Microphone capture unavailable: install the 'sounddevice' package."
            )
            return
        try:
            audio_path = self.results_dir / "microphone.wav"
            self.microphone_recorder = MicrophoneRecorder(audio_path)
            self.microphone_recorder.start()
            self._append_log_entry(
                f"Recording microphone audio to {audio_path.name}"
            )
        except Exception as exc:
            self._append_log_entry(f"Failed to start microphone capture: {exc}")
            self.microphone_recorder = None

    def _stop_microphone_recording(self):
        if self.microphone_recorder is None:
            return
        try:
            self.microphone_recorder.stop()
        except Exception:
            pass
        finally:
            self.microphone_recorder = None

    def _init_timers(self):
        # clock/date-time label
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(100)  # 10 Hz for higher precision elapsed time

        # state machine + progress updates
        self.state_timer = QTimer(self)
        self.state_timer.timeout.connect(self._update_state)
        self.state_timer.start(50)  # 20 Hz approx

    def _init_beep_thread(self):
        self.beep_thread = QtCore.QThread(self)
        self.beep_worker = BeepWorker()
        self.beep_worker.moveToThread(self.beep_thread)
        self.request_beep.connect(self.beep_worker.do_beep)
        self.beep_thread.start()

    def _shutdown_beep_thread(self):
        thread = getattr(self, "beep_thread", None)
        if thread is None:
            return
        worker = getattr(self, "beep_worker", None)
        try:
            if worker is not None:
                self.request_beep.disconnect(worker.do_beep)
        except Exception:
            pass
        if thread.isRunning():
            thread.quit()
            thread.wait()
        try:
            thread.deleteLater()
        except Exception:
            pass
        if worker is not None:
            try:
                worker.deleteLater()
            except Exception:
                pass
        self.beep_thread = None
        self.beep_worker = None

    @staticmethod
    def _sanitize_token(value: str, fallback: str) -> str:
        cleaned = "".join(
            ch if str(ch).isalnum() or ch in "-_" else "_" for ch in str(value)
        )
        cleaned = cleaned.strip("_") or fallback
        return cleaned

    @staticmethod
    def _combined_token(primary: str, secondary: str, fallback: str) -> str:
        primary_clean = "".join(
            ch if str(ch).isalnum() or ch in "-_" else "_" for ch in str(primary)
        ).strip("_")
        secondary_clean = (
            "".join(ch if str(ch).isalnum() or ch in "-_" else "_" for ch in str(secondary)).strip("_")
            if secondary
            else ""
        )
        if secondary_clean:
            if primary_clean:
                return f"{primary_clean}-and-{secondary_clean}"
            return secondary_clean
        cleaned_primary = primary_clean or ""
        return cleaned_primary or MainWindow._sanitize_token(fallback, fallback)

    @staticmethod
    def _combined_value(subject_dict: dict, primary_key: str, secondary_key: str, fallback: str) -> str:
        primary = subject_dict.get(primary_key, fallback)
        secondary = subject_dict.get(secondary_key, "") if subject_dict.get("has_second_participant") else ""
        return MainWindow._combined_token(primary, secondary, fallback)

    def _combined_participant_name(self) -> str:
        return self._combined_value(self.subject_dict, "name", "second_name", "participant")

    def _combined_participant_id(self) -> str:
        return self._combined_value(self.subject_dict, "participant_id", "second_participant_id", "pid")

    def _combined_age_group(self) -> str:
        return self._combined_value(self.subject_dict, "age_group", "second_age_group", "no_age")

    @staticmethod
    def _coerce_positive_int(value, default: int | None = None) -> int | None:
        try:
            val = int(float(value))
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
        return default

    def _now_ns(self) -> int:
        if self.time_reference:
            return self.time_reference.now_ns()
        return time.time_ns()

    def _now(self) -> float:
        if self.time_reference:
            return self.time_reference.now()
        return time.time()

    def _now_datetime(self):
        if self.time_reference:
            return self.time_reference.now_datetime()
        return datetime.now()

    def _client_datetime_iso(self) -> str:
        dt = self._now_datetime()
        if dt.tzinfo is None:
            return dt.isoformat() + "Z"
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def build_results_dir(
        subject_dict: dict, *, time_reference: time_reference_module.TimeReference | None = None
    ) -> Path:
        name = MainWindow._combined_value(subject_dict, "name", "second_name", "subject")
        age_group = MainWindow._combined_value(
            subject_dict, "age_group", "second_age_group", "no_age"
        )
        exp_id = MainWindow._sanitize_token(
            subject_dict.get("experiment_id", "0"), "0"
        )
        participant_id = MainWindow._combined_value(
            subject_dict, "participant_id", "second_participant_id", "pid"
        )
        if time_reference is None:
            time_reference = time_reference_module.get_global_time_reference()
        timestamp = time_reference.now_datetime().strftime("%Y%m%d_%H%M%S")
        rnd = f"{random.randint(0, 999):03d}"

        base_dir = Path("results")
        base_dir.mkdir(parents=True, exist_ok=True)
        folder_name = f"{name}_{age_group}_{exp_id}_{participant_id}_{timestamp}_{rnd}"
        results_dir = base_dir / folder_name
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def _init_logging(self):
        """
        Create a unique results directory and open a CSV log file.
        """
        try:
            if self.results_dir is None:
                self.results_dir = self.build_results_dir(
                    self.subject_dict, time_reference=self.time_reference
                )
            else:
                self.results_dir = Path(self.results_dir)
                self.results_dir.mkdir(parents=True, exist_ok=True)

            log_path = self.results_dir / "events.csv"
            self.log_file_handle = log_path.open("w", encoding="utf-8")
            self.log_file_handle.write(
                "timestamp,os_time_local,os_time_utc,reference_time_local,"
                "reference_time_utc,mode,action_window,action_index,rep_index\n"
            )
            self._text_log_path = self.results_dir / "session.log"
            self._text_log_handle = self._text_log_path.open("w", encoding="utf-8")
            self._save_configuration_snapshot()
        except Exception:
            # logging is optional; ignore errors
            self.log_file_handle = None
            if self._text_log_handle is not None:
                try:
                    self._text_log_handle.close()
                except Exception:
                    pass
            self._text_log_handle = None

    def _save_configuration_snapshot(self):
        """Write participant, experiment, and action info alongside events."""

        if not getattr(self, "results_dir", None):
            return

        wifi_rows = self._wifi_profile_snapshot_rows()

        try:
            snapshot_path = self.results_dir / "experiment_snapshot.csv"
            with snapshot_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["category", "key", "value"])

                for key, value in (self.subject_dict or {}).items():
                    writer.writerow(["participant", key, str(value)])

                for key, value in (self.experiment_dict or {}).items():
                    writer.writerow(["experiment", key, str(value)])

                for action_name in self.action_names:
                    video_path = self.actions_dict.get(action_name, "")
                    writer.writerow(["action", action_name, str(video_path)])

                for category, key, value in wifi_rows:
                    writer.writerow([category, key, value])
        except Exception:
            # optional metadata file; ignore failures
            pass

    @staticmethod
    def _wifi_access_point_label(ap: dict, index: int) -> str:
        name = str(ap.get("name") or ap.get("ssid") or "").strip()
        if name:
            return name
        return f"AP {index}"

    def _wifi_profile_scenario(self) -> str:
        scenario = str(self.wifi_profile.get("csi_capture_scenario", "")).lower()
        if scenario in {"scenario_1", "scenario_2"}:
            return scenario
        return ""

    def _wifi_profile_snapshot_rows(self) -> list[tuple[str, str, str]]:
        scenario = self._wifi_profile_scenario()
        access_points = self.wifi_profile.get("access_points", [])
        if not scenario or not access_points:
            return []

        rows: list[tuple[str, str, str]] = [("wifi_profile", "scenario", scenario)]
        for idx, ap in enumerate(access_points, start=1):
            label = self._wifi_access_point_label(ap, idx)
            key_prefix = (
                self._sanitize_ap_name(label)
                if hasattr(self, "wifi_manager")
                else f"ap_{idx}"
            )
            key_prefix = key_prefix or f"ap_{idx}"
            channel = str(ap.get("channel", "")).strip()
            frequency = str(ap.get("frequency", "")).strip()
            framework = str(ap.get("framework", "")).strip() or "nexmon"
            rows.extend(
                [
                    ("wifi_router", f"{key_prefix}_name", label),
                    ("wifi_router", f"{key_prefix}_channel", channel),
                    ("wifi_router", f"{key_prefix}_frequency", frequency),
                    ("wifi_router", f"{key_prefix}_framework", framework),
                ]
            )
        return rows

    def _wifi_profile_environment_metadata(self) -> dict[str, str]:
        scenario = self._wifi_profile_scenario()
        access_points = self.wifi_profile.get("access_points", [])
        if not scenario or not access_points:
            return {}

        metadata: dict[str, str] = {"wifi_scenario": scenario}
        for idx, ap in enumerate(access_points, start=1):
            label = self._wifi_access_point_label(ap, idx)
            channel = str(ap.get("channel", "")).strip()
            frequency = str(ap.get("frequency", "")).strip()
            framework = str(ap.get("framework", "")).strip() or "nexmon"
            metadata[f"wifi_ap_{idx}"] = (
                f"{label} (channel={channel}, frequency={frequency}, framework={framework})"
            )
        return metadata

    def _open_webcam_frame_log(self):
        if self._webcam_frame_log is not None:
            return
        if not getattr(self, "results_dir", None):
            return

        try:
            log_path = self.results_dir / "webcam_frames.csv"
            self._webcam_frame_log = log_path.open("w", encoding="utf-8", newline="")
            self._webcam_frame_log.write("frame_index,timestamp_ns,timestamp_iso\n")
        except Exception:
            self._webcam_frame_log = None

    def _log_webcam_frame(self):
        if self._webcam_frame_log is None:
            return

        try:
            timestamp_ns = self._now_ns()
            timestamp_iso = datetime.fromtimestamp(timestamp_ns / 1_000_000_000).isoformat()
            line = f"{self._webcam_frame_index},{timestamp_ns},{timestamp_iso}\n"
            self._webcam_frame_log.write(line)
            self._webcam_frame_log.flush()
            self._webcam_frame_index += 1
        except Exception:
            pass

    def _close_webcam_frame_log(self):
        if self._webcam_frame_log is None:
            return

        try:
            self._webcam_frame_log.close()
        except Exception:
            pass
        finally:
            self._webcam_frame_log = None

    def _make_connections(self):
        # connect menu "Close" action if present
        if hasattr(self, "actionClose"):
            self.actionClose.triggered.connect(self.close)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def _apply_status_color(self, color_name: str):
        """Apply the chunk/background color for the mode progress bar."""

        if not hasattr(self, "label_action"):
            return

        try:
            # keep the title text centered and bold while tinting the chunk
            profile = self._ui_profile_with_defaults()
            font_size = int(profile.get("status_font_size", 20))
            text_color = QColor(str(profile.get("status_font_color", "#000000")))
            if not text_color.isValid():
                text_color = QColor("#000000")
            offset_x = int(profile.get("status_offset_x", 0))
            offset_y = int(profile.get("status_offset_y", 0))
            self.label_action.setStyleSheet(
                "QProgressBar { text-align: center; font: bold "
                f"{font_size}pt; color: {text_color.name()}; padding-left: {offset_x}px; "
                f"padding-top: {offset_y}px; }} "
                f"QProgressBar::chunk {{ background-color: {color_name}; }}"
            )
        except Exception:
            # fallback for environments where label_action is not a progress bar
            self.label_action.setStyleSheet(f"background-color: {color_name}")

    def _update_mode_progress(self, progress: float):
        """Update the progress bar behind the mode title."""

        if not hasattr(self, "label_action"):
            return

        try:
            clamped = max(0.0, min(1.0, progress))
            self.label_action.setValue(int(clamped * 100))
        except Exception:
            # Ignore errors if the widget cannot display progress
            pass

    def set_status(self, text: str, color_name: str):
        """
        Update the main status label only when text or color changes.
        """
        if not hasattr(self, "label_action"):
            return

        text_changed = text != self._current_status_text
        color_changed = color_name != self._current_status_color

        if text_changed:
            try:
                # For QProgressBar we want the text rendered as the format string
                self.label_action.setFormat(text)
            except Exception:
                self.label_action.setText(text)
            self._current_status_text = text

        if color_changed:
            self._apply_status_color(color_name)
            self._current_status_color = color_name

    def _append_log_entry(self, message: str):
        """Append a log entry on the GUI thread.

        Some callers originate from worker threads (e.g., CSI capture threads).
        Directly touching Qt widgets from those threads can crash the app, so we
        marshal the update back onto the main thread when necessary.
        """

        if not message:
            return

        if QtCore.QThread.currentThread() != self.thread():
            QtCore.QMetaObject.invokeMethod(
                self,
                "_append_log_entry_main_thread",
                Qt.QueuedConnection,
                QtCore.Q_ARG(str, message),
            )
            return

        self._append_log_entry_main_thread(message)

    @QtCore.pyqtSlot(str)
    def _append_log_entry_main_thread(self, message: str):
        timestamp = self._now_datetime().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{timestamp}] {message}"
        self._ui_log_entries.append(entry)
        if len(self._ui_log_entries) > 500:
            self._ui_log_entries = self._ui_log_entries[-500:]

        if self._text_log_handle is not None:
            try:
                self._text_log_handle.write(entry + "\n")
                self._text_log_handle.flush()
            except Exception:
                pass

        if self.log_widget is not None:
            try:
                self.log_widget.setPlainText("\n".join(self._ui_log_entries))
                scrollbar = self.log_widget.verticalScrollBar()
                if scrollbar is not None:
                    scrollbar.setValue(scrollbar.maximum())
            except Exception:
                pass

    def _should_show_transcript(self):
        if not self._current_transcript_text:
            return False
        if self.mode == self.MODE_ACTIONS and self.action_window == 1:
            return False
        if self.media_player is not None:
            try:
                if self.media_player.state() == QMediaPlayer.PlayingState:
                    return False
            except Exception:
                pass
        return True

    def _update_transcript_label(self):
        if self.transcript_label is None:
            return

        if self._should_show_transcript():
            self.transcript_label.setText(self._current_transcript_text)
            self.transcript_label.show()
            self.transcript_label.raise_()
            if (
                self.webcam_preview_label is not None
                and self.webcam_preview_label.isVisible()
            ):
                self.webcam_preview_label.raise_()
        else:
            self.transcript_label.hide()

    def _set_transcript_message(self, message: str):
        self._current_transcript_text = (message or "").strip()
        self._update_transcript_label()

    def _clear_transcript_message(self):
        if not self._current_transcript_text:
            return
        self._current_transcript_text = ""
        self._update_transcript_label()

    def _on_media_state_changed(self, state):
        # Whenever media playback changes we may need to hide/show transcript overlay
        self._update_transcript_label()

    def _speak(self, message: str, log_message: bool = True):
        if log_message:
            self._append_log_entry(message)

        self._set_transcript_message(message)

        if not self.voice_assistant_enabled or self.voice_assistant is None:
            return

        try:
            self.voice_assistant.stop()
        except Exception:
            pass

        try:
            self.voice_assistant.speak(message)
        except Exception:
            self.voice_assistant_enabled = False
            self.voice_assistant = None

    def _update_clock(self):
        # date & time
        if hasattr(self, "label_time"):
            now_dt = self._now_datetime()
            date_str = now_dt.strftime("%Y-%m-%d")
            time_str = now_dt.strftime("%H:%M:%S")
            self.label_time.setText(f"{date_str}  {time_str}")

        # elapsed experiment time
        if (
            hasattr(self, "label_elapsed_time")
            and self.mode != self.MODE_FINISHED
            and self.experiment_start_time is not None
        ):
            elapsed = self._now() - self.experiment_start_time
            total_ms = int(elapsed * 1000)
            hours, remainder = divmod(total_ms, 3600_000)
            minutes, remainder = divmod(remainder, 60_000)
            seconds, milliseconds = divmod(remainder, 1000)
            elapsed_str = f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
            self.label_elapsed_time.setText(elapsed_str)

        # remaining time display
        if hasattr(self, "label_remain_time"):
            if self.experiment_start_time is None or self._expected_experiment_duration <= 0:
                remaining_seconds = 0.0
            elif self.mode == self.MODE_FINISHED and self._experiment_end_time is not None:
                remaining_seconds = max(
                    0.0, self._expected_experiment_duration - (self._experiment_end_time - self.experiment_start_time)
                )
            else:
                remaining_seconds = self._expected_experiment_duration - max(
                    0.0, self._now() - self.experiment_start_time
                )
            self.label_remain_time.setText(self._format_time_remaining(remaining_seconds))

    # ------------------------------------------------------------------
    # Core state machine
    # ------------------------------------------------------------------
    def _update_state(self):
        if self.mode == self.MODE_FINISHED:
            return

        now = self._now()

        if self.mode in (
            self.MODE_BEGIN_BASELINE,
            self.MODE_BETWEEN_BASELINE,
            self.MODE_END_BASELINE,
        ):
            self._update_baseline_phase(now)
        elif self.mode == self.MODE_ACTIONS:
            self._update_action_phase(now)

        # lightweight logging at the same cadence
        self._log_sample(now)

    def _update_baseline_phase(self, now: float):
        if self.mode == self.MODE_BEGIN_BASELINE:
            total = self.beginning_baseline_duration
        elif self.mode == self.MODE_BETWEEN_BASELINE:
            total = self.between_actions_baseline_duration
        else:
            total = self.ending_baseline_duration

        elapsed = now - self.phase_start_time
        progress = max(0.0, min(1.0, elapsed / total))
        self._update_mode_progress(progress)

        # baseline visual cues
        self.set_status(
            self._ui_profile_text("status_baseline_text", "Baseline mode - Please do not move!"),
            "yellow",
        )
        if hasattr(self, "label_curaction"):
            self.label_curaction.setText("Baseline")
        if hasattr(self, "label_count"):
            self.label_count.setText("--")

        # check for phase completion
        if elapsed >= total:
            if self.mode in (self.MODE_BEGIN_BASELINE, self.MODE_BETWEEN_BASELINE):
                self._start_actions(now)
            elif self.mode == self.MODE_END_BASELINE:
                self._finish_experiment()

    def _update_action_phase(self, now: float):
        # no actions configured?
        if not self.action_names or self.current_action_index >= len(self.action_names):
            self._start_end_baseline(now)
            return

        cycle_duration = self.stop_time + self.action_time
        if cycle_duration <= 0:
            # degenerate, avoid division by zero
            cycle_duration = 1.0

        t_in_cycle = (now - self.phase_start_time) % cycle_duration
        window_progress = max(0.0, min(1.0, t_in_cycle / cycle_duration))

        # determine whether we are in stop or action sub-window
        new_window = 0 if t_in_cycle < self.stop_time else 1

        self._maybe_schedule_action_captures(t_in_cycle)

        # window transition logic
        if new_window != self.action_window:
            # update state before announcing transitions so transcript overlay visibility
            # logic sees the latest window value
            self.action_window = new_window
            if new_window == 1:
                # entering action window
                self.current_repetition += 1
                self._on_enter_action_window()
            else:
                # entering stop window
                self._trigger_post_action_capture()
                if self.current_repetition >= self.repetitions_per_action:
                    # finished this action; move to next action / baseline
                    self.current_repetition = 0
                    self.media_player.stop()
                    self.current_action_index += 1
                    if self.current_action_index >= len(self.action_names):
                        self._start_end_baseline(now)
                    else:
                        self._start_between_baseline(now)
                    return
                self._on_enter_stop_window()

        # update labels for current action + repetition
        self._update_action_labels()

        # progress within the current stop/action window
        if self.action_window == 0:
            if self.stop_time <= 0:
                mode_progress = 1.0
            else:
                mode_progress = max(0.0, min(1.0, t_in_cycle / self.stop_time))
        else:
            if self.action_time <= 0:
                mode_progress = 1.0
            else:
                mode_progress = max(
                    0.0, min(1.0, (t_in_cycle - self.stop_time) / self.action_time)
                )
        self._update_mode_progress(mode_progress)

    def _start_beginning_baseline(self, now=None):
        """Start the initial baseline and synchronise the elapsed timer."""

        now = now or self._now()
        self.mode = self.MODE_BEGIN_BASELINE
        self.phase_start_time = now
        self.experiment_start_time = now
        self.action_window = 0
        self.current_repetition = 0

    def _start_actions(self, now: float):
        self.mode = self.MODE_ACTIONS
        self.phase_start_time = now
        self.action_window = 0  # start with stop window
        self.current_repetition = 0

        if self.current_action_index >= len(self.action_names):
            # no available actions -> directly go to end baseline
            self._start_end_baseline(now)
            return

        # ensure correct labels
        self._on_enter_stop_window()
        self._update_action_labels()

    def _start_between_baseline(self, now: float):
        self.mode = self.MODE_BETWEEN_BASELINE
        self.phase_start_time = now
        self.action_window = 0
        # progress + labels will be handled by _update_baseline_phase
        self._announce_between_baseline()

    def _start_end_baseline(self, now: float):
        self.mode = self.MODE_END_BASELINE
        self.phase_start_time = now
        self.action_window = 0
        # progress + labels will be handled by _update_baseline_phase
        self._announce_end_baseline()

    def _compute_expected_duration(self) -> float:
        """Estimate how long the configured experiment should last."""

        total = self.beginning_baseline_duration + self.ending_baseline_duration
        action_block = max(0.0, self.stop_time + self.action_time)
        action_count = len(self.action_names)
        if action_count:
            total += action_count * self.repetitions_per_action * action_block
            if action_count > 1:
                total += (action_count - 1) * self.between_actions_baseline_duration
        return max(total, 0.0)

    def _finish_experiment(self):
        self.mode = self.MODE_FINISHED
        self.set_status(self._ui_profile_text("status_finished_text", "Experiment finished"), "yellow")
        self._update_mode_progress(1.0)
        self.media_player.stop()
        self._save_action_signal()
        self._save_csi_capture_signal()
        self._save_hand_landmarks()
        self._shutdown_webcam()
        self._speak(
            self._ui_profile_text(
                "guide_experiment_finished",
                "The experiment is finished. Thank you for participating.",
            ),
            log_message=False,
        )
        self._append_log_entry("Experiment finished.")
        self._experiment_end_time = self._now()
        self._summary_window_pending = True
        self._finalize_wifi_captures()
        if not self._wifi_routers:
            self._maybe_show_summary_window()

    # ------------------------------------------------------------------
    # Action window helpers
    # ------------------------------------------------------------------
    def _get_next_action_name(self):
        if not self.action_names:
            return None
        if self.current_action_index >= len(self.action_names):
            return None
        if self.current_repetition < self.repetitions_per_action:
            return self.action_names[self.current_action_index]
        next_index = self.current_action_index + 1
        if next_index < len(self.action_names):
            return self.action_names[next_index]
        return None

    def _announce_next_action(self):
        next_action = self._get_next_action_name()
        if next_action:
            message = self._format_guidance_message(
                self._ui_profile_text(
                    "guide_next_action_with_name",
                    "Next action: {action}. Perform the action only once after the beep.",
                ),
                next_action,
            )
        else:
            message = self._ui_profile_text(
                "guide_next_action_none",
                "No action. Please remain still.",
            )
        self._speak(message)

    def _announce_beginning_baseline(self):
        next_action = self._get_next_action_name()
        if next_action:
            message = self._format_guidance_message(
                self._ui_profile_text(
                    "guide_beginning_with_action",
                    "The experiment starts soon. Get ready to perform the activity {action}. "
                    "This is a preview of the next action.",
                ),
                next_action,
            )
        else:
            message = self._ui_profile_text(
                "guide_beginning_no_action",
                "The experiment starts soon. Get ready to perform the activity.",
            )
        self._speak(message)
        self._show_next_action_preview()

    def _announce_between_baseline(self):
        next_action = self._get_next_action_name()
        if next_action:
            message = self._format_guidance_message(
                self._ui_profile_text(
                    "guide_between_with_action",
                    "Wait for the next action to start soon. Next action: {action}. "
                    "This is a preview of the next action.",
                ),
                next_action,
            )
        else:
            message = self._ui_profile_text(
                "guide_between_no_action",
                "Wait for the next action to start soon.",
            )
        self._speak(message)
        self._show_next_action_preview()

    def _announce_end_baseline(self):
        message = self._ui_profile_text(
            "guide_end_baseline",
            "Remain seated and do not move until the experiment is finished.",
        )
        self._speak(message)
        self._stop_next_action_preview()

    def _on_enter_stop_window(self):
        self.set_status(
            self._ui_profile_text("status_stop_text", "Stop - Please do not move!"),
            "red",
        )
        # stop video playback
        try:
            self.media_player.stop()
        except Exception:
            pass
        self._announce_next_action()
        self._stop_next_action_preview()
        next_action = self._get_next_action_name()
        if next_action:
            self._append_log_entry(
                f"Stop window - preparing for action '{next_action}'."
            )
        else:
            self._append_log_entry("Stop window - remain still.")
        self._pre_action_capture_started = False
        self._post_action_capture_started = False

    def _on_enter_action_window(self):
        self.set_status(self._ui_profile_text("status_action_text", "Action"), "green")
        self._clear_transcript_message()

        # trigger asynchronous beep
        self.request_beep.emit()

        # show current action video's clip
        try:
            action_name = self.action_names[self.current_action_index]
            self._append_log_entry(
                f"Perform action '{action_name}' (repetition {self.current_repetition}/{self.repetitions_per_action})."
            )
            video_path = self.actions_dict.get(action_name, "")
            if video_path:
                # support both absolute and relative paths
                abs_path = str(Path(video_path).resolve())
                url = QUrl.fromLocalFile(abs_path)
                self.media_player.setMedia(QMediaContent(url))
                self.media_player.play()
        except Exception:
            # never crash on media errors
            pass

        # Kick off depth capture for the action window when enabled, independent of CSI capture.
        try:
            self._start_depth_capture_if_enabled(
                label=self.action_names[self.current_action_index]
                if self.current_action_index < len(self.action_names)
                else "action",
                action_index=self.current_action_index,
                repetition=max(1, self.current_repetition),
            )
        except Exception:
            # Depth capture should not interrupt experiment flow.
            pass

    def _update_action_labels(self):
        if hasattr(self, "label_curaction"):
            if self.current_action_index < len(self.action_names):
                self.label_curaction.setText(self.action_names[self.current_action_index])
            else:
                self.label_curaction.setText("")

        if hasattr(self, "label_count"):
            # show at least 1 / N while in first action window
            rep_display = max(1, min(self.current_repetition, self.repetitions_per_action))
            self.label_count.setText(f"{rep_display} / {self.repetitions_per_action}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_sample(self, now: float):
        try:
            if self.log_file_handle is not None:
                ts = datetime.fromtimestamp(now).isoformat()
                os_ns = time.time_ns()
                os_time_local = time_reference_module.fmt_local_from_ns(os_ns)
                os_time_utc = time_reference_module.fmt_utc_from_ns(os_ns)
                ref_time_local = ""
                ref_time_utc = ""
                if (
                    self.time_reference
                    and self.time_reference.enabled
                    and self.time_reference.ref_utc_ns is not None
                ):
                    ref_ns = self.time_reference.now_ns()
                    ref_time_local = time_reference_module.fmt_local_from_ns(ref_ns)
                    ref_time_utc = time_reference_module.fmt_utc_from_ns(ref_ns)
                line = (
                    f"{ts},{os_time_local},{os_time_utc},{ref_time_local},"
                    f"{ref_time_utc},{self.mode},{self.action_window},"
                    f"{self.current_action_index},{self.current_repetition}\n"
                )
                self.log_file_handle.write(line)
        except Exception:
            # logging is best-effort
            pass

        try:
            # record binary action signal sample with nanosecond resolution
            timestamp_ns = self._now_ns()
            action_flag = int(self.mode == self.MODE_ACTIONS and self.action_window == 1)
            self._action_signal_samples.append((timestamp_ns, action_flag))
        except Exception:
            # logging is best-effort
            pass

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if is_password_required():
            password, ok = QInputDialog.getText(
                self,
                "Confirm exit",
                "Enter password to close the window:",
                QLineEdit.Password,
            )
            if not ok:
                event.ignore()
                return

            if not verify_password(password):
                QMessageBox.warning(
                    self,
                    "Incorrect password",
                    "The provided password is incorrect. The window will remain open.",
                )
                event.ignore()
                return

        # stop timers
        if hasattr(self, "state_timer"):
            self.state_timer.stop()
        if hasattr(self, "clock_timer"):
            self.clock_timer.stop()

        # stop media
        try:
            self.media_player.stop()
        except Exception:
            pass

        # stop beep thread
        self._shutdown_beep_thread()

        # finalize CSI downloads
        self._finalize_wifi_captures()

        # close log file
        if self.log_file_handle is not None:
            try:
                self.log_file_handle.close()
            except Exception:
                pass
            self.log_file_handle = None
        if self._text_log_handle is not None:
            try:
                self._text_log_handle.close()
            except Exception:
                pass
            self._text_log_handle = None

        # persist action signal artefacts
        self._save_action_signal()
        self._save_csi_capture_signal()
        self._save_hand_landmarks()
        self._shutdown_webcam()
        if self.voice_assistant is not None:
            try:
                self.voice_assistant.stop()
            except Exception:
                pass

        self._stop_next_action_preview()
        self._show_summary_window()

        super().closeEvent(event)

    def _save_action_signal(self):
        """Persist the captured action-mode signal as numpy array and PNG plot."""

        if self._signal_saved or not getattr(self, "results_dir", None):
            return

        if not self._action_signal_samples:
            # nothing to save
            self._signal_saved = True
            return

        try:
            data = np.array(self._action_signal_samples, dtype=np.int64)
            signal_path = self.results_dir / "action_signal.npy"
            np.save(signal_path, data)

            fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
            ax.step(data[:, 0], data[:, 1], where="post", linewidth=1.5)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Time (ns since epoch)")
            ax.set_ylabel("Action mode")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Off", "On"])
            ax.set_title("Action Mode Signal")
            ax.grid(True, axis="x", linestyle="--", alpha=0.3)
            fig.tight_layout()

            plot_path = self.results_dir / "action_signal.png"
            fig.savefig(plot_path, dpi=300)
            plt.close(fig)
        except Exception:
            # never crash because of plotting / saving errors
            pass
        finally:
            self._signal_saved = True

    def _save_csi_capture_signal(self):
        """Persist CSI capture on/off signal as numpy array and PNG plot."""

        if self._csi_signal_saved or not getattr(self, "results_dir", None):
            return

        if not self._csi_capture_intervals:
            self._csi_signal_saved = True
            return

        try:
            ordered = [
                (int(start), int(end))
                for start, end in self._csi_capture_intervals
                if isinstance(start, (int, float))
                and isinstance(end, (int, float))
                and end >= start
            ]
            ordered.sort(key=lambda pair: pair[0])

            if not ordered:
                self._csi_signal_saved = True
                return

            samples: list[tuple[int, int]] = []
            for start, end in ordered:
                samples.append((start, 1))
                samples.append((end, 0))

            data = np.array(samples, dtype=np.int64)
            signal_path = self.results_dir / "csi_capture_signal.npy"
            np.save(signal_path, data)

            fig, ax = plt.subplots(figsize=(12, 3.8), dpi=300)
            ax.step(data[:, 0], data[:, 1], where="post", linewidth=1.6, color="tab:green")
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Time (ns since epoch)")
            ax.set_ylabel("CSI capture")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Idle", "Recording"])
            ax.set_title("CSI Capture Signal")
            ax.grid(True, axis="x", linestyle="--", alpha=0.3)
            fig.tight_layout()

            plot_path = self.results_dir / "csi_capture_signal.png"
            fig.savefig(plot_path, dpi=300)
            plt.close(fig)
            self._save_combined_signals()
        except Exception:
            # never crash because of plotting / saving errors
            pass
        finally:
            self._csi_signal_saved = True

    def _save_combined_signals(self):
        """Persist a combined plot of action and CSI capture signals."""

        if self._combined_signal_saved or not getattr(self, "results_dir", None):
            return

        action_path = self.results_dir / "action_signal.npy"
        csi_path = self.results_dir / "csi_capture_signal.npy"
        if not action_path.exists() or not csi_path.exists():
            return

        try:
            action_data = np.load(action_path)
            csi_data = np.load(csi_path)
            if not action_data.size or not csi_data.size:
                return

            fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
            ax.step(
                action_data[:, 0],
                action_data[:, 1],
                where="post",
                linewidth=1.6,
                color="tab:blue",
                label="Action mode",
            )
            ax.step(
                csi_data[:, 0],
                csi_data[:, 1],
                where="post",
                linewidth=1.6,
                color="tab:green",
                label="CSI capture",
            )
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Time (ns since epoch)")
            ax.set_ylabel("Signal state")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Low", "High"])
            ax.set_title("Action mode and CSI capture signals")
            ax.grid(True, axis="x", linestyle="--", alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()

            plot_path = self.results_dir / "action_and_csi_signals.png"
            fig.savefig(plot_path, dpi=300)
            plt.close(fig)
        except Exception:
            # never crash because of plotting / saving errors
            pass
        finally:
            self._combined_signal_saved = True

    def _save_hand_landmarks(self):
        """Persist captured hand landmark coordinates and summary plot."""

        if self._hand_data_saved or not getattr(self, "results_dir", None):
            return

        if not self.hand_recognition_enabled:
            self._hand_data_saved = True
            return

        if self.hand_recognition_engine is None:
            self._hand_data_saved = True
            return

        try:
            saved = self.hand_recognition_engine.save_results(self.results_dir)
        except Exception:
            saved = True
        if saved:
            self._hand_data_saved = True

    def _on_preview_media_status_changed(self, status):
        try:
            if status == QMediaPlayer.EndOfMedia and self.preview_player is not None:
                self.preview_player.setPosition(0)
                self.preview_player.play()
        except Exception:
            pass

    def _show_next_action_preview(self):
        if self.preview_widget is None or self.preview_player is None:
            return
        action_name = self._get_next_action_name()
        if not action_name:
            self._stop_next_action_preview()
            return
        video_path = self.actions_dict.get(action_name)
        if not video_path:
            self._stop_next_action_preview()
            return
        abs_path = Path(video_path).expanduser().resolve()
        if not abs_path.exists():
            self._stop_next_action_preview()
            return
        try:
            url = QUrl.fromLocalFile(str(abs_path))
            self.preview_player.setMedia(QMediaContent(url))
            self.preview_widget.show()
            self._layout_preview_widget()
            self.preview_widget.raise_()
            self.preview_player.play()
        except Exception:
            self._stop_next_action_preview()

    def _stop_next_action_preview(self):
        if self.preview_player is not None:
            try:
                self.preview_player.stop()
            except Exception:
                pass
        if self.preview_widget is not None:
            self.preview_widget.hide()

    def _show_summary_window(self):
        if self._summary_window_shown:
            return
        self._summary_window_shown = True
        try:
            from experiment_summary_window import (
                ExperimentSummaryWindow,
                register_summary_window,
            )
        except Exception:
            return

        results_dir = getattr(self, "results_dir", None)
        elapsed = 0.0
        if self.experiment_start_time is not None:
            end_time = self._experiment_end_time or self._now()
            elapsed = max(0.0, end_time - self.experiment_start_time)

        summary_window = ExperimentSummaryWindow(
            results_dir=results_dir,
            subject_info=self.subject_dict,
            experiment_info=self.experiment_dict,
            environment_info={
                **getattr(self, "environment_info", {}),
                **self._wifi_profile_environment_metadata(),
            },
            actions_profile_name=self.experiment_dict.get("actions_profile", ""),
            actions_list=list(self.actions_dict.keys()),
            elapsed_time=elapsed,
            expected_duration=self._compute_expected_duration(),
        )
        summary_window.show()
        register_summary_window(summary_window)
        self._maybe_reboot_access_points()
