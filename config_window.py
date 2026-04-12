# config_window.py
import csv
import datetime
import hashlib
import json
import random
import re
import requests
import subprocess
import threading
import time
from copy import deepcopy
from pathlib import Path

from PyQt5.QtCore import QEvent, QEventLoop, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QButtonGroup,
    QRadioButton,
    QDoubleSpinBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QProgressDialog,
    QCheckBox,
    QAction,
    QMenuBar,
    QWidget,
    QTabWidget,
    QHeaderView,
    QAbstractItemView,
    QFontComboBox,
    QColorDialog,
    QScrollArea,
    QTextEdit,
    QSizePolicy,
)
from action_preview_dialog import ActionPreviewDialog
from csi_capture_window import RouterConnectionTestDialog, RouterPcapCleanupDialog
from wifi_csi_manager import WiFiCSIManager, WiFiRouter
from voice_assistant import GTTSVoiceAssistant
from password_manager import is_password_required, verify_password
from help.help_explorer import HelpExplorerDialog
from time_reference import (
    DEFAULT_TIME_SERVERS,
    best_startup_sync,
    fmt_local_from_ns,
    fmt_utc_from_ns,
)

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

CONFIGS_DIR = Path("configs")
PARTICIPANTS_FILE = CONFIGS_DIR / "participants_profiles.csv"
EXPERIMENTS_FILE = CONFIGS_DIR / "experiment_profiles.csv"
ACTIONS_FILE = CONFIGS_DIR / "action_profiles.csv"
VOICE_FILE = CONFIGS_DIR / "voice_profiles.csv"
CAMERA_FILE = CONFIGS_DIR / "camera_profiles.csv"
UI_FILE = CONFIGS_DIR / "ui_profiles.csv"
WIFI_FILE = CONFIGS_DIR / "wifi_profiles.csv"
ENVIRONMENT_FILE = CONFIGS_DIR / "environment_profiles.csv"
TIME_FILE = CONFIGS_DIR / "time_profiles.csv"
DEMO_FILE = CONFIGS_DIR / "demo_profiles.csv"
DEPTH_CAMERA_FILE = CONFIGS_DIR / "depth_camera_profiles.csv"
SELECTED_PROFILES_FILE = CONFIGS_DIR / "selected_profiles.json"
SCRIPTS_DIR = Path("scripts")

SUPPORTED_LANGUAGES = [
    ("en", "English"),
    ("de", "German"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("hi", "Hindi"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("ru", "Russian"),
    ("ar", "Arabic"),
    ("fa", "Persian"),
    ("zh-cn", "Chinese (Simplified)"),
    ("zh-tw", "Chinese (Traditional)"),
]



# ----------------------------------------------------------------------
# Table widget with built-in row dragging
# ----------------------------------------------------------------------
class RowDragDropTableWidget(QTableWidget):
    def __init__(self, *args, drop_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._drop_callback = drop_callback
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropOverwriteMode(False)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)

    def dropEvent(self, event):
        super().dropEvent(event)
        if callable(self._drop_callback):
            self._drop_callback()


# ----------------------------------------------------------------------
# Default profile (used when nothing is saved yet)
# ----------------------------------------------------------------------
def generate_participant_id(name: str, age_group: str, gender: str) -> str:
    seed = f"{(name or '').lower()}|{(age_group or '').lower()}|{(gender or '').lower()}"
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:12]


def generate_experiment_id() -> str:
    return f"{random.randint(100000, 999999)}"


def _generate_default_participant_profile() -> dict:
    random.seed(time.time_ns())

    random_suffix = random.randint(10000, 99999)
    gender = random.choice(["Female", "Male", "Non-binary", "Prefer not to say", ""])
    age_group = "Blank"
    name = f"Participant_{random_suffix}"
    experiment_id = generate_experiment_id()

    return {
        "name": name,
        "age_group": age_group,
        "participant_id": generate_participant_id(name, age_group, gender),
        "experiment_id": experiment_id,
        "gender": gender,
        "age_value": 0,
        "height_cm": round(random.uniform(150.0, 190.0), 1),
        "weight_kg": round(random.uniform(50.0, 95.0), 1),
        "dominant_hand": random.choice(["left", "right"]),
        "description": "",
        "has_second_participant": False,
        "second_name": "",
        "second_age_group": age_group,
        "second_age_value": 0,
        "second_participant_id": generate_participant_id("", age_group, ""),
        "second_gender": "",
        "second_dominant_hand": "",
        "second_height_cm": 0.0,
        "second_weight_kg": 0.0,
        "second_description": "",
    }


DEFAULT_PROFILE = {
    "name": "default_profile",
    "subject": _generate_default_participant_profile(),
    "experiment": {
        "beginning_baseline_recording": 15.0,
        "between_actions_baseline_recording": 10.0,
        "ending_baseline_recording": 20.0,
        "stop_time": 6.0,
        "preview_pause": 2.0,
        "action_time": 4.0,
        "each_action_repitition_times": 20,
        "save_location": "./experiments/",
    },
    "actions": {
        "circle_clockwise": "videos/circle_clockwise.mp4",
        "left_right": "videos/right_left.mp4",
        "up_down": "videos/up_down.mp4",
        "push_pull": "videos/push_pull.mp4",
    },
}

DEFAULT_PARTICIPANT_PROFILE = deepcopy(DEFAULT_PROFILE["subject"])
DEFAULT_EXPERIMENT_PROFILE = deepcopy(DEFAULT_PROFILE["experiment"])
DEFAULT_ACTION_PROFILE = deepcopy(DEFAULT_PROFILE["actions"])
DEFAULT_VOICE_PROFILE = {
    "use_voice_assistant": False,
    "language": "en",
    "preview_actions_voice": True,
}
DEFAULT_CAMERA_PROFILE = {
    "use_webcam": False,
    "camera_device": "0",
    "use_hand_recognition": False,
    "hand_recognition_mode": "live",
    "hand_model_complexity": "light",
    "hand_left_wrist_color": "#0000ff",
    "hand_right_wrist_color": "#ff0000",
    "hand_wrist_circle_radius": 18,
}
UI_LABEL_TEXT_FIELDS = [
    {"prefix": "name_title", "label": "Name label", "text": "Name:", "font_size": 12},
    {
        "prefix": "participant_title",
        "label": "Participant ID label",
        "text": "Participant ID:",
        "font_size": 12,
    },
    {
        "prefix": "experiment_title",
        "label": "Experiment ID label",
        "text": "Experiment ID:",
        "font_size": 12,
    },
    {
        "prefix": "second_name_title",
        "label": "Second name label",
        "text": "Second Name:",
        "font_size": 12,
    },
    {
        "prefix": "second_participant_title",
        "label": "Second participant ID label",
        "text": "Second Participant ID:",
        "font_size": 12,
    },
    {
        "prefix": "remaining_time_title",
        "label": "Remaining time label",
        "text": "Remaining Time:",
        "font_size": 12,
    },
    {
        "prefix": "elapsed_time_title",
        "label": "Elapsed time label",
        "text": "Elapsed Time:",
        "font_size": 12,
    },
    {"prefix": "count_title", "label": "Count label", "text": "Count:", "font_size": 12},
    {
        "prefix": "current_action_title",
        "label": "Current action label",
        "text": "Current action:",
        "font_size": 12,
    },
    {"prefix": "brand_header", "label": "Brand header", "text": "Wirlab", "font_size": 26},
]
UI_VALUE_LABEL_FIELDS = [
    {"prefix": "name_value", "label": "Name value", "font_size": 12},
    {"prefix": "participant_value", "label": "Participant ID value", "font_size": 12},
    {"prefix": "experiment_value", "label": "Experiment ID value", "font_size": 12},
    {"prefix": "second_name_value", "label": "Second name value", "font_size": 12},
    {
        "prefix": "second_participant_value",
        "label": "Second participant ID value",
        "font_size": 12,
    },
]
UI_STATUS_MESSAGE_FIELDS = [
    ("status_baseline_text", "Baseline mode - Please do not move!"),
    ("status_stop_text", "Stop - Please do not move!"),
    ("status_action_text", "Action"),
    ("status_finished_text", "Experiment finished"),
]
UI_GUIDE_MESSAGE_FIELDS = [
    (
        "guide_next_action_with_name",
        "Next action: {action}. Perform the action only once after the beep.",
    ),
    ("guide_next_action_none", "No action. Please remain still."),
    (
        "guide_beginning_with_action",
        "The experiment starts soon. Get ready to perform the activity {action}. "
        "This is a preview of the next action.",
    ),
    (
        "guide_beginning_no_action",
        "The experiment starts soon. Get ready to perform the activity.",
    ),
    (
        "guide_between_with_action",
        "Wait for the next action to start soon. Next action: {action}. "
        "This is a preview of the next action.",
    ),
    ("guide_between_no_action", "Wait for the next action to start soon."),
    (
        "guide_end_baseline",
        "Remain seated and do not move until the experiment is finished.",
    ),
    (
        "guide_experiment_finished",
        "The experiment is finished. Thank you for participating.",
    ),
]
UI_WEBCAM_MESSAGE_FIELDS = [
    ("webcam_disabled_text", "Webcam disabled"),
    ("webcam_starting_text", "Starting webcam..."),
    ("webcam_unavailable_text", "Webcam unavailable"),
    ("webcam_opencv_unavailable_text", "OpenCV unavailable"),
]
DEFAULT_DEPTH_CAMERA_PROFILE = {
    "enabled": False,
    "api_ip": "127.0.0.1",
    "fps": 30,
    "rgb_resolution": {"width": 640, "height": 480},
    "depth_resolution": {"width": 640, "height": 480},
    "save_raw_npz": True,
    "save_location": "recordings",
}
DEFAULT_UI_PROFILE = {
    "camera_frame_percent": 100.0,
    "preview_frame_percent": 100.0,
    "action_frame_percent": 100.0,
    "details_pane_percent": 30.0,
    "show_log": True,
    "log_frame_percent": 100.0,
    "start_fullscreen": False,
    "count_font_family": "",
    "count_font_size": 12,
    "count_font_color": "#000000",
    "action_font_family": "",
    "action_font_size": 20,
    "action_font_color": "#000000",
    "time_font_family": "",
    "time_font_size": 20,
    "time_font_color": "#000000",
    "remaining_time_value_font_family": "",
    "remaining_time_value_font_size": 12,
    "remaining_time_value_font_color": "#000000",
    "elapsed_time_value_font_family": "",
    "elapsed_time_value_font_size": 12,
    "elapsed_time_value_font_color": "#000000",
    "transcript_position": "middle",
    "transcript_font_size": 24,
    "transcript_font_color": "#ffffff",
    "transcript_offset_x": 0,
    "transcript_offset_y": 0,
    "status_font_size": 20,
    "status_font_color": "#000000",
    "status_offset_x": 0,
    "status_offset_y": 0,
    "log_placeholder_text": "Log output will appear here.",
    "log_font_size": 10,
    "log_font_color": "#000000",
    "log_offset_x": 0,
    "log_offset_y": 0,
    "webcam_font_size": 12,
    "webcam_font_color": "#ffffff",
    "webcam_offset_x": 0,
    "webcam_offset_y": 0,
}
for item in UI_LABEL_TEXT_FIELDS:
    DEFAULT_UI_PROFILE.setdefault(f"{item['prefix']}_text", item["text"])
    DEFAULT_UI_PROFILE.setdefault(f"{item['prefix']}_font_size", item["font_size"])
    DEFAULT_UI_PROFILE.setdefault(f"{item['prefix']}_color", "#000000")
    DEFAULT_UI_PROFILE.setdefault(f"{item['prefix']}_offset_x", 0)
    DEFAULT_UI_PROFILE.setdefault(f"{item['prefix']}_offset_y", 0)
for item in UI_VALUE_LABEL_FIELDS:
    DEFAULT_UI_PROFILE.setdefault(f"{item['prefix']}_font_size", item["font_size"])
    DEFAULT_UI_PROFILE.setdefault(f"{item['prefix']}_color", "#000000")
    DEFAULT_UI_PROFILE.setdefault(f"{item['prefix']}_offset_x", 0)
    DEFAULT_UI_PROFILE.setdefault(f"{item['prefix']}_offset_y", 0)
for key, message in UI_STATUS_MESSAGE_FIELDS:
    DEFAULT_UI_PROFILE.setdefault(key, message)
for key, message in UI_GUIDE_MESSAGE_FIELDS:
    DEFAULT_UI_PROFILE.setdefault(key, message)
for key, message in UI_WEBCAM_MESSAGE_FIELDS:
    DEFAULT_UI_PROFILE.setdefault(key, message)

# Minimum number of rows kept in the actions table so profiles immediately
# populate visible slots without requiring users to add rows manually.
MIN_ACTION_TABLE_ROWS = 20

DEFAULT_ENVIRONMENT_PROFILE = {
    "length_m": 5.0,
    "width_m": 5.0,
    "height_m": 3.0,
    "description": "",
}

DEFAULT_TIME_PROFILE = {
    "use_time_server": False,
    "time_server": "time.cloudflare.com",
}
DEFAULT_DEMO_PROFILE = {
    "demo_capture_mode": "router_live",
    "capture_duration_seconds": 5.0,
    "effective_capture_samples": 0,
    "apply_hampel_to_ratio_magnitude": False,
    "apply_hampel_to_ratio_phase": False,
    "demo_title_text": "Doppler Radiance Fields (DoRF) for Robust Wi-Fi Sensing and Human Activity Recognition",
    "university_logo_image_path": "",
    "university_logo_image_size_px": 160,
    "icassp_logo_image_path": "",
    "website_url": "https://dorf.navidhasanzadeh.com",
    "icassp_title_text": "IEEE ICASSP 2026",
    "icassp_logo_text_vertical_gap": 0,
    "demo_title_font_size_px": 22,
    "authors_text": "Authors: Navid Hasanzadeh, Shahrokh Valaee",
    "authors_font_size_px": 12,
    "university_text": "University of Toronto",
    "wirlab_text": "WIRLab",
    "university_font_size_px": 12,
    "title_authors_vertical_gap": 0,
    "authors_university_vertical_gap": 0,
    "capture_guidance_title": "CSI Capture Guidance",
    "capture_guidance_message": "Please perform one of these gestures.",
    "capture_guidance_video_left_label": "Left/Right",
    "capture_guidance_video_left_path": "videos/right_left.mp4",
    "capture_guidance_video_right_label": "Up/Down",
    "capture_guidance_video_right_path": "videos/up_down.mp4",
    "activity_class_names": [],
    "subplot_settings": {
        "csi_ratio_magnitude": {"visible": True, "title": "CSI Ratio Magnitude", "xlabel": "Time (s)", "ylabel": "|Ratio|", "info": "Shows CSI magnitude ratio between two TX antennas."},
        "csi_ratio_phase": {"visible": True, "title": "CSI Ratio Phase", "xlabel": "Time (s)", "ylabel": "Phase (rad)", "info": "Shows CSI phase ratio between two TX antennas."},
        "doppler_music": {"visible": True, "title": "Doppler MUSIC Projection", "xlabel": "Time (s)", "ylabel": "Norm. Doppler", "info": "Shows Doppler projections extracted from CSI streams."},
        "dorf_loss": {"visible": True, "title": "DoRF DTW Loss", "xlabel": "Iteration", "ylabel": "Loss", "info": "Shows optimization loss across DoRF iterations."},
        "dorf_velocity": {"visible": True, "title": "Estimated Velocity Components", "xlabel": "Time index", "ylabel": "Velocity", "info": "Shows estimated 3D velocity components."},
        "dorf_energy": {"visible": True, "title": "Energy Envelope", "xlabel": "Time index", "ylabel": "Energy", "info": "Shows squared velocity magnitude over time."},
        "dorf_projection_map": {"visible": True, "title": "Average DoRF Projection Map", "xlabel": "Longitude bins", "ylabel": "Latitude bins", "info": "Shows average DoRF projection map."},
        "dorf_cluster_fit": {"visible": True, "title": "Cluster Fit", "xlabel": "Time index", "ylabel": "Amplitude", "info": "Compares observed and predicted cluster Doppler."},
        "dorf_vmf_clusters": {"visible": True, "title": "vMF Clusters", "xlabel": "", "ylabel": "", "info": "Shows directional clusters on the unit sphere."},
    },
    "dorf_plot_order": [
        "dorf_loss",
        "dorf_velocity",
        "dorf_energy",
        "dorf_projection_map",
        "dorf_cluster_fit",
        "dorf_vmf_clusters",
    ],
}

SUPPORTED_DEMO_CAPTURE_MODES = {"router_live", "synthetic_random"}

DEFAULT_WIFI_AP = {
    "name": "",
    "ssid": "",
    "password": "",
    "router_ssh_ip": "",
    "router_ssh_username": "",
    "router_ssh_password": "",
    "ssh_key_address": "",
    "framework": "",
    "type": "",
    "frequency": "2.4 GHz",
    "channel": "1",
    "bandwidth": "20MHz",
    "transmitter_macs": "",
    "init_test_save_directory": "/mnt/CSI_USB/",
    "use_ethernet": False,
    "order": 1,
    "download_mode": "SFTP",
}

DEFAULT_WIFI_PROFILE = {
    "access_points": [deepcopy(DEFAULT_WIFI_AP)],
    "init_test_duration": 5.0,
    "init_test_save_directory": "/mnt/CSI_USB/",
    "csi_capture_scenario": "scenario_2",
    "pre_action_capture_duration": 2.0,
    "post_action_capture_duration": 2.0,
    "delete_prev_pcap": False,
    "count_packets": False,
    "reboot_after_summary": False,
}

DEFAULT_PARTICIPANT_PROFILE_NAME = "default_participant"
DEFAULT_EXPERIMENT_PROFILE_NAME = "default_experiment"
DEFAULT_ACTION_PROFILE_NAME = "default_actions"
DEFAULT_VOICE_PROFILE_NAME = "default_voice"
DEFAULT_CAMERA_PROFILE_NAME = "default_camera"
DEFAULT_DEPTH_CAMERA_PROFILE_NAME = "default_depth_camera"
DEFAULT_UI_PROFILE_NAME = "default_ui"
DEFAULT_WIFI_PROFILE_NAME = "default_wifi"
DEFAULT_ENVIRONMENT_PROFILE_NAME = "default_environment"
DEFAULT_TIME_PROFILE_NAME = "default_time"
DEFAULT_DEMO_PROFILE_NAME = "default_demo"

FREQUENCY_CHANNELS = {
    "2.4 GHz": [str(ch) for ch in range(1, 15)],
    "5 GHz": [
        "36",
        "40",
        "44",
        "48",
        "52",
        "56",
        "60",
        "64",
        "100",
        "104",
        "108",
        "112",
        "116",
        "120",
        "124",
        "128",
        "132",
        "136",
        "140",
        "144",
        "149",
        "153",
        "157",
        "161",
        "165",
    ],
}

FREQUENCY_BANDWIDTHS = {
    "2.4 GHz": ["20MHz", "40MHz"],
    "5 GHz": ["20MHz", "40MHz", "80MHz"],
}


def _ensure_configs_dir():
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_default_profile_file(path: Path, profiles: dict, save_func):
    _ensure_configs_dir()
    if path.exists():
        return
    save_func(profiles)


def _profiles_with_default(
    profiles: dict, default_name: str, default_data: dict
) -> dict:
    profiles = profiles or {}
    profiles.pop(default_name, None)
    merged = {default_name: deepcopy(default_data)}
    merged.update(profiles)
    return merged


DEFAULT_PROFILE_NAMES = {
    "participant": DEFAULT_PARTICIPANT_PROFILE_NAME,
    "experiment": DEFAULT_EXPERIMENT_PROFILE_NAME,
    "actions": DEFAULT_ACTION_PROFILE_NAME,
    "voice": DEFAULT_VOICE_PROFILE_NAME,
    "camera": DEFAULT_CAMERA_PROFILE_NAME,
    "depth_camera": DEFAULT_DEPTH_CAMERA_PROFILE_NAME,
    "ui": DEFAULT_UI_PROFILE_NAME,
    "wifi": DEFAULT_WIFI_PROFILE_NAME,
    "environment": DEFAULT_ENVIRONMENT_PROFILE_NAME,
    "time": DEFAULT_TIME_PROFILE_NAME,
    "demo": DEFAULT_DEMO_PROFILE_NAME,
}


def _is_default_profile(profile_type: str, name: str) -> bool:
    return name == DEFAULT_PROFILE_NAMES.get(profile_type)


def _as_bool(value):
    """Return True if value represents an affirmative flag."""

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_model_complexity(value: str | None) -> str:
    """Return a supported MediaPipe model complexity value."""

    normalized = (value or "").strip().lower()
    if normalized in {"full", "1"}:
        return "full"
    return "light"


def _normalize_hand_mode(value: str | None) -> str:
    """Return a supported hand recognition mode."""

    normalized = (value or "").strip().lower()
    if normalized and normalized not in {"live"}:
        return "live"
    return "live"


def _normalize_demo_capture_mode(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if normalized not in SUPPORTED_DEMO_CAPTURE_MODES:
        return DEFAULT_DEMO_PROFILE["demo_capture_mode"]
    return normalized


def _normalize_hex_color(value: str | None, default: str) -> str:
    """Normalize a hex color string to #RRGGBB format."""

    value = (value or "").strip()
    if not value:
        return default
    value = value.lstrip("#")
    if re.fullmatch(r"[0-9a-fA-F]{6}", value):
        return f"#{value.lower()}"
    return default


def _parse_resolution(value: str | dict | None, *, default: dict) -> dict:
    """Normalize resolution into a dict with width/height keys."""

    if isinstance(value, dict):
        width = int(value.get("width", default.get("width", 0)) or 0)
        height = int(value.get("height", default.get("height", 0)) or 0)
        if width > 0 and height > 0:
            return {"width": width, "height": height}

    if isinstance(value, str):
        cleaned = value.strip().lower().replace("@", "x")
        match = re.match(r"^(\d+)x(\d+)", cleaned)
        if match:
            try:
                width = int(match.group(1))
                height = int(match.group(2))
                if width > 0 and height > 0:
                    return {"width": width, "height": height}
            except (TypeError, ValueError):
                pass

    return deepcopy(default)


def _resolution_to_text(value: dict | None) -> str:
    """Return a compact WxH string for the given resolution dict."""

    if not isinstance(value, dict):
        return ""
    width = value.get("width")
    height = value.get("height")
    if not width or not height:
        return ""
    return f"{int(width)}x{int(height)}"


def filter_blank_actions(actions: dict | None) -> dict:
    """Return a new actions dict containing only rows with non-empty names."""

    cleaned = {}
    if not isinstance(actions, dict):
        return cleaned

    for name, video in actions.items():
        action_name = (name or "").strip()
        if not action_name:
            continue
        cleaned[action_name] = (video or "").strip() if isinstance(video, str) else video
    return cleaned


# ----------------------------------------------------------------------
# Helpers to load legacy text files (subject.txt, experiment.txt, actions.txt)
# ----------------------------------------------------------------------
def _load_kv_file(path: Path):
    """Load key=value file into a dict."""
    d = {}
    if not path.exists():
        return d
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip()
    return d


def _load_actions_file(path: Path):
    """
    Load actions.txt:
    actions...
    ---------------
    video paths...
    """
    if not path.exists():
        return {}

    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)

    split_idx = None
    for i, l in enumerate(lines):
        # treat any line of only '-' as the separator
        if set(l) == {"-"}:
            split_idx = i
            break

    if split_idx is None:
        return {}

    names = lines[:split_idx]
    videos = lines[split_idx + 1 :]
    n = min(len(names), len(videos))

    actions = {}
    for i in range(n):
        actions[names[i]] = videos[i]
    return actions


def _load_legacy_profile_from_txt():
    """
    Try to create a profile using subject.txt, experiment.txt, actions.txt.
    If something fails, fall back to DEFAULT_PROFILE (deep copy).
    """
    subject_txt = Path("subject.txt")
    experiment_txt = Path("experiment.txt")
    actions_txt = Path("actions.txt")

    subject = _load_kv_file(subject_txt)
    experiment = _load_kv_file(experiment_txt)
    actions = _load_actions_file(actions_txt)

    if not subject and not experiment and not actions:
        return deepcopy(DEFAULT_PROFILE)

    # Fill in missing fields with default
    profile = deepcopy(DEFAULT_PROFILE)

    if subject:
        profile["subject"].update(subject)
        for key in ("height_cm", "weight_kg"):
            if key in subject:
                try:
                    profile["subject"][key] = float(subject[key])
                except ValueError:
                    pass

    if experiment:
        # convert numeric fields safely
        for key in [
            "beginning_baseline_recording",
            "between_actions_baseline_recording",
            "ending_baseline_recording",
            "stop_time",
            "preview_pause",
            "action_time",
        ]:
            if key in experiment:
                try:
                    profile["experiment"][key] = float(experiment[key])
                except ValueError:
                    pass
        if "each_action_repitition_times" in experiment:
            try:
                profile["experiment"]["each_action_repitition_times"] = int(
                    experiment["each_action_repitition_times"]
                )
            except ValueError:
                pass
        if "save_location" in experiment:
            profile["experiment"]["save_location"] = experiment["save_location"]

    if actions:
        profile["actions"] = actions

    # use subject name as profile name if present
    pname = profile["subject"].get("name", DEFAULT_PARTICIPANT_PROFILE_NAME)
    profile["name"] = pname or DEFAULT_PARTICIPANT_PROFILE_NAME
    return profile


# ----------------------------------------------------------------------
# New CSV-based storage helpers
# ----------------------------------------------------------------------
def _load_participant_profiles_from_csv():
    profiles = {}
    if not PARTICIPANTS_FILE.exists():
        return profiles

    try:
        with PARTICIPANTS_FILE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile_name = (row.get("profile_name") or "").strip()
                if not profile_name:
                    continue
                subject = deepcopy(DEFAULT_PARTICIPANT_PROFILE)
                subject["name"] = (row.get("participant_name") or "").strip()
                subject["age_group"] = (
                    (row.get("age_group") or "Blank").strip()
                )
                subject["gender"] = (row.get("gender") or "").strip()
                subject["participant_id"] = row.get("participant_id", "").strip()
                subject["dominant_hand"] = (row.get("dominant_hand") or "").strip()
                try:
                    subject["age_value"] = int(row.get("age_value", 0))
                except (TypeError, ValueError):
                    subject["age_value"] = 0
                try:
                    subject["height_cm"] = float(row.get("height_cm", subject["height_cm"]))
                except (TypeError, ValueError):
                    pass
                try:
                    subject["weight_kg"] = float(row.get("weight_kg", subject["weight_kg"]))
                except (TypeError, ValueError):
                    pass
                subject["description"] = (row.get("description") or "").strip()
                subject["has_second_participant"] = _as_bool(
                    row.get("has_second_participant", subject.get("has_second_participant", False))
                )
                subject["second_name"] = (row.get("second_name") or "").strip()
                subject["second_age_group"] = (row.get("second_age_group") or "Blank").strip()
                subject["second_gender"] = (row.get("second_gender") or "").strip()
                subject["second_participant_id"] = (row.get("second_participant_id") or "").strip()
                subject["second_dominant_hand"] = (row.get("second_dominant_hand") or "").strip()
                try:
                    subject["second_age_value"] = int(row.get("second_age_value", 0))
                except (TypeError, ValueError):
                    subject["second_age_value"] = 0
                try:
                    subject["second_height_cm"] = float(
                        row.get("second_height_cm", subject["second_height_cm"])
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    subject["second_weight_kg"] = float(
                        row.get("second_weight_kg", subject["second_weight_kg"])
                    )
                except (TypeError, ValueError):
                    pass
                subject["second_description"] = (row.get("second_description") or "").strip()
                subject["participant_id"] = subject["participant_id"] or generate_participant_id(
                    subject.get("name", ""),
                    subject.get("age_group", ""),
                    subject.get("gender", ""),
                )
                subject["second_participant_id"] = subject["second_participant_id"] or generate_participant_id(
                    subject.get("second_name", ""),
                    subject.get("second_age_group", ""),
                    subject.get("second_gender", ""),
                )
                profiles[profile_name] = subject
    except Exception:
        return {}

    return profiles


def _load_environment_profiles_from_csv():
    profiles = {}
    if not ENVIRONMENT_FILE.exists():
        return profiles

    try:
        with ENVIRONMENT_FILE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile_name = (row.get("profile_name") or "").strip()
                if not profile_name:
                    continue
                profile = deepcopy(DEFAULT_ENVIRONMENT_PROFILE)
                for key in ("length_m", "width_m", "height_m"):
                    try:
                        profile[key] = float(row.get(key, profile[key]))
                    except (TypeError, ValueError):
                        pass
                profile["description"] = (row.get("description") or "").strip()
                profiles[profile_name] = profile
    except Exception:
        return {}

    return profiles


def _load_time_profiles_from_csv():
    profiles = {}
    if not TIME_FILE.exists():
        return profiles

    try:
        with TIME_FILE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile_name = (row.get("profile_name") or "").strip()
                if not profile_name:
                    continue
                profile = deepcopy(DEFAULT_TIME_PROFILE)
                profile["use_time_server"] = _as_bool(
                    row.get("use_time_server", profile["use_time_server"])
                )
                time_server = (row.get("time_server") or "").strip()
                if time_server:
                    profile["time_server"] = time_server
                profiles[profile_name] = profile
    except Exception:
        return {}

    return profiles


def _load_demo_profiles_from_csv():
    profiles = {}
    if not DEMO_FILE.exists():
        return profiles

    try:
        with DEMO_FILE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile_name = (row.get("profile_name") or "").strip()
                if not profile_name:
                    continue
                profile = deepcopy(DEFAULT_DEMO_PROFILE)
                profile["demo_capture_mode"] = _normalize_demo_capture_mode(
                    row.get("demo_capture_mode", profile["demo_capture_mode"])
                )
                try:
                    profile["capture_duration_seconds"] = float(
                        row.get(
                            "capture_duration_seconds",
                            profile["capture_duration_seconds"],
                        )
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    profile["effective_capture_samples"] = max(
                        0,
                        int(
                            row.get(
                                "effective_capture_samples",
                                profile["effective_capture_samples"],
                            )
                        ),
                    )
                except (TypeError, ValueError):
                    pass
                profile["apply_hampel_to_ratio_magnitude"] = _as_bool(
                    row.get(
                        "apply_hampel_to_ratio_magnitude",
                        profile["apply_hampel_to_ratio_magnitude"],
                    )
                )
                profile["apply_hampel_to_ratio_phase"] = _as_bool(
                    row.get(
                        "apply_hampel_to_ratio_phase",
                        profile["apply_hampel_to_ratio_phase"],
                    )
                )
                profile["demo_title_text"] = (
                    row.get("demo_title_text")
                    or profile["demo_title_text"]
                ).strip() or DEFAULT_DEMO_PROFILE["demo_title_text"]
                profile["university_logo_image_path"] = (
                    row.get("university_logo_image_path")
                    or row.get("qr_code_image_path")
                    or profile["university_logo_image_path"]
                ).strip()
                try:
                    profile["university_logo_image_size_px"] = max(
                        20,
                        min(
                            600,
                            int(
                                row.get(
                                    "university_logo_image_size_px",
                                    row.get(
                                        "qr_code_image_size_px",
                                        profile["university_logo_image_size_px"],
                                    ),
                                )
                            ),
                        ),
                    )
                except (TypeError, ValueError):
                    pass
                profile["icassp_logo_image_path"] = (
                    row.get("icassp_logo_image_path")
                    or profile["icassp_logo_image_path"]
                ).strip()
                profile["website_url"] = (
                    row.get("website_url")
                    or row.get("qr_website_url")
                    or profile["website_url"]
                ).strip() or DEFAULT_DEMO_PROFILE["website_url"]
                profile["icassp_title_text"] = (
                    row.get("icassp_title_text", profile["icassp_title_text"]) or ""
                ).strip()
                try:
                    profile["demo_title_font_size_px"] = max(
                        10,
                        min(
                            96,
                            int(
                                row.get(
                                    "demo_title_font_size_px",
                                    profile["demo_title_font_size_px"],
                                )
                            ),
                        ),
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    profile["icassp_logo_text_vertical_gap"] = int(
                        row.get(
                            "icassp_logo_text_vertical_gap",
                            profile["icassp_logo_text_vertical_gap"],
                        )
                    )
                except (TypeError, ValueError):
                    pass
                profile["authors_text"] = (
                    row.get("authors_text")
                    or profile["authors_text"]
                ).strip() or DEFAULT_DEMO_PROFILE["authors_text"]
                try:
                    profile["authors_font_size_px"] = max(
                        8,
                        min(
                            72,
                            int(
                                row.get(
                                    "authors_font_size_px",
                                    profile["authors_font_size_px"],
                                )
                            ),
                        ),
                    )
                except (TypeError, ValueError):
                    pass
                profile["university_text"] = (
                    row.get("university_text", profile["university_text"]) or ""
                ).strip()
                profile["wirlab_text"] = (
                    row.get("wirlab_text", profile.get("wirlab_text", "WIRLab")) or ""
                ).strip()
                try:
                    profile["university_font_size_px"] = max(
                        8,
                        min(
                            72,
                            int(
                                row.get(
                                    "university_font_size_px",
                                    profile["university_font_size_px"],
                                )
                            ),
                        ),
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    profile["title_authors_vertical_gap"] = int(
                        row.get(
                            "title_authors_vertical_gap",
                            row.get(
                                "title_authors_university_vertical_gap",
                                profile["title_authors_vertical_gap"],
                            ),
                        )
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    profile["authors_university_vertical_gap"] = int(
                        row.get(
                            "authors_university_vertical_gap",
                            row.get(
                                "title_authors_university_vertical_gap",
                                profile["authors_university_vertical_gap"],
                            ),
                        )
                    )
                except (TypeError, ValueError):
                    pass
                profile["capture_guidance_title"] = (
                    row.get("capture_guidance_title")
                    or profile["capture_guidance_title"]
                ).strip() or DEFAULT_DEMO_PROFILE["capture_guidance_title"]
                profile["capture_guidance_message"] = (
                    row.get("capture_guidance_message")
                    or profile["capture_guidance_message"]
                ).strip() or DEFAULT_DEMO_PROFILE["capture_guidance_message"]
                profile["capture_guidance_video_left_label"] = (
                    row.get("capture_guidance_video_left_label")
                    or profile["capture_guidance_video_left_label"]
                ).strip() or DEFAULT_DEMO_PROFILE["capture_guidance_video_left_label"]
                profile["capture_guidance_video_left_path"] = (
                    row.get("capture_guidance_video_left_path")
                    or profile["capture_guidance_video_left_path"]
                ).strip()
                profile["capture_guidance_video_right_label"] = (
                    row.get("capture_guidance_video_right_label")
                    or profile["capture_guidance_video_right_label"]
                ).strip() or DEFAULT_DEMO_PROFILE["capture_guidance_video_right_label"]
                profile["capture_guidance_video_right_path"] = (
                    row.get("capture_guidance_video_right_path")
                    or profile["capture_guidance_video_right_path"]
                ).strip()
                class_names_text = (row.get("activity_class_names") or "").strip()
                if class_names_text:
                    profile["activity_class_names"] = [
                        part.strip() for part in class_names_text.split(",") if part.strip()
                    ]
                raw_subplot_json = (row.get("subplot_settings_json") or "").strip()
                if raw_subplot_json:
                    try:
                        parsed = json.loads(raw_subplot_json)
                        if isinstance(parsed, dict):
                            profile["subplot_settings"] = parsed
                    except Exception:
                        pass
                order_text = (row.get("dorf_plot_order") or "").strip()
                if order_text:
                    profile["dorf_plot_order"] = [
                        item.strip() for item in order_text.split(",") if item.strip()
                    ]
                profiles[profile_name] = profile
    except Exception:
        return {}

    return profiles


def _load_experiment_profiles_from_csv():
    profiles = {}
    if not EXPERIMENTS_FILE.exists():
        return profiles

    try:
        with EXPERIMENTS_FILE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile_name = (row.get("profile_name") or "").strip()
                if not profile_name:
                    continue
                exp = deepcopy(DEFAULT_EXPERIMENT_PROFILE)
                for key in [
                    "beginning_baseline_recording",
                    "between_actions_baseline_recording",
                    "ending_baseline_recording",
                    "stop_time",
                    "preview_pause",
                    "action_time",
                ]:
                    try:
                        exp[key] = float(row.get(key, exp[key]))
                    except (TypeError, ValueError):
                        pass
                try:
                    exp["each_action_repitition_times"] = int(
                        row.get("each_action_repitition_times", exp["each_action_repitition_times"])
                    )
                except (TypeError, ValueError):
                    pass
                exp["save_location"] = (row.get("save_location") or exp["save_location"]).strip()
                profiles[profile_name] = exp
    except Exception:
        return {}

    return profiles


def _load_action_profiles_from_csv():
    profiles = {}
    if not ACTIONS_FILE.exists():
        return profiles

    try:
        with ACTIONS_FILE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile_name = (row.get("profile_name") or "").strip()
                if not profile_name:
                    continue
                action_name = (row.get("action_name") or "").strip()
                video_path = (row.get("video_path") or "").strip()
                actions = profiles.setdefault(profile_name, {})
                if action_name:
                    actions[action_name] = video_path
    except Exception:
        return {}

    return {name: filter_blank_actions(actions) for name, actions in profiles.items()}


def _load_voice_profiles_from_csv():
    profiles = {}
    if not VOICE_FILE.exists():
        return profiles

    try:
        with VOICE_FILE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile_name = (row.get("profile_name") or "").strip()
                if not profile_name:
                    continue
                voice = deepcopy(DEFAULT_VOICE_PROFILE)
                voice["use_voice_assistant"] = _as_bool(
                    row.get("use_voice_assistant", voice["use_voice_assistant"])
                )
                voice["language"] = (
                    row.get("language", voice["language"])
                    or voice["language"]
                ).strip()
                voice["preview_actions_voice"] = _as_bool(
                    row.get(
                        "preview_actions_voice",
                        voice["preview_actions_voice"],
                    )
                )
                profiles[profile_name] = voice
    except Exception:
        return {}

    return profiles


def _load_camera_profiles_from_csv():
    profiles = {}
    if not CAMERA_FILE.exists():
        return profiles

    try:
        with CAMERA_FILE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile_name = (row.get("profile_name") or "").strip()
                if not profile_name:
                    continue
                camera = deepcopy(DEFAULT_CAMERA_PROFILE)
                camera["use_webcam"] = _as_bool(
                    row.get("use_webcam", camera["use_webcam"])
                )
                camera["camera_device"] = (
                    row.get("camera_device", camera["camera_device"])
                    or camera["camera_device"]
                )
                camera["use_hand_recognition"] = _as_bool(
                    row.get("use_hand_recognition", camera["use_hand_recognition"])
                )
                camera["hand_recognition_mode"] = _normalize_hand_mode(
                    row.get("hand_recognition_mode", camera["hand_recognition_mode"])
                )
                camera["hand_model_complexity"] = _normalize_model_complexity(
                    row.get("hand_model_complexity", camera["hand_model_complexity"])
                )
                camera["hand_left_wrist_color"] = _normalize_hex_color(
                    row.get("hand_left_wrist_color", camera["hand_left_wrist_color"]),
                    DEFAULT_CAMERA_PROFILE["hand_left_wrist_color"],
                )
                camera["hand_right_wrist_color"] = _normalize_hex_color(
                    row.get("hand_right_wrist_color", camera["hand_right_wrist_color"]),
                    DEFAULT_CAMERA_PROFILE["hand_right_wrist_color"],
                )
                try:
                    camera["hand_wrist_circle_radius"] = max(
                        1,
                        int(
                            row.get(
                                "hand_wrist_circle_radius",
                                camera["hand_wrist_circle_radius"],
                            )
                        ),
                    )
                except (TypeError, ValueError):
                    camera["hand_wrist_circle_radius"] = DEFAULT_CAMERA_PROFILE[
                        "hand_wrist_circle_radius"
                    ]
                profiles[profile_name] = camera
    except Exception:
        return {}

    return profiles


def _load_depth_camera_profiles_from_csv():
    profiles = {}
    if not DEPTH_CAMERA_FILE.exists():
        return profiles

    try:
        with DEPTH_CAMERA_FILE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile_name = (row.get("profile_name") or "").strip()
                if not profile_name:
                    continue
                profile = deepcopy(DEFAULT_DEPTH_CAMERA_PROFILE)
                profile["enabled"] = _as_bool(
                    row.get("enabled", profile["enabled"])
                )
                profile["api_ip"] = (
                    row.get("api_ip", profile["api_ip"]) or profile["api_ip"]
                ).strip()
                try:
                    profile["fps"] = int(row.get("fps", profile["fps"]))
                except (TypeError, ValueError):
                    profile["fps"] = DEFAULT_DEPTH_CAMERA_PROFILE["fps"]
                profile["rgb_resolution"] = _parse_resolution(
                    row.get("rgb_resolution", profile["rgb_resolution"]),
                    default=DEFAULT_DEPTH_CAMERA_PROFILE["rgb_resolution"],
                )
                profile["depth_resolution"] = _parse_resolution(
                    row.get("depth_resolution", profile["depth_resolution"]),
                    default=DEFAULT_DEPTH_CAMERA_PROFILE["depth_resolution"],
                )
                profile["save_raw_npz"] = _as_bool(
                    row.get("save_raw_npz", profile["save_raw_npz"])
                )
                profile["save_location"] = (
                    row.get("save_location", profile["save_location"])
                    or profile["save_location"]
                )
                profiles[profile_name] = profile
    except Exception:
        return {}

    return profiles


def _load_ui_profiles_from_csv():
    profiles = {}
    if not UI_FILE.exists():
        return profiles

    try:
        with UI_FILE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile_name = (row.get("profile_name") or "").strip()
                if not profile_name:
                    continue
                profile = deepcopy(DEFAULT_UI_PROFILE)
                for key in (
                    "camera_frame_percent",
                    "preview_frame_percent",
                    "action_frame_percent",
                    "details_pane_percent",
                    "log_frame_percent",
                ):
                    try:
                        profile[key] = float(row.get(key, profile[key]))
                    except (TypeError, ValueError):
                        pass
                profile["start_fullscreen"] = _as_bool(
                    row.get("start_fullscreen", profile["start_fullscreen"])
                )
                profile["show_log"] = _as_bool(
                    row.get("show_log", profile["show_log"])
                )
                for key in (
                    "count_font_family",
                    "action_font_family",
                    "time_font_family",
                    "remaining_time_value_font_family",
                    "elapsed_time_value_font_family",
                    "count_font_color",
                    "action_font_color",
                    "time_font_color",
                    "remaining_time_value_font_color",
                    "elapsed_time_value_font_color",
                    "transcript_position",
                    "transcript_font_color",
                    "status_font_color",
                    "log_font_color",
                    "webcam_font_color",
                ):
                    profile[key] = (row.get(key, profile[key]) or profile[key]).strip()
                for key in (
                    "count_font_color",
                    "action_font_color",
                    "time_font_color",
                    "remaining_time_value_font_color",
                    "elapsed_time_value_font_color",
                    "transcript_font_color",
                    "status_font_color",
                    "log_font_color",
                    "webcam_font_color",
                ):
                    profile[key] = _normalize_hex_color(
                        profile.get(key, DEFAULT_UI_PROFILE.get(key, "#000000")),
                        DEFAULT_UI_PROFILE.get(key, "#000000"),
                    )
                for key in (
                    "count_font_size",
                    "action_font_size",
                    "time_font_size",
                    "remaining_time_value_font_size",
                    "elapsed_time_value_font_size",
                    "transcript_font_size",
                    "status_font_size",
                    "log_font_size",
                    "webcam_font_size",
                    "transcript_offset_x",
                    "transcript_offset_y",
                    "status_offset_x",
                    "status_offset_y",
                    "log_offset_x",
                    "log_offset_y",
                    "webcam_offset_x",
                    "webcam_offset_y",
                ):
                    try:
                        profile[key] = int(row.get(key, profile[key]))
                    except (TypeError, ValueError):
                        pass
                for item in UI_LABEL_TEXT_FIELDS:
                    prefix = item["prefix"]
                    text_key = f"{prefix}_text"
                    size_key = f"{prefix}_font_size"
                    color_key = f"{prefix}_color"
                    x_key = f"{prefix}_offset_x"
                    y_key = f"{prefix}_offset_y"
                    profile[text_key] = (row.get(text_key, profile[text_key]) or profile[text_key]).strip()
                    try:
                        profile[size_key] = int(row.get(size_key, profile[size_key]))
                    except (TypeError, ValueError):
                        pass
                    profile[color_key] = _normalize_hex_color(
                        row.get(color_key, profile[color_key]),
                        profile[color_key],
                    )
                    try:
                        profile[x_key] = int(row.get(x_key, profile[x_key]))
                    except (TypeError, ValueError):
                        pass
                    try:
                        profile[y_key] = int(row.get(y_key, profile[y_key]))
                    except (TypeError, ValueError):
                        pass
                for item in UI_VALUE_LABEL_FIELDS:
                    prefix = item["prefix"]
                    size_key = f"{prefix}_font_size"
                    color_key = f"{prefix}_color"
                    x_key = f"{prefix}_offset_x"
                    y_key = f"{prefix}_offset_y"
                    try:
                        profile[size_key] = int(row.get(size_key, profile[size_key]))
                    except (TypeError, ValueError):
                        pass
                    profile[color_key] = _normalize_hex_color(
                        row.get(color_key, profile[color_key]),
                        profile[color_key],
                    )
                    try:
                        profile[x_key] = int(row.get(x_key, profile[x_key]))
                    except (TypeError, ValueError):
                        pass
                    try:
                        profile[y_key] = int(row.get(y_key, profile[y_key]))
                    except (TypeError, ValueError):
                        pass
                for key, _message in UI_STATUS_MESSAGE_FIELDS:
                    profile[key] = (row.get(key, profile[key]) or profile[key]).strip()
                for key, _message in UI_GUIDE_MESSAGE_FIELDS:
                    profile[key] = (row.get(key, profile[key]) or profile[key]).strip()
                for key, _message in UI_WEBCAM_MESSAGE_FIELDS:
                    profile[key] = (row.get(key, profile[key]) or profile[key]).strip()
                profile["log_placeholder_text"] = (
                    row.get("log_placeholder_text", profile["log_placeholder_text"])
                    or profile["log_placeholder_text"]
                ).strip()
                profiles[profile_name] = profile
    except Exception:
        return {}

    return profiles


def _load_wifi_profiles_from_csv():
    profiles = {}
    if not WIFI_FILE.exists():
        return profiles

    try:
        with WIFI_FILE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile_name = (row.get("profile_name") or "").strip()
                if not profile_name:
                    continue

                profile = profiles.setdefault(
                    profile_name,
                    {
                        "access_points": [],
                        "init_test_duration": DEFAULT_WIFI_PROFILE["init_test_duration"],
                        "init_test_save_directory": DEFAULT_WIFI_PROFILE[
                            "init_test_save_directory"
                        ],
                        "csi_capture_scenario": DEFAULT_WIFI_PROFILE[
                            "csi_capture_scenario"
                        ],
                        "pre_action_capture_duration": DEFAULT_WIFI_PROFILE[
                            "pre_action_capture_duration"
                        ],
                        "post_action_capture_duration": DEFAULT_WIFI_PROFILE[
                            "post_action_capture_duration"
                        ],
                        "delete_prev_pcap": DEFAULT_WIFI_PROFILE["delete_prev_pcap"],
                        "count_packets": DEFAULT_WIFI_PROFILE["count_packets"],
                        "reboot_after_summary": DEFAULT_WIFI_PROFILE[
                            "reboot_after_summary"
                        ],
                    },
                )

                try:
                    profile["init_test_duration"] = float(
                        row.get("init_test_duration", profile["init_test_duration"]) or profile["init_test_duration"]
                    )
                except (TypeError, ValueError):
                    profile["init_test_duration"] = profile["init_test_duration"]

                profile["init_test_save_directory"] = (
                    row.get("init_test_save_directory", profile["init_test_save_directory"])
                    or profile["init_test_save_directory"]
                )
                profile["csi_capture_scenario"] = row.get(
                    "csi_capture_scenario", profile["csi_capture_scenario"]
                ) or profile["csi_capture_scenario"]
                profile["delete_prev_pcap"] = _as_bool(
                    row.get("delete_prev_pcap", profile["delete_prev_pcap"])
                )
                try:
                    profile["pre_action_capture_duration"] = float(
                        row.get(
                            "pre_action_capture_duration",
                            profile["pre_action_capture_duration"],
                        )
                        or profile["pre_action_capture_duration"]
                    )
                except (TypeError, ValueError):
                    profile["pre_action_capture_duration"] = profile[
                        "pre_action_capture_duration"
                    ]
                try:
                    profile["post_action_capture_duration"] = float(
                        row.get(
                            "post_action_capture_duration",
                            profile["post_action_capture_duration"],
                        )
                        or profile["post_action_capture_duration"]
                    )
                except (TypeError, ValueError):
                    profile["post_action_capture_duration"] = profile[
                        "post_action_capture_duration"
                    ]
                profile["count_packets"] = _as_bool(
                    row.get("count_packets", profile["count_packets"])
                )
                profile["reboot_after_summary"] = _as_bool(
                    row.get("reboot_after_summary", profile["reboot_after_summary"])
                )

                ap = deepcopy(DEFAULT_WIFI_AP)
                ap["name"] = (row.get("name") or "").strip() or profile_name
                ap["ssid"] = row.get("ssid", "")
                ap["password"] = row.get("password", "")
                ap["router_ssh_ip"] = row.get("router_ssh_ip", "")
                ap["router_ssh_username"] = row.get("router_ssh_username", "")
                ap["router_ssh_password"] = row.get("router_ssh_password", "")
                ap["ssh_key_address"] = row.get("ssh_key_address", "")
                ap["framework"] = (
                    row.get("framework")
                    or row.get("beginning_script")
                    or ""
                ).strip()
                ap["type"] = (
                    row.get("type")
                    or row.get("ending_script")
                    or ""
                ).strip()
                ap["frequency"] = row.get("frequency", ap["frequency"]).strip() or ap[
                    "frequency"
                ]
                ap["channel"] = row.get("channel", ap["channel"]).strip() or ap[
                    "channel"
                ]
                ap["bandwidth"] = row.get("bandwidth", ap["bandwidth"]).strip() or ap[
                    "bandwidth"
                ]
                ap["transmitter_macs"] = row.get("transmitter_macs", "")
                ap["init_test_save_directory"] = (
                    row.get("init_test_save_directory")
                    or profile.get("init_test_save_directory")
                    or DEFAULT_WIFI_PROFILE["init_test_save_directory"]
                )
                ap["use_ethernet"] = _as_bool(row.get("use_ethernet", ap["use_ethernet"]))
                ap["download_mode"] = (
                    row.get("download_mode", ap["download_mode"]).strip() or ap[
                        "download_mode"
                    ]
                )
                try:
                    ap["order"] = int(row.get("order", ap["order"]))
                except (TypeError, ValueError):
                    ap["order"] = ap["order"]

                profile["access_points"].append(ap)
    except Exception:
        return {}

    return profiles


def load_participant_profiles():
    _ensure_default_profile_file(
        PARTICIPANTS_FILE,
        {DEFAULT_PARTICIPANT_PROFILE_NAME: deepcopy(DEFAULT_PARTICIPANT_PROFILE)},
        save_participant_profiles,
    )
    profiles = _profiles_with_default(
        _load_participant_profiles_from_csv(),
        DEFAULT_PARTICIPANT_PROFILE_NAME,
        DEFAULT_PARTICIPANT_PROFILE,
    )
    for subject in profiles.values():
        if isinstance(subject, dict):
            subject["experiment_id"] = generate_experiment_id()
            subject["age_group"] = subject.get("age_group", "Blank") or "Blank"
            subject["age_value"] = 0 if subject.get("age_group") == "Blank" else subject.get("age_value", 0)
            subject.setdefault("has_second_participant", False)
            subject.setdefault("second_name", "")
            subject["second_age_group"] = subject.get("second_age_group", "Blank") or "Blank"
            subject["second_age_value"] = (
                0 if subject.get("second_age_group") == "Blank" else subject.get("second_age_value", 0)
            )
            subject.setdefault("second_participant_id", generate_participant_id("", subject["second_age_group"], ""))
            subject.setdefault("second_gender", "")
            subject.setdefault("second_dominant_hand", "")
            subject.setdefault("second_height_cm", 0.0)
            subject.setdefault("second_weight_kg", 0.0)
            subject.setdefault("second_description", "")
    if len(profiles) > 1:
        return profiles

    legacy = _load_legacy_profile_from_txt()
    merged = deepcopy(DEFAULT_PARTICIPANT_PROFILE)
    merged.update(deepcopy(legacy["subject"]))
    profiles[DEFAULT_PARTICIPANT_PROFILE_NAME] = merged
    return profiles


def load_environment_profiles():
    _ensure_default_profile_file(
        ENVIRONMENT_FILE,
        {DEFAULT_ENVIRONMENT_PROFILE_NAME: deepcopy(DEFAULT_ENVIRONMENT_PROFILE)},
        save_environment_profiles,
    )
    return _profiles_with_default(
        _load_environment_profiles_from_csv(),
        DEFAULT_ENVIRONMENT_PROFILE_NAME,
        DEFAULT_ENVIRONMENT_PROFILE,
    )


def load_time_profiles():
    _ensure_default_profile_file(
        TIME_FILE,
        {DEFAULT_TIME_PROFILE_NAME: deepcopy(DEFAULT_TIME_PROFILE)},
        save_time_profiles,
    )
    return _profiles_with_default(
        _load_time_profiles_from_csv(),
        DEFAULT_TIME_PROFILE_NAME,
        DEFAULT_TIME_PROFILE,
    )


def load_demo_profiles():
    _ensure_default_profile_file(
        DEMO_FILE,
        {DEFAULT_DEMO_PROFILE_NAME: deepcopy(DEFAULT_DEMO_PROFILE)},
        save_demo_profiles,
    )
    return _profiles_with_default(
        _load_demo_profiles_from_csv(),
        DEFAULT_DEMO_PROFILE_NAME,
        DEFAULT_DEMO_PROFILE,
    )


def load_experiment_profiles():
    _ensure_default_profile_file(
        EXPERIMENTS_FILE,
        {DEFAULT_EXPERIMENT_PROFILE_NAME: deepcopy(DEFAULT_EXPERIMENT_PROFILE)},
        save_experiment_profiles,
    )
    profiles = _profiles_with_default(
        _load_experiment_profiles_from_csv(),
        DEFAULT_EXPERIMENT_PROFILE_NAME,
        DEFAULT_EXPERIMENT_PROFILE,
    )
    if len(profiles) > 1:
        return profiles

    legacy = _load_legacy_profile_from_txt()
    merged = deepcopy(DEFAULT_EXPERIMENT_PROFILE)
    merged.update(deepcopy(legacy["experiment"]))
    profiles[DEFAULT_EXPERIMENT_PROFILE_NAME] = merged
    return profiles


def load_action_profiles():
    _ensure_default_profile_file(
        ACTIONS_FILE,
        {DEFAULT_ACTION_PROFILE_NAME: deepcopy(DEFAULT_ACTION_PROFILE)},
        save_action_profiles,
    )
    profiles = _profiles_with_default(
        _load_action_profiles_from_csv(),
        DEFAULT_ACTION_PROFILE_NAME,
        DEFAULT_ACTION_PROFILE,
    )
    profiles = {name: filter_blank_actions(actions) for name, actions in profiles.items()}
    if len(profiles) > 1:
        return profiles

    legacy = _load_legacy_profile_from_txt()
    merged = deepcopy(DEFAULT_ACTION_PROFILE)
    merged.update(deepcopy(legacy["actions"]))
    profiles[DEFAULT_ACTION_PROFILE_NAME] = filter_blank_actions(merged)
    return profiles


def load_voice_profiles():
    _ensure_default_profile_file(
        VOICE_FILE,
        {DEFAULT_VOICE_PROFILE_NAME: deepcopy(DEFAULT_VOICE_PROFILE)},
        save_voice_profiles,
    )
    return _profiles_with_default(
        _load_voice_profiles_from_csv(),
        DEFAULT_VOICE_PROFILE_NAME,
        DEFAULT_VOICE_PROFILE,
    )


def load_ui_profiles():
    _ensure_default_profile_file(
        UI_FILE,
        {DEFAULT_UI_PROFILE_NAME: deepcopy(DEFAULT_UI_PROFILE)},
        save_ui_profiles,
    )
    return _profiles_with_default(
        _load_ui_profiles_from_csv(),
        DEFAULT_UI_PROFILE_NAME,
        DEFAULT_UI_PROFILE,
    )


def _create_camera_profiles_from_experiments(experiments: dict) -> dict:
    camera_profiles = {}
    for name, exp in experiments.items():
        camera_profiles[name] = {
            "use_webcam": _as_bool(
                exp.get("use_webcam", DEFAULT_CAMERA_PROFILE["use_webcam"])
            ),
            "use_hand_recognition": _as_bool(
                exp.get(
                    "use_hand_recognition", DEFAULT_CAMERA_PROFILE["use_hand_recognition"]
                )
            ),
            "hand_recognition_mode": _normalize_hand_mode(
                exp.get(
                    "hand_recognition_mode",
                    DEFAULT_CAMERA_PROFILE["hand_recognition_mode"],
                )
            ),
            "hand_model_complexity": _normalize_model_complexity(
                exp.get(
                    "hand_model_complexity",
                    DEFAULT_CAMERA_PROFILE["hand_model_complexity"],
                )
            ),
        }
    return camera_profiles


def load_camera_profiles(experiments: dict | None = None):
    _ensure_default_profile_file(
        CAMERA_FILE,
        {DEFAULT_CAMERA_PROFILE_NAME: deepcopy(DEFAULT_CAMERA_PROFILE)},
        save_camera_profiles,
    )
    profiles = _profiles_with_default(
        _load_camera_profiles_from_csv(),
        DEFAULT_CAMERA_PROFILE_NAME,
        DEFAULT_CAMERA_PROFILE,
    )
    return profiles


def load_depth_camera_profiles():
    _ensure_default_profile_file(
        DEPTH_CAMERA_FILE,
        {DEFAULT_DEPTH_CAMERA_PROFILE_NAME: deepcopy(DEFAULT_DEPTH_CAMERA_PROFILE)},
        save_depth_camera_profiles,
    )
    return _profiles_with_default(
        _load_depth_camera_profiles_from_csv(),
        DEFAULT_DEPTH_CAMERA_PROFILE_NAME,
        DEFAULT_DEPTH_CAMERA_PROFILE,
    )


def load_wifi_profiles():
    _ensure_default_profile_file(
        WIFI_FILE,
        {DEFAULT_WIFI_PROFILE_NAME: deepcopy(DEFAULT_WIFI_PROFILE)},
        save_wifi_profiles,
    )
    profiles = _profiles_with_default(
        _load_wifi_profiles_from_csv(),
        DEFAULT_WIFI_PROFILE_NAME,
        DEFAULT_WIFI_PROFILE,
    )

    for name, profile in profiles.items():
        profile.setdefault("access_points", [])
        profile.setdefault(
            "init_test_duration", DEFAULT_WIFI_PROFILE["init_test_duration"]
        )
        profile.setdefault(
            "init_test_save_directory",
            DEFAULT_WIFI_PROFILE["init_test_save_directory"],
        )
        profile.setdefault(
            "pre_action_capture_duration",
            DEFAULT_WIFI_PROFILE["pre_action_capture_duration"],
        )
        profile.setdefault(
            "post_action_capture_duration",
            DEFAULT_WIFI_PROFILE["post_action_capture_duration"],
        )
        profile.setdefault("delete_prev_pcap", DEFAULT_WIFI_PROFILE["delete_prev_pcap"])
        profile.setdefault(
            "reboot_after_summary", DEFAULT_WIFI_PROFILE["reboot_after_summary"]
        )
        for ap in profile.get("access_points", []):
            ap.setdefault(
                "init_test_save_directory", profile.get("init_test_save_directory", "")
            )

    if profiles:
        return profiles
    return {}


def load_selected_profile_choices():
    if not SELECTED_PROFILES_FILE.exists():
        return {}

    try:
        with SELECTED_PROFILES_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_selected_profile_choices(choices: dict):
    try:
        SELECTED_PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with SELECTED_PROFILES_FILE.open("w", encoding="utf-8") as handle:
            json.dump(choices, handle, indent=2)
    except Exception:
        pass


def save_participant_profiles(profiles: dict):
    _ensure_configs_dir()
    try:
        with PARTICIPANTS_FILE.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "profile_name",
                "participant_name",
                "age_group",
                "age_value",
                "participant_id",
                "gender",
                "dominant_hand",
                "height_cm",
                "weight_kg",
                "description",
                "has_second_participant",
                "second_name",
                "second_age_group",
                "second_age_value",
                "second_participant_id",
                "second_gender",
                "second_dominant_hand",
                "second_height_cm",
                "second_weight_kg",
                "second_description",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for profile_name, subject in profiles.items():
                if not isinstance(subject, dict):
                    continue
                gender = subject.get("gender", "")
                second_gender = subject.get("second_gender", "")
                row = {
                    "profile_name": profile_name,
                    "participant_name": subject.get("name", ""),
                    "age_group": subject.get("age_group", "Blank"),
                    "age_value": subject.get("age_value", 0),
                    "participant_id": generate_participant_id(
                        subject.get("name", ""),
                        subject.get("age_group", ""),
                        gender,
                    ),
                    "gender": gender,
                    "dominant_hand": subject.get("dominant_hand", ""),
                    "height_cm": subject.get("height_cm", 0.0),
                    "weight_kg": subject.get("weight_kg", 0.0),
                    "description": subject.get("description", ""),
                    "has_second_participant": bool(subject.get("has_second_participant", False)),
                    "second_name": subject.get("second_name", ""),
                    "second_age_group": subject.get("second_age_group", "Blank"),
                    "second_age_value": subject.get("second_age_value", 0),
                    "second_participant_id": generate_participant_id(
                        subject.get("second_name", ""),
                        subject.get("second_age_group", ""),
                        second_gender,
                    ),
                    "second_gender": second_gender,
                    "second_dominant_hand": subject.get("second_dominant_hand", ""),
                    "second_height_cm": subject.get("second_height_cm", 0.0),
                    "second_weight_kg": subject.get("second_weight_kg", 0.0),
                    "second_description": subject.get("second_description", ""),
                }
                writer.writerow(row)
    except Exception:
        pass


def save_environment_profiles(profiles: dict):
    _ensure_configs_dir()
    try:
        with ENVIRONMENT_FILE.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "profile_name",
                "length_m",
                "width_m",
                "height_m",
                "description",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for profile_name, profile in profiles.items():
                if not isinstance(profile, dict):
                    continue
                row = {"profile_name": profile_name}
                for key in ("length_m", "width_m", "height_m", "description"):
                    row[key] = profile.get(key, DEFAULT_ENVIRONMENT_PROFILE.get(key))
                writer.writerow(row)
    except Exception:
        pass


def save_time_profiles(profiles: dict):
    _ensure_configs_dir()
    try:
        with TIME_FILE.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "profile_name",
                "use_time_server",
                "time_server",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for profile_name, profile in profiles.items():
                if not isinstance(profile, dict):
                    continue
                writer.writerow(
                    {
                        "profile_name": profile_name,
                        "use_time_server": bool(
                            profile.get(
                                "use_time_server", DEFAULT_TIME_PROFILE["use_time_server"]
                            )
                        ),
                        "time_server": profile.get(
                            "time_server", DEFAULT_TIME_PROFILE["time_server"]
                        ),
                    }
                )
    except Exception:
        pass


def save_demo_profiles(profiles: dict):
    _ensure_configs_dir()
    try:
        with DEMO_FILE.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "profile_name",
                "demo_capture_mode",
                "capture_duration_seconds",
                "effective_capture_samples",
                "apply_hampel_to_ratio_magnitude",
                "apply_hampel_to_ratio_phase",
                "demo_title_text",
                "university_logo_image_path",
                "university_logo_image_size_px",
                "icassp_logo_image_path",
                "website_url",
                "icassp_title_text",
                "icassp_logo_text_vertical_gap",
                "demo_title_font_size_px",
                "authors_text",
                "authors_font_size_px",
                "university_text",
                "wirlab_text",
                "university_font_size_px",
                "title_authors_vertical_gap",
                "authors_university_vertical_gap",
                "capture_guidance_title",
                "capture_guidance_message",
                "capture_guidance_video_left_label",
                "capture_guidance_video_left_path",
                "capture_guidance_video_right_label",
                "capture_guidance_video_right_path",
                "activity_class_names",
                "subplot_settings_json",
                "dorf_plot_order",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for profile_name, profile in profiles.items():
                if not isinstance(profile, dict):
                    continue
                writer.writerow(
                    {
                        "profile_name": profile_name,
                        "demo_capture_mode": _normalize_demo_capture_mode(
                            profile.get(
                                "demo_capture_mode",
                                DEFAULT_DEMO_PROFILE["demo_capture_mode"],
                            )
                        ),
                        "capture_duration_seconds": float(
                            profile.get(
                                "capture_duration_seconds",
                                DEFAULT_DEMO_PROFILE["capture_duration_seconds"],
                            )
                        ),
                        "effective_capture_samples": int(
                            profile.get(
                                "effective_capture_samples",
                                DEFAULT_DEMO_PROFILE["effective_capture_samples"],
                            )
                        ),
                        "apply_hampel_to_ratio_magnitude": bool(
                            profile.get(
                                "apply_hampel_to_ratio_magnitude",
                                DEFAULT_DEMO_PROFILE["apply_hampel_to_ratio_magnitude"],
                            )
                        ),
                        "apply_hampel_to_ratio_phase": bool(
                            profile.get(
                                "apply_hampel_to_ratio_phase",
                                DEFAULT_DEMO_PROFILE["apply_hampel_to_ratio_phase"],
                            )
                        ),
                        "demo_title_text": (
                            str(
                                profile.get(
                                    "demo_title_text",
                                    DEFAULT_DEMO_PROFILE["demo_title_text"],
                                )
                            ).strip()
                            or DEFAULT_DEMO_PROFILE["demo_title_text"]
                        ),
                        "university_logo_image_path": str(
                            profile.get(
                                "university_logo_image_path",
                                DEFAULT_DEMO_PROFILE["university_logo_image_path"],
                            )
                        ).strip(),
                        "university_logo_image_size_px": max(
                            20,
                            min(
                                600,
                                int(
                                    profile.get(
                                        "university_logo_image_size_px",
                                        DEFAULT_DEMO_PROFILE["university_logo_image_size_px"],
                                    )
                                ),
                            ),
                        ),
                        "icassp_logo_image_path": str(
                            profile.get(
                                "icassp_logo_image_path",
                                DEFAULT_DEMO_PROFILE["icassp_logo_image_path"],
                            )
                        ).strip(),
                        "website_url": (
                            str(
                                profile.get(
                                    "website_url",
                                    DEFAULT_DEMO_PROFILE["website_url"],
                                )
                            ).strip()
                            or DEFAULT_DEMO_PROFILE["website_url"]
                        ),
                        "icassp_title_text": (
                            str(
                                profile.get(
                                    "icassp_title_text",
                                    DEFAULT_DEMO_PROFILE["icassp_title_text"],
                                )
                            ).strip()
                        ),
                        "icassp_logo_text_vertical_gap": int(
                            profile.get(
                                "icassp_logo_text_vertical_gap",
                                DEFAULT_DEMO_PROFILE["icassp_logo_text_vertical_gap"],
                            )
                        ),
                        "demo_title_font_size_px": max(
                            10,
                            min(
                                96,
                                int(
                                    profile.get(
                                        "demo_title_font_size_px",
                                        DEFAULT_DEMO_PROFILE["demo_title_font_size_px"],
                                    )
                                ),
                            ),
                        ),
                        "authors_text": (
                            str(
                                profile.get(
                                    "authors_text",
                                    DEFAULT_DEMO_PROFILE["authors_text"],
                                )
                            ).strip()
                            or DEFAULT_DEMO_PROFILE["authors_text"]
                        ),
                        "authors_font_size_px": max(
                            8,
                            min(
                                72,
                                int(
                                    profile.get(
                                        "authors_font_size_px",
                                        DEFAULT_DEMO_PROFILE["authors_font_size_px"],
                                    )
                                ),
                            ),
                        ),
                        "university_text": (
                            str(
                                profile.get(
                                    "university_text",
                                    DEFAULT_DEMO_PROFILE["university_text"],
                                )
                            ).strip()
                        ),
                        "wirlab_text": str(
                            profile.get(
                                "wirlab_text",
                                DEFAULT_DEMO_PROFILE["wirlab_text"],
                            )
                        ).strip(),
                        "university_font_size_px": max(
                            8,
                            min(
                                72,
                                int(
                                    profile.get(
                                        "university_font_size_px",
                                        DEFAULT_DEMO_PROFILE["university_font_size_px"],
                                    )
                                ),
                            ),
                        ),
                        "title_authors_vertical_gap": int(
                            profile.get(
                                "title_authors_vertical_gap",
                                DEFAULT_DEMO_PROFILE["title_authors_vertical_gap"],
                            )
                        ),
                        "authors_university_vertical_gap": int(
                            profile.get(
                                "authors_university_vertical_gap",
                                DEFAULT_DEMO_PROFILE["authors_university_vertical_gap"],
                            )
                        ),
                        "capture_guidance_title": (
                            str(
                                profile.get(
                                    "capture_guidance_title",
                                    DEFAULT_DEMO_PROFILE["capture_guidance_title"],
                                )
                            ).strip()
                            or DEFAULT_DEMO_PROFILE["capture_guidance_title"]
                        ),
                        "capture_guidance_message": (
                            str(
                                profile.get(
                                    "capture_guidance_message",
                                    DEFAULT_DEMO_PROFILE["capture_guidance_message"],
                                )
                            ).strip()
                            or DEFAULT_DEMO_PROFILE["capture_guidance_message"]
                        ),
                        "capture_guidance_video_left_label": (
                            str(
                                profile.get(
                                    "capture_guidance_video_left_label",
                                    DEFAULT_DEMO_PROFILE["capture_guidance_video_left_label"],
                                )
                            ).strip()
                            or DEFAULT_DEMO_PROFILE["capture_guidance_video_left_label"]
                        ),
                        "capture_guidance_video_left_path": str(
                            profile.get(
                                "capture_guidance_video_left_path",
                                DEFAULT_DEMO_PROFILE["capture_guidance_video_left_path"],
                            )
                        ).strip(),
                        "capture_guidance_video_right_label": (
                            str(
                                profile.get(
                                    "capture_guidance_video_right_label",
                                    DEFAULT_DEMO_PROFILE["capture_guidance_video_right_label"],
                                )
                            ).strip()
                            or DEFAULT_DEMO_PROFILE["capture_guidance_video_right_label"]
                        ),
                        "capture_guidance_video_right_path": str(
                            profile.get(
                                "capture_guidance_video_right_path",
                                DEFAULT_DEMO_PROFILE["capture_guidance_video_right_path"],
                            )
                        ).strip(),
                        "activity_class_names": ",".join(
                            str(item).strip()
                            for item in profile.get(
                                "activity_class_names",
                                DEFAULT_DEMO_PROFILE["activity_class_names"],
                            )
                            if str(item).strip()
                        ),
                        "subplot_settings_json": json.dumps(
                            profile.get(
                                "subplot_settings",
                                DEFAULT_DEMO_PROFILE["subplot_settings"],
                            ),
                            ensure_ascii=False,
                        ),
                        "dorf_plot_order": ",".join(
                            str(item).strip()
                            for item in profile.get(
                                "dorf_plot_order",
                                DEFAULT_DEMO_PROFILE["dorf_plot_order"],
                            )
                            if str(item).strip()
                        ),
                    }
                )
    except Exception:
        pass


def save_ui_profiles(profiles: dict):
    _ensure_configs_dir()
    try:
        with UI_FILE.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "profile_name",
                "camera_frame_percent",
                "preview_frame_percent",
                "action_frame_percent",
                "details_pane_percent",
                "show_log",
                "log_frame_percent",
                "start_fullscreen",
                "count_font_family",
                "count_font_size",
                "count_font_color",
                "action_font_family",
                "action_font_size",
                "action_font_color",
                "time_font_family",
                "time_font_size",
                "time_font_color",
                "remaining_time_value_font_family",
                "remaining_time_value_font_size",
                "remaining_time_value_font_color",
                "elapsed_time_value_font_family",
                "elapsed_time_value_font_size",
                "elapsed_time_value_font_color",
                "transcript_position",
                "transcript_font_size",
                "transcript_font_color",
                "transcript_offset_x",
                "transcript_offset_y",
                "status_font_size",
                "status_font_color",
                "status_offset_x",
                "status_offset_y",
                "log_placeholder_text",
                "log_font_size",
                "log_font_color",
                "log_offset_x",
                "log_offset_y",
                "webcam_font_size",
                "webcam_font_color",
                "webcam_offset_x",
                "webcam_offset_y",
            ]
            for item in UI_LABEL_TEXT_FIELDS:
                prefix = item["prefix"]
                fieldnames.extend(
                    [
                        f"{prefix}_text",
                        f"{prefix}_font_size",
                        f"{prefix}_color",
                        f"{prefix}_offset_x",
                        f"{prefix}_offset_y",
                    ]
                )
            for item in UI_VALUE_LABEL_FIELDS:
                prefix = item["prefix"]
                fieldnames.extend(
                    [
                        f"{prefix}_font_size",
                        f"{prefix}_color",
                        f"{prefix}_offset_x",
                        f"{prefix}_offset_y",
                    ]
                )
            fieldnames.extend([key for key, _message in UI_STATUS_MESSAGE_FIELDS])
            fieldnames.extend([key for key, _message in UI_GUIDE_MESSAGE_FIELDS])
            fieldnames.extend([key for key, _message in UI_WEBCAM_MESSAGE_FIELDS])
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for profile_name, profile in profiles.items():
                if not isinstance(profile, dict):
                    continue
                row = {"profile_name": profile_name}
                for key in (
                    "camera_frame_percent",
                    "preview_frame_percent",
                    "action_frame_percent",
                    "details_pane_percent",
                    "log_frame_percent",
                ):
                    row[key] = profile.get(key, DEFAULT_UI_PROFILE.get(key, 100.0))
                row["start_fullscreen"] = "1" if _as_bool(profile.get("start_fullscreen", False)) else "0"
                row["show_log"] = "1" if _as_bool(profile.get("show_log", True)) else "0"
                for key in (
                    "count_font_family",
                    "action_font_family",
                    "time_font_family",
                    "remaining_time_value_font_family",
                    "elapsed_time_value_font_family",
                    "count_font_color",
                    "action_font_color",
                    "time_font_color",
                    "remaining_time_value_font_color",
                    "elapsed_time_value_font_color",
                    "transcript_position",
                    "transcript_font_color",
                    "status_font_color",
                    "log_font_color",
                    "webcam_font_color",
                ):
                    row[key] = profile.get(key, DEFAULT_UI_PROFILE.get(key, ""))
                for key in (
                    "count_font_size",
                    "action_font_size",
                    "time_font_size",
                    "remaining_time_value_font_size",
                    "elapsed_time_value_font_size",
                    "transcript_font_size",
                    "transcript_offset_x",
                    "transcript_offset_y",
                    "status_font_size",
                    "status_offset_x",
                    "status_offset_y",
                    "log_font_size",
                    "log_offset_x",
                    "log_offset_y",
                    "webcam_font_size",
                    "webcam_offset_x",
                    "webcam_offset_y",
                ):
                    row[key] = profile.get(key, DEFAULT_UI_PROFILE.get(key, 12))
                row["log_placeholder_text"] = profile.get(
                    "log_placeholder_text",
                    DEFAULT_UI_PROFILE.get("log_placeholder_text", ""),
                )
                for item in UI_LABEL_TEXT_FIELDS:
                    prefix = item["prefix"]
                    row[f"{prefix}_text"] = profile.get(
                        f"{prefix}_text", DEFAULT_UI_PROFILE.get(f"{prefix}_text", "")
                    )
                    row[f"{prefix}_font_size"] = profile.get(
                        f"{prefix}_font_size",
                        DEFAULT_UI_PROFILE.get(f"{prefix}_font_size", 12),
                    )
                    row[f"{prefix}_color"] = profile.get(
                        f"{prefix}_color",
                        DEFAULT_UI_PROFILE.get(f"{prefix}_color", "#000000"),
                    )
                    row[f"{prefix}_offset_x"] = profile.get(
                        f"{prefix}_offset_x",
                        DEFAULT_UI_PROFILE.get(f"{prefix}_offset_x", 0),
                    )
                    row[f"{prefix}_offset_y"] = profile.get(
                        f"{prefix}_offset_y",
                        DEFAULT_UI_PROFILE.get(f"{prefix}_offset_y", 0),
                    )
                for item in UI_VALUE_LABEL_FIELDS:
                    prefix = item["prefix"]
                    row[f"{prefix}_font_size"] = profile.get(
                        f"{prefix}_font_size",
                        DEFAULT_UI_PROFILE.get(f"{prefix}_font_size", 12),
                    )
                    row[f"{prefix}_color"] = profile.get(
                        f"{prefix}_color",
                        DEFAULT_UI_PROFILE.get(f"{prefix}_color", "#000000"),
                    )
                    row[f"{prefix}_offset_x"] = profile.get(
                        f"{prefix}_offset_x",
                        DEFAULT_UI_PROFILE.get(f"{prefix}_offset_x", 0),
                    )
                    row[f"{prefix}_offset_y"] = profile.get(
                        f"{prefix}_offset_y",
                        DEFAULT_UI_PROFILE.get(f"{prefix}_offset_y", 0),
                    )
                for key, _message in UI_STATUS_MESSAGE_FIELDS:
                    row[key] = profile.get(key, DEFAULT_UI_PROFILE.get(key, ""))
                for key, _message in UI_GUIDE_MESSAGE_FIELDS:
                    row[key] = profile.get(key, DEFAULT_UI_PROFILE.get(key, ""))
                for key, _message in UI_WEBCAM_MESSAGE_FIELDS:
                    row[key] = profile.get(key, DEFAULT_UI_PROFILE.get(key, ""))
                writer.writerow(row)
    except Exception:
        pass


def save_experiment_profiles(profiles: dict):
    _ensure_configs_dir()
    try:
        with EXPERIMENTS_FILE.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "profile_name",
                "beginning_baseline_recording",
                "between_actions_baseline_recording",
                "ending_baseline_recording",
                "stop_time",
                "preview_pause",
                "action_time",
                "each_action_repitition_times",
                "save_location",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for profile_name, exp in profiles.items():
                if not isinstance(exp, dict):
                    continue
                row = {"profile_name": profile_name}
                for key in fieldnames[1:]:
                    value = exp.get(key, DEFAULT_EXPERIMENT_PROFILE.get(key))
                    row[key] = value
                writer.writerow(row)
    except Exception:
        pass


def save_action_profiles(profiles: dict):
    _ensure_configs_dir()
    try:
        with ACTIONS_FILE.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["profile_name", "action_name", "video_path"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for profile_name, actions in profiles.items():
                if not isinstance(actions, dict):
                    continue
                if not actions:
                    writer.writerow({"profile_name": profile_name, "action_name": "", "video_path": ""})
                    continue
                for action_name, video_path in actions.items():
                    writer.writerow(
                        {
                            "profile_name": profile_name,
                            "action_name": action_name,
                            "video_path": video_path,
                        }
                    )
    except Exception:
        pass


def save_voice_profiles(profiles: dict):
    _ensure_configs_dir()
    try:
        with VOICE_FILE.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "profile_name",
                "use_voice_assistant",
                "language",
                "preview_actions_voice",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for profile_name, voice in profiles.items():
                if not isinstance(voice, dict):
                    continue
                writer.writerow(
                    {
                        "profile_name": profile_name,
                        "use_voice_assistant": "1"
                        if _as_bool(voice.get("use_voice_assistant", False))
                        else "0",
                        "language": (voice.get("language") or "en").strip(),
                        "preview_actions_voice": "1"
                        if _as_bool(voice.get("preview_actions_voice", True))
                        else "0",
                    }
                )
    except Exception:
        pass


def save_camera_profiles(profiles: dict):
    _ensure_configs_dir()
    try:
        with CAMERA_FILE.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "profile_name",
                "use_webcam",
                "camera_device",
                "use_hand_recognition",
                "hand_recognition_mode",
                "hand_model_complexity",
                "hand_left_wrist_color",
                "hand_right_wrist_color",
                "hand_wrist_circle_radius",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for profile_name, camera in profiles.items():
                if not isinstance(camera, dict):
                    continue
                writer.writerow(
                    {
                        "profile_name": profile_name,
                        "use_webcam": "1"
                        if _as_bool(camera.get("use_webcam", False))
                        else "0",
                        "camera_device": str(camera.get("camera_device") or "0").strip(),
                        "use_hand_recognition": "1"
                        if _as_bool(camera.get("use_hand_recognition", False))
                        else "0",
                        "hand_recognition_mode": _normalize_hand_mode(
                            camera.get(
                                "hand_recognition_mode",
                                DEFAULT_CAMERA_PROFILE["hand_recognition_mode"],
                            )
                        ),
                        "hand_model_complexity": _normalize_model_complexity(
                            camera.get("hand_model_complexity", DEFAULT_CAMERA_PROFILE["hand_model_complexity"])
                        ),
                        "hand_left_wrist_color": _normalize_hex_color(
                            camera.get(
                                "hand_left_wrist_color",
                                DEFAULT_CAMERA_PROFILE["hand_left_wrist_color"],
                            ),
                            DEFAULT_CAMERA_PROFILE["hand_left_wrist_color"],
                        ),
                        "hand_right_wrist_color": _normalize_hex_color(
                            camera.get(
                                "hand_right_wrist_color",
                                DEFAULT_CAMERA_PROFILE["hand_right_wrist_color"],
                            ),
                            DEFAULT_CAMERA_PROFILE["hand_right_wrist_color"],
                        ),
                        "hand_wrist_circle_radius": max(
                            1,
                            int(
                                camera.get(
                                    "hand_wrist_circle_radius",
                                    DEFAULT_CAMERA_PROFILE["hand_wrist_circle_radius"],
                                )
                                or DEFAULT_CAMERA_PROFILE["hand_wrist_circle_radius"]
                            ),
                        ),
                    }
                )
    except Exception:
        pass


def save_depth_camera_profiles(profiles: dict):
    _ensure_configs_dir()
    try:
        with DEPTH_CAMERA_FILE.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "profile_name",
                "enabled",
                "api_ip",
                "fps",
                "rgb_resolution",
                "depth_resolution",
                "save_raw_npz",
                "save_location",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for profile_name, profile in profiles.items():
                if not isinstance(profile, dict):
                    continue
                writer.writerow(
                    {
                        "profile_name": profile_name,
                        "enabled": "1"
                        if _as_bool(profile.get("enabled", False))
                        else "0",
                        "api_ip": (profile.get("api_ip") or "").strip(),
                        "fps": int(profile.get("fps", DEFAULT_DEPTH_CAMERA_PROFILE["fps"])),
                        "rgb_resolution": _resolution_to_text(
                            profile.get(
                                "rgb_resolution",
                                DEFAULT_DEPTH_CAMERA_PROFILE["rgb_resolution"],
                            )
                        ),
                        "depth_resolution": _resolution_to_text(
                            profile.get(
                                "depth_resolution",
                                DEFAULT_DEPTH_CAMERA_PROFILE["depth_resolution"],
                            )
                        ),
                        "save_raw_npz": "1"
                        if _as_bool(profile.get("save_raw_npz", False))
                        else "0",
                        "save_location": profile.get("save_location", "")
                        or DEFAULT_DEPTH_CAMERA_PROFILE["save_location"],
                    }
                )
    except Exception:
        pass


def save_wifi_profiles(profiles: dict):
    _ensure_configs_dir()
    try:
        with WIFI_FILE.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "profile_name",
                "name",
                "framework",
                "type",
                "order",
                "ssid",
                "password",
                "router_ssh_ip",
                "router_ssh_username",
                "router_ssh_password",
                "ssh_key_address",
                "frequency",
                "channel",
                "bandwidth",
                "transmitter_macs",
                "use_ethernet",
                "download_mode",
                "init_test_duration",
                "init_test_save_directory",
                "csi_capture_scenario",
                "pre_action_capture_duration",
                "post_action_capture_duration",
                "delete_prev_pcap",
                "count_packets",
                "reboot_after_summary",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for profile_name, profile in profiles.items():
                if isinstance(profile, list):
                    profile = {
                        "access_points": profile,
                    }
                if not isinstance(profile, dict):
                    continue

                aps = profile.get("access_points", [])
                init_duration = profile.get(
                    "init_test_duration", DEFAULT_WIFI_PROFILE["init_test_duration"]
                )
                init_save_directory = profile.get(
                    "init_test_save_directory",
                    DEFAULT_WIFI_PROFILE["init_test_save_directory"],
                )
                csi_capture_scenario = profile.get(
                    "csi_capture_scenario", DEFAULT_WIFI_PROFILE["csi_capture_scenario"]
                )
                pre_action_duration = profile.get(
                    "pre_action_capture_duration",
                    DEFAULT_WIFI_PROFILE["pre_action_capture_duration"],
                )
                post_action_duration = profile.get(
                    "post_action_capture_duration",
                    DEFAULT_WIFI_PROFILE["post_action_capture_duration"],
                )
                delete_prev_pcap = profile.get(
                    "delete_prev_pcap", DEFAULT_WIFI_PROFILE["delete_prev_pcap"]
                )
                if not aps:
                    writer.writerow(
                        {
                            "profile_name": profile_name,
                            "name": "",
                            "framework": "",
                            "type": "",
                            "order": 1,
                            "use_ethernet": "0",
                            "init_test_duration": init_duration,
                            "init_test_save_directory": init_save_directory,
                            "csi_capture_scenario": csi_capture_scenario,
                            "pre_action_capture_duration": pre_action_duration,
                            "post_action_capture_duration": post_action_duration,
                            "delete_prev_pcap": "1"
                            if _as_bool(delete_prev_pcap)
                            else "0",
                            "count_packets": "1"
                            if _as_bool(profile.get("count_packets"))
                            else "0",
                            "reboot_after_summary": "1"
                            if _as_bool(profile.get("reboot_after_summary"))
                            else "0",
                        }
                    )
                    continue

                for ap in aps:
                    row = {
                        "profile_name": profile_name,
                        "name": ap.get("name", ""),
                        "framework": ap.get("framework", ""),
                        "type": ap.get("type", ""),
                    }
                    for key in fieldnames:
                        if key in {"profile_name", "name", "framework", "type"}:
                            continue
                        if key == "use_ethernet":
                            row[key] = "1" if _as_bool(ap.get(key)) else "0"
                        elif key == "init_test_duration":
                            row[key] = init_duration
                        elif key == "init_test_save_directory":
                            row[key] = ap.get("init_test_save_directory", init_save_directory)
                        elif key == "csi_capture_scenario":
                            row[key] = csi_capture_scenario
                        elif key == "pre_action_capture_duration":
                            row[key] = pre_action_duration
                        elif key == "post_action_capture_duration":
                            row[key] = post_action_duration
                        elif key == "delete_prev_pcap":
                            row[key] = "1" if _as_bool(delete_prev_pcap) else "0"
                        elif key == "count_packets":
                            row[key] = "1" if _as_bool(profile.get(key)) else "0"
                        elif key == "reboot_after_summary":
                            row[key] = (
                                "1" if _as_bool(profile.get("reboot_after_summary")) else "0"
                            )
                        else:
                            row[key] = ap.get(key, DEFAULT_WIFI_AP.get(key, ""))
                    writer.writerow(row)
    except Exception:
        pass


# ----------------------------------------------------------------------
# Configuration dialog
# ----------------------------------------------------------------------
class ConfigDialog(QDialog):
    """Dialog to manage participant, experiment, and action profiles separately."""

    def __init__(
        self,
        participant_profiles: dict,
        experiment_profiles: dict,
        environment_profiles: dict,
        action_profiles: dict,
        voice_profiles: dict = None,
        camera_profiles: dict = None,
        depth_camera_profiles: dict = None,
        ui_profiles: dict = None,
        wifi_profiles: dict = None,
        time_profiles: dict = None,
        demo_profiles: dict = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("WIRLab - Experiment Configuration")
        self.setModal(True)
        self.resize(900, 650)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        self.voice_assistant = None
        self._voice_language = None
        self.session_experiment_id = generate_experiment_id()

        self.age_group_options = [
            "Blank",
            "Under 14",
            "14-18",
            "18-30",
            "30-40",
            "40-50",
            "50-60",
            "60-70",
            ">70",
        ]

        self.gender_options = [
            "",
            "Female",
            "Male",
            "Non-binary",
            "Prefer not to say",
        ]

        # Deep copies so edits do not leak out of the dialog
        self.participant_profiles = deepcopy(participant_profiles) or {
            DEFAULT_PARTICIPANT_PROFILE_NAME: deepcopy(DEFAULT_PARTICIPANT_PROFILE)
        }
        self.experiment_profiles = deepcopy(experiment_profiles) or {
            DEFAULT_EXPERIMENT_PROFILE_NAME: deepcopy(DEFAULT_EXPERIMENT_PROFILE)
        }
        self.environment_profiles = deepcopy(environment_profiles) or {
            DEFAULT_ENVIRONMENT_PROFILE_NAME: deepcopy(DEFAULT_ENVIRONMENT_PROFILE)
        }
        self.action_profiles = deepcopy(action_profiles) or {
            DEFAULT_ACTION_PROFILE_NAME: deepcopy(DEFAULT_ACTION_PROFILE)
        }
        if camera_profiles:
            self.camera_profiles = deepcopy(camera_profiles)
        else:
            self.camera_profiles = {
                DEFAULT_CAMERA_PROFILE_NAME: deepcopy(DEFAULT_CAMERA_PROFILE)
            }
        if depth_camera_profiles:
            self.depth_camera_profiles = deepcopy(depth_camera_profiles)
        else:
            self.depth_camera_profiles = {
                DEFAULT_DEPTH_CAMERA_PROFILE_NAME: deepcopy(DEFAULT_DEPTH_CAMERA_PROFILE)
            }
        if voice_profiles:
            self.voice_profiles = deepcopy(voice_profiles)
        else:
            self.voice_profiles = {"default_voice": deepcopy(DEFAULT_VOICE_PROFILE)}
        if ui_profiles:
            self.ui_profiles = deepcopy(ui_profiles)
        else:
            self.ui_profiles = {"default_ui": deepcopy(DEFAULT_UI_PROFILE)}
        if wifi_profiles:
            self.wifi_profiles = {}
            for key, profile in deepcopy(wifi_profiles).items():
                if isinstance(profile, list):
                    self.wifi_profiles[key] = {
                        "name": key,
                        "framework": "",
                        "type": "",
                        "access_points": profile,
                    }
                elif isinstance(profile, dict):
                    normalized = deepcopy(profile)
                    normalized.setdefault("access_points", [])
                    normalized.setdefault("delete_prev_pcap", False)
                    normalized.setdefault("count_packets", False)
                    self.wifi_profiles[key] = normalized
        else:
            self.wifi_profiles = {"default_wifi": deepcopy(DEFAULT_WIFI_PROFILE)}
        if time_profiles:
            self.time_profiles = deepcopy(time_profiles)
        else:
            self.time_profiles = {
                DEFAULT_TIME_PROFILE_NAME: deepcopy(DEFAULT_TIME_PROFILE)
            }
        if demo_profiles:
            self.demo_profiles = deepcopy(demo_profiles)
        else:
            self.demo_profiles = {
                DEFAULT_DEMO_PROFILE_NAME: deepcopy(DEFAULT_DEMO_PROFILE)
            }

        # Ensure gender options contain the genders already stored
        for subject in self.participant_profiles.values():
            gender = (subject or {}).get("gender", "")
            self._ensure_gender_option(gender)

        saved_choices = load_selected_profile_choices()

        self.current_participant_name = self._resolve_initial_profile_choice(
            saved_choices.get("participant"),
            self.participant_profiles,
            DEFAULT_PARTICIPANT_PROFILE_NAME,
        )
        self.current_experiment_name = self._resolve_initial_profile_choice(
            saved_choices.get("experiment"),
            self.experiment_profiles,
            DEFAULT_EXPERIMENT_PROFILE_NAME,
        )
        self.current_environment_name = self._resolve_initial_profile_choice(
            saved_choices.get("environment"),
            self.environment_profiles,
            DEFAULT_ENVIRONMENT_PROFILE_NAME,
        )
        self.current_action_profile_name = self._resolve_initial_profile_choice(
            saved_choices.get("actions"),
            self.action_profiles,
            DEFAULT_ACTION_PROFILE_NAME,
        )
        self.current_camera_profile_name = self._resolve_initial_profile_choice(
            saved_choices.get("camera"),
            self.camera_profiles,
            DEFAULT_CAMERA_PROFILE_NAME,
        )
        self.current_depth_camera_profile_name = self._resolve_initial_profile_choice(
            saved_choices.get("depth_camera"),
            self.depth_camera_profiles,
            DEFAULT_DEPTH_CAMERA_PROFILE_NAME,
        )
        self.current_voice_profile_name = self._resolve_initial_profile_choice(
            saved_choices.get("voice"),
            self.voice_profiles,
            DEFAULT_VOICE_PROFILE_NAME,
        )
        self.current_ui_profile_name = self._resolve_initial_profile_choice(
            saved_choices.get("ui"),
            self.ui_profiles,
            DEFAULT_UI_PROFILE_NAME,
        )
        self.current_wifi_profile_name = self._resolve_initial_profile_choice(
            saved_choices.get("wifi"),
            self.wifi_profiles,
            DEFAULT_WIFI_PROFILE_NAME,
        )
        self.current_time_profile_name = self._resolve_initial_profile_choice(
            saved_choices.get("time"),
            self.time_profiles,
            DEFAULT_TIME_PROFILE_NAME,
        )
        self.current_demo_profile_name = self._resolve_initial_profile_choice(
            saved_choices.get("demo"),
            self.demo_profiles,
            DEFAULT_DEMO_PROFILE_NAME,
        )

        self.wifi_manager = WiFiCSIManager()

        self._reboot_dialog = None
        self._reboot_thread = None

        self._settings_locked = is_password_required()

        self._camera_preview_timer = None
        self._camera_preview_capture = None
        self.cmb_camera_device = None
        self.btn_refresh_cameras = None
        self.btn_preview_camera = None
        self.lbl_camera_preview = None
        self._depth_status_timer = None
        self._depth_status_thread = None
        self._depth_status_cancel = False
        self._depth_status_refreshing = False
        self._time_preview_timer = None
        self._time_sync_thread = None
        self._time_sync_result = None
        self._time_sync_error = None
        self._help_dialog = None

        self._build_ui()
        self._populate_participant_combo()
        self._populate_experiment_combo()
        self._populate_environment_combo()
        self._populate_action_combo()
        self._populate_camera_combo()
        self._populate_depth_camera_combo()
        self._populate_voice_combo()
        self._populate_ui_combo()
        self._populate_wifi_combo()
        self._populate_time_combo()
        self._populate_demo_combo()

        self._load_participant_into_ui(self.current_participant_name)
        self._load_experiment_into_ui(self.current_experiment_name)
        self._load_environment_into_ui(self.current_environment_name)
        self._load_actions_into_ui(self.current_action_profile_name)
        self._load_camera_into_ui(self.current_camera_profile_name)
        self._load_depth_camera_into_ui(self.current_depth_camera_profile_name)
        self._load_voice_into_ui(self.current_voice_profile_name)
        self._load_ui_into_ui(self.current_ui_profile_name)
        self._load_wifi_into_ui(self.current_wifi_profile_name)
        self._load_time_into_ui(self.current_time_profile_name)
        self._load_demo_into_ui(self.current_demo_profile_name)
        self._set_locked_state(self._settings_locked)

    def _ensure_gender_option(self, value: str):
        value = (value or "").strip()
        if not value:
            return
        if value not in self.gender_options:
            self.gender_options.append(value)
        if hasattr(self, "cmb_gender") and self.cmb_gender.findText(value) < 0:
            self.cmb_gender.addItem(value)

    def _resolve_initial_profile_choice(
        self, preferred: str | None, profiles: dict, default_name: str
    ) -> str:
        if preferred and preferred in profiles:
            return preferred
        if default_name in profiles:
            return default_name
        return next(iter(profiles.keys()), "")

    # ------------------------------------------------------------------
    # Voice helpers
    # ------------------------------------------------------------------
    def _populate_language_options(self):
        combo = getattr(self, "cmb_voice_language", None)
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        for code, label in SUPPORTED_LANGUAGES:
            combo.addItem(f"{label} ({code})", code)
        combo.blockSignals(False)

    def _apply_button_color(self, button: QPushButton | None, color_hex: str):
        if button is None:
            return
        color = QColor(color_hex)
        if not color.isValid():
            color = QColor("#000000")
        text_color = "#000000" if color.lightness() > 128 else "#ffffff"
        button.setProperty("color_hex", color.name())
        button.setStyleSheet(
            f"background-color: {color.name()}; color: {text_color}; padding: 4px;"
        )
        button.setText(color.name())

    def _pick_button_color(self, button: QPushButton | None):
        if button is None:
            return
        current = QColor(button.property("color_hex") or "#000000")
        if not current.isValid():
            current = QColor("#000000")
        chosen = QColorDialog.getColor(current, self, "Select text color")
        if chosen.isValid():
            self._apply_button_color(button, chosen.name())

    def _update_log_controls_state(self):
        if hasattr(self, "spn_log_frame_percent") and hasattr(self, "chk_show_log"):
            self.spn_log_frame_percent.setEnabled(self.chk_show_log.isChecked())

    def _update_second_controls_state(self):
        enabled = bool(getattr(self, "chk_has_second_participant", None)) and (
            self.chk_has_second_participant.isEnabled()
            and self.chk_has_second_participant.isChecked()
        )
        if hasattr(self, "grp_second_subject"):
            self.grp_second_subject.setEnabled(enabled)
        if hasattr(self, "scr_second_subject"):
            self.scr_second_subject.setEnabled(enabled)
        self._refresh_second_participant_id_display()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _get_git_commit_version(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        menu_bar = QMenuBar(self)
        help_menu = menu_bar.addMenu("Help")
        help_action = QAction("Help Explorer", self)
        help_action.triggered.connect(self._open_help_explorer)
        help_menu.addAction(help_action)
        about_action = QAction("About Us", self)
        about_action.triggered.connect(self._open_about_us)
        help_menu.addAction(about_action)
        main_layout.setMenuBar(menu_bar)

        lock_row = QHBoxLayout()
        self.lock_status_label = QLabel(
            "Settings are locked. Enter the password to edit.", self
        )
        lock_row.addWidget(self.lock_status_label, stretch=1)
        self.btn_toggle_lock = QPushButton("Unlock Settings", self)
        self.btn_toggle_lock.clicked.connect(self._on_toggle_settings_lock)
        lock_row.addWidget(self.btn_toggle_lock)
        self.btn_help = QPushButton("Help", self)
        self.btn_help.clicked.connect(self._open_help_explorer)
        lock_row.addWidget(self.btn_help)
        main_layout.addLayout(lock_row)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(self._create_participant_tab(), "Participant")
        self.tabs.addTab(self._create_experiment_tab(), "Experiment")
        self.tabs.addTab(self._create_environment_tab(), "Environment")
        self.tabs.addTab(self._create_camera_tab(), "2D Camera")
        self.tabs.addTab(self._create_depth_camera_tab(), "Depth Camera")
        self.tabs.addTab(self._create_voice_tab(), "Voice")
        self.tabs.addTab(self._create_ui_tab(), "User Interface")
        self.tabs.addTab(self._create_time_tab(), "Time")
        self.tabs.addTab(self._create_demo_tab(), "Demo")
        self.tabs.addTab(self._create_wifi_tab(), "Wi-Fi")
        self.tabs.addTab(self._create_actions_tab(), "Actions")
        main_layout.addWidget(self.tabs, stretch=1)

        self.edt_name.textChanged.connect(self._refresh_participant_id_display)
        self.cmb_gender.currentTextChanged.connect(
            lambda *_: self._refresh_participant_id_display()
        )
        self.age_button_group.buttonClicked.connect(
            lambda *_: self._refresh_participant_id_display()
        )
        self.edt_second_name.textChanged.connect(self._refresh_second_participant_id_display)
        self.cmb_second_gender.currentTextChanged.connect(
            lambda *_: self._refresh_second_participant_id_display()
        )
        self.second_age_button_group.buttonClicked.connect(
            lambda *_: self._refresh_second_participant_id_display()
        )

        bottom_row = QHBoxLayout()
        git_version = self._get_git_commit_version()
        self.git_version_label = QLabel(f"Version: {git_version}", self)
        self.btn_test_connections = QPushButton("Test Connections", self)
        self.btn_test_connections.clicked.connect(self._on_test_connections_clicked)
        bottom_row.addWidget(self.btn_test_connections)

        self.btn_delete_pcaps = QPushButton("Delete PCAPs", self)
        self.btn_delete_pcaps.clicked.connect(self._on_delete_pcaps_clicked)
        bottom_row.addWidget(self.btn_delete_pcaps)

        self.btn_reboot_access_points = QPushButton("Reboot Access Points", self)
        self.btn_reboot_access_points.clicked.connect(
            self._on_reboot_access_points_clicked
        )
        bottom_row.addWidget(self.btn_reboot_access_points)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.git_version_label, alignment=Qt.AlignmentFlag.AlignCenter)
        bottom_row.addStretch(1)
        self.btn_start = QPushButton("Start Experiment", self)
        self.btn_start.clicked.connect(self._on_start_clicked)
        bottom_row.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.clicked.connect(self.reject)
        bottom_row.addWidget(self.btn_cancel)

        main_layout.addLayout(bottom_row)

    def _show_help_dialog(self, topic: str | None = None) -> None:
        if self._help_dialog is None or not self._help_dialog.isVisible():
            self._help_dialog = HelpExplorerDialog(self, initial_topic=topic)
            self._help_dialog.show()
        else:
            if topic:
                self._help_dialog.show_topic(topic)
            self._help_dialog.raise_()
            self._help_dialog.activateWindow()

    def _open_help_explorer(self) -> None:
        self._show_help_dialog()

    def _open_about_us(self) -> None:
        self._show_help_dialog("about_us.html")

    def _set_locked_state(self, locked: bool):
        locked = locked and is_password_required()
        self._settings_locked = locked
        if hasattr(self, "tabs"):
            self.tabs.setEnabled(not locked)
        if hasattr(self, "btn_test_connections"):
            self.btn_test_connections.setEnabled(not locked)
        if hasattr(self, "btn_delete_pcaps"):
            self.btn_delete_pcaps.setEnabled(not locked)
        if hasattr(self, "btn_reboot_access_points"):
            self.btn_reboot_access_points.setEnabled(not locked)
        if hasattr(self, "btn_start"):
            self.btn_start.setEnabled(not locked)

        status_text = (
            "Settings are locked. Enter the password to edit."
            if locked
            else "Settings unlocked. You can edit and start experiments."
        )
        if hasattr(self, "lock_status_label"):
            self.lock_status_label.setText(status_text)
        if hasattr(self, "btn_toggle_lock"):
            self.btn_toggle_lock.setText(
                "Unlock Settings" if locked else "Lock Settings"
            )

    def _on_toggle_settings_lock(self):
        if not self._settings_locked:
            self._set_locked_state(True)
            return

        if not is_password_required():
            self._set_locked_state(False)
            QMessageBox.information(
                self,
                "Settings unlocked",
                "Admin mode enabled. Password bypassed.",
            )
            return

        password, ok = QInputDialog.getText(
            self,
            "Unlock settings",
            "Enter password to edit settings:",
            QLineEdit.Password,
        )
        if not ok:
            return

        if verify_password(password):
            self._set_locked_state(False)
            QMessageBox.information(
                self,
                "Settings unlocked",
                "Password accepted. You can now modify configuration settings.",
            )
        else:
            QMessageBox.warning(
                self,
                "Incorrect password",
                "The provided password is incorrect. Settings remain locked.",
            )

    def _create_script_line_edit(self, text: str = ""):
        line_edit = QLineEdit(text)
        line_edit.setPlaceholderText("Double-click to choose a file…")
        line_edit.setProperty("script_line_edit", True)
        line_edit.installEventFilter(self)
        return line_edit

    def _list_script_profiles(self) -> list[str]:
        if not SCRIPTS_DIR.is_dir():
            return []
        return sorted([p.name for p in SCRIPTS_DIR.iterdir() if p.is_dir()])

    def _list_script_types(self, profile: str) -> list[str]:
        profile_dir = SCRIPTS_DIR / profile
        if not profile_dir.is_dir():
            return []
        return sorted([p.name for p in profile_dir.iterdir() if p.is_dir()])

    def _create_script_profile_combo(self, text: str = "") -> QComboBox:
        combo = QComboBox(self)
        combo.setEditable(True)
        profiles = self._list_script_profiles()
        combo.addItems(profiles)
        if text and combo.findText(text) == -1:
            combo.addItem(text)
        if text:
            combo.setCurrentText(text)
        elif profiles:
            combo.setCurrentIndex(0)
        return combo

    def _populate_script_type_combo(
        self, combo: QComboBox, profile: str, selected_type: str | None = None
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        types = self._list_script_types(profile)
        combo.addItems(types)

        if selected_type:
            if combo.findText(selected_type) == -1:
                combo.addItem(selected_type)
            combo.setCurrentText(selected_type)
        elif types:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _create_script_type_combo(
        self, profile_combo: QComboBox, text: str = ""
    ) -> QComboBox:
        combo = QComboBox(self)
        combo.setEditable(True)
        self._populate_script_type_combo(combo, profile_combo.currentText(), text)
        return combo

    def _create_participant_tab(self):
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        participant_row = QHBoxLayout()
        participant_row.addWidget(QLabel("Participant profile:", tab))

        self.cmb_participant_profile = QComboBox(tab)
        self.cmb_participant_profile.currentTextChanged.connect(
            self._on_participant_profile_changed
        )
        participant_row.addWidget(self.cmb_participant_profile, stretch=1)

        self.btn_participant_new = QPushButton("New", tab)
        self.btn_participant_new.clicked.connect(self._on_new_participant_profile)
        participant_row.addWidget(self.btn_participant_new)

        self.btn_participant_duplicate = QPushButton("Duplicate", tab)
        self.btn_participant_duplicate.clicked.connect(
            self._on_duplicate_participant_profile
        )
        participant_row.addWidget(self.btn_participant_duplicate)

        self.btn_participant_rename = QPushButton("Rename", tab)
        self.btn_participant_rename.clicked.connect(self._on_rename_participant_profile)
        participant_row.addWidget(self.btn_participant_rename)

        self.btn_participant_delete = QPushButton("Delete", tab)
        self.btn_participant_delete.clicked.connect(self._on_delete_participant_profile)
        participant_row.addWidget(self.btn_participant_delete)

        self.btn_participant_save = QPushButton("Save", tab)
        self.btn_participant_save.clicked.connect(self._on_save_participant_profiles)
        participant_row.addWidget(self.btn_participant_save)

        layout.addLayout(participant_row)

        self.grp_subject = QGroupBox("Participant Details", tab)
        subject_layout = QFormLayout(self.grp_subject)

        self.edt_name = QLineEdit(self.grp_subject)
        self.edt_exp_id = QLineEdit(self.grp_subject)
        self.edt_exp_id.setReadOnly(True)
        self.edt_participant_id = QLineEdit(self.grp_subject)
        self.edt_participant_id.setReadOnly(True)
        self.cmb_gender = QComboBox(self.grp_subject)
        self.cmb_gender.setEditable(True)
        self.cmb_gender.addItems(self.gender_options)
        self.cmb_dominant_hand = QComboBox(self.grp_subject)
        self.cmb_dominant_hand.addItems(["", "Right-handed", "Left-handed"])
        self.spn_height = QDoubleSpinBox(self.grp_subject)
        self.spn_height.setRange(0.0, 300.0)
        self.spn_height.setDecimals(1)
        self.spn_height.setSuffix(" cm")
        self.spn_weight = QDoubleSpinBox(self.grp_subject)
        self.spn_weight.setRange(0.0, 500.0)
        self.spn_weight.setDecimals(1)
        self.spn_weight.setSuffix(" kg")
        self.txt_participant_description = QTextEdit(self.grp_subject)
        self.txt_participant_description.setPlaceholderText(
            "Optional notes about the participant"
        )
        self.txt_participant_description.setMinimumHeight(60)

        self.age_button_group = QButtonGroup(self.grp_subject)
        self.age_group_buttons: dict[str, QRadioButton] = {}
        age_widget = QWidget(self.grp_subject)
        age_layout = QVBoxLayout(age_widget)
        age_layout.setContentsMargins(0, 0, 0, 0)
        for option in self.age_group_options:
            btn = QRadioButton(option, age_widget)
            self.age_button_group.addButton(btn)
            self.age_group_buttons[option] = btn
            age_layout.addWidget(btn)
        # Default selection
        if self.age_group_options:
            self.age_group_buttons[self.age_group_options[0]].setChecked(True)

        subject_layout.addRow("Name:", self.edt_name)
        subject_layout.addRow("Age:", age_widget)
        subject_layout.addRow("Experiment ID:", self.edt_exp_id)
        subject_layout.addRow("Participant ID:", self.edt_participant_id)
        subject_layout.addRow("Gender:", self.cmb_gender)
        subject_layout.addRow("Dominant hand:", self.cmb_dominant_hand)
        subject_layout.addRow("Height:", self.spn_height)
        subject_layout.addRow("Weight:", self.spn_weight)
        subject_layout.addRow("Description:", self.txt_participant_description)

        self.scr_subject = QScrollArea(tab)
        self.scr_subject.setWidgetResizable(True)
        self.scr_subject.setWidget(self.grp_subject)
        layout.addWidget(self.scr_subject)

        self.chk_has_second_participant = QCheckBox("Include a second participant", tab)
        self.chk_has_second_participant.stateChanged.connect(self._update_second_controls_state)
        layout.addWidget(self.chk_has_second_participant)

        self.grp_second_subject = QGroupBox("Second Participant Details", tab)
        second_layout = QFormLayout(self.grp_second_subject)

        self.edt_second_name = QLineEdit(self.grp_second_subject)
        self.edt_second_participant_id = QLineEdit(self.grp_second_subject)
        self.edt_second_participant_id.setReadOnly(True)
        self.cmb_second_gender = QComboBox(self.grp_second_subject)
        self.cmb_second_gender.setEditable(True)
        self.cmb_second_gender.addItems(self.gender_options)
        self.cmb_second_dominant_hand = QComboBox(self.grp_second_subject)
        self.cmb_second_dominant_hand.addItems(["", "Right-handed", "Left-handed"])
        self.spn_second_height = QDoubleSpinBox(self.grp_second_subject)
        self.spn_second_height.setRange(0.0, 300.0)
        self.spn_second_height.setDecimals(1)
        self.spn_second_height.setSuffix(" cm")
        self.spn_second_weight = QDoubleSpinBox(self.grp_second_subject)
        self.spn_second_weight.setRange(0.0, 500.0)
        self.spn_second_weight.setDecimals(1)
        self.spn_second_weight.setSuffix(" kg")
        self.txt_second_description = QTextEdit(self.grp_second_subject)
        self.txt_second_description.setPlaceholderText(
            "Optional notes about the second participant"
        )
        self.txt_second_description.setMinimumHeight(60)

        self.second_age_button_group = QButtonGroup(self.grp_second_subject)
        self.second_age_group_buttons: dict[str, QRadioButton] = {}
        second_age_widget = QWidget(self.grp_second_subject)
        second_age_layout = QVBoxLayout(second_age_widget)
        second_age_layout.setContentsMargins(0, 0, 0, 0)
        for option in self.age_group_options:
            btn = QRadioButton(option, second_age_widget)
            self.second_age_button_group.addButton(btn)
            self.second_age_group_buttons[option] = btn
            second_age_layout.addWidget(btn)
        if self.age_group_options:
            self.second_age_group_buttons[self.age_group_options[0]].setChecked(True)

        second_layout.addRow("Name:", self.edt_second_name)
        second_layout.addRow("Age:", second_age_widget)
        second_layout.addRow("Participant ID:", self.edt_second_participant_id)
        second_layout.addRow("Gender:", self.cmb_second_gender)
        second_layout.addRow("Dominant hand:", self.cmb_second_dominant_hand)
        second_layout.addRow("Height:", self.spn_second_height)
        second_layout.addRow("Weight:", self.spn_second_weight)
        second_layout.addRow("Description:", self.txt_second_description)

        self.grp_second_subject.setEnabled(False)
        self.scr_second_subject = QScrollArea(tab)
        self.scr_second_subject.setWidgetResizable(True)
        self.scr_second_subject.setWidget(self.grp_second_subject)
        self.scr_second_subject.setEnabled(False)
        layout.addWidget(self.scr_second_subject)
        layout.addStretch(1)
        return tab

    def _create_experiment_tab(self):
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        experiment_row = QHBoxLayout()
        experiment_row.addWidget(QLabel("Experiment timings profile:", tab))

        self.cmb_experiment_profile = QComboBox(tab)
        self.cmb_experiment_profile.currentTextChanged.connect(
            self._on_experiment_profile_changed
        )
        experiment_row.addWidget(self.cmb_experiment_profile, stretch=1)

        self.btn_experiment_new = QPushButton("New", tab)
        self.btn_experiment_new.clicked.connect(self._on_new_experiment_profile)
        experiment_row.addWidget(self.btn_experiment_new)

        self.btn_experiment_duplicate = QPushButton("Duplicate", tab)
        self.btn_experiment_duplicate.clicked.connect(
            self._on_duplicate_experiment_profile
        )
        experiment_row.addWidget(self.btn_experiment_duplicate)

        self.btn_experiment_rename = QPushButton("Rename", tab)
        self.btn_experiment_rename.clicked.connect(self._on_rename_experiment_profile)
        experiment_row.addWidget(self.btn_experiment_rename)

        self.btn_experiment_delete = QPushButton("Delete", tab)
        self.btn_experiment_delete.clicked.connect(self._on_delete_experiment_profile)
        experiment_row.addWidget(self.btn_experiment_delete)

        self.btn_experiment_save = QPushButton("Save", tab)
        self.btn_experiment_save.clicked.connect(self._on_save_experiment_profiles)
        experiment_row.addWidget(self.btn_experiment_save)

        layout.addLayout(experiment_row)

        self.grp_experiment = QGroupBox("Experiment Timings", tab)
        exp_layout = QFormLayout(self.grp_experiment)

        self.spn_begin_baseline = QDoubleSpinBox(self.grp_experiment)
        self.spn_begin_baseline.setRange(0.0, 10_000.0)
        self.spn_begin_baseline.setDecimals(1)
        self.spn_begin_baseline.valueChanged.connect(self._update_total_duration_label)

        self.spn_between_baseline = QDoubleSpinBox(self.grp_experiment)
        self.spn_between_baseline.setRange(0.0, 10_000.0)
        self.spn_between_baseline.setDecimals(1)
        self.spn_between_baseline.valueChanged.connect(self._update_total_duration_label)

        self.spn_end_baseline = QDoubleSpinBox(self.grp_experiment)
        self.spn_end_baseline.setRange(0.0, 10_000.0)
        self.spn_end_baseline.setDecimals(1)
        self.spn_end_baseline.valueChanged.connect(self._update_total_duration_label)

        self.spn_stop_time = QDoubleSpinBox(self.grp_experiment)
        self.spn_stop_time.setRange(0.0, 10_000.0)
        self.spn_stop_time.setDecimals(1)
        self.spn_stop_time.valueChanged.connect(self._update_total_duration_label)

        self.spn_preview_pause = QDoubleSpinBox(self.grp_experiment)
        self.spn_preview_pause.setRange(0.0, 10_000.0)
        self.spn_preview_pause.setDecimals(1)

        self.spn_action_time = QDoubleSpinBox(self.grp_experiment)
        self.spn_action_time.setRange(0.0, 10_000.0)
        self.spn_action_time.setDecimals(1)
        self.spn_action_time.valueChanged.connect(self._update_total_duration_label)

        self.spn_reps = QSpinBox(self.grp_experiment)
        self.spn_reps.setRange(1, 10_000)
        self.spn_reps.valueChanged.connect(self._update_total_duration_label)

        self.edt_save_location = QLineEdit(self.grp_experiment)
        exp_layout.addRow("Beginning baseline (s):", self.spn_begin_baseline)
        exp_layout.addRow("Between-actions baseline (s):", self.spn_between_baseline)
        exp_layout.addRow("Ending baseline (s):", self.spn_end_baseline)
        exp_layout.addRow("Stop time (s):", self.spn_stop_time)
        exp_layout.addRow("Preview stop time (s):", self.spn_preview_pause)
        exp_layout.addRow("Action time (s):", self.spn_action_time)
        exp_layout.addRow("Repetitions per action:", self.spn_reps)
        exp_layout.addRow("Save location:", self.edt_save_location)
        self.lbl_total_duration = QLabel("--", self.grp_experiment)
        exp_layout.addRow("Total duration:", self.lbl_total_duration)

        layout.addWidget(self.grp_experiment)
        layout.addStretch(1)
        return tab

    def _create_environment_tab(self):
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        env_row = QHBoxLayout()
        env_row.addWidget(QLabel("Environment profile:", tab))

        self.cmb_environment_profile = QComboBox(tab)
        self.cmb_environment_profile.currentTextChanged.connect(
            self._on_environment_profile_changed
        )
        env_row.addWidget(self.cmb_environment_profile, stretch=1)

        self.btn_environment_new = QPushButton("New", tab)
        self.btn_environment_new.clicked.connect(self._on_new_environment_profile)
        env_row.addWidget(self.btn_environment_new)

        self.btn_environment_duplicate = QPushButton("Duplicate", tab)
        self.btn_environment_duplicate.clicked.connect(
            self._on_duplicate_environment_profile
        )
        env_row.addWidget(self.btn_environment_duplicate)

        self.btn_environment_rename = QPushButton("Rename", tab)
        self.btn_environment_rename.clicked.connect(self._on_rename_environment_profile)
        env_row.addWidget(self.btn_environment_rename)

        self.btn_environment_delete = QPushButton("Delete", tab)
        self.btn_environment_delete.clicked.connect(self._on_delete_environment_profile)
        env_row.addWidget(self.btn_environment_delete)

        self.btn_environment_save = QPushButton("Save", tab)
        self.btn_environment_save.clicked.connect(self._on_save_environment_profiles)
        env_row.addWidget(self.btn_environment_save)

        layout.addLayout(env_row)

        self.grp_environment = QGroupBox("Environment Details", tab)
        env_layout = QFormLayout(self.grp_environment)

        self.spn_room_length = QDoubleSpinBox(self.grp_environment)
        self.spn_room_length.setRange(0.0, 1000.0)
        self.spn_room_length.setDecimals(2)
        self.spn_room_length.setSuffix(" m")

        self.spn_room_width = QDoubleSpinBox(self.grp_environment)
        self.spn_room_width.setRange(0.0, 1000.0)
        self.spn_room_width.setDecimals(2)
        self.spn_room_width.setSuffix(" m")

        self.spn_room_height = QDoubleSpinBox(self.grp_environment)
        self.spn_room_height.setRange(0.0, 1000.0)
        self.spn_room_height.setDecimals(2)
        self.spn_room_height.setSuffix(" m")

        self.txt_environment_description = QTextEdit(self.grp_environment)
        self.txt_environment_description.setPlaceholderText(
            "Describe the room layout, furniture, or other notes"
        )
        self.txt_environment_description.setMinimumHeight(80)

        env_layout.addRow("Room length:", self.spn_room_length)
        env_layout.addRow("Room width:", self.spn_room_width)
        env_layout.addRow("Room height:", self.spn_room_height)
        env_layout.addRow("Description:", self.txt_environment_description)

        layout.addWidget(self.grp_environment)
        layout.addStretch(1)
        return tab

    def _create_camera_tab(self):
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("2D Camera profile:", tab))

        self.cmb_camera_profile = QComboBox(tab)
        self.cmb_camera_profile.currentTextChanged.connect(
            self._on_camera_profile_changed
        )
        camera_row.addWidget(self.cmb_camera_profile, stretch=1)

        self.btn_camera_new = QPushButton("New", tab)
        self.btn_camera_new.clicked.connect(self._on_new_camera_profile)
        camera_row.addWidget(self.btn_camera_new)

        self.btn_camera_duplicate = QPushButton("Duplicate", tab)
        self.btn_camera_duplicate.clicked.connect(self._on_duplicate_camera_profile)
        camera_row.addWidget(self.btn_camera_duplicate)

        self.btn_camera_rename = QPushButton("Rename", tab)
        self.btn_camera_rename.clicked.connect(self._on_rename_camera_profile)
        camera_row.addWidget(self.btn_camera_rename)

        self.btn_camera_delete = QPushButton("Delete", tab)
        self.btn_camera_delete.clicked.connect(self._on_delete_camera_profile)
        camera_row.addWidget(self.btn_camera_delete)

        self.btn_camera_save = QPushButton("Save", tab)
        self.btn_camera_save.clicked.connect(self._on_save_camera_profiles)
        camera_row.addWidget(self.btn_camera_save)

        layout.addLayout(camera_row)

        info = QLabel(
            "Camera settings control webcam recording and optional hand recognition "
            "overlay. These settings are stored in a separate profile so they can "
            "be reused across experiments.",
            tab,
        )
        info.setWordWrap(True)
        self.lbl_hand_context = QLabel("", tab)
        self.lbl_hand_context.setWordWrap(True)

        self.chk_use_webcam = QCheckBox("Camera", tab)
        self.chk_use_webcam.toggled.connect(self._on_camera_checkbox_toggled)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Camera device:", tab))
        self.cmb_camera_device = QComboBox(tab)
        self.cmb_camera_device.setEditable(True)
        self.cmb_camera_device.setInsertPolicy(QComboBox.NoInsert)
        device_row.addWidget(self.cmb_camera_device, stretch=1)
        self.btn_refresh_cameras = QPushButton("Refresh", tab)
        self.btn_refresh_cameras.clicked.connect(self._refresh_camera_device_list)
        device_row.addWidget(self.btn_refresh_cameras)
        self.btn_preview_camera = QPushButton("Preview", tab)
        self.btn_preview_camera.clicked.connect(self._on_toggle_camera_preview)
        device_row.addWidget(self.btn_preview_camera)

        self.lbl_camera_preview = QLabel("Camera preview disabled", tab)
        self.lbl_camera_preview.setAlignment(Qt.AlignCenter)
        self.lbl_camera_preview.setMinimumSize(320, 240)
        self.lbl_camera_preview.setStyleSheet(
            "background-color: black; color: white; padding: 8px;"
        )

        self.chk_hand_recognition = QCheckBox(
            "Enable hand recognition overlay (requires webcam and MediaPipe)", tab
        )
        self.chk_hand_recognition.toggled.connect(
            lambda _checked: self._update_hand_controls_state()
        )

        self.cmb_hand_mode = QComboBox(tab)
        self.cmb_hand_mode.addItem(
            "Process hand recognition during the experiment (live)", userData="live"
        )
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Hand recognition timing:", tab))
        mode_row.addWidget(self.cmb_hand_mode, stretch=1)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model complexity:", tab))
        self.cmb_hand_model_complexity = QComboBox(tab)
        self.cmb_hand_model_complexity.addItem("Light (fast)", userData="light")
        self.cmb_hand_model_complexity.addItem("Full (high accuracy)", userData="full")
        model_row.addWidget(self.cmb_hand_model_complexity, stretch=1)

        appearance_group = QGroupBox("Hand overlay appearance", tab)
        appearance_layout = QFormLayout(appearance_group)
        self.btn_left_wrist_color = QPushButton(tab)
        self.btn_left_wrist_color.clicked.connect(
            lambda: self._pick_button_color(self.btn_left_wrist_color)
        )
        self.btn_right_wrist_color = QPushButton(tab)
        self.btn_right_wrist_color.clicked.connect(
            lambda: self._pick_button_color(self.btn_right_wrist_color)
        )
        self.spn_wrist_radius = QSpinBox(tab)
        self.spn_wrist_radius.setRange(1, 200)
        self.spn_wrist_radius.setValue(
            int(DEFAULT_CAMERA_PROFILE.get("hand_wrist_circle_radius", 18))
        )
        self._apply_button_color(
            self.btn_left_wrist_color, DEFAULT_CAMERA_PROFILE["hand_left_wrist_color"]
        )
        self._apply_button_color(
            self.btn_right_wrist_color, DEFAULT_CAMERA_PROFILE["hand_right_wrist_color"]
        )
        appearance_layout.addRow("Left wrist color:", self.btn_left_wrist_color)
        appearance_layout.addRow("Right wrist color:", self.btn_right_wrist_color)
        appearance_layout.addRow("Wrist circle radius:", self.spn_wrist_radius)

        layout.addWidget(self.lbl_hand_context)
        layout.addWidget(info)
        layout.addWidget(self.chk_use_webcam)
        layout.addLayout(device_row)
        layout.addWidget(self.lbl_camera_preview)
        layout.addWidget(self.chk_hand_recognition)
        layout.addLayout(mode_row)
        layout.addLayout(model_row)
        layout.addWidget(appearance_group)
        layout.addStretch(1)
        return tab

    def _create_depth_camera_tab(self):
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Depth camera profile:", tab))

        self.cmb_depth_camera_profile = QComboBox(tab)
        self.cmb_depth_camera_profile.currentTextChanged.connect(
            self._on_depth_camera_profile_changed
        )
        depth_row.addWidget(self.cmb_depth_camera_profile, stretch=1)

        self.btn_depth_camera_new = QPushButton("New", tab)
        self.btn_depth_camera_new.clicked.connect(self._on_new_depth_camera_profile)
        depth_row.addWidget(self.btn_depth_camera_new)

        self.btn_depth_camera_duplicate = QPushButton("Duplicate", tab)
        self.btn_depth_camera_duplicate.clicked.connect(
            self._on_duplicate_depth_camera_profile
        )
        depth_row.addWidget(self.btn_depth_camera_duplicate)

        self.btn_depth_camera_rename = QPushButton("Rename", tab)
        self.btn_depth_camera_rename.clicked.connect(self._on_rename_depth_camera_profile)
        depth_row.addWidget(self.btn_depth_camera_rename)

        self.btn_depth_camera_delete = QPushButton("Delete", tab)
        self.btn_depth_camera_delete.clicked.connect(self._on_delete_depth_camera_profile)
        depth_row.addWidget(self.btn_depth_camera_delete)

        self.btn_depth_camera_save = QPushButton("Save", tab)
        self.btn_depth_camera_save.clicked.connect(self._on_save_depth_camera_profiles)
        depth_row.addWidget(self.btn_depth_camera_save)

        layout.addLayout(depth_row)

        info = QLabel(
            "Configure the RealSense D455 depth camera via its REST API. When enabled, "
            "the app will poll the /api/status endpoint every second to show live status.",
            tab,
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.chk_depth_enabled = QCheckBox("Enable depth camera", tab)
        self.chk_depth_enabled.toggled.connect(self._on_depth_checkbox_toggled)

        form_group = QGroupBox("Connection and recording options", tab)
        form_layout = QFormLayout(form_group)

        self.edt_depth_api_ip = QLineEdit(tab)
        self.edt_depth_api_ip.setPlaceholderText("e.g., 10.211.55.8 or 10.211.55.8:5000")

        self.spn_depth_fps = QSpinBox(tab)
        self.spn_depth_fps.setRange(1, 120)
        self.spn_depth_fps.setValue(int(DEFAULT_DEPTH_CAMERA_PROFILE["fps"]))

        self.cmb_depth_rgb_res = QComboBox(tab)
        self.cmb_depth_rgb_res.setEditable(True)
        for label in ["640x480", "848x480", "1280x720"]:
            self.cmb_depth_rgb_res.addItem(label)

        self.cmb_depth_depth_res = QComboBox(tab)
        self.cmb_depth_depth_res.setEditable(True)
        for label in ["640x480", "848x480", "1280x720"]:
            self.cmb_depth_depth_res.addItem(label)

        self.chk_depth_save_raw = QCheckBox("Save raw NPZ files", tab)

        save_row = QHBoxLayout()
        self.edt_depth_save_location = QLineEdit(tab)
        self.edt_depth_save_location.setPlaceholderText("recordings/...")
        self.btn_depth_browse_location = QPushButton("Browse…", tab)
        self.btn_depth_browse_location.clicked.connect(self._on_browse_depth_save_location)
        save_row.addWidget(self.edt_depth_save_location, stretch=1)
        save_row.addWidget(self.btn_depth_browse_location)

        form_layout.addRow("API IP / host:", self.edt_depth_api_ip)
        form_layout.addRow("FPS:", self.spn_depth_fps)
        form_layout.addRow("RGB resolution:", self.cmb_depth_rgb_res)
        form_layout.addRow("Depth resolution:", self.cmb_depth_depth_res)
        form_layout.addRow(self.chk_depth_save_raw)
        form_layout.addRow("Save location:", save_row)

        layout.addWidget(self.chk_depth_enabled)
        layout.addWidget(form_group)

        status_group = QGroupBox("Camera status", tab)
        status_layout = QVBoxLayout(status_group)
        status_header = QHBoxLayout()
        self.lbl_depth_status = QLabel("Status updates appear here.", tab)
        status_header.addWidget(self.lbl_depth_status)
        status_header.addStretch(1)
        self.btn_depth_refresh_status = QPushButton("Refresh now", tab)
        self.btn_depth_refresh_status.clicked.connect(
            lambda *_: self._refresh_depth_status(force=True)
        )
        status_header.addWidget(self.btn_depth_refresh_status)
        status_layout.addLayout(status_header)

        self.txt_depth_status = QTextEdit(tab)
        self.txt_depth_status.setReadOnly(True)
        self.txt_depth_status.setMinimumHeight(140)
        self.txt_depth_status.setPlaceholderText("Waiting for depth camera status…")
        status_layout.addWidget(self.txt_depth_status)

        layout.addWidget(status_group)
        layout.addStretch(1)
        return tab

    def _create_voice_tab(self):
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel("Voice profile:", tab))

        self.cmb_voice_profile = QComboBox(tab)
        self.cmb_voice_profile.currentTextChanged.connect(
            self._on_voice_profile_changed
        )
        voice_row.addWidget(self.cmb_voice_profile, stretch=1)

        self.btn_voice_new = QPushButton("New", tab)
        self.btn_voice_new.clicked.connect(self._on_new_voice_profile)
        voice_row.addWidget(self.btn_voice_new)

        self.btn_voice_duplicate = QPushButton("Duplicate", tab)
        self.btn_voice_duplicate.clicked.connect(self._on_duplicate_voice_profile)
        voice_row.addWidget(self.btn_voice_duplicate)

        self.btn_voice_rename = QPushButton("Rename", tab)
        self.btn_voice_rename.clicked.connect(self._on_rename_voice_profile)
        voice_row.addWidget(self.btn_voice_rename)

        self.btn_voice_delete = QPushButton("Delete", tab)
        self.btn_voice_delete.clicked.connect(self._on_delete_voice_profile)
        voice_row.addWidget(self.btn_voice_delete)

        self.btn_voice_save = QPushButton("Save", tab)
        self.btn_voice_save.clicked.connect(self._on_save_voice_profiles)
        voice_row.addWidget(self.btn_voice_save)

        layout.addLayout(voice_row)

        self.grp_voice = QGroupBox("Voice Assistant", tab)
        voice_layout = QFormLayout(self.grp_voice)

        self.chk_voice_enabled = QCheckBox(
            "Enable spoken instructions", self.grp_voice
        )
        voice_layout.addRow("Voice assistant:", self.chk_voice_enabled)

        self.chk_preview_voice_enabled = QCheckBox(
            "Play preview instructions", self.grp_voice
        )
        voice_layout.addRow("Preview actions:", self.chk_preview_voice_enabled)

        self.cmb_voice_language = QComboBox(self.grp_voice)
        self._populate_language_options()
        voice_layout.addRow("Language:", self.cmb_voice_language)

        layout.addWidget(self.grp_voice)
        layout.addStretch(1)
        return tab

    def _create_ui_tab(self):
        tab = QWidget(self)
        outer_layout = QVBoxLayout(tab)
        scroll_area = QScrollArea(tab)
        scroll_area.setWidgetResizable(True)

        content = QWidget(tab)
        layout = QVBoxLayout(content)

        ui_row = QHBoxLayout()
        ui_row.addWidget(QLabel("User interface profile:", tab))

        self.cmb_ui_profile = QComboBox(tab)
        self.cmb_ui_profile.currentTextChanged.connect(self._on_ui_profile_changed)
        ui_row.addWidget(self.cmb_ui_profile, stretch=1)

        self.btn_ui_new = QPushButton("New", tab)
        self.btn_ui_new.clicked.connect(self._on_new_ui_profile)
        ui_row.addWidget(self.btn_ui_new)

        self.btn_ui_duplicate = QPushButton("Duplicate", tab)
        self.btn_ui_duplicate.clicked.connect(self._on_duplicate_ui_profile)
        ui_row.addWidget(self.btn_ui_duplicate)

        self.btn_ui_rename = QPushButton("Rename", tab)
        self.btn_ui_rename.clicked.connect(self._on_rename_ui_profile)
        ui_row.addWidget(self.btn_ui_rename)

        self.btn_ui_delete = QPushButton("Delete", tab)
        self.btn_ui_delete.clicked.connect(self._on_delete_ui_profile)
        ui_row.addWidget(self.btn_ui_delete)

        self.btn_ui_save = QPushButton("Save", tab)
        self.btn_ui_save.clicked.connect(self._on_save_ui_profiles)
        ui_row.addWidget(self.btn_ui_save)

        layout.addLayout(ui_row)

        self.grp_ui_frames = QGroupBox("Frames", tab)
        frame_layout = QFormLayout(self.grp_ui_frames)

        self.spn_camera_frame_percent = QDoubleSpinBox(self.grp_ui_frames)
        self.spn_camera_frame_percent.setRange(10.0, 400.0)
        self.spn_camera_frame_percent.setSingleStep(5.0)
        self.spn_camera_frame_percent.setSuffix(" %")
        frame_layout.addRow("Camera preview size:", self.spn_camera_frame_percent)

        self.spn_preview_frame_percent = QDoubleSpinBox(self.grp_ui_frames)
        self.spn_preview_frame_percent.setRange(10.0, 400.0)
        self.spn_preview_frame_percent.setSingleStep(5.0)
        self.spn_preview_frame_percent.setSuffix(" %")
        frame_layout.addRow("Next action preview size:", self.spn_preview_frame_percent)

        self.spn_action_frame_percent = QDoubleSpinBox(self.grp_ui_frames)
        self.spn_action_frame_percent.setRange(50.0, 400.0)
        self.spn_action_frame_percent.setSingleStep(10.0)
        self.spn_action_frame_percent.setSuffix(" %")
        frame_layout.addRow("Action video size:", self.spn_action_frame_percent)

        self.spn_details_pane_percent = QDoubleSpinBox(self.grp_ui_frames)
        self.spn_details_pane_percent.setRange(10.0, 90.0)
        self.spn_details_pane_percent.setSingleStep(5.0)
        self.spn_details_pane_percent.setSuffix(" %")
        frame_layout.addRow("Details pane width:", self.spn_details_pane_percent)

        self.chk_show_log = QCheckBox("Show log", self.grp_ui_frames)
        self.chk_show_log.toggled.connect(lambda _checked: self._update_log_controls_state())
        frame_layout.addRow("Log visibility:", self.chk_show_log)

        self.spn_log_frame_percent = QDoubleSpinBox(self.grp_ui_frames)
        self.spn_log_frame_percent.setRange(10.0, 400.0)
        self.spn_log_frame_percent.setSingleStep(5.0)
        self.spn_log_frame_percent.setSuffix(" %")
        frame_layout.addRow("Log frame size:", self.spn_log_frame_percent)

        layout.addWidget(self.grp_ui_frames)

        self.grp_window_options = QGroupBox("Window", tab)
        window_layout = QFormLayout(self.grp_window_options)
        self.chk_start_fullscreen = QCheckBox(
            "Open the main window in full screen", self.grp_window_options
        )
        window_layout.addRow(self.chk_start_fullscreen)
        layout.addWidget(self.grp_window_options)

        self.grp_transcript = QGroupBox("Instruction overlay", tab)
        transcript_layout = QFormLayout(self.grp_transcript)
        self.cmb_transcript_position = QComboBox(self.grp_transcript)
        self.cmb_transcript_position.addItem("Center", "middle")
        self.cmb_transcript_position.addItem("Top", "top")
        transcript_layout.addRow("Position:", self.cmb_transcript_position)

        self.spn_transcript_font_size = QSpinBox(self.grp_transcript)
        self.spn_transcript_font_size.setRange(10, 96)
        transcript_layout.addRow("Font size:", self.spn_transcript_font_size)
        self.btn_transcript_color = QPushButton("#ffffff", self.grp_transcript)
        self.btn_transcript_color.clicked.connect(
            lambda _checked=False, btn=self.btn_transcript_color: self._pick_button_color(btn)
        )
        transcript_layout.addRow("Text color:", self.btn_transcript_color)
        self.spn_transcript_offset_x = QSpinBox(self.grp_transcript)
        self.spn_transcript_offset_x.setRange(-800, 800)
        transcript_layout.addRow("Offset X:", self.spn_transcript_offset_x)
        self.spn_transcript_offset_y = QSpinBox(self.grp_transcript)
        self.spn_transcript_offset_y.setRange(-800, 800)
        transcript_layout.addRow("Offset Y:", self.spn_transcript_offset_y)
        layout.addWidget(self.grp_transcript)

        font_groups = QVBoxLayout()

        def expand_line_edit(line: QLineEdit):
            line.setMinimumHeight(28)
            line.setMinimumWidth(360)
            line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        def build_font_group(title: str):
            group = QGroupBox(title, tab)
            form = QFormLayout(group)
            font_combo = QFontComboBox(group)
            font_combo.setEditable(True)
            size_spin = QSpinBox(group)
            size_spin.setRange(6, 96)
            color_button = QPushButton("#000000", group)
            color_button.clicked.connect(lambda _checked=False, btn=color_button: self._pick_button_color(btn))
            form.addRow("Font:", font_combo)
            form.addRow("Size:", size_spin)
            form.addRow("Color:", color_button)
            return group, font_combo, size_spin, color_button

        (
            self.grp_count_font,
            self.cmb_count_font,
            self.spn_count_font_size,
            self.btn_count_color,
        ) = build_font_group("Count number text")
        (
            self.grp_action_font,
            self.cmb_action_font,
            self.spn_action_font_size,
            self.btn_action_color,
        ) = build_font_group("Current action text")
        (
            self.grp_time_font,
            self.cmb_time_font,
            self.spn_time_font_size,
            self.btn_time_color,
        ) = build_font_group("Elapsed time text")
        (
            self.grp_remaining_time_value_font,
            self.cmb_remaining_time_value_font,
            self.spn_remaining_time_value_font_size,
            self.btn_remaining_time_value_color,
        ) = build_font_group("Remaining time value")
        (
            self.grp_elapsed_time_value_font,
            self.cmb_elapsed_time_value_font,
            self.spn_elapsed_time_value_font_size,
            self.btn_elapsed_time_value_color,
        ) = build_font_group("Elapsed time value")

        font_groups.addWidget(self.grp_count_font)
        font_groups.addWidget(self.grp_action_font)
        font_groups.addWidget(self.grp_time_font)
        font_groups.addWidget(self.grp_remaining_time_value_font)
        font_groups.addWidget(self.grp_elapsed_time_value_font)

        fonts_container = QGroupBox("Text appearance", tab)
        fonts_container.setLayout(font_groups)
        layout.addWidget(fonts_container)

        self.grp_status_messages = QGroupBox("Status messages", tab)
        status_layout = QFormLayout(self.grp_status_messages)
        self.ui_status_edits = {}
        for key, message in UI_STATUS_MESSAGE_FIELDS:
            line = QLineEdit(self.grp_status_messages)
            expand_line_edit(line)
            line.setText(message)
            self.ui_status_edits[key] = line
            status_layout.addRow(key.replace("_", " ").title() + ":", line)
        self.spn_status_font_size = QSpinBox(self.grp_status_messages)
        self.spn_status_font_size.setRange(10, 96)
        status_layout.addRow("Font size:", self.spn_status_font_size)
        self.btn_status_color = QPushButton("#000000", self.grp_status_messages)
        self.btn_status_color.clicked.connect(
            lambda _checked=False, btn=self.btn_status_color: self._pick_button_color(btn)
        )
        status_layout.addRow("Text color:", self.btn_status_color)
        self.spn_status_offset_x = QSpinBox(self.grp_status_messages)
        self.spn_status_offset_x.setRange(-800, 800)
        status_layout.addRow("Offset X:", self.spn_status_offset_x)
        self.spn_status_offset_y = QSpinBox(self.grp_status_messages)
        self.spn_status_offset_y.setRange(-800, 800)
        status_layout.addRow("Offset Y:", self.spn_status_offset_y)
        layout.addWidget(self.grp_status_messages)

        self.grp_instruction_messages = QGroupBox("Guiding messages", tab)
        guide_layout = QFormLayout(self.grp_instruction_messages)
        self.ui_guide_edits = {}
        for key, message in UI_GUIDE_MESSAGE_FIELDS:
            line = QLineEdit(self.grp_instruction_messages)
            expand_line_edit(line)
            line.setText(message)
            self.ui_guide_edits[key] = line
            label = key.replace("_", " ").replace("guide", "Guide").strip().title()
            guide_layout.addRow(f"{label}:", line)
        guide_hint = QLabel(
            "Use {action} to insert the next action name.", self.grp_instruction_messages
        )
        guide_hint.setWordWrap(True)
        guide_layout.addRow("", guide_hint)
        layout.addWidget(self.grp_instruction_messages)

        self.grp_main_text = QGroupBox("Main window labels", tab)
        main_text_layout = QVBoxLayout(self.grp_main_text)
        self.tbl_ui_text = QTableWidget(self.grp_main_text)
        self.tbl_ui_text.setColumnCount(6)
        self.tbl_ui_text.setHorizontalHeaderLabels(
            ["Label", "Text", "Font size", "Text color", "Offset X", "Offset Y"]
        )
        self.tbl_ui_text.verticalHeader().setVisible(False)
        self.tbl_ui_text.setRowCount(len(UI_LABEL_TEXT_FIELDS))
        self.tbl_ui_text.setSelectionMode(QAbstractItemView.NoSelection)
        self.ui_text_controls = []
        for row_index, item in enumerate(UI_LABEL_TEXT_FIELDS):
            label_item = QTableWidgetItem(item["label"])
            label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
            self.tbl_ui_text.setItem(row_index, 0, label_item)
            text_edit = QLineEdit(self.tbl_ui_text)
            text_edit.setMinimumHeight(28)
            text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.tbl_ui_text.setCellWidget(row_index, 1, text_edit)
            self.tbl_ui_text.setRowHeight(row_index, 32)
            font_spin = QSpinBox(self.tbl_ui_text)
            font_spin.setRange(6, 96)
            self.tbl_ui_text.setCellWidget(row_index, 2, font_spin)
            color_button = QPushButton("#000000", self.tbl_ui_text)
            color_button.clicked.connect(
                lambda _checked=False, btn=color_button: self._pick_button_color(btn)
            )
            self.tbl_ui_text.setCellWidget(row_index, 3, color_button)
            offset_x = QSpinBox(self.tbl_ui_text)
            offset_x.setRange(-800, 800)
            self.tbl_ui_text.setCellWidget(row_index, 4, offset_x)
            offset_y = QSpinBox(self.tbl_ui_text)
            offset_y.setRange(-800, 800)
            self.tbl_ui_text.setCellWidget(row_index, 5, offset_y)
            self.ui_text_controls.append(
                {
                    "prefix": item["prefix"],
                    "text": text_edit,
                    "font_size": font_spin,
                    "color": color_button,
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                }
            )
        self.tbl_ui_text.horizontalHeader().setStretchLastSection(True)
        self.tbl_ui_text.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        main_text_layout.addWidget(self.tbl_ui_text)
        layout.addWidget(self.grp_main_text)

        self.grp_main_values = QGroupBox("Main window values", tab)
        main_value_layout = QVBoxLayout(self.grp_main_values)
        self.tbl_ui_values = QTableWidget(self.grp_main_values)
        self.tbl_ui_values.setColumnCount(5)
        self.tbl_ui_values.setHorizontalHeaderLabels(
            ["Value", "Font size", "Text color", "Offset X", "Offset Y"]
        )
        self.tbl_ui_values.verticalHeader().setVisible(False)
        self.tbl_ui_values.setRowCount(len(UI_VALUE_LABEL_FIELDS))
        self.tbl_ui_values.setSelectionMode(QAbstractItemView.NoSelection)
        self.ui_value_controls = []
        for row_index, item in enumerate(UI_VALUE_LABEL_FIELDS):
            label_item = QTableWidgetItem(item["label"])
            label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
            self.tbl_ui_values.setItem(row_index, 0, label_item)
            self.tbl_ui_values.setRowHeight(row_index, 32)
            font_spin = QSpinBox(self.tbl_ui_values)
            font_spin.setRange(6, 96)
            self.tbl_ui_values.setCellWidget(row_index, 1, font_spin)
            color_button = QPushButton("#000000", self.tbl_ui_values)
            color_button.clicked.connect(
                lambda _checked=False, btn=color_button: self._pick_button_color(btn)
            )
            self.tbl_ui_values.setCellWidget(row_index, 2, color_button)
            offset_x = QSpinBox(self.tbl_ui_values)
            offset_x.setRange(-800, 800)
            self.tbl_ui_values.setCellWidget(row_index, 3, offset_x)
            offset_y = QSpinBox(self.tbl_ui_values)
            offset_y.setRange(-800, 800)
            self.tbl_ui_values.setCellWidget(row_index, 4, offset_y)
            self.ui_value_controls.append(
                {
                    "prefix": item["prefix"],
                    "font_size": font_spin,
                    "color": color_button,
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                }
            )
        self.tbl_ui_values.horizontalHeader().setStretchLastSection(True)
        main_value_layout.addWidget(self.tbl_ui_values)
        layout.addWidget(self.grp_main_values)

        self.grp_log_text = QGroupBox("Log text", tab)
        log_layout = QFormLayout(self.grp_log_text)
        self.edt_log_placeholder_text = QLineEdit(self.grp_log_text)
        expand_line_edit(self.edt_log_placeholder_text)
        log_layout.addRow("Placeholder text:", self.edt_log_placeholder_text)
        self.spn_log_font_size = QSpinBox(self.grp_log_text)
        self.spn_log_font_size.setRange(6, 48)
        log_layout.addRow("Font size:", self.spn_log_font_size)
        self.btn_log_color = QPushButton("#000000", self.grp_log_text)
        self.btn_log_color.clicked.connect(
            lambda _checked=False, btn=self.btn_log_color: self._pick_button_color(btn)
        )
        log_layout.addRow("Text color:", self.btn_log_color)
        self.spn_log_offset_x = QSpinBox(self.grp_log_text)
        self.spn_log_offset_x.setRange(-800, 800)
        log_layout.addRow("Offset X:", self.spn_log_offset_x)
        self.spn_log_offset_y = QSpinBox(self.grp_log_text)
        self.spn_log_offset_y.setRange(-800, 800)
        log_layout.addRow("Offset Y:", self.spn_log_offset_y)
        layout.addWidget(self.grp_log_text)

        self.grp_webcam_text = QGroupBox("Webcam preview text", tab)
        webcam_layout = QFormLayout(self.grp_webcam_text)
        self.ui_webcam_edits = {}
        for key, message in UI_WEBCAM_MESSAGE_FIELDS:
            line = QLineEdit(self.grp_webcam_text)
            expand_line_edit(line)
            line.setText(message)
            self.ui_webcam_edits[key] = line
            webcam_layout.addRow(key.replace("_", " ").title() + ":", line)
        self.spn_webcam_font_size = QSpinBox(self.grp_webcam_text)
        self.spn_webcam_font_size.setRange(6, 48)
        webcam_layout.addRow("Font size:", self.spn_webcam_font_size)
        self.btn_webcam_color = QPushButton("#ffffff", self.grp_webcam_text)
        self.btn_webcam_color.clicked.connect(
            lambda _checked=False, btn=self.btn_webcam_color: self._pick_button_color(btn)
        )
        webcam_layout.addRow("Text color:", self.btn_webcam_color)
        self.spn_webcam_offset_x = QSpinBox(self.grp_webcam_text)
        self.spn_webcam_offset_x.setRange(-800, 800)
        webcam_layout.addRow("Offset X:", self.spn_webcam_offset_x)
        self.spn_webcam_offset_y = QSpinBox(self.grp_webcam_text)
        self.spn_webcam_offset_y.setRange(-800, 800)
        webcam_layout.addRow("Offset Y:", self.spn_webcam_offset_y)
        layout.addWidget(self.grp_webcam_text)

        layout.addStretch(1)
        self._update_log_controls_state()

        scroll_area.setWidget(content)
        outer_layout.addWidget(scroll_area)
        return tab

    def _create_time_tab(self):
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Time profile:", tab))

        self.cmb_time_profile = QComboBox(tab)
        self.cmb_time_profile.currentTextChanged.connect(self._on_time_profile_changed)
        time_row.addWidget(self.cmb_time_profile, stretch=1)

        self.btn_time_new = QPushButton("New", tab)
        self.btn_time_new.clicked.connect(self._on_new_time_profile)
        time_row.addWidget(self.btn_time_new)

        self.btn_time_duplicate = QPushButton("Duplicate", tab)
        self.btn_time_duplicate.clicked.connect(self._on_duplicate_time_profile)
        time_row.addWidget(self.btn_time_duplicate)

        self.btn_time_rename = QPushButton("Rename", tab)
        self.btn_time_rename.clicked.connect(self._on_rename_time_profile)
        time_row.addWidget(self.btn_time_rename)

        self.btn_time_delete = QPushButton("Delete", tab)
        self.btn_time_delete.clicked.connect(self._on_delete_time_profile)
        time_row.addWidget(self.btn_time_delete)

        self.btn_time_save = QPushButton("Save", tab)
        self.btn_time_save.clicked.connect(self._on_save_time_profiles)
        time_row.addWidget(self.btn_time_save)

        layout.addLayout(time_row)

        self.grp_time_server = QGroupBox("Time Server", tab)
        time_layout = QFormLayout(self.grp_time_server)

        self.chk_use_time_server = QCheckBox(
            "Use time server for timestamps and file names", self.grp_time_server
        )
        self.chk_use_time_server.toggled.connect(self._on_time_server_toggled)
        time_layout.addRow("Enable:", self.chk_use_time_server)

        server_row = QHBoxLayout()
        self.cmb_time_server = QComboBox(self.grp_time_server)
        self.cmb_time_server.setEditable(True)
        for server in DEFAULT_TIME_SERVERS:
            self.cmb_time_server.addItem(server)
        self.cmb_time_server.currentTextChanged.connect(self._on_time_server_changed)
        server_row.addWidget(self.cmb_time_server, stretch=1)
        self.btn_time_sync = QPushButton("Sync now", self.grp_time_server)
        self.btn_time_sync.clicked.connect(self._on_time_sync_clicked)
        server_row.addWidget(self.btn_time_sync)
        server_widget = QWidget(self.grp_time_server)
        server_widget.setLayout(server_row)
        time_layout.addRow("Time server:", server_widget)

        self.lbl_time_status = QLabel("Not synced", self.grp_time_server)
        self.lbl_time_status.setWordWrap(True)
        time_layout.addRow("Status:", self.lbl_time_status)

        self.lbl_os_time = QLabel("—", self.grp_time_server)
        time_layout.addRow("OS time:", self.lbl_os_time)

        self.lbl_server_time = QLabel("—", self.grp_time_server)
        time_layout.addRow("Time server:", self.lbl_server_time)

        self.lbl_time_offset = QLabel("—", self.grp_time_server)
        time_layout.addRow("Difference (ms):", self.lbl_time_offset)

        layout.addWidget(self.grp_time_server)
        layout.addStretch(1)

        self._time_preview_timer = QTimer(self)
        self._time_preview_timer.timeout.connect(self._update_time_preview)
        self._time_preview_timer.start(250)
        self._update_time_preview()

        return tab

    def _create_wifi_tab(self):
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        wifi_row = QHBoxLayout()
        wifi_row.addWidget(QLabel("Wi-Fi profile:", tab))

        self.cmb_wifi_profile = QComboBox(tab)
        self.cmb_wifi_profile.currentTextChanged.connect(self._on_wifi_profile_changed)
        wifi_row.addWidget(self.cmb_wifi_profile, stretch=1)

        self.btn_wifi_new = QPushButton("New", tab)
        self.btn_wifi_new.clicked.connect(self._on_new_wifi_profile)
        wifi_row.addWidget(self.btn_wifi_new)

        self.btn_wifi_duplicate = QPushButton("Duplicate", tab)
        self.btn_wifi_duplicate.clicked.connect(self._on_duplicate_wifi_profile)
        wifi_row.addWidget(self.btn_wifi_duplicate)

        self.btn_wifi_rename = QPushButton("Rename", tab)
        self.btn_wifi_rename.clicked.connect(self._on_rename_wifi_profile)
        wifi_row.addWidget(self.btn_wifi_rename)

        self.btn_wifi_delete = QPushButton("Delete", tab)
        self.btn_wifi_delete.clicked.connect(self._on_delete_wifi_profile)
        wifi_row.addWidget(self.btn_wifi_delete)

        self.btn_wifi_save = QPushButton("Save", tab)
        self.btn_wifi_save.clicked.connect(self._on_save_wifi_profiles)
        wifi_row.addWidget(self.btn_wifi_save)

        layout.addLayout(wifi_row)

        self.grp_wifi_init_options = QGroupBox("Initialization Test Settings", tab)
        options_layout = QFormLayout(self.grp_wifi_init_options)
        self.spn_init_test_duration = QDoubleSpinBox(self.grp_wifi_init_options)
        self.spn_init_test_duration.setRange(1.0, 3600.0)
        self.spn_init_test_duration.setSingleStep(1.0)
        self.spn_init_test_duration.setSuffix(" s")
        options_layout.addRow("Test duration:", self.spn_init_test_duration)

        self.txt_init_save_directory = QLineEdit(self.grp_wifi_init_options)
        options_layout.addRow("Remote save directory:", self.txt_init_save_directory)

        self.cmb_csi_scenario = QComboBox(self.grp_wifi_init_options)
        self.cmb_csi_scenario.addItem(
            "Scenario 1: Experiment-long CSI capture", userData="scenario_1"
        )
        self.cmb_csi_scenario.addItem(
            "Scenario 2: Action-adjacent CSI captures", userData="scenario_2"
        )
        self.cmb_csi_scenario.addItem(
            "Demo: Real-time manual CSI capture", userData="demo"
        )
        self.cmb_csi_scenario.addItem(
            "No CSI collection (skip router connections)",
            userData="no_collection",
        )
        options_layout.addRow("CSI capture scenario:", self.cmb_csi_scenario)
        self.cmb_csi_scenario.currentIndexChanged.connect(
            self._on_csi_scenario_changed
        )

        self.chk_delete_prev_pcap = QCheckBox(
            "Delete previous pcap files before capture",
            self.grp_wifi_init_options,
        )
        options_layout.addRow("Delete previous pcaps:", self.chk_delete_prev_pcap)

        self.spn_pre_action_capture = QDoubleSpinBox(self.grp_wifi_init_options)
        self.spn_pre_action_capture.setRange(0.0, 60.0)
        self.spn_pre_action_capture.setSingleStep(0.5)
        self.spn_pre_action_capture.setSuffix(" s")
        options_layout.addRow(
            "CSI capture before action:", self.spn_pre_action_capture
        )

        self.spn_post_action_capture = QDoubleSpinBox(self.grp_wifi_init_options)
        self.spn_post_action_capture.setRange(0.0, 60.0)
        self.spn_post_action_capture.setSingleStep(0.5)
        self.spn_post_action_capture.setSuffix(" s")
        options_layout.addRow(
            "CSI capture after action:", self.spn_post_action_capture
        )

        self.chk_count_packets = QCheckBox(
            "Count packets after capture download", self.grp_wifi_init_options
        )
        options_layout.addRow("Count packets:", self.chk_count_packets)

        self.chk_reboot_after_summary = QCheckBox(
            "Reboot access points after showing CSI summary", self.grp_wifi_init_options
        )
        options_layout.addRow("Reboot after summary:", self.chk_reboot_after_summary)

        layout.addWidget(self.grp_wifi_init_options)

        self.grp_wifi = QGroupBox("Access Points", tab)
        wifi_layout = QVBoxLayout(self.grp_wifi)

        self.tbl_wifi = RowDragDropTableWidget(self.grp_wifi)
        self.tbl_wifi.setColumnCount(17)
        self.tbl_wifi.setHorizontalHeaderLabels(
            [
                "Order",
                "Access Point Name",
                "Framework",
                "Type",
                "Wi-Fi SSID",
                "Wi-Fi Password",
                "Router SSH IP",
                "Router SSH Username",
                "Router SSH Password",
                "SSH Key Address",
                "Frequency",
                "Channel",
                "Bandwidth",
                "Transmitter MACs",
                "Remote Save Directory",
                "Use Ethernet",
                "Download Mode",
            ]
        )
        self.tbl_wifi.horizontalHeader().setStretchLastSection(True)
        self.tbl_wifi.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        wifi_layout.addWidget(self.tbl_wifi)

        btn_row_wifi = QHBoxLayout()
        self.btn_add_wifi_ap = QPushButton("Add Access Point", self.grp_wifi)
        self.btn_add_wifi_ap.clicked.connect(self._on_add_wifi_ap)
        btn_row_wifi.addWidget(self.btn_add_wifi_ap)

        self.btn_remove_wifi_ap = QPushButton("Remove Selected", self.grp_wifi)
        self.btn_remove_wifi_ap.clicked.connect(self._on_remove_wifi_ap)
        btn_row_wifi.addWidget(self.btn_remove_wifi_ap)

        btn_row_wifi.addStretch(1)
        wifi_layout.addLayout(btn_row_wifi)

        layout.addWidget(self.grp_wifi)
        layout.addStretch(1)
        return tab

    def _create_demo_tab(self):
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        row = QHBoxLayout()
        row.addWidget(QLabel("Demo profile:", tab))
        self.cmb_demo_profile = QComboBox(tab)
        self.cmb_demo_profile.currentTextChanged.connect(self._on_demo_profile_changed)
        row.addWidget(self.cmb_demo_profile, stretch=1)

        self.btn_demo_new = QPushButton("New", tab)
        self.btn_demo_new.clicked.connect(self._on_new_demo_profile)
        row.addWidget(self.btn_demo_new)

        self.btn_demo_duplicate = QPushButton("Duplicate", tab)
        self.btn_demo_duplicate.clicked.connect(self._on_duplicate_demo_profile)
        row.addWidget(self.btn_demo_duplicate)

        self.btn_demo_rename = QPushButton("Rename", tab)
        self.btn_demo_rename.clicked.connect(self._on_rename_demo_profile)
        row.addWidget(self.btn_demo_rename)

        self.btn_demo_delete = QPushButton("Delete", tab)
        self.btn_demo_delete.clicked.connect(self._on_delete_demo_profile)
        row.addWidget(self.btn_demo_delete)

        self.btn_demo_save = QPushButton("Save", tab)
        self.btn_demo_save.clicked.connect(self._on_save_demo_profiles)
        row.addWidget(self.btn_demo_save)

        layout.addLayout(row)

        demo_scroll = QScrollArea(tab)
        demo_scroll.setWidgetResizable(True)
        demo_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        demo_scroll_content = QWidget(demo_scroll)
        demo_scroll_layout = QVBoxLayout(demo_scroll_content)
        demo_scroll_layout.setContentsMargins(0, 0, 0, 0)

        self.grp_demo = QGroupBox("Demo Settings", demo_scroll_content)
        form = QFormLayout(self.grp_demo)
        self.spn_demo_capture_duration = QDoubleSpinBox(self.grp_demo)
        self.spn_demo_capture_duration.setRange(1.0, 300.0)
        self.spn_demo_capture_duration.setSingleStep(0.5)
        self.spn_demo_capture_duration.setSuffix(" s")
        form.addRow("Demo CSI capture duration:", self.spn_demo_capture_duration)
        self.cmb_demo_capture_mode = QComboBox(self.grp_demo)
        self.cmb_demo_capture_mode.addItem(
            "Live capture from routers", userData="router_live"
        )
        self.cmb_demo_capture_mode.addItem(
            "Synthetic random CSI (no router connection)",
            userData="synthetic_random",
        )
        form.addRow("Demo capture mode:", self.cmb_demo_capture_mode)
        self.spn_demo_effective_samples = QSpinBox(self.grp_demo)
        self.spn_demo_effective_samples.setRange(0, 1_000_000)
        self.spn_demo_effective_samples.setSingleStep(10)
        self.spn_demo_effective_samples.setSpecialValueText("Use all samples")
        form.addRow("Effective CSI capture samples:", self.spn_demo_effective_samples)
        self.chk_demo_hampel_ratio_magnitude = QCheckBox(
            "Apply Hampel filter to CSI ratio magnitude before plotting",
            self.grp_demo,
        )
        form.addRow("", self.chk_demo_hampel_ratio_magnitude)
        self.chk_demo_hampel_ratio_phase = QCheckBox(
            "Apply Hampel filter to CSI ratio phase before plotting",
            self.grp_demo,
        )
        form.addRow("", self.chk_demo_hampel_ratio_phase)

        self.txt_demo_title = QLineEdit(self.grp_demo)
        self.txt_demo_title.setPlaceholderText(DEFAULT_DEMO_PROFILE["demo_title_text"])
        form.addRow("Demo window title:", self.txt_demo_title)

        self.txt_demo_university_logo_image_path = QLineEdit(self.grp_demo)
        self.txt_demo_university_logo_image_path.setPlaceholderText("/path/to/UofT-logo.png")
        form.addRow("University logo image path:", self.txt_demo_university_logo_image_path)
        self.spn_demo_university_logo_image_size = QSpinBox(self.grp_demo)
        self.spn_demo_university_logo_image_size.setRange(20, 600)
        self.spn_demo_university_logo_image_size.setSuffix(" px")
        self.spn_demo_university_logo_image_size.setSingleStep(10)
        form.addRow("University logo image height:", self.spn_demo_university_logo_image_size)

        self.txt_demo_icassp_logo_image_path = QLineEdit(self.grp_demo)
        self.txt_demo_icassp_logo_image_path.setPlaceholderText("/path/to/icassp-logo.png")
        form.addRow("ICASSP logo image path:", self.txt_demo_icassp_logo_image_path)

        self.txt_demo_website_url = QLineEdit(self.grp_demo)
        self.txt_demo_website_url.setPlaceholderText(DEFAULT_DEMO_PROFILE["website_url"])
        form.addRow("Website URL (status bar):", self.txt_demo_website_url)

        self.txt_demo_icassp_title = QLineEdit(self.grp_demo)
        self.txt_demo_icassp_title.setPlaceholderText(DEFAULT_DEMO_PROFILE["icassp_title_text"])
        form.addRow("Top-left conference text:", self.txt_demo_icassp_title)
        self.spn_demo_icassp_logo_text_vertical_gap = QSpinBox(self.grp_demo)
        self.spn_demo_icassp_logo_text_vertical_gap.setRange(-80, 80)
        self.spn_demo_icassp_logo_text_vertical_gap.setSuffix(" px")
        form.addRow(
            "Top-left logo/text vertical gap:",
            self.spn_demo_icassp_logo_text_vertical_gap,
        )
        self.spn_demo_title_font_size = QSpinBox(self.grp_demo)
        self.spn_demo_title_font_size.setRange(10, 96)
        self.spn_demo_title_font_size.setSuffix(" px")
        form.addRow("Title font size:", self.spn_demo_title_font_size)

        self.txt_demo_authors = QLineEdit(self.grp_demo)
        self.txt_demo_authors.setPlaceholderText(DEFAULT_DEMO_PROFILE["authors_text"])
        form.addRow("Authors text:", self.txt_demo_authors)
        self.spn_demo_authors_font_size = QSpinBox(self.grp_demo)
        self.spn_demo_authors_font_size.setRange(8, 72)
        self.spn_demo_authors_font_size.setSuffix(" px")
        form.addRow("Authors font size:", self.spn_demo_authors_font_size)

        self.txt_demo_university = QLineEdit(self.grp_demo)
        self.txt_demo_university.setPlaceholderText(DEFAULT_DEMO_PROFILE["university_text"])
        form.addRow("University text:", self.txt_demo_university)
        self.txt_demo_wirlab = QLineEdit(self.grp_demo)
        self.txt_demo_wirlab.setPlaceholderText(DEFAULT_DEMO_PROFILE["wirlab_text"])
        form.addRow("WIRLab text:", self.txt_demo_wirlab)
        self.spn_demo_university_font_size = QSpinBox(self.grp_demo)
        self.spn_demo_university_font_size.setRange(8, 72)
        self.spn_demo_university_font_size.setSuffix(" px")
        form.addRow("University font size:", self.spn_demo_university_font_size)
        self.spn_demo_title_authors_vertical_gap = QSpinBox(self.grp_demo)
        self.spn_demo_title_authors_vertical_gap.setRange(-80, 80)
        self.spn_demo_title_authors_vertical_gap.setSuffix(" px")
        form.addRow(
            "Title → authors vertical gap:",
            self.spn_demo_title_authors_vertical_gap,
        )

        self.spn_demo_authors_university_vertical_gap = QSpinBox(self.grp_demo)
        self.spn_demo_authors_university_vertical_gap.setRange(-80, 80)
        self.spn_demo_authors_university_vertical_gap.setSuffix(" px")
        form.addRow(
            "Authors → university vertical gap:",
            self.spn_demo_authors_university_vertical_gap,
        )

        self.txt_demo_capture_guidance_title = QLineEdit(self.grp_demo)
        self.txt_demo_capture_guidance_title.setPlaceholderText(
            DEFAULT_DEMO_PROFILE["capture_guidance_title"]
        )
        form.addRow("Guidance window title:", self.txt_demo_capture_guidance_title)

        self.txt_demo_capture_guidance_message = QLineEdit(self.grp_demo)
        self.txt_demo_capture_guidance_message.setPlaceholderText(
            DEFAULT_DEMO_PROFILE["capture_guidance_message"]
        )
        form.addRow("Guidance message:", self.txt_demo_capture_guidance_message)

        self.txt_demo_capture_guidance_left_label = QLineEdit(self.grp_demo)
        self.txt_demo_capture_guidance_left_label.setPlaceholderText(
            DEFAULT_DEMO_PROFILE["capture_guidance_video_left_label"]
        )
        form.addRow("Guidance left video label:", self.txt_demo_capture_guidance_left_label)

        self.txt_demo_capture_guidance_left_path = QLineEdit(self.grp_demo)
        self.txt_demo_capture_guidance_left_path.setPlaceholderText(
            DEFAULT_DEMO_PROFILE["capture_guidance_video_left_path"]
        )
        form.addRow("Guidance left video path:", self.txt_demo_capture_guidance_left_path)

        self.txt_demo_capture_guidance_right_label = QLineEdit(self.grp_demo)
        self.txt_demo_capture_guidance_right_label.setPlaceholderText(
            DEFAULT_DEMO_PROFILE["capture_guidance_video_right_label"]
        )
        form.addRow("Guidance right video label:", self.txt_demo_capture_guidance_right_label)

        self.txt_demo_capture_guidance_right_path = QLineEdit(self.grp_demo)
        self.txt_demo_capture_guidance_right_path.setPlaceholderText(
            DEFAULT_DEMO_PROFILE["capture_guidance_video_right_path"]
        )
        form.addRow("Guidance right video path:", self.txt_demo_capture_guidance_right_path)

        self.txt_demo_activity_class_names = QLineEdit(self.grp_demo)
        self.txt_demo_activity_class_names.setPlaceholderText(
            "Class 1 name, Class 2 name, Class 3 name"
        )
        form.addRow(
            "Activity class names (comma-separated):",
            self.txt_demo_activity_class_names,
        )

        self.txt_demo_subplot_settings = QTextEdit(self.grp_demo)
        self.txt_demo_subplot_settings.setPlaceholderText(
            "JSON object keyed by subplot category with: visible, title, xlabel, ylabel, info"
        )
        self.txt_demo_subplot_settings.setMinimumHeight(180)
        form.addRow("Subplot settings (JSON):", self.txt_demo_subplot_settings)

        self.txt_demo_dorf_plot_order = QLineEdit(self.grp_demo)
        self.txt_demo_dorf_plot_order.setPlaceholderText(
            ",".join(DEFAULT_DEMO_PROFILE["dorf_plot_order"])
        )
        form.addRow("DoRF plot order (comma-separated categories):", self.txt_demo_dorf_plot_order)

        help_label = QLabel(
            "Used by Wi-Fi Scenario 'Demo' for each CSI Capture button press.",
            self.grp_demo,
        )
        help_label.setWordWrap(True)
        form.addRow("", help_label)

        demo_scroll_layout.addWidget(self.grp_demo)
        demo_scroll_layout.addStretch(1)
        demo_scroll.setWidget(demo_scroll_content)
        layout.addWidget(demo_scroll)
        layout.addStretch(1)
        return tab

    def _create_actions_tab(self):
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        actions_row = QHBoxLayout()
        actions_row.addWidget(QLabel("Actions profile:", tab))

        self.cmb_actions_profile = QComboBox(tab)
        self.cmb_actions_profile.currentTextChanged.connect(
            self._on_action_profile_changed
        )
        actions_row.addWidget(self.cmb_actions_profile, stretch=1)

        self.btn_actions_new = QPushButton("New", tab)
        self.btn_actions_new.clicked.connect(self._on_new_action_profile)
        actions_row.addWidget(self.btn_actions_new)

        self.btn_actions_duplicate = QPushButton("Duplicate", tab)
        self.btn_actions_duplicate.clicked.connect(self._on_duplicate_action_profile)
        actions_row.addWidget(self.btn_actions_duplicate)

        self.btn_actions_rename = QPushButton("Rename", tab)
        self.btn_actions_rename.clicked.connect(self._on_rename_action_profile)
        actions_row.addWidget(self.btn_actions_rename)

        self.btn_actions_delete = QPushButton("Delete", tab)
        self.btn_actions_delete.clicked.connect(self._on_delete_action_profile)
        actions_row.addWidget(self.btn_actions_delete)

        self.btn_actions_save = QPushButton("Save", tab)
        self.btn_actions_save.clicked.connect(self._on_save_action_profiles)
        actions_row.addWidget(self.btn_actions_save)

        layout.addLayout(actions_row)

        self.grp_actions = QGroupBox("Actions", tab)
        actions_layout = QVBoxLayout(self.grp_actions)

        self.tbl_actions = QTableWidget(self.grp_actions)
        self.tbl_actions.setColumnCount(2)
        self.tbl_actions.setHorizontalHeaderLabels(["Action name", "Video path"])
        self.tbl_actions.horizontalHeader().setStretchLastSection(True)
        self.tbl_actions.verticalHeader().setVisible(False)
        self.tbl_actions.setSelectionBehavior(self.tbl_actions.SelectRows)
        self.tbl_actions.itemChanged.connect(self._on_actions_table_item_changed)
        model = self.tbl_actions.model()
        if model is not None:
            model.rowsInserted.connect(lambda *args: self._update_total_duration_label())
            model.rowsRemoved.connect(lambda *args: self._update_total_duration_label())
        self._ensure_action_table_capacity(MIN_ACTION_TABLE_ROWS)
        actions_layout.addWidget(self.tbl_actions)

        btn_row_actions = QHBoxLayout()
        self.btn_add_action = QPushButton("Add Action", self.grp_actions)
        self.btn_add_action.clicked.connect(self._on_add_action)
        btn_row_actions.addWidget(self.btn_add_action)

        self.btn_remove_action = QPushButton("Remove Action", self.grp_actions)
        self.btn_remove_action.clicked.connect(self._on_remove_action)
        btn_row_actions.addWidget(self.btn_remove_action)

        self.btn_browse_video = QPushButton("Browse Video…", self.grp_actions)
        self.btn_browse_video.clicked.connect(self._on_browse_video)
        btn_row_actions.addWidget(self.btn_browse_video)

        btn_row_actions.addStretch(1)
        actions_layout.addLayout(btn_row_actions)

        layout.addWidget(self.grp_actions)
        layout.addStretch(1)
        return tab



    # ------------------------------------------------------------------
    # Combo helpers
    # ------------------------------------------------------------------
    def _populate_participant_combo(self):
        self.cmb_participant_profile.blockSignals(True)
        self.cmb_participant_profile.clear()
        for name in self.participant_profiles.keys():
            self.cmb_participant_profile.addItem(name)
        idx = self.cmb_participant_profile.findText(self.current_participant_name)
        if idx < 0 and self.cmb_participant_profile.count() > 0:
            idx = 0
            self.current_participant_name = self.cmb_participant_profile.itemText(idx)
        self.cmb_participant_profile.setCurrentIndex(max(idx, 0))
        self.cmb_participant_profile.blockSignals(False)

    def _populate_experiment_combo(self):
        self.cmb_experiment_profile.blockSignals(True)
        self.cmb_experiment_profile.clear()
        for name in self.experiment_profiles.keys():
            self.cmb_experiment_profile.addItem(name)
        idx = self.cmb_experiment_profile.findText(self.current_experiment_name)
        if idx < 0 and self.cmb_experiment_profile.count() > 0:
            idx = 0
            self.current_experiment_name = self.cmb_experiment_profile.itemText(idx)
        self.cmb_experiment_profile.setCurrentIndex(max(idx, 0))
        self.cmb_experiment_profile.blockSignals(False)

    def _populate_environment_combo(self):
        self.cmb_environment_profile.blockSignals(True)
        self.cmb_environment_profile.clear()
        for name in self.environment_profiles.keys():
            self.cmb_environment_profile.addItem(name)
        idx = self.cmb_environment_profile.findText(self.current_environment_name)
        if idx < 0 and self.cmb_environment_profile.count() > 0:
            idx = 0
            self.current_environment_name = self.cmb_environment_profile.itemText(idx)
        self.cmb_environment_profile.setCurrentIndex(max(idx, 0))
        self.cmb_environment_profile.blockSignals(False)

    def _populate_action_combo(self):
        self.cmb_actions_profile.blockSignals(True)
        self.cmb_actions_profile.clear()
        for name in self.action_profiles.keys():
            self.cmb_actions_profile.addItem(name)
        idx = self.cmb_actions_profile.findText(self.current_action_profile_name)
        if idx < 0 and self.cmb_actions_profile.count() > 0:
            idx = 0
            self.current_action_profile_name = self.cmb_actions_profile.itemText(idx)
        self.cmb_actions_profile.setCurrentIndex(max(idx, 0))
        self.cmb_actions_profile.blockSignals(False)

    def _populate_camera_combo(self):
        self.cmb_camera_profile.blockSignals(True)
        self.cmb_camera_profile.clear()
        for name in self.camera_profiles.keys():
            self.cmb_camera_profile.addItem(name)
        idx = self.cmb_camera_profile.findText(self.current_camera_profile_name)
        if idx < 0 and self.cmb_camera_profile.count() > 0:
            idx = 0
            self.current_camera_profile_name = self.cmb_camera_profile.itemText(idx)
        self.cmb_camera_profile.setCurrentIndex(max(idx, 0))
        self.cmb_camera_profile.blockSignals(False)

    def _populate_depth_camera_combo(self):
        self.cmb_depth_camera_profile.blockSignals(True)
        self.cmb_depth_camera_profile.clear()
        for name in self.depth_camera_profiles.keys():
            self.cmb_depth_camera_profile.addItem(name)
        idx = self.cmb_depth_camera_profile.findText(self.current_depth_camera_profile_name)
        if idx < 0 and self.cmb_depth_camera_profile.count() > 0:
            idx = 0
            self.current_depth_camera_profile_name = self.cmb_depth_camera_profile.itemText(idx)
        self.cmb_depth_camera_profile.setCurrentIndex(max(idx, 0))
        self.cmb_depth_camera_profile.blockSignals(False)

    def _on_participant_profile_changed(self, new_name: str):
        if not new_name:
            return
        self._update_current_participant_from_ui()
        self.current_participant_name = new_name
        self._load_participant_into_ui(new_name)

    def _on_experiment_profile_changed(self, new_name: str):
        if not new_name:
            return
        self._update_current_experiment_from_ui()
        self.current_experiment_name = new_name
        self._load_experiment_into_ui(new_name)

    def _on_environment_profile_changed(self, new_name: str):
        if not new_name:
            return
        self._update_current_environment_from_ui()
        self.current_environment_name = new_name
        self._load_environment_into_ui(new_name)

    def _on_action_profile_changed(self, new_name: str):
        if not new_name:
            return
        self._update_current_actions_from_ui()
        self.current_action_profile_name = new_name
        self._load_actions_into_ui(new_name)

    def _on_camera_profile_changed(self, new_name: str):
        if not new_name:
            return
        self._update_current_camera_from_ui()
        self.current_camera_profile_name = new_name
        self._load_camera_into_ui(new_name)

    def _on_depth_camera_profile_changed(self, new_name: str):
        if not new_name:
            return
        self._update_current_depth_camera_from_ui()
        self.current_depth_camera_profile_name = new_name
        self._load_depth_camera_into_ui(new_name)

    def _populate_voice_combo(self):
        self.cmb_voice_profile.blockSignals(True)
        self.cmb_voice_profile.clear()
        for name in self.voice_profiles.keys():
            self.cmb_voice_profile.addItem(name)
        idx = self.cmb_voice_profile.findText(self.current_voice_profile_name)
        if idx < 0 and self.cmb_voice_profile.count() > 0:
            idx = 0
            self.current_voice_profile_name = self.cmb_voice_profile.itemText(idx)
        self.cmb_voice_profile.setCurrentIndex(max(idx, 0))
        self.cmb_voice_profile.blockSignals(False)

    def _populate_ui_combo(self):
        self.cmb_ui_profile.blockSignals(True)
        self.cmb_ui_profile.clear()
        for name in self.ui_profiles.keys():
            self.cmb_ui_profile.addItem(name)
        idx = self.cmb_ui_profile.findText(self.current_ui_profile_name)
        if idx < 0 and self.cmb_ui_profile.count() > 0:
            idx = 0
            self.current_ui_profile_name = self.cmb_ui_profile.itemText(idx)
        self.cmb_ui_profile.setCurrentIndex(max(idx, 0))
        self.cmb_ui_profile.blockSignals(False)

    def _populate_wifi_combo(self):
        self.cmb_wifi_profile.blockSignals(True)
        self.cmb_wifi_profile.clear()
        for name in self.wifi_profiles.keys():
            self.cmb_wifi_profile.addItem(name)
        idx = self.cmb_wifi_profile.findText(self.current_wifi_profile_name)
        if idx < 0 and self.cmb_wifi_profile.count() > 0:
            idx = 0
            self.current_wifi_profile_name = self.cmb_wifi_profile.itemText(idx)
        self.cmb_wifi_profile.setCurrentIndex(max(idx, 0))
        self.cmb_wifi_profile.blockSignals(False)

    def _populate_time_combo(self):
        self.cmb_time_profile.blockSignals(True)
        self.cmb_time_profile.clear()
        for name in self.time_profiles.keys():
            self.cmb_time_profile.addItem(name)
        idx = self.cmb_time_profile.findText(self.current_time_profile_name)
        if idx < 0 and self.cmb_time_profile.count() > 0:
            idx = 0
            self.current_time_profile_name = self.cmb_time_profile.itemText(idx)
        self.cmb_time_profile.setCurrentIndex(max(idx, 0))
        self.cmb_time_profile.blockSignals(False)

    def _populate_demo_combo(self):
        self.cmb_demo_profile.blockSignals(True)
        self.cmb_demo_profile.clear()
        for name in self.demo_profiles.keys():
            self.cmb_demo_profile.addItem(name)
        idx = self.cmb_demo_profile.findText(self.current_demo_profile_name)
        if idx < 0 and self.cmb_demo_profile.count() > 0:
            idx = 0
            self.current_demo_profile_name = self.cmb_demo_profile.itemText(idx)
        self.cmb_demo_profile.setCurrentIndex(max(idx, 0))
        self.cmb_demo_profile.blockSignals(False)

    def _set_profile_editable(self, profile_type: str, editable: bool):
        widget_map = {
            "participant": [
                getattr(self, "grp_subject", None),
                getattr(self, "grp_second_subject", None),
                getattr(self, "scr_subject", None),
                getattr(self, "scr_second_subject", None),
                getattr(self, "chk_has_second_participant", None),
            ],
            "experiment": [getattr(self, "grp_experiment", None)],
            "environment": [getattr(self, "grp_environment", None)],
            "actions": [getattr(self, "grp_actions", None)],
            "voice": [getattr(self, "grp_voice", None)],
            "camera": [
                getattr(self, "chk_use_webcam", None),
                getattr(self, "cmb_camera_device", None),
                getattr(self, "btn_refresh_cameras", None),
                getattr(self, "btn_preview_camera", None),
                getattr(self, "lbl_camera_preview", None),
                getattr(self, "chk_hand_recognition", None),
                getattr(self, "cmb_hand_mode", None),
                getattr(self, "cmb_hand_model_complexity", None),
            ],
            "depth_camera": [
                getattr(self, "chk_depth_enabled", None),
                getattr(self, "edt_depth_api_ip", None),
                getattr(self, "spn_depth_fps", None),
                getattr(self, "cmb_depth_rgb_res", None),
                getattr(self, "cmb_depth_depth_res", None),
                getattr(self, "chk_depth_save_raw", None),
                getattr(self, "edt_depth_save_location", None),
                getattr(self, "btn_depth_browse_location", None),
                getattr(self, "btn_depth_refresh_status", None),
                getattr(self, "txt_depth_status", None),
            ],
            "ui": [
                getattr(self, "grp_ui_frames", None),
                getattr(self, "grp_count_font", None),
                getattr(self, "grp_action_font", None),
                getattr(self, "grp_time_font", None),
                getattr(self, "grp_remaining_time_value_font", None),
                getattr(self, "grp_elapsed_time_value_font", None),
                getattr(self, "grp_transcript", None),
                getattr(self, "grp_window_options", None),
                getattr(self, "grp_status_messages", None),
                getattr(self, "grp_instruction_messages", None),
                getattr(self, "grp_main_text", None),
                getattr(self, "grp_main_values", None),
                getattr(self, "grp_log_text", None),
                getattr(self, "grp_webcam_text", None),
            ],
            "wifi": [
                getattr(self, "grp_wifi_init_options", None),
                getattr(self, "grp_wifi", None),
            ],
            "time": [getattr(self, "grp_time_server", None)],
            "demo": [getattr(self, "grp_demo", None)],
        }
        button_map = {
            "participant": [
                getattr(self, "btn_participant_delete", None),
                getattr(self, "btn_participant_rename", None),
                getattr(self, "btn_participant_save", None),
            ],
            "experiment": [
                getattr(self, "btn_experiment_delete", None),
                getattr(self, "btn_experiment_rename", None),
                getattr(self, "btn_experiment_save", None),
            ],
            "environment": [
                getattr(self, "btn_environment_delete", None),
                getattr(self, "btn_environment_rename", None),
                getattr(self, "btn_environment_save", None),
            ],
            "actions": [
                getattr(self, "btn_actions_delete", None),
                getattr(self, "btn_actions_rename", None),
                getattr(self, "btn_actions_save", None),
                getattr(self, "btn_add_action", None),
                getattr(self, "btn_remove_action", None),
                getattr(self, "btn_browse_video", None),
            ],
            "voice": [
                getattr(self, "btn_voice_delete", None),
                getattr(self, "btn_voice_rename", None),
                getattr(self, "btn_voice_save", None),
            ],
            "camera": [
                getattr(self, "btn_camera_delete", None),
                getattr(self, "btn_camera_rename", None),
                getattr(self, "btn_camera_save", None),
            ],
            "depth_camera": [
                getattr(self, "btn_depth_camera_delete", None),
                getattr(self, "btn_depth_camera_rename", None),
                getattr(self, "btn_depth_camera_save", None),
            ],
            "ui": [
                getattr(self, "btn_ui_delete", None),
                getattr(self, "btn_ui_rename", None),
                getattr(self, "btn_ui_save", None),
            ],
            "wifi": [
                getattr(self, "btn_wifi_delete", None),
                getattr(self, "btn_wifi_rename", None),
                getattr(self, "btn_wifi_save", None),
                getattr(self, "btn_add_wifi_ap", None),
                getattr(self, "btn_remove_wifi_ap", None),
            ],
            "time": [
                getattr(self, "btn_time_delete", None),
                getattr(self, "btn_time_rename", None),
                getattr(self, "btn_time_save", None),
            ],
            "demo": [
                getattr(self, "btn_demo_delete", None),
                getattr(self, "btn_demo_rename", None),
                getattr(self, "btn_demo_save", None),
            ],
        }

        for widget in widget_map.get(profile_type, []):
            if widget is not None:
                widget.setEnabled(editable)

        for button in button_map.get(profile_type, []):
            if button is not None:
                button.setEnabled(editable)

    def _on_voice_profile_changed(self, new_name: str):
        if not new_name:
            return
        self._update_current_voice_from_ui()
        self.current_voice_profile_name = new_name
        self._load_voice_into_ui(new_name)

    def _on_ui_profile_changed(self, new_name: str):
        if not new_name:
            return
        self._update_current_ui_from_ui()
        self.current_ui_profile_name = new_name
        self._load_ui_into_ui(new_name)

    def _on_wifi_profile_changed(self, new_name: str):
        if not new_name:
            return
        self._update_current_wifi_from_ui()
        self.current_wifi_profile_name = new_name
        self._load_wifi_into_ui(new_name)

    def _on_time_profile_changed(self, new_name: str):
        if not new_name:
            return
        self._update_current_time_from_ui()
        self.current_time_profile_name = new_name
        self._load_time_into_ui(new_name)

    def _on_demo_profile_changed(self, new_name: str):
        if not new_name:
            return
        self._update_current_demo_from_ui()
        self.current_demo_profile_name = new_name
        self._load_demo_into_ui(new_name)

    # ------------------------------------------------------------------
    # New / duplicate / delete / save helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _create_unique_name(base: str, existing: dict) -> str:
        candidate = base
        counter = 1
        while candidate in existing:
            candidate = f"{base}_{counter}"
            counter += 1
        return candidate

    def _rename_profile(
        self,
        profile_type: str,
        profiles: dict,
        current_name_attr: str,
        populate_func,
        load_func,
        *,
        update_inner_name: bool = False,
    ):
        current_name = getattr(self, current_name_attr, "")
        if not current_name:
            return
        if _is_default_profile(profile_type, current_name):
            QMessageBox.information(
                self,
                "Protected Profile",
                "The default profile cannot be renamed. Duplicate it to make changes.",
            )
            return

        new_name, ok = QInputDialog.getText(
            self,
            "Rename Profile",
            "Enter a new name for the profile:",
            text=current_name,
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == current_name:
            return
        if new_name in profiles:
            QMessageBox.warning(
                self,
                "Duplicate Name",
                f"A profile called '{new_name}' already exists.",
            )
            return

        profiles[new_name] = profiles.pop(current_name)
        if update_inner_name and isinstance(profiles[new_name], dict):
            profiles[new_name]["name"] = new_name
        setattr(self, current_name_attr, new_name)
        populate_func()
        load_func(new_name)

    def _on_new_participant_profile(self):
        name, ok = QInputDialog.getText(
            self,
            "New Participant Profile",
            "Enter a name for the new participant profile:",
            text="participant_profile",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self.participant_profiles:
            QMessageBox.warning(
                self,
                "Duplicate Name",
                f"A participant profile called '{name}' already exists.",
            )
            return

        self.participant_profiles[name] = deepcopy(DEFAULT_PARTICIPANT_PROFILE)
        self.current_participant_name = name
        self._populate_participant_combo()
        self._load_participant_into_ui(name)

    def _on_duplicate_participant_profile(self):
        if not self.current_participant_name:
            return
        base_name = self.current_participant_name
        new_name = self._create_unique_name(f"{base_name}_copy", self.participant_profiles)
        self.participant_profiles[new_name] = deepcopy(
            self.participant_profiles[base_name]
        )
        self.current_participant_name = new_name
        self._populate_participant_combo()
        self._load_participant_into_ui(new_name)

    def _on_rename_participant_profile(self):
        self._rename_profile(
            "participant",
            self.participant_profiles,
            "current_participant_name",
            self._populate_participant_combo,
            self._load_participant_into_ui,
        )

    def _on_delete_participant_profile(self):
        if not self.current_participant_name:
            return
        if _is_default_profile("participant", self.current_participant_name):
            QMessageBox.information(
                self,
                "Protected Profile",
                "The default participant profile cannot be modified directly. Duplicate it to make changes.",
            )
            return
        if len(self.participant_profiles) <= 1:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "At least one participant profile must exist.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete Participant Profile",
            f"Delete participant profile '{self.current_participant_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.participant_profiles.pop(self.current_participant_name, None)
        self.current_participant_name = next(iter(self.participant_profiles.keys()))
        self._populate_participant_combo()
        self._load_participant_into_ui(self.current_participant_name)

    def _on_save_participant_profiles(self):
        self._update_current_participant_from_ui()
        save_participant_profiles(self.participant_profiles)
        QMessageBox.information(self, "Saved", "Participant profiles have been saved.")

    def _on_new_experiment_profile(self):
        name, ok = QInputDialog.getText(
            self,
            "New Experiment Profile",
            "Enter a name for the new experiment timings profile:",
            text="experiment_profile",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self.experiment_profiles:
            QMessageBox.warning(
                self,
                "Duplicate Name",
                f"An experiment profile called '{name}' already exists.",
            )
            return
        self.experiment_profiles[name] = deepcopy(DEFAULT_EXPERIMENT_PROFILE)
        self.current_experiment_name = name
        self._populate_experiment_combo()
        self._load_experiment_into_ui(name)

    def _on_duplicate_experiment_profile(self):
        if not self.current_experiment_name:
            return
        base_name = self.current_experiment_name
        new_name = self._create_unique_name(
            f"{base_name}_copy", self.experiment_profiles
        )
        self.experiment_profiles[new_name] = deepcopy(
            self.experiment_profiles[base_name]
        )
        self.current_experiment_name = new_name
        self._populate_experiment_combo()
        self._load_experiment_into_ui(new_name)

    def _on_duplicate_environment_profile(self):
        if not self.current_environment_name:
            return
        base_name = self.current_environment_name
        new_name = self._create_unique_name(
            f"{base_name}_copy", self.environment_profiles
        )
        self.environment_profiles[new_name] = deepcopy(
            self.environment_profiles[base_name]
        )
        self.current_environment_name = new_name
        self._populate_environment_combo()
        self._load_environment_into_ui(new_name)

    def _on_new_environment_profile(self):
        name, ok = QInputDialog.getText(
            self,
            "New Environment Profile",
            "Enter a name for the new environment profile:",
            text="environment_profile",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self.environment_profiles:
            QMessageBox.warning(
                self,
                "Duplicate Name",
                f"An environment profile called '{name}' already exists.",
            )
            return

        self.environment_profiles[name] = deepcopy(DEFAULT_ENVIRONMENT_PROFILE)
        self.current_environment_name = name
        self._populate_environment_combo()
        self._load_environment_into_ui(name)

    def _on_rename_experiment_profile(self):
        self._rename_profile(
            "experiment",
            self.experiment_profiles,
            "current_experiment_name",
            self._populate_experiment_combo,
            self._load_experiment_into_ui,
        )

    def _on_rename_environment_profile(self):
        self._rename_profile(
            "environment",
            self.environment_profiles,
            "current_environment_name",
            self._populate_environment_combo,
            self._load_environment_into_ui,
        )

    def _on_delete_experiment_profile(self):
        if not self.current_experiment_name:
            return
        if _is_default_profile("experiment", self.current_experiment_name):
            QMessageBox.information(
                self,
                "Protected Profile",
                "The default experiment profile cannot be modified directly. Duplicate it to make changes.",
            )
            return
        if len(self.experiment_profiles) <= 1:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "At least one experiment profile must exist.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete Experiment Profile",
            f"Delete experiment profile '{self.current_experiment_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.experiment_profiles.pop(self.current_experiment_name, None)
        self.current_experiment_name = next(iter(self.experiment_profiles.keys()))
        self._populate_experiment_combo()
        self._load_experiment_into_ui(self.current_experiment_name)

    def _on_delete_environment_profile(self):
        if not self.current_environment_name:
            return
        if _is_default_profile("environment", self.current_environment_name):
            QMessageBox.information(
                self,
                "Protected Profile",
                "The default environment profile cannot be modified directly. Duplicate it to make changes.",
            )
            return
        if len(self.environment_profiles) <= 1:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "At least one environment profile must exist.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete Environment Profile",
            f"Delete environment profile '{self.current_environment_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.environment_profiles.pop(self.current_environment_name, None)
        self.current_environment_name = next(iter(self.environment_profiles.keys()))
        self._populate_environment_combo()
        self._load_environment_into_ui(self.current_environment_name)

    def _on_save_experiment_profiles(self):
        self._update_current_experiment_from_ui()
        save_experiment_profiles(self.experiment_profiles)
        QMessageBox.information(self, "Saved", "Experiment profiles have been saved.")

    def _on_save_environment_profiles(self):
        self._update_current_environment_from_ui()
        save_environment_profiles(self.environment_profiles)
        QMessageBox.information(self, "Saved", "Environment profiles have been saved.")

    def _on_new_action_profile(self):
        name, ok = QInputDialog.getText(
            self,
            "New Actions Profile",
            "Enter a name for the new actions profile:",
            text="actions_profile",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self.action_profiles:
            QMessageBox.warning(
                self,
                "Duplicate Name",
                f"An actions profile called '{name}' already exists.",
            )
            return
        self.action_profiles[name] = deepcopy(DEFAULT_ACTION_PROFILE)
        self.current_action_profile_name = name
        self._populate_action_combo()
        self._load_actions_into_ui(name)

    def _on_duplicate_action_profile(self):
        if not self.current_action_profile_name:
            return
        base_name = self.current_action_profile_name
        new_name = self._create_unique_name(f"{base_name}_copy", self.action_profiles)
        self.action_profiles[new_name] = deepcopy(
            self.action_profiles[base_name]
        )
        self.current_action_profile_name = new_name
        self._populate_action_combo()
        self._load_actions_into_ui(new_name)

    def _on_rename_action_profile(self):
        self._rename_profile(
            "actions",
            self.action_profiles,
            "current_action_profile_name",
            self._populate_action_combo,
            self._load_actions_into_ui,
        )

    def _on_delete_action_profile(self):
        if not self.current_action_profile_name:
            return
        if _is_default_profile("actions", self.current_action_profile_name):
            QMessageBox.information(
                self,
                "Protected Profile",
                "The default actions profile cannot be modified directly. Duplicate it to make changes.",
            )
            return
        if len(self.action_profiles) <= 1:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "At least one actions profile must exist.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete Actions Profile",
            f"Delete actions profile '{self.current_action_profile_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.action_profiles.pop(self.current_action_profile_name, None)
        self.current_action_profile_name = next(iter(self.action_profiles.keys()))
        self._populate_action_combo()
        self._load_actions_into_ui(self.current_action_profile_name)

    def _on_save_action_profiles(self):
        self._update_current_actions_from_ui()
        save_action_profiles(self.action_profiles)
        QMessageBox.information(self, "Saved", "Actions profiles have been saved.")

    def _on_new_voice_profile(self):
        name, ok = QInputDialog.getText(
            self,
            "New Voice Profile",
            "Enter a name for the new voice profile:",
            text="voice_profile",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self.voice_profiles:
            QMessageBox.warning(
                self,
                "Duplicate Name",
                f"A voice profile called '{name}' already exists.",
            )
            return
        self.voice_profiles[name] = deepcopy(DEFAULT_VOICE_PROFILE)
        self.current_voice_profile_name = name
        self._populate_voice_combo()
        self._load_voice_into_ui(name)

    def _on_new_camera_profile(self):
        name, ok = QInputDialog.getText(
            self,
            "New 2D Camera Profile",
            "Enter a name for the new camera profile:",
            text="camera_profile",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self.camera_profiles:
            QMessageBox.warning(
                self,
                "Duplicate Name",
                f"A camera profile called '{name}' already exists.",
            )
            return
        self.camera_profiles[name] = deepcopy(DEFAULT_CAMERA_PROFILE)
        self.current_camera_profile_name = name
        self._populate_camera_combo()
        self._load_camera_into_ui(name)

    def _on_new_depth_camera_profile(self):
        name, ok = QInputDialog.getText(
            self,
            "New Depth Camera Profile",
            "Enter a name for the new depth camera profile:",
            text="depth_camera_profile",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self.depth_camera_profiles:
            QMessageBox.warning(
                self,
                "Duplicate Name",
                f"A depth camera profile called '{name}' already exists.",
            )
            return
        self.depth_camera_profiles[name] = deepcopy(DEFAULT_DEPTH_CAMERA_PROFILE)
        self.current_depth_camera_profile_name = name
        self._populate_depth_camera_combo()
        self._load_depth_camera_into_ui(name)

    def _on_duplicate_voice_profile(self):
        if not self.current_voice_profile_name:
            return
        base_name = self.current_voice_profile_name
        new_name = self._create_unique_name(f"{base_name}_copy", self.voice_profiles)
        self.voice_profiles[new_name] = deepcopy(self.voice_profiles[base_name])
        self.current_voice_profile_name = new_name
        self._populate_voice_combo()
        self._load_voice_into_ui(new_name)

    def _on_rename_voice_profile(self):
        self._rename_profile(
            "voice",
            self.voice_profiles,
            "current_voice_profile_name",
            self._populate_voice_combo,
            self._load_voice_into_ui,
        )

    def _on_duplicate_camera_profile(self):
        if not self.current_camera_profile_name:
            return
        base_name = self.current_camera_profile_name
        new_name = self._create_unique_name(f"{base_name}_copy", self.camera_profiles)
        self.camera_profiles[new_name] = deepcopy(self.camera_profiles[base_name])
        self.current_camera_profile_name = new_name
        self._populate_camera_combo()
        self._load_camera_into_ui(new_name)

    def _on_duplicate_depth_camera_profile(self):
        if not self.current_depth_camera_profile_name:
            return
        base_name = self.current_depth_camera_profile_name
        new_name = self._create_unique_name(
            f"{base_name}_copy", self.depth_camera_profiles
        )
        self.depth_camera_profiles[new_name] = deepcopy(
            self.depth_camera_profiles[base_name]
        )
        self.current_depth_camera_profile_name = new_name
        self._populate_depth_camera_combo()
        self._load_depth_camera_into_ui(new_name)

    def _on_rename_camera_profile(self):
        self._rename_profile(
            "camera",
            self.camera_profiles,
            "current_camera_profile_name",
            self._populate_camera_combo,
            self._load_camera_into_ui,
        )

    def _on_rename_depth_camera_profile(self):
        self._rename_profile(
            "depth_camera",
            self.depth_camera_profiles,
            "current_depth_camera_profile_name",
            self._populate_depth_camera_combo,
            self._load_depth_camera_into_ui,
        )

    def _on_delete_voice_profile(self):
        if not self.current_voice_profile_name:
            return
        if _is_default_profile("voice", self.current_voice_profile_name):
            QMessageBox.information(
                self,
                "Protected Profile",
                "The default voice profile cannot be modified directly. Duplicate it to make changes.",
            )
            return
        if len(self.voice_profiles) <= 1:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "At least one voice profile must exist.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete Voice Profile",
            f"Delete voice profile '{self.current_voice_profile_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.voice_profiles.pop(self.current_voice_profile_name, None)
        self.current_voice_profile_name = next(iter(self.voice_profiles.keys()))
        self._populate_voice_combo()
        self._load_voice_into_ui(self.current_voice_profile_name)

    def _on_save_voice_profiles(self):
        self._update_current_voice_from_ui()
        save_voice_profiles(self.voice_profiles)
        QMessageBox.information(
            self,
            "Saved",
            "Voice profiles have been saved.",
        )

    def _on_new_ui_profile(self):
        name, ok = QInputDialog.getText(
            self,
            "New UI Profile",
            "Enter a name for the new UI profile:",
            text="ui_profile",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self.ui_profiles:
            QMessageBox.warning(
                self,
                "Duplicate Name",
                f"A UI profile called '{name}' already exists.",
            )
            return
        self.ui_profiles[name] = deepcopy(DEFAULT_UI_PROFILE)
        self.current_ui_profile_name = name
        self._populate_ui_combo()
        self._load_ui_into_ui(name)

    def _on_duplicate_ui_profile(self):
        if not self.current_ui_profile_name:
            return
        base_name = self.current_ui_profile_name
        new_name = self._create_unique_name(f"{base_name}_copy", self.ui_profiles)
        self.ui_profiles[new_name] = deepcopy(self.ui_profiles[base_name])
        self.current_ui_profile_name = new_name
        self._populate_ui_combo()
        self._load_ui_into_ui(new_name)

    def _on_rename_ui_profile(self):
        self._rename_profile(
            "ui",
            self.ui_profiles,
            "current_ui_profile_name",
            self._populate_ui_combo,
            self._load_ui_into_ui,
        )

    def _on_delete_ui_profile(self):
        if not self.current_ui_profile_name:
            return
        if _is_default_profile("ui", self.current_ui_profile_name):
            QMessageBox.information(
                self,
                "Protected Profile",
                "The default UI profile cannot be modified directly. Duplicate it to make changes.",
            )
            return
        if len(self.ui_profiles) <= 1:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "At least one UI profile must exist.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete UI Profile",
            f"Delete UI profile '{self.current_ui_profile_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.ui_profiles.pop(self.current_ui_profile_name, None)
        self.current_ui_profile_name = next(iter(self.ui_profiles.keys()))
        self._populate_ui_combo()
        self._load_ui_into_ui(self.current_ui_profile_name)

    def _on_save_ui_profiles(self):
        self._update_current_ui_from_ui()
        save_ui_profiles(self.ui_profiles)
        QMessageBox.information(self, "Saved", "UI profiles have been saved.")

    def _on_delete_camera_profile(self):
        if not self.current_camera_profile_name:
            return
        if _is_default_profile("camera", self.current_camera_profile_name):
            QMessageBox.information(
                self,
                "Protected Profile",
                "The default camera profile cannot be modified directly. Duplicate it to make changes.",
            )
            return
        if len(self.camera_profiles) <= 1:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "At least one camera profile must exist.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete 2D Camera Profile",
            f"Delete camera profile '{self.current_camera_profile_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.camera_profiles.pop(self.current_camera_profile_name, None)
        self.current_camera_profile_name = next(iter(self.camera_profiles.keys()))
        self._populate_camera_combo()
        self._load_camera_into_ui(self.current_camera_profile_name)

    def _on_save_camera_profiles(self):
        self._update_current_camera_from_ui()
        save_camera_profiles(self.camera_profiles)
        QMessageBox.information(
            self,
            "Saved",
            "Camera profiles have been saved.",
        )

    def _on_delete_depth_camera_profile(self):
        if not self.current_depth_camera_profile_name:
            return
        if _is_default_profile("depth_camera", self.current_depth_camera_profile_name):
            QMessageBox.information(
                self,
                "Protected Profile",
                "The default depth camera profile cannot be modified directly. Duplicate it to make changes.",
            )
            return
        if len(self.depth_camera_profiles) <= 1:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "At least one depth camera profile must exist.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete Depth Camera Profile",
            f"Delete depth camera profile '{self.current_depth_camera_profile_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.depth_camera_profiles.pop(self.current_depth_camera_profile_name, None)
        self.current_depth_camera_profile_name = next(iter(self.depth_camera_profiles.keys()))
        self._populate_depth_camera_combo()
        self._load_depth_camera_into_ui(self.current_depth_camera_profile_name)

    def _on_save_depth_camera_profiles(self):
        self._update_current_depth_camera_from_ui()
        save_depth_camera_profiles(self.depth_camera_profiles)
        QMessageBox.information(
            self,
            "Saved",
            "Depth camera profiles have been saved.",
        )

    def _on_new_wifi_profile(self):
        name, ok = QInputDialog.getText(
            self,
            "New Wi-Fi Profile",
            "Enter a name for the new Wi-Fi profile:",
            text="wifi_profile",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self.wifi_profiles:
            QMessageBox.warning(
                self,
                "Duplicate Name",
                f"A Wi-Fi profile called '{name}' already exists.",
            )
            return

        new_profile = deepcopy(DEFAULT_WIFI_PROFILE)
        new_profile["name"] = name
        self.wifi_profiles[name] = new_profile
        self.current_wifi_profile_name = name
        self._populate_wifi_combo()
        self._load_wifi_into_ui(name)

    def _on_duplicate_wifi_profile(self):
        if not self.current_wifi_profile_name:
            return
        base_name = self.current_wifi_profile_name
        new_name = self._create_unique_name(f"{base_name}_copy", self.wifi_profiles)
        duplicated = deepcopy(self.wifi_profiles[base_name])
        duplicated["name"] = new_name
        self.wifi_profiles[new_name] = duplicated
        self.current_wifi_profile_name = new_name
        self._populate_wifi_combo()
        self._load_wifi_into_ui(new_name)

    def _on_rename_wifi_profile(self):
        self._rename_profile(
            "wifi",
            self.wifi_profiles,
            "current_wifi_profile_name",
            self._populate_wifi_combo,
            self._load_wifi_into_ui,
            update_inner_name=True,
        )

    def _on_delete_wifi_profile(self):
        if not self.current_wifi_profile_name:
            return
        if _is_default_profile("wifi", self.current_wifi_profile_name):
            QMessageBox.information(
                self,
                "Protected Profile",
                "The default Wi-Fi profile cannot be modified directly. Duplicate it to make changes.",
            )
            return
        if len(self.wifi_profiles) <= 1:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "At least one Wi-Fi profile must exist.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete Wi-Fi Profile",
            f"Delete Wi-Fi profile '{self.current_wifi_profile_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.wifi_profiles.pop(self.current_wifi_profile_name, None)
        self.current_wifi_profile_name = next(iter(self.wifi_profiles.keys()))
        self._populate_wifi_combo()
        self._load_wifi_into_ui(self.current_wifi_profile_name)

    def _on_save_wifi_profiles(self):
        self._update_current_wifi_from_ui()
        save_wifi_profiles(self.wifi_profiles)
        QMessageBox.information(
            self,
            "Wi-Fi Profiles Saved",
            "Wi-Fi profiles have been saved successfully.",
        )

    def _on_new_time_profile(self):
        name, ok = QInputDialog.getText(
            self,
            "New Time Profile",
            "Enter a name for the new time profile:",
            text="time_profile",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self.time_profiles:
            QMessageBox.warning(
                self,
                "Duplicate Name",
                f"A time profile called '{name}' already exists.",
            )
            return

        self.time_profiles[name] = deepcopy(DEFAULT_TIME_PROFILE)
        self.current_time_profile_name = name
        self._populate_time_combo()
        self._load_time_into_ui(name)

    def _on_duplicate_time_profile(self):
        if not self.current_time_profile_name:
            return
        base_name = self.current_time_profile_name
        new_name = self._create_unique_name(f"{base_name}_copy", self.time_profiles)
        self.time_profiles[new_name] = deepcopy(self.time_profiles[base_name])
        self.current_time_profile_name = new_name
        self._populate_time_combo()
        self._load_time_into_ui(new_name)

    def _on_rename_time_profile(self):
        self._rename_profile(
            "time",
            self.time_profiles,
            "current_time_profile_name",
            self._populate_time_combo,
            self._load_time_into_ui,
        )

    def _on_delete_time_profile(self):
        if not self.current_time_profile_name:
            return
        if _is_default_profile("time", self.current_time_profile_name):
            QMessageBox.information(
                self,
                "Protected Profile",
                "The default time profile cannot be modified directly. Duplicate it to make changes.",
            )
            return
        if len(self.time_profiles) <= 1:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "At least one time profile must exist.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete Time Profile",
            f"Delete time profile '{self.current_time_profile_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.time_profiles.pop(self.current_time_profile_name, None)
        self.current_time_profile_name = next(iter(self.time_profiles.keys()))
        self._populate_time_combo()
        self._load_time_into_ui(self.current_time_profile_name)

    def _on_save_time_profiles(self):
        self._update_current_time_from_ui()
        save_time_profiles(self.time_profiles)
        QMessageBox.information(self, "Saved", "Time profiles have been saved.")

    def _on_new_demo_profile(self):
        name, ok = QInputDialog.getText(
            self,
            "New Demo Profile",
            "Enter a name for the new demo profile:",
            text="demo_profile",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self.demo_profiles:
            QMessageBox.warning(self, "Duplicate Name", f"A demo profile called '{name}' already exists.")
            return
        self.demo_profiles[name] = deepcopy(DEFAULT_DEMO_PROFILE)
        self.current_demo_profile_name = name
        self._populate_demo_combo()
        self._load_demo_into_ui(name)

    def _on_duplicate_demo_profile(self):
        if not self.current_demo_profile_name:
            return
        base_name = self.current_demo_profile_name
        new_name = self._create_unique_name(f"{base_name}_copy", self.demo_profiles)
        self.demo_profiles[new_name] = deepcopy(self.demo_profiles[base_name])
        self.current_demo_profile_name = new_name
        self._populate_demo_combo()
        self._load_demo_into_ui(new_name)

    def _on_rename_demo_profile(self):
        self._rename_profile(
            "demo",
            self.demo_profiles,
            "current_demo_profile_name",
            self._populate_demo_combo,
            self._load_demo_into_ui,
        )

    def _on_delete_demo_profile(self):
        if not self.current_demo_profile_name:
            return
        if _is_default_profile("demo", self.current_demo_profile_name):
            QMessageBox.information(
                self,
                "Protected Profile",
                "The default demo profile cannot be modified directly. Duplicate it to make changes.",
            )
            return
        if len(self.demo_profiles) <= 1:
            QMessageBox.warning(self, "Cannot Delete", "At least one demo profile must exist.")
            return
        reply = QMessageBox.question(
            self,
            "Delete Demo Profile",
            f"Delete demo profile '{self.current_demo_profile_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.demo_profiles.pop(self.current_demo_profile_name, None)
        self.current_demo_profile_name = next(iter(self.demo_profiles.keys()))
        self._populate_demo_combo()
        self._load_demo_into_ui(self.current_demo_profile_name)

    def _on_save_demo_profiles(self):
        self._update_current_demo_from_ui()
        save_demo_profiles(self.demo_profiles)
        QMessageBox.information(self, "Saved", "Demo profiles have been saved.")

    def _on_add_wifi_ap(self):
        self._add_wifi_row()

    def _on_remove_wifi_ap(self):
        selection = self.tbl_wifi.selectionModel()
        if not selection:
            return
        rows = sorted({idx.row() for idx in selection.selectedRows()}, reverse=True)
        for row in rows:
            self.tbl_wifi.removeRow(row)

    # ------------------------------------------------------------------
    # Actions table
    # ------------------------------------------------------------------
    def _on_add_action(self):
        row = self.tbl_actions.rowCount()
        self.tbl_actions.insertRow(row)
        self.tbl_actions.setItem(row, 0, QTableWidgetItem("new_action"))
        self.tbl_actions.setItem(row, 1, QTableWidgetItem(""))
        self._update_total_duration_label()

    def _on_remove_action(self):
        rows = self.tbl_actions.selectionModel().selectedRows()
        for idx in sorted(rows, key=lambda x: x.row(), reverse=True):
            self.tbl_actions.removeRow(idx.row())
        if rows:
            self._update_total_duration_label()

    def _on_browse_video(self):
        row = self.tbl_actions.currentRow()
        if row < 0:
            QMessageBox.information(
                self, "Select Action", "Please select an action row first."
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
        )
        if not path:
            return

        item = self.tbl_actions.item(row, 1)
        if item is None:
            item = QTableWidgetItem()
            self.tbl_actions.setItem(row, 1, item)
        item.setText(path)
        self._update_total_duration_label()

    def _on_browse_depth_save_location(self):
        current = self.edt_depth_save_location.text().strip()
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select depth recording folder",
            current or ".",
        )
        if directory:
            self.edt_depth_save_location.setText(directory)

    def _on_actions_table_item_changed(self, _item):
        self._update_total_duration_label()

    # ------------------------------------------------------------------
    # Experiment duration helpers
    # ------------------------------------------------------------------
    def _get_action_count(self) -> int:
        count = 0
        for row in range(self.tbl_actions.rowCount()):
            item = self.tbl_actions.item(row, 0)
            if item and item.text().strip():
                count += 1
        return count

    def _calculate_total_duration_seconds(self) -> float:
        begin = float(self.spn_begin_baseline.value())
        between = float(self.spn_between_baseline.value())
        end = float(self.spn_end_baseline.value())
        stop_time = float(self.spn_stop_time.value())
        action_time = float(self.spn_action_time.value())
        repetitions = max(0, int(self.spn_reps.value()))
        action_count = self._get_action_count()

        total = begin + end
        if action_count > 0 and repetitions > 0:
            total += action_count * repetitions * (stop_time + action_time)
            total += max(0, action_count - 1) * between
        return total

    @staticmethod
    def _format_seconds_to_hms(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _update_total_duration_label(self):
        if not hasattr(self, "lbl_total_duration"):
            return
        total_seconds = self._calculate_total_duration_seconds()
        self.lbl_total_duration.setText(self._format_seconds_to_hms(total_seconds))

    def _refresh_experiment_id_display(self):
        if hasattr(self, "edt_exp_id"):
            self.edt_exp_id.setText(self.session_experiment_id)

    def _refresh_participant_id_display(self):
        if not hasattr(self, "edt_participant_id"):
            return
        name = self.edt_name.text().strip() if hasattr(self, "edt_name") else ""
        gender = self.cmb_gender.currentText().strip() if hasattr(self, "cmb_gender") else ""
        age_button = self.age_button_group.checkedButton() if hasattr(self, "age_button_group") else None
        age_group = age_button.text().strip() if age_button is not None else "Blank"
        participant_id = generate_participant_id(name, age_group, gender)
        self.edt_participant_id.setText(participant_id)

    def _refresh_second_participant_id_display(self):
        if not hasattr(self, "edt_second_participant_id"):
            return
        name = self.edt_second_name.text().strip() if hasattr(self, "edt_second_name") else ""
        gender = self.cmb_second_gender.currentText().strip() if hasattr(self, "cmb_second_gender") else ""
        age_button = self.second_age_button_group.checkedButton() if hasattr(self, "second_age_button_group") else None
        age_group = age_button.text().strip() if age_button is not None else "Blank"
        participant_id = generate_participant_id(name, age_group, gender)
        self.edt_second_participant_id.setText(participant_id)

    # ------------------------------------------------------------------
    # Loading/saving data for each profile type
    # ------------------------------------------------------------------
    def _load_participant_into_ui(self, name: str):
        subject = self.participant_profiles.get(name)
        if not subject:
            subject = deepcopy(DEFAULT_PARTICIPANT_PROFILE)
            self.participant_profiles[name] = subject

        subject["experiment_id"] = self.session_experiment_id
        self.edt_name.setText(subject.get("name", ""))
        age_group = subject.get("age_group") or "Blank"
        if age_group not in self.age_group_options:
            age_group = "Blank"
        if age_group in self.age_group_buttons:
            self.age_group_buttons[age_group].setChecked(True)
        self._refresh_experiment_id_display()
        gender = subject.get("gender", "")
        self._ensure_gender_option(gender)
        self.cmb_gender.setCurrentText(gender)
        hand = (subject.get("dominant_hand") or "").strip()
        hand_idx = self.cmb_dominant_hand.findText(hand, Qt.MatchFixedString)
        if hand_idx < 0:
            hand_idx = 0
        self.cmb_dominant_hand.setCurrentIndex(hand_idx)
        try:
            self.spn_height.setValue(float(subject.get("height_cm", 0.0)))
        except (TypeError, ValueError):
            self.spn_height.setValue(0.0)
        try:
            self.spn_weight.setValue(float(subject.get("weight_kg", 0.0)))
        except (TypeError, ValueError):
            self.spn_weight.setValue(0.0)
        if hasattr(self, "txt_participant_description"):
            self.txt_participant_description.blockSignals(True)
            self.txt_participant_description.setPlainText(subject.get("description", ""))
            self.txt_participant_description.blockSignals(False)
        subject["participant_id"] = generate_participant_id(
            subject.get("name", ""), age_group, gender
        )
        subject["age_value"] = 0 if age_group == "Blank" else subject.get("age_value", 0)
        self._refresh_participant_id_display()
        has_second = bool(subject.get("has_second_participant", False))
        self.chk_has_second_participant.setChecked(has_second)
        second_age_group = subject.get("second_age_group") or "Blank"
        if second_age_group not in self.age_group_options:
            second_age_group = "Blank"
        if second_age_group in self.second_age_group_buttons:
            self.second_age_group_buttons[second_age_group].setChecked(True)
        second_gender = subject.get("second_gender", "")
        self._ensure_gender_option(second_gender)
        self.cmb_second_gender.setCurrentText(second_gender)
        hand_second = (subject.get("second_dominant_hand") or "").strip()
        hand_second_idx = self.cmb_second_dominant_hand.findText(hand_second, Qt.MatchFixedString)
        if hand_second_idx < 0:
            hand_second_idx = 0
        self.cmb_second_dominant_hand.setCurrentIndex(hand_second_idx)
        self.edt_second_name.setText(subject.get("second_name", ""))
        try:
            self.spn_second_height.setValue(float(subject.get("second_height_cm", 0.0)))
        except (TypeError, ValueError):
            self.spn_second_height.setValue(0.0)
        try:
            self.spn_second_weight.setValue(float(subject.get("second_weight_kg", 0.0)))
        except (TypeError, ValueError):
            self.spn_second_weight.setValue(0.0)
        if hasattr(self, "txt_second_description"):
            self.txt_second_description.blockSignals(True)
            self.txt_second_description.setPlainText(subject.get("second_description", ""))
            self.txt_second_description.blockSignals(False)
        subject["second_participant_id"] = generate_participant_id(
            subject.get("second_name", ""), second_age_group, second_gender
        )
        subject["second_age_value"] = (
            0 if second_age_group == "Blank" else subject.get("second_age_value", 0)
        )
        self._refresh_second_participant_id_display()
        self._set_profile_editable(
            "participant", not _is_default_profile("participant", name)
        )
        self._update_second_controls_state()

    def _load_experiment_into_ui(self, name: str):
        exp = self.experiment_profiles.get(name)
        if not exp:
            exp = deepcopy(DEFAULT_EXPERIMENT_PROFILE)
            self.experiment_profiles[name] = exp

        self.spn_begin_baseline.setValue(
            float(exp.get("beginning_baseline_recording", 15.0))
        )
        self.spn_between_baseline.setValue(
            float(exp.get("between_actions_baseline_recording", 10.0))
        )
        self.spn_end_baseline.setValue(
            float(exp.get("ending_baseline_recording", 20.0))
        )
        self.spn_stop_time.setValue(float(exp.get("stop_time", 6.0)))
        self.spn_preview_pause.setValue(float(exp.get("preview_pause", 2.0)))
        self.spn_action_time.setValue(float(exp.get("action_time", 4.0)))
        self.spn_reps.setValue(int(exp.get("each_action_repitition_times", 20)))
        self.edt_save_location.setText(exp.get("save_location", "./experiments/"))
        self._update_total_duration_label()
        self._set_profile_editable(
            "experiment", not _is_default_profile("experiment", name)
        )

    def _load_environment_into_ui(self, name: str):
        profile = self.environment_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_ENVIRONMENT_PROFILE)
            self.environment_profiles[name] = profile

        for spin, key in [
            (self.spn_room_length, "length_m"),
            (self.spn_room_width, "width_m"),
            (self.spn_room_height, "height_m"),
        ]:
            if spin is None:
                continue
            try:
                spin.blockSignals(True)
                spin.setValue(
                    float(profile.get(key, DEFAULT_ENVIRONMENT_PROFILE.get(key, 0.0)))
                )
            except Exception:
                spin.setValue(DEFAULT_ENVIRONMENT_PROFILE.get(key, 0.0))
            finally:
                spin.blockSignals(False)

        if hasattr(self, "txt_environment_description"):
            self.txt_environment_description.blockSignals(True)
            self.txt_environment_description.setPlainText(profile.get("description", ""))
            self.txt_environment_description.blockSignals(False)

        self._set_profile_editable(
            "environment", not _is_default_profile("environment", name)
        )

    def _load_actions_into_ui(self, name: str):
        actions = self.action_profiles.get(name)
        if not isinstance(actions, dict):
            actions = deepcopy(DEFAULT_ACTION_PROFILE)
            self.action_profiles[name] = actions

        action_rows = list(actions.items())
        table = self.tbl_actions
        model = table.model()

        table.setUpdatesEnabled(False)
        table.blockSignals(True)
        if model is not None:
            model.blockSignals(True)

        table.setRowCount(0)
        table.clearContents()
        for act_name, video in action_rows:
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(act_name))
            table.setItem(row, 1, QTableWidgetItem(video))

        self._ensure_action_table_capacity(MIN_ACTION_TABLE_ROWS)

        if model is not None:
            model.blockSignals(False)
        table.blockSignals(False)
        table.setUpdatesEnabled(True)
        table.scrollToTop()
        table.viewport().update()
        self._update_total_duration_label()
        self._set_profile_editable("actions", not _is_default_profile("actions", name))

    def _ensure_action_table_capacity(self, minimum_rows: int):
        """Ensure the actions table always has a visible set of blank rows."""
        table = self.tbl_actions
        current_rows = table.rowCount()
        if current_rows >= minimum_rows:
            return
        for row in range(current_rows, minimum_rows):
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(""))
            table.setItem(row, 1, QTableWidgetItem(""))

    def _load_camera_into_ui(self, name: str):
        profile = self.camera_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_CAMERA_PROFILE)
            self.camera_profiles[name] = profile
        if hasattr(self, "chk_use_webcam"):
            self.chk_use_webcam.blockSignals(True)
            self.chk_use_webcam.setChecked(
                _as_bool(profile.get("use_webcam", DEFAULT_CAMERA_PROFILE["use_webcam"]))
            )
            self.chk_use_webcam.blockSignals(False)
        if hasattr(self, "chk_hand_recognition"):
            self.chk_hand_recognition.blockSignals(True)
            self.chk_hand_recognition.setChecked(
                _as_bool(
                    profile.get(
                        "use_hand_recognition", DEFAULT_CAMERA_PROFILE["use_hand_recognition"]
                    )
                )
            )
            self.chk_hand_recognition.blockSignals(False)
        if hasattr(self, "cmb_hand_mode"):
            self.cmb_hand_mode.blockSignals(True)
            mode = _normalize_hand_mode(
                profile.get(
                    "hand_recognition_mode", DEFAULT_CAMERA_PROFILE["hand_recognition_mode"]
                )
            )
            idx = self.cmb_hand_mode.findData(mode)
            self.cmb_hand_mode.setCurrentIndex(idx if idx >= 0 else 0)
            self.cmb_hand_mode.blockSignals(False)
        combo = getattr(self, "cmb_hand_model_complexity", None)
        if combo is not None:
            combo.blockSignals(True)
            value = _normalize_model_complexity(
                profile.get(
                    "hand_model_complexity", DEFAULT_CAMERA_PROFILE["hand_model_complexity"]
                )
            )
            idx = combo.findData(value)
            if idx < 0:
                idx = 0
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        if hasattr(self, "btn_left_wrist_color"):
            self._apply_button_color(
                self.btn_left_wrist_color,
                profile.get(
                    "hand_left_wrist_color", DEFAULT_CAMERA_PROFILE["hand_left_wrist_color"]
                ),
            )
        if hasattr(self, "btn_right_wrist_color"):
            self._apply_button_color(
                self.btn_right_wrist_color,
                profile.get(
                    "hand_right_wrist_color", DEFAULT_CAMERA_PROFILE["hand_right_wrist_color"]
                ),
            )
        if hasattr(self, "spn_wrist_radius"):
            try:
                self.spn_wrist_radius.blockSignals(True)
                radius_value = int(
                    profile.get(
                        "hand_wrist_circle_radius",
                        DEFAULT_CAMERA_PROFILE["hand_wrist_circle_radius"],
                    )
                )
                if radius_value <= 0:
                    radius_value = DEFAULT_CAMERA_PROFILE["hand_wrist_circle_radius"]
                self.spn_wrist_radius.setValue(radius_value)
            except Exception:
                self.spn_wrist_radius.setValue(
                    int(DEFAULT_CAMERA_PROFILE["hand_wrist_circle_radius"])
                )
            finally:
                self.spn_wrist_radius.blockSignals(False)
        if hasattr(self, "lbl_hand_context"):
            self.lbl_hand_context.setText(f"Editing camera profile: {name}")
        if hasattr(self, "cmb_camera_device"):
            self._refresh_camera_device_list(force_refresh=False)
            device_value = str(
                profile.get("camera_device", DEFAULT_CAMERA_PROFILE["camera_device"])
            ).strip()
            self.cmb_camera_device.blockSignals(True)
            if device_value:
                idx = self.cmb_camera_device.findText(device_value)
                if idx < 0:
                    self.cmb_camera_device.addItem(device_value)
                    idx = self.cmb_camera_device.findText(device_value)
                self.cmb_camera_device.setCurrentIndex(max(idx, 0))
            self.cmb_camera_device.blockSignals(False)
        self._update_hand_controls_state()
        self._set_profile_editable("camera", not _is_default_profile("camera", name))

    def _load_depth_camera_into_ui(self, name: str):
        profile = self.depth_camera_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_DEPTH_CAMERA_PROFILE)
            self.depth_camera_profiles[name] = profile

        enabled = _as_bool(profile.get("enabled", DEFAULT_DEPTH_CAMERA_PROFILE["enabled"]))
        if hasattr(self, "chk_depth_enabled"):
            self.chk_depth_enabled.blockSignals(True)
            self.chk_depth_enabled.setChecked(enabled)
            self.chk_depth_enabled.blockSignals(False)

        if hasattr(self, "edt_depth_api_ip"):
            self.edt_depth_api_ip.setText(profile.get("api_ip", ""))

        if hasattr(self, "spn_depth_fps"):
            try:
                fps_value = int(profile.get("fps", DEFAULT_DEPTH_CAMERA_PROFILE["fps"]))
            except (TypeError, ValueError):
                fps_value = DEFAULT_DEPTH_CAMERA_PROFILE["fps"]
            self.spn_depth_fps.blockSignals(True)
            self.spn_depth_fps.setValue(fps_value)
            self.spn_depth_fps.blockSignals(False)

        def _set_combo_value(combo: QComboBox | None, value: str):
            if combo is None:
                return
            combo.blockSignals(True)
            if value and combo.findText(value) < 0:
                combo.addItem(value)
            idx = combo.findText(value)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

        _set_combo_value(
            getattr(self, "cmb_depth_rgb_res", None),
            _resolution_to_text(
                profile.get("rgb_resolution", DEFAULT_DEPTH_CAMERA_PROFILE["rgb_resolution"])
            ),
        )
        _set_combo_value(
            getattr(self, "cmb_depth_depth_res", None),
            _resolution_to_text(
                profile.get("depth_resolution", DEFAULT_DEPTH_CAMERA_PROFILE["depth_resolution"])
            ),
        )

        if hasattr(self, "chk_depth_save_raw"):
            self.chk_depth_save_raw.blockSignals(True)
            self.chk_depth_save_raw.setChecked(_as_bool(profile.get("save_raw_npz", True)))
            self.chk_depth_save_raw.blockSignals(False)
        if hasattr(self, "edt_depth_save_location"):
            self.edt_depth_save_location.setText(
                profile.get("save_location", DEFAULT_DEPTH_CAMERA_PROFILE["save_location"])
            )

        self._set_profile_editable("depth_camera", not _is_default_profile("depth_camera", name))
        self._restart_depth_status_timer()

    def _load_voice_into_ui(self, name: str):
        profile = self.voice_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_VOICE_PROFILE)
            self.voice_profiles[name] = profile
        self.chk_voice_enabled.setChecked(
            _as_bool(profile.get("use_voice_assistant", False))
        )
        self.chk_preview_voice_enabled.setChecked(
            _as_bool(profile.get("preview_actions_voice", True))
        )
        language = (profile.get("language") or "en").strip()
        combo = getattr(self, "cmb_voice_language", None)
        if combo is not None:
            idx = combo.findData(language)
            if idx < 0:
                idx = 0
            combo.setCurrentIndex(idx)
        self._set_profile_editable("voice", not _is_default_profile("voice", name))

    def _load_ui_into_ui(self, name: str):
        profile = self.ui_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_UI_PROFILE)
            self.ui_profiles[name] = profile

        for spin, key in [
            (self.spn_camera_frame_percent, "camera_frame_percent"),
            (self.spn_preview_frame_percent, "preview_frame_percent"),
            (self.spn_action_frame_percent, "action_frame_percent"),
            (self.spn_details_pane_percent, "details_pane_percent"),
            (self.spn_log_frame_percent, "log_frame_percent"),
        ]:
            if spin is not None:
                spin.blockSignals(True)
                spin.setValue(float(profile.get(key, DEFAULT_UI_PROFILE.get(key, 100.0))))
                spin.blockSignals(False)

        if hasattr(self, "chk_show_log"):
            self.chk_show_log.blockSignals(True)
            self.chk_show_log.setChecked(_as_bool(profile.get("show_log", True)))
            self.chk_show_log.blockSignals(False)

        if hasattr(self, "chk_start_fullscreen"):
            self.chk_start_fullscreen.blockSignals(True)
            self.chk_start_fullscreen.setChecked(
                _as_bool(profile.get("start_fullscreen", False))
            )
            self.chk_start_fullscreen.blockSignals(False)

        if hasattr(self, "cmb_transcript_position"):
            self.cmb_transcript_position.blockSignals(True)
            position = str(profile.get("transcript_position", "middle")).lower()
            idx = self.cmb_transcript_position.findData(position)
            if idx < 0:
                idx = 0
            self.cmb_transcript_position.setCurrentIndex(idx)
            self.cmb_transcript_position.blockSignals(False)

        if hasattr(self, "spn_transcript_font_size"):
            self.spn_transcript_font_size.blockSignals(True)
            self.spn_transcript_font_size.setValue(
                int(profile.get("transcript_font_size", DEFAULT_UI_PROFILE["transcript_font_size"]))
            )
            self.spn_transcript_font_size.blockSignals(False)
        if hasattr(self, "btn_transcript_color"):
            self._apply_button_color(
                self.btn_transcript_color,
                profile.get("transcript_font_color", DEFAULT_UI_PROFILE["transcript_font_color"]),
            )
        if hasattr(self, "spn_transcript_offset_x"):
            self.spn_transcript_offset_x.blockSignals(True)
            self.spn_transcript_offset_x.setValue(
                int(profile.get("transcript_offset_x", DEFAULT_UI_PROFILE["transcript_offset_x"]))
            )
            self.spn_transcript_offset_x.blockSignals(False)
        if hasattr(self, "spn_transcript_offset_y"):
            self.spn_transcript_offset_y.blockSignals(True)
            self.spn_transcript_offset_y.setValue(
                int(profile.get("transcript_offset_y", DEFAULT_UI_PROFILE["transcript_offset_y"]))
            )
            self.spn_transcript_offset_y.blockSignals(False)

        font_mappings = [
            ("count", self.cmb_count_font, self.spn_count_font_size, self.btn_count_color),
            ("action", self.cmb_action_font, self.spn_action_font_size, self.btn_action_color),
            ("time", self.cmb_time_font, self.spn_time_font_size, self.btn_time_color),
            (
                "remaining_time_value",
                self.cmb_remaining_time_value_font,
                self.spn_remaining_time_value_font_size,
                self.btn_remaining_time_value_color,
            ),
            (
                "elapsed_time_value",
                self.cmb_elapsed_time_value_font,
                self.spn_elapsed_time_value_font_size,
                self.btn_elapsed_time_value_color,
            ),
        ]

        for prefix, combo, size_spin, color_button in font_mappings:
            family_key = f"{prefix}_font_family"
            size_key = f"{prefix}_font_size"
            color_key = f"{prefix}_font_color"
            if combo is not None:
                combo.blockSignals(True)
                combo.setCurrentText(profile.get(family_key, DEFAULT_UI_PROFILE.get(family_key, "")))
                combo.blockSignals(False)
            if size_spin is not None:
                size_spin.blockSignals(True)
                size_spin.setValue(int(profile.get(size_key, DEFAULT_UI_PROFILE.get(size_key, 12))))
                size_spin.blockSignals(False)
            self._apply_button_color(
                color_button, profile.get(color_key, DEFAULT_UI_PROFILE.get(color_key, "#000000"))
            )

        if hasattr(self, "ui_status_edits"):
            for key, message in UI_STATUS_MESSAGE_FIELDS:
                line = self.ui_status_edits.get(key)
                if line is not None:
                    line.setText(profile.get(key, message))
        if hasattr(self, "spn_status_font_size"):
            self.spn_status_font_size.setValue(
                int(profile.get("status_font_size", DEFAULT_UI_PROFILE["status_font_size"]))
            )
        if hasattr(self, "btn_status_color"):
            self._apply_button_color(
                self.btn_status_color,
                profile.get("status_font_color", DEFAULT_UI_PROFILE["status_font_color"]),
            )
        if hasattr(self, "spn_status_offset_x"):
            self.spn_status_offset_x.setValue(
                int(profile.get("status_offset_x", DEFAULT_UI_PROFILE["status_offset_x"]))
            )
        if hasattr(self, "spn_status_offset_y"):
            self.spn_status_offset_y.setValue(
                int(profile.get("status_offset_y", DEFAULT_UI_PROFILE["status_offset_y"]))
            )

        if hasattr(self, "ui_guide_edits"):
            for key, message in UI_GUIDE_MESSAGE_FIELDS:
                line = self.ui_guide_edits.get(key)
                if line is not None:
                    line.setText(profile.get(key, message))

        if hasattr(self, "ui_text_controls"):
            for control in self.ui_text_controls:
                prefix = control["prefix"]
                control["text"].setText(
                    profile.get(f"{prefix}_text", DEFAULT_UI_PROFILE.get(f"{prefix}_text", ""))
                )
                control["font_size"].setValue(
                    int(profile.get(f"{prefix}_font_size", DEFAULT_UI_PROFILE.get(f"{prefix}_font_size", 12)))
                )
                self._apply_button_color(
                    control["color"],
                    profile.get(f"{prefix}_color", DEFAULT_UI_PROFILE.get(f"{prefix}_color", "#000000")),
                )
                control["offset_x"].setValue(
                    int(profile.get(f"{prefix}_offset_x", DEFAULT_UI_PROFILE.get(f"{prefix}_offset_x", 0)))
                )
                control["offset_y"].setValue(
                    int(profile.get(f"{prefix}_offset_y", DEFAULT_UI_PROFILE.get(f"{prefix}_offset_y", 0)))
                )

        if hasattr(self, "ui_value_controls"):
            for control in self.ui_value_controls:
                prefix = control["prefix"]
                control["font_size"].setValue(
                    int(profile.get(f"{prefix}_font_size", DEFAULT_UI_PROFILE.get(f"{prefix}_font_size", 12)))
                )
                self._apply_button_color(
                    control["color"],
                    profile.get(f"{prefix}_color", DEFAULT_UI_PROFILE.get(f"{prefix}_color", "#000000")),
                )
                control["offset_x"].setValue(
                    int(profile.get(f"{prefix}_offset_x", DEFAULT_UI_PROFILE.get(f"{prefix}_offset_x", 0)))
                )
                control["offset_y"].setValue(
                    int(profile.get(f"{prefix}_offset_y", DEFAULT_UI_PROFILE.get(f"{prefix}_offset_y", 0)))
                )

        if hasattr(self, "edt_log_placeholder_text"):
            self.edt_log_placeholder_text.setText(
                profile.get("log_placeholder_text", DEFAULT_UI_PROFILE["log_placeholder_text"])
            )
        if hasattr(self, "spn_log_font_size"):
            self.spn_log_font_size.setValue(
                int(profile.get("log_font_size", DEFAULT_UI_PROFILE["log_font_size"]))
            )
        if hasattr(self, "btn_log_color"):
            self._apply_button_color(
                self.btn_log_color,
                profile.get("log_font_color", DEFAULT_UI_PROFILE["log_font_color"]),
            )
        if hasattr(self, "spn_log_offset_x"):
            self.spn_log_offset_x.setValue(
                int(profile.get("log_offset_x", DEFAULT_UI_PROFILE["log_offset_x"]))
            )
        if hasattr(self, "spn_log_offset_y"):
            self.spn_log_offset_y.setValue(
                int(profile.get("log_offset_y", DEFAULT_UI_PROFILE["log_offset_y"]))
            )

        if hasattr(self, "ui_webcam_edits"):
            for key, message in UI_WEBCAM_MESSAGE_FIELDS:
                line = self.ui_webcam_edits.get(key)
                if line is not None:
                    line.setText(profile.get(key, message))
        if hasattr(self, "spn_webcam_font_size"):
            self.spn_webcam_font_size.setValue(
                int(profile.get("webcam_font_size", DEFAULT_UI_PROFILE["webcam_font_size"]))
            )
        if hasattr(self, "btn_webcam_color"):
            self._apply_button_color(
                self.btn_webcam_color,
                profile.get("webcam_font_color", DEFAULT_UI_PROFILE["webcam_font_color"]),
            )
        if hasattr(self, "spn_webcam_offset_x"):
            self.spn_webcam_offset_x.setValue(
                int(profile.get("webcam_offset_x", DEFAULT_UI_PROFILE["webcam_offset_x"]))
            )
        if hasattr(self, "spn_webcam_offset_y"):
            self.spn_webcam_offset_y.setValue(
                int(profile.get("webcam_offset_y", DEFAULT_UI_PROFILE["webcam_offset_y"]))
            )

        self._update_log_controls_state()
        self._set_profile_editable("ui", not _is_default_profile("ui", name))

    def _load_time_into_ui(self, name: str):
        profile = self.time_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_TIME_PROFILE)
            self.time_profiles[name] = profile

        if hasattr(self, "chk_use_time_server"):
            self.chk_use_time_server.blockSignals(True)
            self.chk_use_time_server.setChecked(
                _as_bool(profile.get("use_time_server", False))
            )
            self.chk_use_time_server.blockSignals(False)
        if hasattr(self, "cmb_time_server"):
            server = (profile.get("time_server") or "").strip()
            self.cmb_time_server.blockSignals(True)
            if server:
                idx = self.cmb_time_server.findText(server)
                if idx < 0:
                    self.cmb_time_server.addItem(server)
                    idx = self.cmb_time_server.findText(server)
                self.cmb_time_server.setCurrentIndex(max(idx, 0))
            self.cmb_time_server.blockSignals(False)

        self._start_time_sync()
        self._set_profile_editable("time", not _is_default_profile("time", name))

    def _load_demo_into_ui(self, name: str):
        profile = self.demo_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_DEMO_PROFILE)
            self.demo_profiles[name] = profile

        if hasattr(self, "spn_demo_capture_duration"):
            self.spn_demo_capture_duration.setValue(
                float(
                    profile.get(
                        "capture_duration_seconds",
                        DEFAULT_DEMO_PROFILE["capture_duration_seconds"],
                    )
                )
            )
        if hasattr(self, "cmb_demo_capture_mode"):
            mode = _normalize_demo_capture_mode(
                profile.get("demo_capture_mode", DEFAULT_DEMO_PROFILE["demo_capture_mode"])
            )
            idx = self.cmb_demo_capture_mode.findData(mode)
            self.cmb_demo_capture_mode.setCurrentIndex(idx if idx >= 0 else 0)
        if hasattr(self, "spn_demo_effective_samples"):
            self.spn_demo_effective_samples.setValue(
                int(
                    profile.get(
                        "effective_capture_samples",
                        DEFAULT_DEMO_PROFILE["effective_capture_samples"],
                    )
                )
            )
        if hasattr(self, "chk_demo_hampel_ratio_magnitude"):
            self.chk_demo_hampel_ratio_magnitude.setChecked(
                bool(
                    profile.get(
                        "apply_hampel_to_ratio_magnitude",
                        DEFAULT_DEMO_PROFILE["apply_hampel_to_ratio_magnitude"],
                    )
                )
            )
        if hasattr(self, "chk_demo_hampel_ratio_phase"):
            self.chk_demo_hampel_ratio_phase.setChecked(
                bool(
                    profile.get(
                        "apply_hampel_to_ratio_phase",
                        DEFAULT_DEMO_PROFILE["apply_hampel_to_ratio_phase"],
                    )
                )
            )
        if hasattr(self, "txt_demo_title"):
            self.txt_demo_title.setText(
                str(
                    profile.get(
                        "demo_title_text",
                        DEFAULT_DEMO_PROFILE["demo_title_text"],
                    )
                )
            )
        if hasattr(self, "txt_demo_university_logo_image_path"):
            self.txt_demo_university_logo_image_path.setText(
                str(
                    profile.get(
                        "university_logo_image_path",
                        profile.get(
                            "qr_code_image_path",
                            DEFAULT_DEMO_PROFILE["university_logo_image_path"],
                        ),
                    )
                )
            )
        if hasattr(self, "spn_demo_university_logo_image_size"):
            self.spn_demo_university_logo_image_size.setValue(
                max(
                    20,
                    min(
                        600,
                        int(
                            profile.get(
                                "university_logo_image_size_px",
                                profile.get(
                                    "qr_code_image_size_px",
                                    DEFAULT_DEMO_PROFILE["university_logo_image_size_px"],
                                ),
                            )
                        ),
                    ),
                )
            )
        if hasattr(self, "txt_demo_icassp_logo_image_path"):
            self.txt_demo_icassp_logo_image_path.setText(
                str(
                    profile.get(
                        "icassp_logo_image_path",
                        DEFAULT_DEMO_PROFILE["icassp_logo_image_path"],
                    )
                )
            )
        if hasattr(self, "txt_demo_website_url"):
            self.txt_demo_website_url.setText(
                str(
                    profile.get(
                        "website_url",
                        profile.get(
                            "qr_website_url",
                            DEFAULT_DEMO_PROFILE["website_url"],
                        ),
                    )
                )
            )
        if hasattr(self, "txt_demo_icassp_title"):
            self.txt_demo_icassp_title.setText(
                str(
                    profile.get(
                        "icassp_title_text",
                        "",
                    )
                )
            )
        if hasattr(self, "spn_demo_icassp_logo_text_vertical_gap"):
            self.spn_demo_icassp_logo_text_vertical_gap.setValue(
                int(
                    profile.get(
                        "icassp_logo_text_vertical_gap",
                        DEFAULT_DEMO_PROFILE["icassp_logo_text_vertical_gap"],
                    )
                )
            )
        if hasattr(self, "spn_demo_title_font_size"):
            self.spn_demo_title_font_size.setValue(
                max(
                    10,
                    min(
                        96,
                        int(
                            profile.get(
                                "demo_title_font_size_px",
                                DEFAULT_DEMO_PROFILE["demo_title_font_size_px"],
                            )
                        ),
                    ),
                )
            )
        if hasattr(self, "txt_demo_authors"):
            self.txt_demo_authors.setText(
                str(
                    profile.get(
                        "authors_text",
                        DEFAULT_DEMO_PROFILE["authors_text"],
                    )
                )
            )
        if hasattr(self, "spn_demo_authors_font_size"):
            self.spn_demo_authors_font_size.setValue(
                max(
                    8,
                    min(
                        72,
                        int(
                            profile.get(
                                "authors_font_size_px",
                                DEFAULT_DEMO_PROFILE["authors_font_size_px"],
                            )
                        ),
                    ),
                )
            )
        if hasattr(self, "txt_demo_university"):
            self.txt_demo_university.setText(
                str(
                    profile.get(
                        "university_text",
                        "",
                    )
                )
            )
        if hasattr(self, "txt_demo_wirlab"):
            self.txt_demo_wirlab.setText(
                str(
                    profile.get(
                        "wirlab_text",
                        DEFAULT_DEMO_PROFILE["wirlab_text"],
                    )
                )
            )
        if hasattr(self, "spn_demo_university_font_size"):
            self.spn_demo_university_font_size.setValue(
                max(
                    8,
                    min(
                        72,
                        int(
                            profile.get(
                                "university_font_size_px",
                                DEFAULT_DEMO_PROFILE["university_font_size_px"],
                            )
                        ),
                    ),
                )
            )
        if hasattr(self, "spn_demo_title_authors_vertical_gap"):
            self.spn_demo_title_authors_vertical_gap.setValue(
                int(
                    profile.get(
                        "title_authors_vertical_gap",
                        DEFAULT_DEMO_PROFILE["title_authors_vertical_gap"],
                    )
                )
            )
        if hasattr(self, "spn_demo_authors_university_vertical_gap"):
            self.spn_demo_authors_university_vertical_gap.setValue(
                int(
                    profile.get(
                        "authors_university_vertical_gap",
                        DEFAULT_DEMO_PROFILE["authors_university_vertical_gap"],
                    )
                )
            )
        if hasattr(self, "txt_demo_capture_guidance_title"):
            self.txt_demo_capture_guidance_title.setText(
                str(
                    profile.get(
                        "capture_guidance_title",
                        DEFAULT_DEMO_PROFILE["capture_guidance_title"],
                    )
                )
            )
        if hasattr(self, "txt_demo_capture_guidance_message"):
            self.txt_demo_capture_guidance_message.setText(
                str(
                    profile.get(
                        "capture_guidance_message",
                        DEFAULT_DEMO_PROFILE["capture_guidance_message"],
                    )
                )
            )
        if hasattr(self, "txt_demo_capture_guidance_left_label"):
            self.txt_demo_capture_guidance_left_label.setText(
                str(
                    profile.get(
                        "capture_guidance_video_left_label",
                        DEFAULT_DEMO_PROFILE["capture_guidance_video_left_label"],
                    )
                )
            )
        if hasattr(self, "txt_demo_capture_guidance_left_path"):
            self.txt_demo_capture_guidance_left_path.setText(
                str(
                    profile.get(
                        "capture_guidance_video_left_path",
                        DEFAULT_DEMO_PROFILE["capture_guidance_video_left_path"],
                    )
                )
            )
        if hasattr(self, "txt_demo_capture_guidance_right_label"):
            self.txt_demo_capture_guidance_right_label.setText(
                str(
                    profile.get(
                        "capture_guidance_video_right_label",
                        DEFAULT_DEMO_PROFILE["capture_guidance_video_right_label"],
                    )
                )
            )
        if hasattr(self, "txt_demo_capture_guidance_right_path"):
            self.txt_demo_capture_guidance_right_path.setText(
                str(
                    profile.get(
                        "capture_guidance_video_right_path",
                        DEFAULT_DEMO_PROFILE["capture_guidance_video_right_path"],
                    )
                )
            )
        if hasattr(self, "txt_demo_activity_class_names"):
            activity_class_names = profile.get(
                "activity_class_names",
                deepcopy(DEFAULT_DEMO_PROFILE["activity_class_names"]),
            )
            if isinstance(activity_class_names, list):
                value = ",".join(
                    str(item).strip() for item in activity_class_names if str(item).strip()
                )
            else:
                value = str(activity_class_names).strip()
            self.txt_demo_activity_class_names.setText(value)
        if hasattr(self, "txt_demo_subplot_settings"):
            subplot_settings = profile.get(
                "subplot_settings", deepcopy(DEFAULT_DEMO_PROFILE["subplot_settings"])
            )
            self.txt_demo_subplot_settings.setPlainText(
                json.dumps(subplot_settings, indent=2, ensure_ascii=False)
            )
        if hasattr(self, "txt_demo_dorf_plot_order"):
            order = profile.get("dorf_plot_order", DEFAULT_DEMO_PROFILE["dorf_plot_order"])
            if isinstance(order, list):
                order_text = ",".join(str(item).strip() for item in order if str(item).strip())
            else:
                order_text = str(order).strip()
            self.txt_demo_dorf_plot_order.setText(order_text)
        self._set_profile_editable("demo", not _is_default_profile("demo", name))

    def _update_hand_controls_state(self):
        camera_available = bool(
            hasattr(self, "chk_use_webcam") and self.chk_use_webcam.isChecked()
        )
        device_widgets = [
            getattr(self, "cmb_camera_device", None),
            getattr(self, "btn_refresh_cameras", None),
            getattr(self, "btn_preview_camera", None),
            getattr(self, "lbl_camera_preview", None),
        ]
        for widget in device_widgets:
            if widget is not None:
                widget.setEnabled(camera_available)
        if not camera_available:
            self._stop_camera_preview()
        if hasattr(self, "chk_hand_recognition"):
            self.chk_hand_recognition.setEnabled(camera_available)
            if not camera_available:
                self.chk_hand_recognition.blockSignals(True)
                self.chk_hand_recognition.setChecked(False)
                self.chk_hand_recognition.blockSignals(False)
        hand_enabled = camera_available and bool(
            hasattr(self, "chk_hand_recognition")
            and self.chk_hand_recognition.isChecked()
        )
        if hasattr(self, "cmb_hand_mode"):
            self.cmb_hand_mode.setEnabled(hand_enabled)
        if hasattr(self, "cmb_hand_model_complexity"):
            self.cmb_hand_model_complexity.setEnabled(hand_enabled)
        for widget_name in [
            "btn_left_wrist_color",
            "btn_right_wrist_color",
            "spn_wrist_radius",
        ]:
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(hand_enabled)

    def _on_time_server_toggled(self, _checked: bool):
        self._start_time_sync()
        self._update_time_preview()

    def _on_time_server_changed(self, _value: str):
        self._start_time_sync()

    def _on_time_sync_clicked(self):
        self._start_time_sync(force=True)

    def _start_time_sync(self, *, force: bool = False):
        if not hasattr(self, "chk_use_time_server") or not self.chk_use_time_server.isChecked():
            self._time_sync_result = None
            self._time_sync_error = None
            return
        if not force and self._time_sync_thread is not None and self._time_sync_thread.is_alive():
            return

        server = ""
        if hasattr(self, "cmb_time_server"):
            server = self.cmb_time_server.currentText().strip()

        self._time_sync_result = None
        self._time_sync_error = None
        if hasattr(self, "lbl_time_status"):
            self.lbl_time_status.setText("Syncing...")

        def _sync():
            try:
                servers = [server] if server else DEFAULT_TIME_SERVERS
                best = best_startup_sync(servers, attempts_per_server=3, timeout_s=1.0)
                self._time_sync_result = best
            except Exception as exc:
                self._time_sync_error = str(exc)

        self._time_sync_thread = threading.Thread(
            target=_sync, name="TimeSyncThread", daemon=True
        )
        self._time_sync_thread.start()

    def _update_time_preview(self):
        if not hasattr(self, "lbl_os_time"):
            return

        os_ns = time.time_ns()
        self.lbl_os_time.setText(fmt_local_from_ns(os_ns))

        if not hasattr(self, "chk_use_time_server") or not self.chk_use_time_server.isChecked():
            if hasattr(self, "lbl_time_status"):
                self.lbl_time_status.setText("Using OS time")
            if hasattr(self, "lbl_server_time"):
                self.lbl_server_time.setText("—")
            if hasattr(self, "lbl_time_offset"):
                self.lbl_time_offset.setText("—")
            return

        if self._time_sync_result:
            server = self._time_sync_result.get("server", "time server")
            offset_ns = self._time_sync_result.get("offset_ns", 0)
            ref_utc_ns = (
                self._time_sync_result.get("t4_unix_ns", 0) + offset_ns
            )
            mono_ref_ns = self._time_sync_result.get("mono_at_t4_ns", time.monotonic_ns())
            server_ns = ref_utc_ns + (time.monotonic_ns() - mono_ref_ns)
            self.lbl_time_status.setText(f"Synced with {server}")
            self.lbl_server_time.setText(fmt_utc_from_ns(server_ns))
            self.lbl_time_offset.setText(f"{offset_ns / 1e6:+.3f} ms")
            return

        if self._time_sync_error:
            self.lbl_time_status.setText(f"Sync failed: {self._time_sync_error}")
            self.lbl_server_time.setText("—")
            self.lbl_time_offset.setText("—")
            return

        self.lbl_time_status.setText("Syncing...")

    def _on_camera_checkbox_toggled(self, checked: bool):
        if not checked and hasattr(self, "chk_hand_recognition"):
            self.chk_hand_recognition.setChecked(False)
        self._update_hand_controls_state()
        self._update_camera_controls_state()

    def _on_depth_checkbox_toggled(self, checked: bool):
        if not checked:
            self._set_depth_status_message("Depth camera disabled.")
        self._restart_depth_status_timer()

    def _refresh_camera_device_list(self, *, force_refresh: bool = True):
        if self.cmb_camera_device is None:
            return
        current_value = self.cmb_camera_device.currentText().strip()
        if not force_refresh and self.cmb_camera_device.count() > 0:
            desired_idx = self.cmb_camera_device.findText(current_value)
            if desired_idx >= 0:
                self.cmb_camera_device.setCurrentIndex(desired_idx)
                return
        devices = self._probe_camera_devices()
        self.cmb_camera_device.blockSignals(True)
        self.cmb_camera_device.clear()
        if devices:
            for device_idx in devices:
                self.cmb_camera_device.addItem(str(device_idx))
        if current_value and self.cmb_camera_device.findText(current_value) < 0:
            self.cmb_camera_device.addItem(current_value)
        if self.cmb_camera_device.count() == 0:
            self.cmb_camera_device.addItem(str(DEFAULT_CAMERA_PROFILE["camera_device"]))
        idx = self.cmb_camera_device.findText(current_value)
        self.cmb_camera_device.setCurrentIndex(idx if idx >= 0 else 0)
        self.cmb_camera_device.blockSignals(False)

    def _probe_camera_devices(self, max_devices: int = 10) -> list[int]:
        if cv2 is None:
            return []
        devices: list[int] = []
        for device_idx in range(max_devices):
            cap = cv2.VideoCapture(device_idx)
            if cap is not None and cap.isOpened():
                devices.append(device_idx)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        return devices

    def _on_toggle_camera_preview(self):
        if self._camera_preview_timer is not None:
            self._stop_camera_preview()
            return
        self._start_camera_preview()

    def _start_camera_preview(self):
        if cv2 is None:
            self._set_camera_preview_message("OpenCV is not available")
            return
        if not self.chk_use_webcam.isChecked():
            self._set_camera_preview_message("Enable the camera to preview")
            return
        device_text = self.cmb_camera_device.currentText().strip()
        device_source: int | str
        try:
            device_source = int(device_text)
        except ValueError:
            device_source = device_text or DEFAULT_CAMERA_PROFILE["camera_device"]
        self._camera_preview_capture = cv2.VideoCapture(device_source)
        if (
            self._camera_preview_capture is None
            or not self._camera_preview_capture.isOpened()
        ):
            self._set_camera_preview_message("Unable to open camera device")
            self._stop_camera_preview()
            return
        self.btn_preview_camera.setText("Stop Preview")
        self._camera_preview_timer = QTimer(self)
        self._camera_preview_timer.timeout.connect(self._update_camera_preview_frame)
        self._camera_preview_timer.start(100)
        self._update_camera_preview_frame()

    def _update_camera_preview_frame(self):
        if (
            self._camera_preview_capture is None
            or cv2 is None
            or self.lbl_camera_preview is None
        ):
            return
        ret, frame = self._camera_preview_capture.read()
        if not ret or frame is None:
            return
        try:
            mirrored = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
        except Exception:
            return
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        scaled = image.scaled(
            self.lbl_camera_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_camera_preview.setPixmap(QPixmap.fromImage(scaled))
        self.lbl_camera_preview.setText("")

    def _stop_camera_preview(self):
        if self._camera_preview_timer is not None:
            try:
                self._camera_preview_timer.stop()
            except Exception:
                pass
            self._camera_preview_timer.deleteLater()
            self._camera_preview_timer = None
        if self._camera_preview_capture is not None:
            try:
                self._camera_preview_capture.release()
            except Exception:
                pass
            self._camera_preview_capture = None
        if hasattr(self, "btn_preview_camera"):
            self.btn_preview_camera.setText("Preview")
        self._set_camera_preview_message("Camera preview disabled")

    def _set_camera_preview_message(self, message: str):
        if self.lbl_camera_preview is None:
            return
        self.lbl_camera_preview.setPixmap(QPixmap())
        self.lbl_camera_preview.setText(message)

    # ------------------------------------------------------------------
    # Depth camera status handling
    # ------------------------------------------------------------------
    def _normalize_depth_api_base(self, ip_text: str) -> str:
        ip = (ip_text or "").strip()
        if not ip:
            return ""
        if not ip.startswith(("http://", "https://")):
            ip = f"http://{ip}"
        scheme, _, remainder = ip.partition("://")
        host_and_path = remainder.split("/", 1)
        host = host_and_path[0]
        if ":" not in host:
            host = f"{host}:5000"
        return f"{scheme}://{host}"

    def _build_depth_status_url(self) -> str:
        base = self._normalize_depth_api_base(self.edt_depth_api_ip.text())
        if not base:
            return ""
        return f"{base}/api/status"

    def _format_depth_status(self, payload: dict | None) -> str:
        if not isinstance(payload, dict):
            return str(payload) if payload is not None else "No status payload"

        lines = []
        summary_fields = []
        for key in ("status", "recorder_state", "recording"):
            if key in payload:
                summary_fields.append(f"{key}: {payload.get(key)}")
        if "queue_depth" in payload:
            summary_fields.append(f"queue depth: {payload.get('queue_depth')}")
        if "rgb_resolution" in payload:
            summary_fields.append(f"RGB: {payload.get('rgb_resolution')}")
        if "depth_resolution" in payload:
            summary_fields.append(f"Depth: {payload.get('depth_resolution')}")
        if "fps" in payload:
            summary_fields.append(f"FPS: {payload.get('fps')}")
        if summary_fields:
            lines.append("Summary:")
            for field in summary_fields:
                lines.append(f" - {field}")
            lines.append("")

        try:
            lines.append("Raw payload:")
            lines.append(json.dumps(payload, indent=2))
        except TypeError:
            lines.append(str(payload))
        return "\n".join(lines)

    def _set_depth_status_message(self, message: str, *, ok: bool | None = None):
        if hasattr(self, "lbl_depth_status"):
            status_text = "Status OK" if ok or ok is None else "Status error"
            self.lbl_depth_status.setText(status_text)
            if ok is False:
                self.lbl_depth_status.setStyleSheet("color: red;")
            elif ok is True:
                self.lbl_depth_status.setStyleSheet("color: green;")
            else:
                self.lbl_depth_status.setStyleSheet("")
        if hasattr(self, "txt_depth_status"):
            self.txt_depth_status.setPlainText(message or "")

    def _refresh_depth_status(self, force: bool = False):
        if self._depth_status_refreshing:
            return
        if not force and (not hasattr(self, "chk_depth_enabled") or not self.chk_depth_enabled.isChecked()):
            return
        url = self._build_depth_status_url()
        if not url:
            self._set_depth_status_message("Enter the depth camera API IP/host to poll status.")
            return

        self._depth_status_refreshing = True
        try:
            response = requests.get(url, timeout=2)
            ok = response.ok
            try:
                payload = response.json()
            except ValueError:
                payload = response.text.strip()
            message = self._format_depth_status(payload)
            self._set_depth_status_message(message, ok=ok)
        except requests.RequestException as exc:
            self._set_depth_status_message(f"Status error: {exc}", ok=False)
        finally:
            self._depth_status_refreshing = False

    def _restart_depth_status_timer(self):
        self._stop_depth_status_timer()
        if hasattr(self, "chk_depth_enabled") and self.chk_depth_enabled.isChecked():
            self._start_depth_status_timer()

    def _start_depth_status_timer(self):
        self._refresh_depth_status(force=True)
        self._depth_status_timer = QTimer(self)
        self._depth_status_timer.timeout.connect(self._refresh_depth_status)
        self._depth_status_timer.start(1000)

    def _stop_depth_status_timer(self):
        if self._depth_status_timer is not None:
            try:
                self._depth_status_timer.stop()
            except Exception:
                pass
            self._depth_status_timer.deleteLater()
            self._depth_status_timer = None

    def _add_wifi_row(self, ap: dict | None = None):
        ap = deepcopy(DEFAULT_WIFI_AP if ap is None else ap)
        row = self.tbl_wifi.rowCount()
        self.tbl_wifi.insertRow(row)

        order_spin = QSpinBox(self.tbl_wifi)
        order_spin.setRange(-1_000_000_000, 1_000_000_000)
        order_spin.setValue(int(ap.get("order", row + 1)))
        self.tbl_wifi.setCellWidget(row, 0, order_spin)

        self.tbl_wifi.setCellWidget(row, 1, QLineEdit(ap.get("name", "")))

        begin_combo = self._create_script_profile_combo(ap.get("framework", ""))
        end_combo = self._create_script_type_combo(
            begin_combo, ap.get("type", "")
        )
        begin_combo.currentTextChanged.connect(
            lambda value, combo=end_combo: self._populate_script_type_combo(
                combo, value, combo.currentText().strip()
            )
        )
        self.tbl_wifi.setCellWidget(row, 2, begin_combo)
        self.tbl_wifi.setCellWidget(row, 3, end_combo)
        self.tbl_wifi.setCellWidget(row, 4, QLineEdit(ap.get("ssid", "")))
        self.tbl_wifi.setCellWidget(row, 5, QLineEdit(ap.get("password", "")))
        self.tbl_wifi.setCellWidget(row, 6, QLineEdit(ap.get("router_ssh_ip", "")))
        self.tbl_wifi.setCellWidget(
            row, 7, QLineEdit(ap.get("router_ssh_username", ""))
        )
        self.tbl_wifi.setCellWidget(
            row, 8, QLineEdit(ap.get("router_ssh_password", ""))
        )
        ssh_key_edit = self._create_script_line_edit(ap.get("ssh_key_address", ""))
        self.tbl_wifi.setCellWidget(row, 9, ssh_key_edit)

        freq_combo = QComboBox(self.tbl_wifi)
        for freq in FREQUENCY_CHANNELS.keys():
            freq_combo.addItem(freq)
        freq = ap.get("frequency", "2.4 GHz")
        idx = freq_combo.findText(freq)
        freq_combo.blockSignals(True)
        freq_combo.setCurrentIndex(max(idx, 0))
        freq_combo.blockSignals(False)
        self.tbl_wifi.setCellWidget(row, 10, freq_combo)

        channel_combo = QComboBox(self.tbl_wifi)
        self.tbl_wifi.setCellWidget(row, 11, channel_combo)

        bandwidth_combo = QComboBox(self.tbl_wifi)
        self.tbl_wifi.setCellWidget(row, 12, bandwidth_combo)

        self._populate_wifi_frequency_dependents(
            row,
            freq_combo.currentText(),
            selected_channel=ap.get("channel"),
            selected_bandwidth=ap.get("bandwidth"),
        )

        freq_combo.currentTextChanged.connect(
            lambda value, combo=freq_combo: self._on_wifi_frequency_changed(
                combo, value
            )
        )

        self.tbl_wifi.setCellWidget(
            row, 13, QLineEdit(ap.get("transmitter_macs", ""))
        )

        remote_dir = ap.get(
            "init_test_save_directory", DEFAULT_WIFI_PROFILE["init_test_save_directory"]
        )
        self.tbl_wifi.setCellWidget(row, 14, QLineEdit(remote_dir))

        ethernet_checkbox = QCheckBox(self.tbl_wifi)
        ethernet_checkbox.setChecked(_as_bool(ap.get("use_ethernet", False)))
        ethernet_checkbox.setProperty("ethernet_checkbox", True)

        # Center the checkbox in the table cell using a layout since QCheckBox
        # itself does not support setAlignment
        ethernet_widget = QWidget(self.tbl_wifi)
        ethernet_layout = QHBoxLayout(ethernet_widget)
        ethernet_layout.setContentsMargins(0, 0, 0, 0)
        ethernet_layout.setAlignment(Qt.AlignCenter)
        ethernet_layout.addWidget(ethernet_checkbox)

        self.tbl_wifi.setCellWidget(row, 15, ethernet_widget)

        download_combo = QComboBox(self.tbl_wifi)
        download_combo.addItems(["SFTP", "FTP"])
        idx = download_combo.findText(ap.get("download_mode", "SFTP"))
        download_combo.setCurrentIndex(max(idx, 0))
        self.tbl_wifi.setCellWidget(row, 16, download_combo)

    def _populate_wifi_frequency_dependents(
        self,
        row: int,
        frequency: str,
        selected_channel: str | None = None,
        selected_bandwidth: str | None = None,
    ):
        channel_combo = self.tbl_wifi.cellWidget(row, 11)
        bandwidth_combo = self.tbl_wifi.cellWidget(row, 12)
        if isinstance(channel_combo, QComboBox):
            channel_combo.blockSignals(True)
            channel_combo.clear()
            for channel in FREQUENCY_CHANNELS.get(frequency, []):
                channel_combo.addItem(channel)
            selected_channel = str(selected_channel) if selected_channel else ""
            idx = channel_combo.findText(selected_channel)
            channel_combo.setCurrentIndex(idx if idx >= 0 else 0)
            channel_combo.blockSignals(False)
        if isinstance(bandwidth_combo, QComboBox):
            bandwidth_combo.blockSignals(True)
            bandwidth_combo.clear()
            for bandwidth in FREQUENCY_BANDWIDTHS.get(frequency, []):
                bandwidth_combo.addItem(bandwidth)
            selected_bandwidth = (
                str(selected_bandwidth) if selected_bandwidth else ""
            )
            idx = bandwidth_combo.findText(selected_bandwidth)
            bandwidth_combo.setCurrentIndex(idx if idx >= 0 else 0)
            bandwidth_combo.blockSignals(False)

    def _on_wifi_frequency_changed(self, frequency_widget: QWidget, frequency: str):
        row = self.tbl_wifi.indexAt(frequency_widget.pos()).row()
        if row < 0:
            return
        self._populate_wifi_frequency_dependents(row, frequency)

    def _apply_delete_prev_pcap_rule(self, scenario_value: str):
        if not hasattr(self, "chk_delete_prev_pcap"):
            return

        is_scenario_two = str(scenario_value).lower() == "scenario_2"
        if is_scenario_two:
            self.chk_delete_prev_pcap.setChecked(False)
        self.chk_delete_prev_pcap.setEnabled(not is_scenario_two)

    def _on_csi_scenario_changed(self):
        scenario_value = self.cmb_csi_scenario.currentData()
        self._apply_delete_prev_pcap_rule(scenario_value)

    def _load_wifi_into_ui(self, name: str):
        profile = self.wifi_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_WIFI_PROFILE)
            self.wifi_profiles[name] = profile

        duration = profile.get("init_test_duration", DEFAULT_WIFI_PROFILE["init_test_duration"])
        save_dir = profile.get(
            "init_test_save_directory", DEFAULT_WIFI_PROFILE["init_test_save_directory"]
        )
        scenario = profile.get(
            "csi_capture_scenario", DEFAULT_WIFI_PROFILE["csi_capture_scenario"]
        )
        delete_prev_pcap = profile.get(
            "delete_prev_pcap", DEFAULT_WIFI_PROFILE["delete_prev_pcap"]
        )
        pre_action_duration = profile.get(
            "pre_action_capture_duration",
            DEFAULT_WIFI_PROFILE["pre_action_capture_duration"],
        )
        post_action_duration = profile.get(
            "post_action_capture_duration",
            DEFAULT_WIFI_PROFILE["post_action_capture_duration"],
        )
        count_packets = profile.get(
            "count_packets", DEFAULT_WIFI_PROFILE["count_packets"]
        )
        reboot_after_summary = profile.get(
            "reboot_after_summary", DEFAULT_WIFI_PROFILE["reboot_after_summary"]
        )
        if hasattr(self, "spn_init_test_duration"):
            self.spn_init_test_duration.setValue(float(duration))
        if hasattr(self, "txt_init_save_directory"):
            self.txt_init_save_directory.setText(save_dir)
        if hasattr(self, "cmb_csi_scenario"):
            idx = self.cmb_csi_scenario.findData(str(scenario))
            self.cmb_csi_scenario.setCurrentIndex(max(idx, 0))
        if hasattr(self, "chk_delete_prev_pcap"):
            self.chk_delete_prev_pcap.setChecked(bool(delete_prev_pcap))
        if hasattr(self, "cmb_csi_scenario"):
            self._apply_delete_prev_pcap_rule(self.cmb_csi_scenario.currentData())
        if hasattr(self, "spn_pre_action_capture"):
            self.spn_pre_action_capture.setValue(float(pre_action_duration))
        if hasattr(self, "spn_post_action_capture"):
            self.spn_post_action_capture.setValue(float(post_action_duration))
        if hasattr(self, "chk_count_packets"):
            self.chk_count_packets.setChecked(bool(count_packets))
        if hasattr(self, "chk_reboot_after_summary"):
            self.chk_reboot_after_summary.setChecked(bool(reboot_after_summary))

        aps = profile.get("access_points", []) if isinstance(profile, dict) else []
        self.tbl_wifi.blockSignals(True)
        self.tbl_wifi.setRowCount(0)
        for ap in aps:
            self._add_wifi_row(ap)
        if not aps:
            self._add_wifi_row()
        self.tbl_wifi.blockSignals(False)

        self._set_profile_editable("wifi", not _is_default_profile("wifi", name))

    def _update_current_participant_from_ui(self):
        name = self.current_participant_name
        if not name:
            return
        if _is_default_profile("participant", name):
            return
        subject = self.participant_profiles.get(name)
        if not subject:
            subject = deepcopy(DEFAULT_PARTICIPANT_PROFILE)
            self.participant_profiles[name] = subject

        subject["name"] = self.edt_name.text().strip()
        age_button = self.age_button_group.checkedButton()
        subject["age_group"] = (
            age_button.text().strip()
            if age_button is not None
            else "Blank"
        )
        subject["experiment_id"] = self.session_experiment_id
        gender = self.cmb_gender.currentText().strip()
        subject["gender"] = gender
        self._ensure_gender_option(gender)
        subject["dominant_hand"] = self.cmb_dominant_hand.currentText().strip()
        subject["height_cm"] = float(self.spn_height.value())
        subject["weight_kg"] = float(self.spn_weight.value())
        if hasattr(self, "txt_participant_description"):
            subject["description"] = self.txt_participant_description.toPlainText().strip()
        subject["age_value"] = 0 if subject["age_group"] == "Blank" else subject.get("age_value", 0)
        subject["participant_id"] = generate_participant_id(
            subject.get("name", ""), subject.get("age_group", ""), gender
        )
        has_second = bool(self.chk_has_second_participant.isChecked()) if hasattr(self, "chk_has_second_participant") else False
        subject["has_second_participant"] = has_second
        second_age_button = self.second_age_button_group.checkedButton() if hasattr(self, "second_age_button_group") else None
        subject["second_age_group"] = (
            second_age_button.text().strip()
            if second_age_button is not None
            else "Blank"
        )
        subject["second_age_value"] = (
            0 if subject["second_age_group"] == "Blank" else subject.get("second_age_value", 0)
        )
        second_gender = self.cmb_second_gender.currentText().strip() if hasattr(self, "cmb_second_gender") else ""
        subject["second_gender"] = second_gender
        self._ensure_gender_option(second_gender)
        subject["second_name"] = self.edt_second_name.text().strip() if hasattr(self, "edt_second_name") else ""
        subject["second_dominant_hand"] = self.cmb_second_dominant_hand.currentText().strip() if hasattr(self, "cmb_second_dominant_hand") else ""
        subject["second_height_cm"] = float(self.spn_second_height.value()) if hasattr(self, "spn_second_height") else 0.0
        subject["second_weight_kg"] = float(self.spn_second_weight.value()) if hasattr(self, "spn_second_weight") else 0.0
        if hasattr(self, "txt_second_description"):
            subject["second_description"] = self.txt_second_description.toPlainText().strip()
        subject["second_participant_id"] = generate_participant_id(
            subject.get("second_name", ""),
            subject.get("second_age_group", ""),
            subject.get("second_gender", ""),
        )
        if not has_second:
            subject["second_name"] = ""
            subject["second_age_group"] = "Blank"
            subject["second_age_value"] = 0
            subject["second_gender"] = ""
            subject["second_dominant_hand"] = ""
            subject["second_height_cm"] = 0.0
            subject["second_weight_kg"] = 0.0
            subject["second_description"] = ""
            subject["second_participant_id"] = generate_participant_id("", "Blank", "")
        self._refresh_participant_id_display()
        self._refresh_second_participant_id_display()

    def _update_current_experiment_from_ui(self):
        name = self.current_experiment_name
        if not name:
            return
        if _is_default_profile("experiment", name):
            return
        exp = self.experiment_profiles.get(name)
        if not exp:
            exp = deepcopy(DEFAULT_EXPERIMENT_PROFILE)
            self.experiment_profiles[name] = exp

        exp["beginning_baseline_recording"] = float(self.spn_begin_baseline.value())
        exp["between_actions_baseline_recording"] = float(
            self.spn_between_baseline.value()
        )
        exp["ending_baseline_recording"] = float(self.spn_end_baseline.value())
        exp["stop_time"] = float(self.spn_stop_time.value())
        exp["preview_pause"] = float(self.spn_preview_pause.value())
        exp["action_time"] = float(self.spn_action_time.value())
        exp["each_action_repitition_times"] = int(self.spn_reps.value())
        exp["save_location"] = (
            self.edt_save_location.text().strip() or "./experiments/"
        )

    def _update_current_environment_from_ui(self):
        name = self.current_environment_name
        if not name:
            return
        if _is_default_profile("environment", name):
            return
        profile = self.environment_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_ENVIRONMENT_PROFILE)
            self.environment_profiles[name] = profile

        for spin, key in [
            (self.spn_room_length, "length_m"),
            (self.spn_room_width, "width_m"),
            (self.spn_room_height, "height_m"),
        ]:
            if spin is not None:
                profile[key] = float(spin.value())

        if hasattr(self, "txt_environment_description"):
            profile["description"] = self.txt_environment_description.toPlainText().strip()

    def _update_current_actions_from_ui(self):
        name = self.current_action_profile_name
        if not name:
            return
        if _is_default_profile("actions", name):
            return

        # Ensure any in-progress edits are finalized before reading values
        try:
            if self.tbl_actions.state() == QAbstractItemView.EditingState:
                self.tbl_actions.closePersistentEditor(self.tbl_actions.currentItem())
        except Exception:
            pass

        actions = {}
        for row in range(self.tbl_actions.rowCount()):
            item_name = self.tbl_actions.item(row, 0)
            item_video = self.tbl_actions.item(row, 1)
            act_name = (item_name.text().strip() if item_name else "")
            video = item_video.text().strip() if item_video else ""
            if act_name:
                actions[act_name] = video
        self.action_profiles[name] = filter_blank_actions(actions)

    def _update_current_camera_from_ui(self):
        name = self.current_camera_profile_name
        if not name:
            return
        if _is_default_profile("camera", name):
            return
        profile = self.camera_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_CAMERA_PROFILE)
            self.camera_profiles[name] = profile
        profile["use_webcam"] = bool(self.chk_use_webcam.isChecked())
        device_value = self.cmb_camera_device.currentText().strip()
        profile["camera_device"] = device_value or DEFAULT_CAMERA_PROFILE["camera_device"]
        profile["use_hand_recognition"] = bool(
            self.chk_hand_recognition.isChecked()
        )
        profile["hand_recognition_mode"] = _normalize_hand_mode(
            self.cmb_hand_mode.currentData()
        )
        profile["hand_model_complexity"] = _normalize_model_complexity(
            self.cmb_hand_model_complexity.currentData()
        )
        left_button = getattr(self, "btn_left_wrist_color", None)
        right_button = getattr(self, "btn_right_wrist_color", None)
        profile["hand_left_wrist_color"] = _normalize_hex_color(
            left_button.property("color_hex") if left_button is not None else None,
            DEFAULT_CAMERA_PROFILE["hand_left_wrist_color"],
        )
        profile["hand_right_wrist_color"] = _normalize_hex_color(
            right_button.property("color_hex") if right_button is not None else None,
            DEFAULT_CAMERA_PROFILE["hand_right_wrist_color"],
        )
        radius_spin = getattr(self, "spn_wrist_radius", None)
        try:
            profile["hand_wrist_circle_radius"] = (
                int(radius_spin.value()) if radius_spin is not None else 0
            )
        except Exception:
            profile["hand_wrist_circle_radius"] = DEFAULT_CAMERA_PROFILE[
                "hand_wrist_circle_radius"
            ]
        if profile["hand_wrist_circle_radius"] <= 0:
            profile["hand_wrist_circle_radius"] = DEFAULT_CAMERA_PROFILE[
                "hand_wrist_circle_radius"
            ]

    def _update_current_depth_camera_from_ui(self):
        name = self.current_depth_camera_profile_name
        if not name:
            return
        if _is_default_profile("depth_camera", name):
            return
        profile = self.depth_camera_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_DEPTH_CAMERA_PROFILE)
            self.depth_camera_profiles[name] = profile
        profile["enabled"] = bool(self.chk_depth_enabled.isChecked())
        profile["api_ip"] = self.edt_depth_api_ip.text().strip()
        profile["save_raw_npz"] = bool(self.chk_depth_save_raw.isChecked())
        profile["save_location"] = (
            self.edt_depth_save_location.text().strip()
            or DEFAULT_DEPTH_CAMERA_PROFILE["save_location"]
        )
        try:
            profile["fps"] = int(self.spn_depth_fps.value())
        except Exception:
            profile["fps"] = DEFAULT_DEPTH_CAMERA_PROFILE["fps"]
        profile["rgb_resolution"] = _parse_resolution(
            self.cmb_depth_rgb_res.currentText(),
            default=DEFAULT_DEPTH_CAMERA_PROFILE["rgb_resolution"],
        )
        profile["depth_resolution"] = _parse_resolution(
            self.cmb_depth_depth_res.currentText(),
            default=DEFAULT_DEPTH_CAMERA_PROFILE["depth_resolution"],
        )

    def _update_current_voice_from_ui(self):
        name = self.current_voice_profile_name
        if not name:
            return
        if _is_default_profile("voice", name):
            return
        profile = self.voice_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_VOICE_PROFILE)
            self.voice_profiles[name] = profile
        profile["use_voice_assistant"] = bool(self.chk_voice_enabled.isChecked())
        profile["preview_actions_voice"] = bool(
            self.chk_preview_voice_enabled.isChecked()
        )
        combo = getattr(self, "cmb_voice_language", None)
        if combo is not None and combo.count() > 0:
            language = combo.currentData()
            profile["language"] = (language or "en").strip()

    def _update_current_ui_from_ui(self):
        name = self.current_ui_profile_name
        if not name:
            return
        if _is_default_profile("ui", name):
            return
        profile = self.ui_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_UI_PROFILE)
            self.ui_profiles[name] = profile
        for spin, key in [
            (self.spn_camera_frame_percent, "camera_frame_percent"),
            (self.spn_preview_frame_percent, "preview_frame_percent"),
            (self.spn_action_frame_percent, "action_frame_percent"),
            (self.spn_details_pane_percent, "details_pane_percent"),
            (self.spn_log_frame_percent, "log_frame_percent"),
        ]:
            if spin is not None:
                profile[key] = float(spin.value())
        if hasattr(self, "chk_show_log"):
            profile["show_log"] = bool(self.chk_show_log.isChecked())

        if hasattr(self, "chk_start_fullscreen"):
            profile["start_fullscreen"] = bool(self.chk_start_fullscreen.isChecked())

        if hasattr(self, "cmb_transcript_position"):
            profile["transcript_position"] = (
                self.cmb_transcript_position.currentData() or "middle"
            )

        if hasattr(self, "spn_transcript_font_size"):
            profile["transcript_font_size"] = int(self.spn_transcript_font_size.value())
        if hasattr(self, "btn_transcript_color"):
            profile["transcript_font_color"] = (
                self.btn_transcript_color.property("color_hex") or "#ffffff"
            )
        if hasattr(self, "spn_transcript_offset_x"):
            profile["transcript_offset_x"] = int(self.spn_transcript_offset_x.value())
        if hasattr(self, "spn_transcript_offset_y"):
            profile["transcript_offset_y"] = int(self.spn_transcript_offset_y.value())

        font_mappings = [
            ("count", self.cmb_count_font, self.spn_count_font_size, self.btn_count_color),
            ("action", self.cmb_action_font, self.spn_action_font_size, self.btn_action_color),
            ("time", self.cmb_time_font, self.spn_time_font_size, self.btn_time_color),
            (
                "remaining_time_value",
                self.cmb_remaining_time_value_font,
                self.spn_remaining_time_value_font_size,
                self.btn_remaining_time_value_color,
            ),
            (
                "elapsed_time_value",
                self.cmb_elapsed_time_value_font,
                self.spn_elapsed_time_value_font_size,
                self.btn_elapsed_time_value_color,
            ),
        ]
        for prefix, combo, size_spin, color_button in font_mappings:
            family_key = f"{prefix}_font_family"
            size_key = f"{prefix}_font_size"
            color_key = f"{prefix}_font_color"
            if combo is not None:
                profile[family_key] = combo.currentText().strip()
            if size_spin is not None:
                profile[size_key] = int(size_spin.value())
            if color_button is not None:
                profile[color_key] = color_button.property("color_hex") or "#000000"

        if hasattr(self, "ui_status_edits"):
            for key, _message in UI_STATUS_MESSAGE_FIELDS:
                line = self.ui_status_edits.get(key)
                if line is not None:
                    profile[key] = line.text().strip()
        if hasattr(self, "spn_status_font_size"):
            profile["status_font_size"] = int(self.spn_status_font_size.value())
        if hasattr(self, "btn_status_color"):
            profile["status_font_color"] = (
                self.btn_status_color.property("color_hex") or "#000000"
            )
        if hasattr(self, "spn_status_offset_x"):
            profile["status_offset_x"] = int(self.spn_status_offset_x.value())
        if hasattr(self, "spn_status_offset_y"):
            profile["status_offset_y"] = int(self.spn_status_offset_y.value())

        if hasattr(self, "ui_guide_edits"):
            for key, _message in UI_GUIDE_MESSAGE_FIELDS:
                line = self.ui_guide_edits.get(key)
                if line is not None:
                    profile[key] = line.text().strip()

        if hasattr(self, "ui_text_controls"):
            for control in self.ui_text_controls:
                prefix = control["prefix"]
                profile[f"{prefix}_text"] = control["text"].text().strip()
                profile[f"{prefix}_font_size"] = int(control["font_size"].value())
                profile[f"{prefix}_color"] = (
                    control["color"].property("color_hex") or "#000000"
                )
                profile[f"{prefix}_offset_x"] = int(control["offset_x"].value())
                profile[f"{prefix}_offset_y"] = int(control["offset_y"].value())

        if hasattr(self, "ui_value_controls"):
            for control in self.ui_value_controls:
                prefix = control["prefix"]
                profile[f"{prefix}_font_size"] = int(control["font_size"].value())
                profile[f"{prefix}_color"] = (
                    control["color"].property("color_hex") or "#000000"
                )
                profile[f"{prefix}_offset_x"] = int(control["offset_x"].value())
                profile[f"{prefix}_offset_y"] = int(control["offset_y"].value())

        if hasattr(self, "edt_log_placeholder_text"):
            profile["log_placeholder_text"] = self.edt_log_placeholder_text.text().strip()
        if hasattr(self, "spn_log_font_size"):
            profile["log_font_size"] = int(self.spn_log_font_size.value())
        if hasattr(self, "btn_log_color"):
            profile["log_font_color"] = (
                self.btn_log_color.property("color_hex") or "#000000"
            )
        if hasattr(self, "spn_log_offset_x"):
            profile["log_offset_x"] = int(self.spn_log_offset_x.value())
        if hasattr(self, "spn_log_offset_y"):
            profile["log_offset_y"] = int(self.spn_log_offset_y.value())

        if hasattr(self, "ui_webcam_edits"):
            for key, _message in UI_WEBCAM_MESSAGE_FIELDS:
                line = self.ui_webcam_edits.get(key)
                if line is not None:
                    profile[key] = line.text().strip()
        if hasattr(self, "spn_webcam_font_size"):
            profile["webcam_font_size"] = int(self.spn_webcam_font_size.value())
        if hasattr(self, "btn_webcam_color"):
            profile["webcam_font_color"] = (
                self.btn_webcam_color.property("color_hex") or "#ffffff"
            )
        if hasattr(self, "spn_webcam_offset_x"):
            profile["webcam_offset_x"] = int(self.spn_webcam_offset_x.value())
        if hasattr(self, "spn_webcam_offset_y"):
            profile["webcam_offset_y"] = int(self.spn_webcam_offset_y.value())

    def _update_current_time_from_ui(self):
        name = self.current_time_profile_name
        if not name:
            return
        if _is_default_profile("time", name):
            return
        profile = self.time_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_TIME_PROFILE)
            self.time_profiles[name] = profile

        if hasattr(self, "chk_use_time_server"):
            profile["use_time_server"] = bool(self.chk_use_time_server.isChecked())
        if hasattr(self, "cmb_time_server"):
            server = self.cmb_time_server.currentText().strip()
            profile["time_server"] = server or DEFAULT_TIME_PROFILE["time_server"]

    def _update_current_demo_from_ui(self):
        name = self.current_demo_profile_name
        if not name:
            return
        if _is_default_profile("demo", name):
            return
        profile = self.demo_profiles.get(name)
        if not profile:
            profile = deepcopy(DEFAULT_DEMO_PROFILE)
            self.demo_profiles[name] = profile
        if hasattr(self, "spn_demo_capture_duration"):
            profile["capture_duration_seconds"] = float(self.spn_demo_capture_duration.value())
        if hasattr(self, "cmb_demo_capture_mode"):
            profile["demo_capture_mode"] = _normalize_demo_capture_mode(
                self.cmb_demo_capture_mode.currentData()
            )
        if hasattr(self, "spn_demo_effective_samples"):
            profile["effective_capture_samples"] = int(self.spn_demo_effective_samples.value())
        if hasattr(self, "chk_demo_hampel_ratio_magnitude"):
            profile["apply_hampel_to_ratio_magnitude"] = bool(
                self.chk_demo_hampel_ratio_magnitude.isChecked()
            )
        if hasattr(self, "chk_demo_hampel_ratio_phase"):
            profile["apply_hampel_to_ratio_phase"] = bool(
                self.chk_demo_hampel_ratio_phase.isChecked()
            )
        if hasattr(self, "txt_demo_title"):
            profile["demo_title_text"] = (
                self.txt_demo_title.text().strip()
                or DEFAULT_DEMO_PROFILE["demo_title_text"]
            )
        if hasattr(self, "txt_demo_university_logo_image_path"):
            profile["university_logo_image_path"] = (
                self.txt_demo_university_logo_image_path.text().strip()
            )
        if hasattr(self, "spn_demo_university_logo_image_size"):
            profile["university_logo_image_size_px"] = int(
                self.spn_demo_university_logo_image_size.value()
            )
        if hasattr(self, "txt_demo_icassp_logo_image_path"):
            profile["icassp_logo_image_path"] = (
                self.txt_demo_icassp_logo_image_path.text().strip()
            )
        if hasattr(self, "txt_demo_website_url"):
            profile["website_url"] = (
                self.txt_demo_website_url.text().strip()
                or DEFAULT_DEMO_PROFILE["website_url"]
            )
        if hasattr(self, "txt_demo_icassp_title"):
            profile["icassp_title_text"] = self.txt_demo_icassp_title.text().strip()
        if hasattr(self, "spn_demo_icassp_logo_text_vertical_gap"):
            profile["icassp_logo_text_vertical_gap"] = int(
                self.spn_demo_icassp_logo_text_vertical_gap.value()
            )
        if hasattr(self, "spn_demo_title_font_size"):
            profile["demo_title_font_size_px"] = int(self.spn_demo_title_font_size.value())
        if hasattr(self, "txt_demo_authors"):
            profile["authors_text"] = (
                self.txt_demo_authors.text().strip()
                or DEFAULT_DEMO_PROFILE["authors_text"]
            )
        if hasattr(self, "spn_demo_authors_font_size"):
            profile["authors_font_size_px"] = int(self.spn_demo_authors_font_size.value())
        if hasattr(self, "txt_demo_university"):
            profile["university_text"] = self.txt_demo_university.text().strip()
        if hasattr(self, "txt_demo_wirlab"):
            profile["wirlab_text"] = self.txt_demo_wirlab.text().strip()
        if hasattr(self, "spn_demo_university_font_size"):
            profile["university_font_size_px"] = int(self.spn_demo_university_font_size.value())
        if hasattr(self, "spn_demo_title_authors_vertical_gap"):
            profile["title_authors_vertical_gap"] = int(
                self.spn_demo_title_authors_vertical_gap.value()
            )
        if hasattr(self, "spn_demo_authors_university_vertical_gap"):
            profile["authors_university_vertical_gap"] = int(
                self.spn_demo_authors_university_vertical_gap.value()
            )
        if hasattr(self, "txt_demo_capture_guidance_title"):
            profile["capture_guidance_title"] = (
                self.txt_demo_capture_guidance_title.text().strip()
                or DEFAULT_DEMO_PROFILE["capture_guidance_title"]
            )
        if hasattr(self, "txt_demo_capture_guidance_message"):
            profile["capture_guidance_message"] = (
                self.txt_demo_capture_guidance_message.text().strip()
                or DEFAULT_DEMO_PROFILE["capture_guidance_message"]
            )
        if hasattr(self, "txt_demo_capture_guidance_left_label"):
            profile["capture_guidance_video_left_label"] = (
                self.txt_demo_capture_guidance_left_label.text().strip()
                or DEFAULT_DEMO_PROFILE["capture_guidance_video_left_label"]
            )
        if hasattr(self, "txt_demo_capture_guidance_left_path"):
            profile["capture_guidance_video_left_path"] = (
                self.txt_demo_capture_guidance_left_path.text().strip()
            )
        if hasattr(self, "txt_demo_capture_guidance_right_label"):
            profile["capture_guidance_video_right_label"] = (
                self.txt_demo_capture_guidance_right_label.text().strip()
                or DEFAULT_DEMO_PROFILE["capture_guidance_video_right_label"]
            )
        if hasattr(self, "txt_demo_capture_guidance_right_path"):
            profile["capture_guidance_video_right_path"] = (
                self.txt_demo_capture_guidance_right_path.text().strip()
            )
        if hasattr(self, "txt_demo_activity_class_names"):
            class_names_text = self.txt_demo_activity_class_names.text().strip()
            if class_names_text:
                profile["activity_class_names"] = [
                    part.strip() for part in class_names_text.split(",") if part.strip()
                ]
            else:
                profile["activity_class_names"] = deepcopy(
                    DEFAULT_DEMO_PROFILE["activity_class_names"]
                )
        if hasattr(self, "txt_demo_subplot_settings"):
            raw_json = self.txt_demo_subplot_settings.toPlainText().strip()
            if raw_json:
                try:
                    parsed = json.loads(raw_json)
                    if isinstance(parsed, dict):
                        profile["subplot_settings"] = parsed
                except Exception:
                    pass
            elif "subplot_settings" not in profile:
                profile["subplot_settings"] = deepcopy(DEFAULT_DEMO_PROFILE["subplot_settings"])
        if hasattr(self, "txt_demo_dorf_plot_order"):
            order_text = self.txt_demo_dorf_plot_order.text().strip()
            if order_text:
                profile["dorf_plot_order"] = [
                    item.strip() for item in order_text.split(",") if item.strip()
                ]
            elif "dorf_plot_order" not in profile:
                profile["dorf_plot_order"] = deepcopy(DEFAULT_DEMO_PROFILE["dorf_plot_order"])

    def _update_current_wifi_from_ui(self):
        name = self.current_wifi_profile_name
        if not name:
            return
        if _is_default_profile("wifi", name):
            return

        profile = self.wifi_profiles.get(name)
        if not isinstance(profile, dict):
            profile = deepcopy(DEFAULT_WIFI_PROFILE)

        if hasattr(self, "spn_init_test_duration"):
            profile["init_test_duration"] = float(self.spn_init_test_duration.value())
        if hasattr(self, "txt_init_save_directory"):
            directory_text = self.txt_init_save_directory.text().strip()
            profile["init_test_save_directory"] = directory_text or DEFAULT_WIFI_PROFILE[
                "init_test_save_directory"
            ]
        if hasattr(self, "cmb_csi_scenario"):
            scenario_value = self.cmb_csi_scenario.currentData()
            profile["csi_capture_scenario"] = scenario_value
        else:
            scenario_value = profile.get(
                "csi_capture_scenario", DEFAULT_WIFI_PROFILE["csi_capture_scenario"]
            )
        if hasattr(self, "spn_pre_action_capture"):
            profile["pre_action_capture_duration"] = float(
                self.spn_pre_action_capture.value()
            )
        if hasattr(self, "spn_post_action_capture"):
            profile["post_action_capture_duration"] = float(
                self.spn_post_action_capture.value()
            )
        if hasattr(self, "chk_delete_prev_pcap"):
            delete_prev_pcap = bool(self.chk_delete_prev_pcap.isChecked())
            if str(scenario_value).lower() == "scenario_2":
                delete_prev_pcap = False
            profile["delete_prev_pcap"] = delete_prev_pcap
        if hasattr(self, "chk_count_packets"):
            profile["count_packets"] = bool(self.chk_count_packets.isChecked())
        if hasattr(self, "chk_reboot_after_summary"):
            profile["reboot_after_summary"] = bool(
                self.chk_reboot_after_summary.isChecked()
            )

        aps = []
        for row in range(self.tbl_wifi.rowCount()):
            spin = self.tbl_wifi.cellWidget(row, 0)
            order = spin.value() if isinstance(spin, QSpinBox) else row + 1

            name_widget = self.tbl_wifi.cellWidget(row, 1)
            begin_script_widget = self.tbl_wifi.cellWidget(row, 2)
            end_script_widget = self.tbl_wifi.cellWidget(row, 3)
            ssid_widget = self.tbl_wifi.cellWidget(row, 4)
            password_widget = self.tbl_wifi.cellWidget(row, 5)
            router_ip_widget = self.tbl_wifi.cellWidget(row, 6)
            router_user_widget = self.tbl_wifi.cellWidget(row, 7)
            router_pass_widget = self.tbl_wifi.cellWidget(row, 8)
            ssh_key_widget = self.tbl_wifi.cellWidget(row, 9)
            freq_widget = self.tbl_wifi.cellWidget(row, 10)
            channel_widget = self.tbl_wifi.cellWidget(row, 11)
            bandwidth_widget = self.tbl_wifi.cellWidget(row, 12)
            mac_widget = self.tbl_wifi.cellWidget(row, 13)
            remote_dir_widget = self.tbl_wifi.cellWidget(row, 14)
            ethernet_widget = self.tbl_wifi.cellWidget(row, 15)
            ethernet_checkbox = None
            if isinstance(ethernet_widget, QWidget):
                ethernet_checkbox = ethernet_widget.findChild(QCheckBox)
            download_widget = self.tbl_wifi.cellWidget(row, 16)

            channel_value = (
                channel_widget.currentText().strip()
                if isinstance(channel_widget, QComboBox)
                else ""
            )
            bandwidth_value = (
                bandwidth_widget.currentText().strip()
                if isinstance(bandwidth_widget, QComboBox)
                else ""
            )
            ap = {
                "order": int(order),
                "name": name_widget.text().strip() if isinstance(name_widget, QLineEdit) else "",
                "framework": (
                    begin_script_widget.currentText().strip()
                    if isinstance(begin_script_widget, QComboBox)
                    else begin_script_widget.text().strip()
                    if isinstance(begin_script_widget, QLineEdit)
                    else ""
                ),
                "type": (
                    end_script_widget.currentText().strip()
                    if isinstance(end_script_widget, QComboBox)
                    else end_script_widget.text().strip()
                    if isinstance(end_script_widget, QLineEdit)
                    else ""
                ),
                "ssid": ssid_widget.text().strip() if isinstance(ssid_widget, QLineEdit) else "",
                "password": password_widget.text().strip()
                if isinstance(password_widget, QLineEdit)
                else "",
                "router_ssh_ip": router_ip_widget.text().strip()
                if isinstance(router_ip_widget, QLineEdit)
                else "",
                "router_ssh_username": router_user_widget.text().strip()
                if isinstance(router_user_widget, QLineEdit)
                else "",
                "router_ssh_password": router_pass_widget.text().strip()
                if isinstance(router_pass_widget, QLineEdit)
                else "",
                "ssh_key_address": ssh_key_widget.text().strip()
                if isinstance(ssh_key_widget, QLineEdit)
                else "",
                "frequency": freq_widget.currentText()
                if isinstance(freq_widget, QComboBox)
                else "2.4 GHz",
                "channel": channel_value or "1",
                "bandwidth": bandwidth_value or "20MHz",
                "transmitter_macs": mac_widget.text().strip()
                if isinstance(mac_widget, QLineEdit)
                else "",
                "init_test_save_directory": (
                    remote_dir_widget.text().strip()
                    if isinstance(remote_dir_widget, QLineEdit)
                    else DEFAULT_WIFI_PROFILE["init_test_save_directory"]
                ),
                "use_ethernet": bool(ethernet_checkbox.isChecked())
                if isinstance(ethernet_checkbox, QCheckBox)
                else False,
                "download_mode": download_widget.currentText()
                if isinstance(download_widget, QComboBox)
                else "SFTP",
            }
            aps.append(ap)

        if not aps:
            aps = [deepcopy(DEFAULT_WIFI_AP)]
        profile["access_points"] = aps

        self.wifi_profiles[name] = profile

    def eventFilter(self, obj, event):
        if (
            isinstance(obj, QLineEdit)
            and obj.property("script_line_edit")
            and event.type() == QEvent.MouseButtonDblClick
        ):
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Script File", "", "All Files (*)"
            )
            if path:
                obj.setText(path)
            return True

        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------
    def get_participant_profiles(self):
        self._update_current_participant_from_ui()
        return deepcopy(self.participant_profiles)

    def get_environment_profiles(self):
        self._update_current_environment_from_ui()
        return deepcopy(self.environment_profiles)

    def get_session_experiment_id(self) -> str:
        return self.session_experiment_id

    def get_experiment_profiles(self):
        self._update_current_experiment_from_ui()
        return deepcopy(self.experiment_profiles)

    def get_action_profiles(self):
        self._update_current_actions_from_ui()
        return deepcopy(
            {name: filter_blank_actions(actions) for name, actions in self.action_profiles.items()}
        )

    def get_voice_profiles(self):
        self._update_current_voice_from_ui()
        return deepcopy(self.voice_profiles)

    def get_camera_profiles(self):
        self._update_current_camera_from_ui()
        return deepcopy(self.camera_profiles)

    def get_depth_camera_profiles(self):
        self._update_current_depth_camera_from_ui()
        return deepcopy(self.depth_camera_profiles)

    def get_wifi_profiles(self):
        self._update_current_wifi_from_ui()
        return deepcopy(self.wifi_profiles)

    def get_ui_profiles(self):
        self._update_current_ui_from_ui()
        return deepcopy(self.ui_profiles)

    def get_time_profiles(self):
        self._update_current_time_from_ui()
        return deepcopy(self.time_profiles)

    def get_demo_profiles(self):
        self._update_current_demo_from_ui()
        return deepcopy(self.demo_profiles)

    def get_selected_participant_name(self):
        return self.current_participant_name

    def get_selected_experiment_name(self):
        return self.current_experiment_name

    def get_selected_environment_name(self):
        return self.current_environment_name

    def get_selected_action_profile_name(self):
        return self.current_action_profile_name

    def get_selected_camera_profile_name(self):
        return self.current_camera_profile_name

    def get_selected_depth_camera_profile_name(self):
        return self.current_depth_camera_profile_name

    def get_selected_voice_profile_name(self):
        return self.current_voice_profile_name

    def get_selected_wifi_profile_name(self):
        return self.current_wifi_profile_name

    def get_selected_ui_profile_name(self):
        return self.current_ui_profile_name

    def get_selected_time_profile_name(self):
        return self.current_time_profile_name

    def get_selected_demo_profile_name(self):
        return self.current_demo_profile_name

    def _on_test_connections_clicked(self):
        self._update_current_wifi_from_ui()
        profile = self.wifi_profiles.get(self.current_wifi_profile_name, {})
        access_points = []
        if isinstance(profile, dict):
            access_points = deepcopy(profile.get("access_points", []))

        if not access_points:
            QMessageBox.information(
                self,
                "No Access Points",
                "Please add at least one access point to test.",
            )
            return

        dialog = RouterConnectionTestDialog(
            self.current_wifi_profile_name,
            access_points,
            wifi_profile=deepcopy(profile),
            parent=self,
        )
        dialog.exec_()

    def _on_reboot_access_points_clicked(self):
        self._update_current_wifi_from_ui()
        profile = self.wifi_profiles.get(self.current_wifi_profile_name, {})
        access_points = []
        if isinstance(profile, dict):
            access_points = deepcopy(profile.get("access_points", []))

        if not access_points:
            QMessageBox.information(
                self,
                "No Access Points",
                "Please add at least one access point to reboot.",
            )
            return

        self._reboot_dialog, self._reboot_thread = (
            self.wifi_manager.reboot_access_points_with_dialog(
                access_points,
                router_cls=WiFiRouter,
                parent=self,
                log=lambda message: print(message),
                thread_name="ConfigAccessPointRebooter",
                on_finished=lambda: setattr(self, "_reboot_dialog", None),
            )
        )

    def _on_delete_pcaps_clicked(self):
        self._update_current_wifi_from_ui()
        profile = self.wifi_profiles.get(self.current_wifi_profile_name, {})
        access_points = []
        if isinstance(profile, dict):
            access_points = deepcopy(profile.get("access_points", []))

        if not access_points:
            QMessageBox.information(
                self,
                "No Access Points",
                "Please add at least one access point to delete PCAPs.",
            )
            return

        response = QMessageBox.question(
            self,
            "Confirm PCAP Deletion",
            "This will delete all .pcap files on every configured router for the "
            f"'{self.current_wifi_profile_name}' Wi-Fi profile.\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return

        dialog = RouterPcapCleanupDialog(
            self.current_wifi_profile_name,
            access_points,
            wifi_profile=deepcopy(profile),
            parent=self,
        )
        dialog.exec_()

    # ------------------------------------------------------------------
    # Start experiment handling
    # ------------------------------------------------------------------
    def _on_start_clicked(self):
        self._update_current_participant_from_ui()
        self._update_current_experiment_from_ui()
        self._update_current_environment_from_ui()
        self._update_current_actions_from_ui()
        self._update_current_voice_from_ui()
        self._update_current_camera_from_ui()
        self._update_current_depth_camera_from_ui()
        self._update_current_ui_from_ui()
        self._update_current_wifi_from_ui()
        self._update_current_time_from_ui()

        actions = filter_blank_actions(
            self.action_profiles.get(self.current_action_profile_name, {})
        )
        voice_profile = self.voice_profiles.get(
            self.current_voice_profile_name, DEFAULT_VOICE_PROFILE
        )
        camera_profile = self.camera_profiles.get(
            self.current_camera_profile_name, DEFAULT_CAMERA_PROFILE
        )
        preview = ActionPreviewDialog(
            actions,
            action_time=float(self.spn_action_time.value()),
            stop_time=float(self.spn_stop_time.value()),
            preview_stop_time=float(self.spn_preview_pause.value()),
            voice_profile=voice_profile,
            camera_profile=camera_profile,
            parent=self,
        )
        result = preview.exec_()
        if result != QDialog.Accepted:
            return

        self.accept()

    def _show_start_progress(self, voice_profile: dict | None = None):
        start_message = (
            "The experiment will start soon. Everyone except the participant should "
            "leave the room; the participant should remain seated."
        )
        progress = QProgressDialog(
            start_message,
            "",
            0,
            100,
            self,
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("WIRLab - Starting Experiment")
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        self._speak_message(start_message, voice_profile)

        loop = QEventLoop()
        start_time = time.monotonic()

        def update_progress():
            elapsed = time.monotonic() - start_time
            percent = min(int((elapsed / 10.0) * 100), 100)
            progress.setValue(percent)
            if percent >= 100:
                timer.stop()
                progress.close()
                loop.quit()

        timer = QTimer(self)
        timer.timeout.connect(update_progress)
        timer.start(100)

        update_progress()
        loop.exec_()

    def _speak_message(self, message: str, voice_profile: dict | None = None):
        if not message:
            return
        profile = voice_profile or self.voice_profiles.get(
            self.current_voice_profile_name, DEFAULT_VOICE_PROFILE
        )
        if not _as_bool(profile.get("use_voice_assistant", False)):
            return

        language = (profile.get("language") or "en").strip() or "en"
        if self._voice_language != language or self.voice_assistant is None:
            try:
                self.voice_assistant = GTTSVoiceAssistant(
                    language=language, parent=self
                )
                self._voice_language = language
            except Exception:
                self.voice_assistant = None
                return
        try:
            self.voice_assistant.stop()
            self.voice_assistant.speak(message)
        except Exception:
            self.voice_assistant = None

    def accept(self):
        self._stop_camera_preview()
        self._stop_depth_status_timer()
        super().accept()

    def reject(self):
        self._stop_camera_preview()
        self._stop_depth_status_timer()
        super().reject()
