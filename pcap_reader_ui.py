import ast
import copy
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import lzma
import pickle
import re
import shutil
import time
import textwrap
import tarfile
from pathlib import Path
from typing import List, Optional

import csv
import struct
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QAction,
    QMenuBar,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy import signal

from pcap_reader import ProcessPcap, ant_processing
from help.help_explorer import HelpExplorerDialog

try:  # pragma: no cover - optional dependency at runtime
    from hampel import hampel
except ImportError:  # pragma: no cover - optional dependency at runtime
    hampel = None

try:  # pragma: no cover - optional dependency at runtime
    import pgzip
except ImportError:  # pragma: no cover - optional dependency at runtime
    pgzip = None


def _unpack_float(H_uint32, nfft):
    nbits = 10
    nman = 12
    nexp = 6

    iq_mask = (1 << (nman - 1)) - 1
    e_mask = (1 << nexp) - 1
    e_p = (1 << (nexp - 1))

    vi = (H_uint32 >> (nexp + nman)) & iq_mask
    vq = (H_uint32 >> nexp) & iq_mask
    e = (H_uint32 & e_mask).astype(np.int8)

    e[e >= e_p] -= (e_p * 2)

    x = vi | vq

    bit_len = np.zeros_like(x, dtype=np.int32)
    mask_nz = x > 0
    float_x = x.astype(np.float64)
    float_x[float_x == 0] = 1
    bits = np.floor(np.log2(float_x)).astype(np.int32) + 1
    bits[~mask_nz] = 0

    temp_e = e.astype(np.int32) + bits
    maxbit = np.max(temp_e)
    shft = nbits - maxbit
    final_e = e + shft

    sgnr_mask = (1 << (nexp + 2 * nman - 1))
    sgni_mask = (sgnr_mask >> nman)

    sign_i = np.ones_like(vi, dtype=np.int32)
    sign_i[(H_uint32 & sgnr_mask) != 0] = -1

    sign_q = np.ones_like(vq, dtype=np.int32)
    sign_q[(H_uint32 & sgni_mask) != 0] = -1

    def apply_shift(val, exp):
        res = np.zeros_like(val)
        mask_neg = (exp < 0) & (exp >= -nman)
        res[mask_neg] = val[mask_neg] >> -exp[mask_neg]
        mask_pos = exp >= 0
        res[mask_pos] = val[mask_pos] << exp[mask_pos]
        return res

    final_vi = apply_shift(vi, final_e) * sign_i
    final_vq = apply_shift(vq, final_e) * sign_q

    return final_vi + 1j * final_vq


class _UbiLocatePcapReader:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.file = None

    def open(self) -> None:
        self.file = open(self.filename, "rb")
        self.file.seek(0, 2)
        self.file_size = self.file.tell()
        self.file.seek(0)
        header = self.file.read(24)
        if len(header) < 24:
            raise ValueError("File too short for PCAP header")

    def next_frame(self):
        if self.file is None:
            return None

        header_bytes = self.file.read(16)
        if not header_bytes or len(header_bytes) < 16:
            return None

        ts_sec, ts_usec, incl_len, orig_len = struct.unpack("<IIII", header_bytes)
        payload = self.file.read(incl_len)
        if len(payload) < incl_len:
            return None

        return {
            "ts_sec": ts_sec,
            "ts_usec": ts_usec,
            "orig_len": orig_len,
            "payload": payload,
        }

    def close(self) -> None:
        if self.file:
            self.file.close()


def _read_ubilocate_csi(
    filename: str, *, bw: int = 80, is_4ss: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    HOFFSET = 16
    nfft = int(bw * 3.2)

    reader = _UbiLocatePcapReader(filename)
    reader.open()

    prevfwcnt = -1

    if is_4ss:
        mask_toprocess = 16
        processedmask = 0
        slice_buffer = None
    else:
        mask_toprocess = 4
        processedcore = 0
        slice_buffer = None

    csi_list: list[np.ndarray] = []
    timestamps_list: list[float] = []
    current_packet_ts = 0.0

    while True:
        frame = reader.next_frame()
        if frame is None:
            break

        payload = frame["payload"]
        frame_ts = float(frame["ts_sec"]) + float(frame["ts_usec"]) * 1e-6

        if len(payload) < (HOFFSET + nfft) * 4:
            continue

        payload_u32 = np.frombuffer(payload, dtype=np.uint32)
        if len(payload_u32) < 16:
            continue

        val_15 = payload_u32[14]
        val_14 = payload_u32[13]
        fwcnt = (val_15 >> 16) & 0xFFFF

        if is_4ss:
            fwmask = (val_14 >> 16) & 0xFF
            current_idx = fwmask

            if fwcnt > prevfwcnt:
                processedmask = 0
                prevfwcnt = fwcnt
                slice_buffer = np.zeros((len(payload_u32), mask_toprocess), dtype=np.uint32)
                current_packet_ts = frame_ts

            processedmask += 1

            if slice_buffer is not None and current_idx < mask_toprocess:
                slice_buffer[:, current_idx] = payload_u32

            if processedmask == mask_toprocess:
                csi_matrix = np.zeros((nfft, mask_toprocess), dtype=np.complex128)
                valid_extraction = True

                for jj in range(mask_toprocess):
                    col_data = slice_buffer[:, jj]
                    H = col_data[15 : 15 + nfft]
                    if len(H) == nfft:
                        c_num = _unpack_float(H, nfft)
                        c_num = np.fft.fftshift(c_num)
                        csi_matrix[:, jj] = c_num
                    else:
                        valid_extraction = False

                if valid_extraction:
                    csi_list.append(csi_matrix)
                    timestamps_list.append(current_packet_ts)

        else:
            tmp = (val_14 >> 16) & 0xFF
            rxcore = tmp // 4

            if fwcnt > prevfwcnt:
                processedcore = 0
                prevfwcnt = fwcnt
                slice_buffer = np.zeros((len(payload_u32), mask_toprocess), dtype=np.uint32)
                current_packet_ts = frame_ts

            processedcore += 1

            if slice_buffer is not None and rxcore < mask_toprocess:
                slice_buffer[:, rxcore] = payload_u32

            if processedcore == mask_toprocess:
                csi_matrix = np.zeros((nfft, mask_toprocess), dtype=np.complex128)
                valid_extraction = True
                for jj in range(mask_toprocess):
                    col_data = slice_buffer[:, jj]
                    H = col_data[15 : 15 + nfft]
                    if len(H) == nfft:
                        c_num = _unpack_float(H, nfft)
                        c_num = np.fft.fftshift(c_num)
                        csi_matrix[:, jj] = c_num
                    else:
                        valid_extraction = False

                if valid_extraction:
                    csi_list.append(csi_matrix)
                    timestamps_list.append(current_packet_ts)

    reader.close()

    if not csi_list:
        return (
            np.empty((0, 0), dtype=np.complex128),
            np.array([], dtype=float),
            np.array([], dtype=str),
        )

    csi_array = np.array(csi_list)
    timestamps = np.array(timestamps_list)
    if timestamps.size:
        timestamps = timestamps - timestamps[0]

    rx_count = 4
    tx_count = mask_toprocess // rx_count if mask_toprocess else 1
    if tx_count == 0:
        tx_count = 1

    try:
        csi_array = csi_array.reshape(csi_array.shape[0], nfft, rx_count, tx_count)
    except ValueError:
        raise ValueError(
            f"Unexpected CSI dimensions: cannot reshape to (packets, {nfft}, {rx_count}, {tx_count})"
        )

    mac_addrs = np.array(["unknown"] * csi_array.shape[0])
    return csi_array, timestamps, mac_addrs


def plot_histogram_with_annotations(
    data,
    ax,
    bins: int = 10,
    title: str = "Histogram with Annotated Centers",
    xlabel: str = "Time (interval)",
    ylabel: str = "Percentage",
):
    """Plot a histogram with percentage annotations on the provided axis."""

    if len(data) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        return

    counts, edges, patches = ax.hist(
        data,
        bins=bins,
        edgecolor="black",
        alpha=0.7,
        weights=np.ones(len(data)) / len(data) * 100,
    )

    bin_centers = (edges[:-1] + edges[1:]) / 2

    for center, count in zip(bin_centers, counts):
        ax.text(center, count + 0.5, f"{count:.2f}%", ha="center", va="bottom", fontsize=8, color="red")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    for patch in patches:
        patch.set_facecolor(np.random.rand(3,))


class PCAPExplorerWindow(QWidget):
    """Browse recorded experiments and visualize CSI magnitudes from PCAPs."""

    def __init__(
        self,
        *,
        results_root: Path | str = "results",
        initial_selection: Optional[dict[str, Path | str]] = None,
        auto_plot: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.results_root = Path(results_root)
        self._initial_selection = initial_selection or {}
        self._auto_plot = auto_plot

        self.mac_timestamp: dict[str, list[float]] = {}
        self.histogram_ax = None
        self._capture_manifest: dict[Path, str] = {}
        self._capture_manifest_entries: list[dict[str, str]] = []
        self._help_dialog: HelpExplorerDialog | None = None

        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self._build_ui()
        self._refresh_experiments()
        self._apply_initial_selection()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle("PCAP CSI Explorer")
        self.setMinimumSize(1100, 800)
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

        root_layout = QHBoxLayout()

        # Left control panel
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_widget.setMinimumWidth(320)

        selectors_box = QGroupBox("Capture selection")
        selectors_layout = QFormLayout(selectors_box)

        self.results_root_label = QLabel(str(self.results_root))
        self.results_root_label.setWordWrap(True)
        results_root_button = QPushButton("Browse…")
        results_root_button.clicked.connect(self._select_results_root)
        results_root_layout = QHBoxLayout()
        results_root_layout.addWidget(self.results_root_label, 1)
        results_root_layout.addWidget(results_root_button)
        results_root_widget = QWidget()
        results_root_widget.setLayout(results_root_layout)

        self.experiment_combo = QComboBox()
        self.experiment_combo.currentIndexChanged.connect(self._populate_access_points)

        self.ap_combo = QComboBox()
        self.ap_combo.currentIndexChanged.connect(self._populate_trials)

        self.trial_combo = QComboBox()
        self.trial_combo.currentIndexChanged.connect(self._populate_pcaps)

        self.pcap_combo = QComboBox()
        self.pcap_combo.currentIndexChanged.connect(self._on_pcap_selection_changed)

        self.framework_combo = QComboBox()
        self.framework_combo.addItem("Nexmon", "nexmon")
        self.framework_combo.addItem("UbiLocate", "ubilocate")

        self.mac_combo = QComboBox()
        self.mac_combo.setEnabled(False)
        self.mac_combo.currentIndexChanged.connect(self._on_mac_selection_changed)

        selectors_layout.addRow("Results folder:", results_root_widget)
        selectors_layout.addRow("Experiment:", self.experiment_combo)
        selectors_layout.addRow("Access point:", self.ap_combo)
        selectors_layout.addRow("Trial:", self.trial_combo)
        selectors_layout.addRow("PCAP file:", self.pcap_combo)
        selectors_layout.addRow("Framework:", self.framework_combo)
        selectors_layout.addRow("MAC address:", self.mac_combo)

        self.details_box = QGroupBox("Experiment details")
        details_layout = QFormLayout(self.details_box)
        self.experiment_datetime_label = QLabel("—")
        self.experiment_id_label = QLabel("—")
        self.participant_id_label = QLabel("—")
        self.experiment_duration_label = QLabel("—")
        details_layout.addRow("Date & time:", self.experiment_datetime_label)
        details_layout.addRow("Experiment ID:", self.experiment_id_label)
        details_layout.addRow("Participant ID:", self.participant_id_label)
        details_layout.addRow("Duration:", self.experiment_duration_label)

        tuning_box = QGroupBox("Visualization settings")
        tuning_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tuning_layout = QVBoxLayout(tuning_box)

        self.bandwidth_spin = QSpinBox()
        self.bandwidth_spin.setRange(20, 160)
        self.bandwidth_spin.setSingleStep(20)
        self.bandwidth_spin.setValue(80)
        self.bandwidth_spin.setSuffix(" MHz")

        bandwidth_form = QFormLayout()
        bandwidth_form.addRow("Bandwidth:", self.bandwidth_spin)
        tuning_layout.addLayout(bandwidth_form)

        stream_box = QGroupBox("RX/TX streams")
        stream_box_layout = QVBoxLayout(stream_box)
        self.stream_scroll = QScrollArea()
        self.stream_scroll.setWidgetResizable(True)
        self.stream_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stream_widget = QWidget()
        self.stream_layout = QVBoxLayout(self.stream_widget)
        self.stream_scroll.setWidget(self.stream_widget)
        stream_box_layout.addWidget(self.stream_scroll, 1)
        self.stream_checkboxes: list[QCheckBox] = []
        self.subcarrier_spins: List[QSpinBox] = []
        tuning_layout.addWidget(stream_box, 1)

        self.hampel_checkbox = QCheckBox("Apply Hampel filter")
        tuning_layout.addWidget(self.hampel_checkbox)

        self.butterworth_checkbox = QCheckBox("Apply Butterworth filter")
        tuning_layout.addWidget(self.butterworth_checkbox)

        self.phase_processing_checkbox = QCheckBox("Process antenna phase")
        tuning_layout.addWidget(self.phase_processing_checkbox)

        self.packet_count_label = QLabel("Packets: -")

        load_button = QPushButton("Load and Plot")
        load_button.clicked.connect(self._load_and_plot)

        export_button = QPushButton("Export Dataset")
        export_button.clicked.connect(self._export_dataset)

        controls_layout.addWidget(selectors_box)
        controls_layout.addWidget(self.details_box)
        controls_layout.addWidget(tuning_box, 1)
        controls_layout.addWidget(self.packet_count_label)
        controls_layout.addWidget(load_button)
        controls_layout.addWidget(export_button)
        controls_layout.addStretch(1)

        # Right plotting panel
        plot_layout = QVBoxLayout()

        plot_layout.addWidget(self.toolbar)

        self.plot_scroll_area = QScrollArea()
        self.plot_scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.addWidget(self.canvas)
        self.plot_scroll_area.setWidget(scroll_content)
        plot_layout.addWidget(self.plot_scroll_area, 1)

        self.status_label = QLabel()
        plot_layout.addWidget(self.status_label)

        root_layout.addWidget(controls_widget)
        root_layout.addLayout(plot_layout, 1)
        root_layout.setStretch(0, 1)
        root_layout.setStretch(1, 4)
        main_layout.addLayout(root_layout, 1)

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

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _select_results_root(self) -> None:
        selected_directory = QFileDialog.getExistingDirectory(
            self,
            "Select results folder",
            str(self.results_root),
        )
        if not selected_directory:
            return

        self.results_root = Path(selected_directory)
        self.results_root_label.setText(str(self.results_root))
        self._refresh_experiments()

    def _load_capture_manifest(self, experiment_path: Path) -> dict[Path, str]:
        manifest_path = experiment_path / "csi_captures" / "captures.csv"
        mapping: dict[Path, str] = {}
        self._capture_manifest_entries = []
        if not manifest_path.exists():
            return mapping

        try:
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    self._capture_manifest_entries.append(row)
                    file_path_raw = row.get("file_path")
                    if not file_path_raw:
                        continue
                    framework = (row.get("framework") or "nexmon").strip().lower()
                    framework = "ubilocate" if framework == "ubilocate" else "nexmon"
                    resolved_path = Path(file_path_raw).expanduser().resolve()
                    mapping[resolved_path] = framework
        except Exception:
            return {}

        return mapping

    def _capture_manifest_entry(self, pcap_path: Path) -> Optional[dict[str, str]]:
        resolved = pcap_path.expanduser().resolve()
        for entry in self._capture_manifest_entries:
            entry_path = entry.get("file_path")
            if not entry_path:
                continue
            try:
                if Path(entry_path).expanduser().resolve() == resolved:
                    return entry
            except OSError:
                continue
        return None

    def _capture_framework(self, pcap_path: Path) -> str:
        resolved = pcap_path.expanduser().resolve()
        return self._capture_manifest.get(resolved, "nexmon")

    def _set_framework_selection(self, framework: str) -> None:
        target = "ubilocate" if framework == "ubilocate" else "nexmon"
        index = self.framework_combo.findData(target)
        if index >= 0:
            self.framework_combo.setCurrentIndex(index)
        else:
            self.framework_combo.setCurrentIndex(0)

    def _default_framework_for_pcap(self, pcap_path: Optional[Path]) -> str:
        if not pcap_path:
            return "nexmon"
        return self._capture_framework(pcap_path)

    def _parse_summary_value(
        self, store: dict[str, str | list[str]], key: str, value: str
    ) -> None:
        key = key.strip()
        value = value.strip()
        if not key:
            return
        current = store.get(key)
        if current is None:
            store[key] = value
        elif isinstance(current, list):
            current.append(value)
        else:
            store[key] = [str(current), value]

    def _parse_experiment_summary(
        self, experiment_path: Optional[Path]
    ) -> tuple[dict[str, object], list[str], str]:
        summary_info: dict[str, object] = {
            "subject_metadata": {},
            "experiment_metadata": {},
            "environment_metadata": {},
            "actions_profile": "",
            "actions_performed": [],
            "info": {},
            "raw_text": "",
        }
        if not experiment_path:
            return summary_info, [], ""
        summary_path = experiment_path / "experiment_summary.txt"
        if not summary_path.exists():
            return summary_info, [], ""

        try:
            content = summary_path.read_text(encoding="utf-8")
        except OSError:
            return summary_info, [], ""

        summary_info["raw_text"] = content.rstrip()
        section = ""
        actions: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.endswith(":") and not stripped.startswith("-"):
                section = stripped.rstrip(":").strip().lower()
                continue

            if section == "subject metadata":
                match = re.match(r"-\s*(.+?):\s*(.*)", stripped)
                if match:
                    self._parse_summary_value(
                        summary_info["subject_metadata"], match.group(1), match.group(2)
                    )
                continue

            if section == "experiment metadata":
                match = re.match(r"-\s*(.+?):\s*(.*)", stripped)
                if match:
                    self._parse_summary_value(
                        summary_info["experiment_metadata"], match.group(1), match.group(2)
                    )
                continue

            if section == "environment metadata":
                match = re.match(r"-\s*(.+?):\s*(.*)", stripped)
                if match:
                    self._parse_summary_value(
                        summary_info["environment_metadata"], match.group(1), match.group(2)
                    )
                continue

            if section == "actions":
                if stripped.lower().startswith("selected profile:"):
                    profile_name = stripped.split(":", 1)[1].strip()
                    summary_info["actions_profile"] = profile_name
                elif stripped.startswith("-"):
                    action_name = stripped.lstrip("-").strip()
                    if action_name:
                        actions.append(action_name)
                continue

            if ":" in stripped and not stripped.endswith(":"):
                key, value = stripped.split(":", 1)
                self._parse_summary_value(summary_info["info"], key, value)

        summary_info["actions_performed"] = actions
        env_profile = ""
        env_metadata = summary_info.get("environment_metadata", {})
        if isinstance(env_metadata, dict):
            env_profile = str(env_metadata.get("profile", "")).strip()
        return summary_info, actions, env_profile

    def _parse_actions_list_value(self, value: str) -> list[str]:
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
        return [part.strip() for part in value.split(",") if part.strip()]

    def _parse_experiment_snapshot(
        self, experiment_path: Optional[Path]
    ) -> tuple[dict[str, dict[str, str]], list[str]]:
        snapshot_info: dict[str, dict[str, str]] = {}
        actions_list: list[str] = []
        if not experiment_path:
            return snapshot_info, actions_list
        snapshot_path = experiment_path / "experiment_snapshot.csv"
        if not snapshot_path.exists():
            return snapshot_info, actions_list

        try:
            with snapshot_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    category = (row.get("category") or "").strip()
                    key = (row.get("key") or "").strip()
                    value = (row.get("value") or "").strip()
                    if not category or not key:
                        continue
                    snapshot_info.setdefault(category, {})[key] = value
                    if category == "experiment" and key == "actions_list":
                        actions_list = self._parse_actions_list_value(value)
        except OSError:
            return {}, []

        return snapshot_info, actions_list

    def _participant_names(self) -> str | list[str]:
        for entry in self._capture_manifest_entries:
            participant = (entry.get("participant") or "").strip()
            second = (entry.get("second_participant") or "").strip()
            if participant:
                if second:
                    return [participant, second]
                return participant
        return "unknown"

    def _bandwidth_for_entry(self, entry: Optional[dict[str, str]]) -> int:
        if entry:
            bandwidth_value = entry.get("bandwidth")
            if bandwidth_value not in (None, ""):
                try:
                    return int(float(bandwidth_value))
                except (TypeError, ValueError):
                    pass
        return int(self.bandwidth_spin.value())

    @staticmethod
    def _sanitize_filename_component(value: object) -> str:
        text = str(value or "").strip()
        if not text or text == "—":
            return "unknown"
        text = text.replace(" ", "_")
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        return text.strip("_") or "unknown"

    @staticmethod
    def _export_action_label(action_name: str) -> str:
        cleaned = str(action_name or "").strip()
        if not cleaned:
            return "unknown"
        match = re.search(r"pre_action[_-]+(?P<label>.+)", cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = match.group("label")
        cleaned = re.sub(r"[_-]C\d+$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[_-]{2,}", "_", cleaned).strip("_- ")
        return cleaned or "unknown"

    def _default_export_filename(self, experiment_path: Path) -> Path:
        metadata = self._extract_experiment_metadata(experiment_path)
        participant_names = self._participant_names()
        if isinstance(participant_names, list):
            participant_name = "_".join(self._sanitize_filename_component(p) for p in participant_names)
        else:
            participant_name = self._sanitize_filename_component(participant_names)

        participant_id = self._sanitize_filename_component(metadata.get("participant_id", "unknown"))
        experiment_id = self._sanitize_filename_component(metadata.get("experiment_id", "unknown"))
        datetime_str = (metadata.get("datetime") or "").strip()
        parsed_dt = None
        if datetime_str and datetime_str != "—":
            try:
                parsed_dt = datetime.fromisoformat(datetime_str)
            except ValueError:
                parsed_dt = None
        if parsed_dt is None:
            parsed_dt = datetime.now()
        timestamp = parsed_dt.strftime("%Y%m%d_%H%M%S")
        filename = f"{participant_name}_{participant_id}_{experiment_id}_{timestamp}.pickle"
        return experiment_path / filename

    def _export_dataset_directory(self, experiment_path: Path, export_path: Path) -> Path:
        metadata = self._extract_experiment_metadata(experiment_path)
        participant_id = self._sanitize_filename_component(
            metadata.get("participant_id", "unknown")
        )
        experiment_id = self._sanitize_filename_component(
            metadata.get("experiment_id", "unknown")
        )
        folder_name = f"raw_data_PID{participant_id}_EXP{experiment_id}"
        return export_path.parent / folder_name

    def _prompt_export_split_by_sniffer(
        self,
    ) -> Optional[tuple[bool, bool, bool, str, bool, int, bool, int, list[str]]]:
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Dataset Options")
        layout = QVBoxLayout(dialog)
        layout.addWidget(
            QLabel(
                "To keep export sizes manageable, you can split the export into one "
                "file per sniffer."
            )
        )
        split_checkbox = QCheckBox("Create a separate file for each sniffer")
        split_checkbox.setChecked(True)
        layout.addWidget(split_checkbox)
        anonymize_checkbox = QCheckBox(
            "Anonymize participant data (replace names with participant IDs and set ages/birthdays to -1)"
        )
        anonymize_checkbox.setChecked(False)
        layout.addWidget(anonymize_checkbox)
        framework_layout = QHBoxLayout()
        framework_layout.addWidget(QLabel("Framework:"))
        framework_combo = QComboBox()
        framework_combo.addItem("UbiLocate", "ubilocate")
        framework_combo.addItem("Nexmon", "nexmon")
        framework_combo.setCurrentIndex(0)
        framework_layout.addWidget(framework_combo, 1)
        layout.addLayout(framework_layout)
        compress_checkbox = QCheckBox("Use high compression (xz)")
        compress_checkbox.setChecked(False)
        layout.addWidget(compress_checkbox)
        threads_checkbox = QCheckBox("Use multi-threaded export")
        threads_checkbox.setChecked(False)
        layout.addWidget(threads_checkbox)
        threads_layout = QHBoxLayout()
        threads_layout.addWidget(QLabel("Threads:"))
        threads_spin = QSpinBox()
        max_threads = max(1, os.cpu_count() or 1)
        threads_spin.setRange(1, max_threads)
        threads_spin.setValue(min(4, max_threads))
        threads_spin.setEnabled(False)
        threads_layout.addWidget(threads_spin, 1)
        layout.addLayout(threads_layout)
        archive_checkbox = QCheckBox("Compress exported folder after export (tar.gz)")
        archive_checkbox.setChecked(False)
        layout.addWidget(archive_checkbox)
        archive_threads_layout = QHBoxLayout()
        archive_threads_layout.addWidget(QLabel("Compression threads:"))
        archive_threads_spin = QSpinBox()
        max_threads = max(1, os.cpu_count() or 1)
        archive_threads_spin.setRange(1, max_threads)
        archive_threads_spin.setValue(min(4, max_threads))
        archive_threads_spin.setEnabled(False)
        archive_threads_layout.addWidget(archive_threads_spin, 1)
        layout.addLayout(archive_threads_layout)

        files_group = QGroupBox("Include additional files")
        files_layout = QVBoxLayout(files_group)
        file_options = [
            ("Experiment snapshot (CSV)", "experiment_snapshot.csv"),
            ("Experiment summary (TXT)", "experiment_summary.txt"),
            ("Events (CSV)", "events.csv"),
            ("Webcam video (MP4)", "webcam.mp4"),
            ("Webcam frames (CSV)", "webcam_frames.csv"),
            ("Hand landmarks (NPY)", "hand_landmarks.npy"),
            ("Hand landmarks preview (PNG)", "hand_landmarks.png"),
        ]
        file_checkboxes: dict[str, QCheckBox] = {}
        for label, filename in file_options:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            files_layout.addWidget(checkbox)
            file_checkboxes[filename] = checkbox
        layout.addWidget(files_group)

        archive_checkbox.toggled.connect(archive_threads_spin.setEnabled)
        threads_checkbox.toggled.connect(threads_spin.setEnabled)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        result = dialog.exec_()
        if result != QDialog.Accepted:
            return None
        selected_files = [
            filename
            for filename, checkbox in file_checkboxes.items()
            if checkbox.isChecked()
        ]
        return (
            split_checkbox.isChecked(),
            anonymize_checkbox.isChecked(),
            compress_checkbox.isChecked(),
            framework_combo.currentData() or "ubilocate",
            archive_checkbox.isChecked(),
            int(archive_threads_spin.value()),
            threads_checkbox.isChecked(),
            int(threads_spin.value()),
            selected_files,
        )

    def _ensure_xz_extension(self, export_path: Path, use_xz: bool) -> Path:
        if not use_xz:
            return export_path
        if export_path.suffix == ".xz":
            return export_path
        if export_path.suffix:
            return export_path.with_suffix(f"{export_path.suffix}.xz")
        return export_path.with_suffix(".xz")

    def _export_name_parts(self, export_path: Path) -> tuple[str, str, str]:
        suffixes = export_path.suffixes
        compression_suffix = ""
        data_suffix = ""
        if suffixes and suffixes[-1] == ".xz":
            compression_suffix = ".xz"
            data_suffix = "".join(suffixes[:-1])
        else:
            data_suffix = "".join(suffixes)
        base_name = export_path.name
        suffix_length = len(data_suffix + compression_suffix)
        if suffix_length:
            base_name = base_name[: -suffix_length]
        return base_name, data_suffix, compression_suffix

    def _export_archive_filename(self, experiment_path: Path) -> str:
        metadata = self._extract_experiment_metadata(experiment_path)
        participant_names = self._participant_names()
        if isinstance(participant_names, list):
            participant_name = "_".join(
                self._sanitize_filename_component(p) for p in participant_names
            )
        else:
            participant_name = self._sanitize_filename_component(participant_names)
        datetime_str = (metadata.get("datetime") or "").strip()
        parsed_dt = None
        if datetime_str and datetime_str != "—":
            try:
                parsed_dt = datetime.fromisoformat(datetime_str)
            except ValueError:
                parsed_dt = None
        if parsed_dt is None:
            parsed_dt = datetime.now()
        timestamp = parsed_dt.strftime("%Y%m%d_%H%M%S")
        return f"{participant_name}_{timestamp}_export.tar.gz"

    def _open_pgzip_handle(self, archive_path: Path, thread_count: int):
        if pgzip is None:
            raise RuntimeError("pgzip is not available")
        pgzip_kwargs = {}
        if thread_count > 1:
            pgzip_kwargs["thread"] = thread_count
        try:
            return pgzip.open(archive_path, "wb", **pgzip_kwargs)
        except TypeError:
            if "thread" in pgzip_kwargs:
                return pgzip.open(archive_path, "wb", threads=thread_count)
            raise

    def _compress_export_directory(
        self,
        export_dir: Path,
        experiment_path: Path,
        thread_count: int,
    ) -> Optional[Path]:
        if not export_dir.exists():
            return None
        if pgzip is None:
            QMessageBox.warning(
                self,
                "Compression unavailable",
                "Unable to compress the export because pgzip is not installed.",
            )
            return None
        archive_path = export_dir.parent / self._export_archive_filename(experiment_path)
        if archive_path.exists():
            archive_path.unlink(missing_ok=True)
        files = sorted([path for path in export_dir.rglob("*") if path.is_file()])
        total_steps = max(1, len(files))
        progress_dialog = QProgressDialog(
            "Compressing export folder...", "Cancel", 0, total_steps, self
        )
        progress_dialog.setWindowTitle("Archive Export")
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)
        base_dir = export_dir.parent

        try:
            with self._open_pgzip_handle(archive_path, thread_count) as archive_handle:
                with tarfile.open(fileobj=archive_handle, mode="w") as tar_handle:
                    if not files:
                        tar_handle.add(export_dir, arcname=export_dir.name)
                    for index, file_path in enumerate(files, start=1):
                        if progress_dialog.wasCanceled():
                            self._set_status("Compression canceled.")
                            archive_path.unlink(missing_ok=True)
                            return None
                        relative_path = file_path.relative_to(base_dir)
                        tar_handle.add(file_path, arcname=relative_path)
                        progress_dialog.setValue(index)
                        QApplication.processEvents()
        finally:
            progress_dialog.setValue(total_steps)
            QApplication.processEvents()
            progress_dialog.close()
        return archive_path

    def _open_export_handle(self, export_path: Path, use_xz: bool):
        if use_xz:
            return lzma.open(export_path, "wb")
        return export_path.open("wb")

    def _export_dataset(self) -> None:
        experiment_path = self._current_experiment_path()
        if not experiment_path:
            self._set_status("Select an experiment to export.")
            return

        export_options = self._prompt_export_split_by_sniffer()
        if export_options is None:
            return
        (
            split_by_sniffer,
            anonymize_data,
            use_xz,
            export_framework,
            archive_export,
            archive_threads,
            use_multithread,
            thread_count,
            selected_files,
        ) = export_options

        default_path = self._default_export_filename(experiment_path)
        default_path = self._ensure_xz_extension(default_path, use_xz)
        save_path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export Dataset",
            str(default_path),
            "Pickle Files (*.pickle *.pkl *.pickle.xz *.pkl.xz *.xz);;All Files (*)",
        )
        if not save_path_str:
            return

        export_path = Path(save_path_str)
        export_dir = self._export_dataset_directory(experiment_path, export_path)
        export_path = self._ensure_xz_extension(export_dir / export_path.name, use_xz)
        csi_root = experiment_path / "csi_captures"
        if not csi_root.exists():
            self._set_status("No csi_captures directory for this experiment.")
            return

        summary_info, actions_list, environment_profile = self._parse_experiment_summary(
            experiment_path
        )
        snapshot_info, snapshot_actions = self._parse_experiment_snapshot(experiment_path)
        participant_names = self._participant_names()
        participant_key: object = (
            tuple(participant_names)
            if isinstance(participant_names, list)
            else participant_names
        )
        primary_id = ""
        secondary_id = ""
        if anonymize_data:
            metadata = self._extract_experiment_metadata(experiment_path)
            fallback_id = str(metadata.get("participant_id", "unknown"))
            primary_id, secondary_id = self._participant_ids_for_anonymization(
                summary_info, fallback_id
            )
            participant_key, summary_info, snapshot_info = self._anonymize_export_payload(
                participant_key,
                summary_info,
                snapshot_info,
                fallback_id,
            )
        dataset: dict[tuple[object, object, object, object, str], list[object]] = {}
        activities = list(snapshot_actions or actions_list or [])

        pcap_entries: list[tuple[Path, Path, Path]] = []
        for ap_dir in sorted([p for p in csi_root.iterdir() if p.is_dir()]):
            for trial_dir in sorted([p for p in ap_dir.iterdir() if p.is_dir()]):
                for pcap_path in sorted(trial_dir.glob("*.pcap")):
                    pcap_entries.append((ap_dir, trial_dir, pcap_path))

        if not pcap_entries:
            self._set_status("No PCAP files found to export.")
            return

        progress_dialog = QProgressDialog(
            "Preparing export...", "Cancel", 0, len(pcap_entries) + 1, self
        )
        progress_dialog.setWindowTitle("Export Dataset")
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)
        start_time = time.monotonic()
        def _load_export_entry(
            ap_dir: Path,
            trial_dir: Path,
            pcap_path: Path,
            entry: Optional[dict[str, str]],
            sniffer_id: str,
        ) -> tuple[str, tuple[object, object, object, object, str], dict[str, object]]:
            framework = export_framework
            bandwidth = self._bandwidth_for_entry(entry)
            if framework == "ubilocate":
                csi_data, time_pkts, _, _ = self._load_ubilocate_csi(
                    pcap_path, bandwidth
                )
            else:
                csi_data, time_pkts, _, _ = self._load_nexmon_csi(
                    pcap_path, bandwidth
                )
            raw_action_name = (entry or {}).get("activity") or pcap_path.stem
            action_name = self._export_action_label(raw_action_name)
            trial_name = trial_dir.name
            key = (
                participant_key,
                sniffer_id,
                environment_profile,
                trial_name,
                action_name,
            )
            payload = {
                "csi": csi_data,
                "timestamps": np.asarray(time_pkts),
                "action": action_name,
                "summary_info": summary_info,
                "snapshot_info": snapshot_info,
                "actions": snapshot_actions or actions_list,
            }
            return sniffer_id, key, payload

        try:
            exported_paths: list[Path] = []
            export_path.parent.mkdir(parents=True, exist_ok=True)

            if split_by_sniffer:
                sniffer_entries: dict[
                    str, list[tuple[Path, Path, Path, Optional[dict[str, str]]]]
                ] = {}
                for ap_dir, trial_dir, pcap_path in pcap_entries:
                    entry = self._capture_manifest_entry(pcap_path)
                    sniffer_id = (entry or {}).get("ap_name") or ap_dir.name
                    sniffer_entries.setdefault(str(sniffer_id), []).append(
                        (ap_dir, trial_dir, pcap_path, entry)
                    )
                processed_index = 0
                dataset_by_sniffer: dict[
                    str, dict[tuple[object, object, object, object, str], dict[str, object]]
                ] = {sniffer_id: {} for sniffer_id in sniffer_entries}
                use_threads = use_multithread and thread_count > 1
                if use_threads:
                    future_map = {}
                    with ThreadPoolExecutor(max_workers=thread_count) as executor:
                        for sniffer_id, entries in sniffer_entries.items():
                            for ap_dir, trial_dir, pcap_path, entry in entries:
                                future = executor.submit(
                                    _load_export_entry,
                                    ap_dir,
                                    trial_dir,
                                    pcap_path,
                                    entry,
                                    sniffer_id,
                                )
                                future_map[future] = pcap_path
                        for future in as_completed(future_map):
                            if progress_dialog.wasCanceled():
                                self._set_status("Export canceled.")
                                return
                            pcap_path = future_map[future]
                            try:
                                sniffer_id, key, payload = future.result()
                            except Exception as exc:  # pragma: no cover - runtime export helper
                                self._set_status(f"Failed to read {pcap_path.name}: {exc}")
                                return
                            dataset_by_sniffer[sniffer_id][key] = payload
                            processed_index += 1
                            elapsed = time.monotonic() - start_time
                            remaining = (elapsed / processed_index) * (
                                len(pcap_entries) - processed_index
                            )
                            progress_dialog.setLabelText(
                                f"Processing {processed_index}/{len(pcap_entries)} "
                                f"(ETA {self._format_duration(remaining)})"
                            )
                            progress_dialog.setValue(processed_index)
                            QApplication.processEvents()
                else:
                    for sniffer_id, entries in sniffer_entries.items():
                        if progress_dialog.wasCanceled():
                            self._set_status("Export canceled.")
                            return
                        for ap_dir, trial_dir, pcap_path, entry in entries:
                            if progress_dialog.wasCanceled():
                                self._set_status("Export canceled.")
                                return
                            try:
                                sniffer_id, key, payload = _load_export_entry(
                                    ap_dir,
                                    trial_dir,
                                    pcap_path,
                                    entry,
                                    sniffer_id,
                                )
                            except Exception as exc:  # pragma: no cover - runtime export helper
                                self._set_status(f"Failed to read {pcap_path.name}: {exc}")
                                return
                            dataset_by_sniffer[sniffer_id][key] = payload
                            processed_index += 1
                            elapsed = time.monotonic() - start_time
                            remaining = (elapsed / processed_index) * (
                                len(pcap_entries) - processed_index
                            )
                            progress_dialog.setLabelText(
                                f"Processing {processed_index}/{len(pcap_entries)} "
                                f"(ETA {self._format_duration(remaining)})"
                            )
                            progress_dialog.setValue(processed_index)
                            QApplication.processEvents()

                for sniffer_id, dataset in dataset_by_sniffer.items():
                    if not dataset:
                        continue
                    progress_dialog.setLabelText(
                        f"Saving sniffer {sniffer_id}..."
                    )
                    QApplication.processEvents()
                    base_name, data_suffix, compression_suffix = self._export_name_parts(
                        export_path
                    )
                    sniffer_component = self._sanitize_filename_component(sniffer_id)
                    sniffer_filename = (
                        f"{base_name}_{sniffer_component}{data_suffix}{compression_suffix}"
                    )
                    sniffer_path = export_path.with_name(sniffer_filename)
                    info = {
                        "sniffer": sniffer_id,
                        "experiment_snapshot": snapshot_info,
                    }
                    with self._open_export_handle(sniffer_path, use_xz) as handle:
                        pickle.dump(
                            (dataset, activities, info),
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    exported_paths.append(sniffer_path)
                    dataset.clear()
            else:
                sniffer_numbers: set[str] = set()
                use_threads = use_multithread and thread_count > 1
                if use_threads:
                    future_map = {}
                    with ThreadPoolExecutor(max_workers=thread_count) as executor:
                        for ap_dir, trial_dir, pcap_path in pcap_entries:
                            entry = self._capture_manifest_entry(pcap_path)
                            sniffer_number = (entry or {}).get("ap_name") or ap_dir.name
                            sniffer_numbers.add(str(sniffer_number))
                            future = executor.submit(
                                _load_export_entry,
                                ap_dir,
                                trial_dir,
                                pcap_path,
                                entry,
                                str(sniffer_number),
                            )
                            future_map[future] = pcap_path
                        for index, future in enumerate(as_completed(future_map), start=1):
                            if progress_dialog.wasCanceled():
                                self._set_status("Export canceled.")
                                return
                            pcap_path = future_map[future]
                            try:
                                _, key, payload = future.result()
                            except Exception as exc:  # pragma: no cover - runtime export helper
                                self._set_status(f"Failed to read {pcap_path.name}: {exc}")
                                return
                            dataset[key] = payload
                            elapsed = time.monotonic() - start_time
                            remaining = (elapsed / index) * (len(pcap_entries) - index)
                            progress_dialog.setLabelText(
                                f"Processing {index}/{len(pcap_entries)} "
                                f"(ETA {self._format_duration(remaining)})"
                            )
                            progress_dialog.setValue(index)
                            QApplication.processEvents()
                else:
                    for index, (ap_dir, trial_dir, pcap_path) in enumerate(
                        pcap_entries, start=1
                    ):
                        if progress_dialog.wasCanceled():
                            self._set_status("Export canceled.")
                            return
                        entry = self._capture_manifest_entry(pcap_path)
                        sniffer_number = (entry or {}).get("ap_name") or ap_dir.name
                        sniffer_numbers.add(str(sniffer_number))
                        try:
                            _, key, payload = _load_export_entry(
                                ap_dir,
                                trial_dir,
                                pcap_path,
                                entry,
                                str(sniffer_number),
                            )
                        except Exception as exc:  # pragma: no cover - runtime export helper
                            self._set_status(f"Failed to read {pcap_path.name}: {exc}")
                            return
                        dataset[key] = payload
                        elapsed = time.monotonic() - start_time
                        remaining = (elapsed / index) * (len(pcap_entries) - index)
                        progress_dialog.setLabelText(
                            f"Processing {index}/{len(pcap_entries)} "
                            f"(ETA {self._format_duration(remaining)})"
                        )
                        progress_dialog.setValue(index)
                        QApplication.processEvents()

                if not dataset:
                    self._set_status("No PCAP files found to export.")
                    return

                progress_dialog.setLabelText("Saving export...")
                progress_dialog.setValue(len(pcap_entries))
                QApplication.processEvents()
                info = {
                    "sniffers": sorted(sniffer_numbers),
                    "experiment_snapshot": snapshot_info,
                }
                with self._open_export_handle(export_path, use_xz) as handle:
                    pickle.dump(
                        (dataset, activities, info),
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Export failed",
                f"Unable to write export file:\n{exc}",
            )
            return
        finally:
            progress_dialog.setValue(len(pcap_entries) + 1)
            QApplication.processEvents()
            progress_dialog.close()

        self._copy_export_metadata_files(
            experiment_path,
            export_path.parent,
            anonymize_data,
            primary_id,
            secondary_id,
            selected_files,
        )
        archive_path = None
        if archive_export:
            archive_path = self._compress_export_directory(
                export_path.parent,
                experiment_path,
                archive_threads,
            )

        if split_by_sniffer:
            status_message = (
                f"Exported dataset to {len(exported_paths)} files in {export_path.parent}"
            )
        else:
            status_message = f"Exported dataset to {export_path}"

        if archive_path:
            status_message = f"{status_message}. Compressed to {archive_path}"

        self._set_status(status_message)

    def _anonymize_export_payload(
        self,
        participant_key: object,
        summary_info: dict[str, object],
        snapshot_info: dict[str, dict[str, str]],
        fallback_id: str,
    ) -> tuple[object, dict[str, object], dict[str, dict[str, str]]]:
        summary_info = copy.deepcopy(summary_info)
        snapshot_info = copy.deepcopy(snapshot_info)
        primary_id, secondary_id = self._participant_ids_for_anonymization(
            summary_info, fallback_id
        )
        summary_info["raw_text"] = ""
        for key in ("subject_metadata", "experiment_metadata", "environment_metadata", "info"):
            section = summary_info.get(key)
            if isinstance(section, dict):
                self._anonymize_metadata_dict(section, primary_id, secondary_id)
        for section in snapshot_info.values():
            if isinstance(section, dict):
                self._anonymize_metadata_dict(section, primary_id, secondary_id)
        if secondary_id and secondary_id != primary_id:
            participant_key = (primary_id, secondary_id)
        else:
            participant_key = primary_id or participant_key
        return participant_key, summary_info, snapshot_info

    def _participant_ids_for_anonymization(
        self, summary_info: dict[str, object], fallback_id: str
    ) -> tuple[str, str]:
        primary_id = fallback_id or "unknown"
        secondary_id = ""
        subject_metadata = summary_info.get("subject_metadata", {})
        if isinstance(subject_metadata, dict):
            for key, value in subject_metadata.items():
                normalized = key.strip().lower().replace(" ", "_")
                if normalized == "participant_id":
                    primary_id = str(value or primary_id)
                if normalized == "second_participant_id":
                    secondary_id = str(value or secondary_id)
        if not secondary_id:
            secondary_id = primary_id
        return primary_id, secondary_id

    @staticmethod
    def _replace_metadata_value(value: object, replacement: object) -> object:
        if isinstance(value, list):
            return [replacement for _ in value]
        return replacement

    def _anonymize_metadata_dict(
        self, metadata: dict[str, object], primary_id: str, secondary_id: str
    ) -> None:
        for key, value in list(metadata.items()):
            lowered = key.lower()
            if "name" in lowered or "participant" in lowered:
                replacement = secondary_id if "second" in lowered else primary_id
                metadata[key] = self._replace_metadata_value(value, replacement)
            if "age" in lowered or "birth" in lowered:
                metadata[key] = self._replace_metadata_value(value, -1)

    def _anonymized_value_for_key(
        self, key: str, primary_id: str, secondary_id: str
    ) -> Optional[str]:
        lowered = key.lower()
        if "name" in lowered or "participant" in lowered:
            replacement = secondary_id if "second" in lowered else primary_id
            return str(replacement)
        if "age" in lowered or "birth" in lowered:
            return "-1"
        return None

    def _anonymize_summary_text(
        self, content: str, primary_id: str, secondary_id: str
    ) -> str:
        lines = []
        for line in content.splitlines():
            match = re.match(r"(\s*-\s*)([^:]+):\s*(.*)", line)
            if match:
                key = match.group(2).strip()
                replacement = self._anonymized_value_for_key(
                    key, primary_id, secondary_id
                )
                if replacement is not None:
                    line = f"{match.group(1)}{key}: {replacement}"
            lines.append(line)
        updated = "\n".join(lines)
        if content.endswith("\n"):
            updated += "\n"
        return updated

    def _write_anonymized_snapshot(
        self, source_path: Path, destination_path: Path, primary_id: str, secondary_id: str
    ) -> None:
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or ["category", "key", "value"]

        for row in rows:
            key = (row.get("key") or "").strip()
            replacement = self._anonymized_value_for_key(key, primary_id, secondary_id)
            if replacement is not None:
                row["value"] = replacement

        with destination_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _copy_export_metadata_files(
        self,
        experiment_path: Path,
        export_dir: Path,
        anonymize_data: bool,
        primary_id: str,
        secondary_id: str,
        selected_files: list[str],
    ) -> None:
        if not experiment_path:
            return
        if anonymize_data:
            if not primary_id:
                primary_id = "unknown"
            if not secondary_id:
                secondary_id = primary_id

        export_dir.mkdir(parents=True, exist_ok=True)
        files_to_copy = {
            "experiment_snapshot.csv",
            "events.csv",
            "experiment_summary.txt",
            "webcam.mp4",
            "webcam_frames.csv",
            "hand_landmarks.npy",
            "hand_landmarks.png",
        }
        selected_set = set(selected_files)
        for filename in sorted(files_to_copy):
            if filename not in selected_set:
                continue
            source_path = experiment_path / filename
            if not source_path.exists():
                continue
            destination_path = export_dir / filename
            try:
                if anonymize_data and filename == "experiment_summary.txt":
                    content = source_path.read_text(encoding="utf-8")
                    anonymized = self._anonymize_summary_text(
                        content, primary_id, secondary_id
                    )
                    destination_path.write_text(anonymized, encoding="utf-8")
                elif anonymize_data and filename == "experiment_snapshot.csv":
                    self._write_anonymized_snapshot(
                        source_path, destination_path, primary_id, secondary_id
                    )
                else:
                    shutil.copy2(source_path, destination_path)
            except OSError:
                continue
        self._write_export_example_script(export_dir)

    def _default_export_pickle_filename(self, export_dir: Path) -> str:
        pickle_files = list(export_dir.glob("*.pickle")) + list(export_dir.glob("*.pkl"))
        if not pickle_files:
            return "your_export_file.pickle"

        def _sniffer_sort_key(path: Path) -> tuple[int, int, str]:
            numbers = [int(value) for value in re.findall(r"\d+", path.stem)]
            if numbers:
                return (0, min(numbers), path.name.lower())
            return (1, 1_000_000_000, path.name.lower())

        return min(pickle_files, key=_sniffer_sort_key).name

    def _write_export_example_script(self, export_dir: Path) -> None:
        script_path = export_dir / "read_exported_pickle.py"
        default_pickle = self._default_export_pickle_filename(export_dir)
        example = textwrap.dedent(
            """\
            \"\"\"Example usage for exported CSI pickle files.

            Pickle structure:
                export = (dataset, activities, info)

                dataset: dict where keys are
                    (participant_id_or_tuple, sniffer_id, environment_profile, trial_name, action_name)
                values are dicts with:
                    "csi": complex numpy array shaped (packets, subcarriers, rx, tx)
                    "timestamps": numpy array shaped (packets,)
                    "action": action label string
                    "summary_info": experiment summary metadata dict
                    "snapshot_info": experiment snapshot metadata dict
                    "actions": list of action labels

                activities: list of activity labels in the export
                info: dict with "sniffers" (or "sniffer") and "experiment_snapshot"
            \"\"\"

            import pickle

            import matplotlib.pyplot as plt
            import numpy as np

            pickle_path = "your_export_file.pickle"

            with open(pickle_path, "rb") as handle:
                dataset, activities, info = pickle.load(handle)

            print(f"Loaded {len(dataset)} entries.")
            print(f"Activities: {activities}")
            print(f"Info: {info}")

            (key, record) = next(iter(dataset.items()))
            print(f"Example key: {key}")

            csi = record["csi"]
            subcarrier_index = 23
            subcarrier_index = min(subcarrier_index, csi.shape[1] - 1)
            magnitude = np.abs(csi[:, subcarrier_index, 0, 0])

            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axes[0].plot(magnitude)
            axes[0].set_title(f"CSI magnitude (subcarrier {subcarrier_index})")
            axes[0].set_ylabel("Magnitude")

            if csi.shape[3] >= 2:
                denominator = csi[:, subcarrier_index, 0, 0] + 1e-6
                ratio = csi[:, subcarrier_index, 0, 1] / denominator
                ratio_angle = np.angle(ratio)
                axes[1].plot(ratio_angle)
                axes[1].set_title(
                    f"CSI ratio angle (RX1 TX2/TX1, subcarrier {subcarrier_index})"
                )
                axes[1].set_ylabel("Angle (radians)")
            else:
                axes[1].text(
                    0.5,
                    0.5,
                    "Ratio angle requires at least 2 TX antennas.",
                    ha="center",
                    va="center",
                )
                axes[1].set_axis_off()

            axes[1].set_xlabel("Packet index")
            fig.tight_layout()
            plt.show()
            """
        ).replace("your_export_file.pickle", default_pickle)
        try:
            script_path.write_text(example, encoding="utf-8")
        except OSError:
            return

    def _load_nexmon_csi(
        self, pcap_path: Path, bandwidth: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        processor = ProcessPcap(str(pcap_path.parent), bw=bandwidth, tx_loc=[0, 0])
        csi_data, time_pkts, _, mac_addrs = processor.process_pcap(
            str(pcap_path), bw=bandwidth
        )
        rx_count = getattr(processor, "num_cores", 1) or 1
        tx_count = 1
        expected_streams = rx_count * tx_count
        expected_cols = expected_streams * processor.nfft
        if csi_data.shape[1] != expected_cols:
            raise ValueError(
                f"Unexpected Nexmon CSI shape {csi_data.shape}; expected second dimension {expected_cols}"
            )
        csi_data = csi_data.reshape((-1, processor.nfft, rx_count, tx_count))
        return csi_data, np.asarray(time_pkts), mac_addrs, processor.nfft

    def _load_ubilocate_csi(
        self, pcap_path: Path, bandwidth: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        csi_data, time_pkts, mac_addrs = _read_ubilocate_csi(
            str(pcap_path), bw=bandwidth, is_4ss=True
        )
        nfft = int(3.2 * bandwidth)
        return csi_data, time_pkts, mac_addrs, nfft

    def _set_combo_to_data(self, combo: QComboBox, target: Path | str | None) -> bool:
        if target is None:
            return False

        target_path = Path(target)
        for index in range(combo.count()):
            data_path = Path(combo.itemData(index) or "")
            if data_path.resolve() == target_path.resolve():
                combo.setCurrentIndex(index)
                return True
        return False

    def _stream_tx_rx_mapping(self, rx_count: int, tx_count: int) -> list[tuple[int, int]]:
        if rx_count <= 0 or tx_count <= 0:
            return []

        mapping: list[tuple[int, int]] = []
        for tx_idx in range(tx_count):
            for rx_idx in range(rx_count):
                mapping.append((tx_idx + 1, rx_idx + 1))
        return mapping

    @staticmethod
    def _farthest_circular_index(current: int, total: int) -> int | None:
        if total < 2:
            return None
        distances: list[tuple[int, int]] = []
        for idx in range(total):
            if idx == current:
                continue
            diff = abs(idx - current)
            circular = min(diff, total - diff)
            distances.append((circular, idx))
        max_distance = max(distance for distance, _ in distances)
        farthest_indices = [idx for distance, idx in distances if distance == max_distance]
        return min(farthest_indices) if farthest_indices else None

    def _populate_stream_checkboxes(self, rx_count: int, tx_count: int) -> None:
        if (rx_count, tx_count) != getattr(self, "_current_stream_shape", (None, None)):
            while self.stream_layout.count():
                item = self.stream_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.stream_checkboxes = []
            self.subcarrier_spins = []
            self._current_stream_shape = (rx_count, tx_count)

            mapping = self._stream_tx_rx_mapping(rx_count, tx_count)
            for idx, (tx_idx, rx_idx) in enumerate(mapping):
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)

                checkbox = QCheckBox(f"Stream {idx + 1} (TX {tx_idx}, RX {rx_idx})")
                checkbox.setChecked(True)

                sub_spin = QSpinBox()
                sub_spin.setRange(1, 256)
                sub_spin.setValue(30)

                row_layout.addWidget(checkbox)
                row_layout.addWidget(QLabel("Subcarrier:"))
                row_layout.addWidget(sub_spin)
                row_layout.addStretch(1)

                self.stream_layout.addWidget(row_widget)
                self.stream_checkboxes.append(checkbox)
                self.subcarrier_spins.append(sub_spin)

            if rx_count * tx_count == 0:
                placeholder = QLabel("No streams detected")
                self.stream_layout.addWidget(placeholder)

            self.stream_layout.addStretch(1)

    def _selected_stream_indices(self) -> list[int]:
        return [idx for idx, cb in enumerate(self.stream_checkboxes) if cb.isChecked()]

    def _on_pcap_selection_changed(self) -> None:
        framework = self._default_framework_for_pcap(self._current_pcap_path())
        self._set_framework_selection(framework)

    def _on_mac_selection_changed(self) -> None:
        if not self.mac_timestamp or self.histogram_ax is None:
            return
        selected_mac = self.mac_combo.currentData()
        if not selected_mac:
            return
        self._plot_mac_histogram(selected_mac)

    def _current_experiment_path(self) -> Optional[Path]:
        data = self.experiment_combo.currentData()
        return Path(data) if data else None

    def _current_ap_path(self) -> Optional[Path]:
        data = self.ap_combo.currentData()
        return Path(data) if data else None

    def _current_trial_path(self) -> Optional[Path]:
        data = self.trial_combo.currentData()
        return Path(data) if data else None

    def _current_pcap_path(self) -> Optional[Path]:
        data = self.pcap_combo.currentData()
        return Path(data) if data else None

    def _extract_experiment_metadata(self, experiment_path: Optional[Path]) -> dict[str, str]:
        metadata = {
            "datetime": "—",
            "experiment_id": "—",
            "participant_id": "—",
            "duration": "—",
        }

        manifest_entries = self._capture_manifest_entries
        snapshot_info: dict[str, dict[str, str]] = {}
        if experiment_path:
            snapshot_info, _ = self._parse_experiment_snapshot(experiment_path)

        def _snapshot_value(section: dict[str, str], *candidates: str) -> str:
            if not section:
                return ""
            normalized = {
                key.strip().lower().replace(" ", "_"): str(value or "").strip()
                for key, value in section.items()
            }
            for candidate in candidates:
                value = normalized.get(candidate)
                if value:
                    return value
            return ""

        participant_from_manifest = next(
            (
                (entry.get("participant_id") or "").strip()
                for entry in manifest_entries
                if entry.get("participant_id")
            ),
            "",
        )

        timestamp_from_manifest = next(
            (
                (entry.get("timestamp") or "").strip()
                for entry in manifest_entries
                if entry.get("timestamp")
            ),
            "",
        )

        experiment_snapshot = snapshot_info.get("experiment", {})
        participant_snapshot = snapshot_info.get("participant", {})
        snapshot_experiment_id = _snapshot_value(
            experiment_snapshot, "experiment_id", "exp_id"
        )
        if not snapshot_experiment_id:
            for section in snapshot_info.values():
                snapshot_experiment_id = _snapshot_value(
                    section, "experiment_id", "exp_id"
                )
                if snapshot_experiment_id:
                    break
        snapshot_participant_id = _snapshot_value(participant_snapshot, "participant_id")
        if snapshot_experiment_id:
            metadata["experiment_id"] = snapshot_experiment_id
        if snapshot_participant_id:
            metadata["participant_id"] = snapshot_participant_id

        if experiment_path:
            parts = experiment_path.name.split("_")
            if len(parts) >= 5:
                if metadata["experiment_id"] in {"—", ""}:
                    metadata["experiment_id"] = parts[2] or "—"
                if metadata["participant_id"] in {"—", ""}:
                    metadata["participant_id"] = parts[3] or "—"
                ts_part = parts[4]
                try:
                    parsed_dt = datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
                    metadata["datetime"] = parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass

        if metadata["participant_id"] in {"—", ""} and participant_from_manifest:
            metadata["participant_id"] = participant_from_manifest

        if metadata["datetime"] == "—" and timestamp_from_manifest:
            try:
                parsed_manifest_ts = datetime.fromisoformat(timestamp_from_manifest)
                metadata["datetime"] = parsed_manifest_ts.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        duration_seconds = self._compute_experiment_duration_seconds(
            experiment_path
        )
        if duration_seconds is not None:
            metadata["duration"] = self._format_duration(duration_seconds)

        return metadata

    @staticmethod
    def _format_duration(total_seconds: float) -> str:
        try:
            total_seconds = max(0.0, float(total_seconds))
        except (TypeError, ValueError):
            return "—"
        delta = timedelta(seconds=total_seconds)
        # Remove microseconds for a cleaner display
        cleaned = delta - timedelta(microseconds=delta.microseconds)
        return str(cleaned)

    def _compute_experiment_duration_seconds(
        self, experiment_path: Optional[Path]
    ) -> Optional[float]:
        summary_duration = self._duration_from_summary_file(experiment_path)
        if summary_duration is not None:
            return summary_duration

        start_times: list[float] = []
        end_times: list[float] = []
        mid_times: list[float] = []

        for entry in self._capture_manifest_entries:
            start_ns = entry.get("start_timestamp_ns")
            end_ns = entry.get("end_timestamp_ns")
            iso_ts = entry.get("timestamp")

            try:
                if start_ns not in (None, ""):
                    start_times.append(int(start_ns) / 1e9)
            except (TypeError, ValueError):
                pass

            try:
                if end_ns not in (None, ""):
                    end_times.append(int(end_ns) / 1e9)
            except (TypeError, ValueError):
                pass

            if iso_ts:
                try:
                    iso_dt = datetime.fromisoformat(str(iso_ts))
                    mid_times.append(iso_dt.timestamp())
                except ValueError:
                    pass

        mid_times.extend(self._log_time_candidates(experiment_path))
        mid_times.extend(self._pcap_time_candidates(experiment_path))

        if start_times and end_times:
            return max(end_times) - min(start_times)

        if mid_times:
            return max(mid_times) - min(mid_times)

        return None

    def _duration_from_summary_file(self, experiment_path: Optional[Path]) -> Optional[float]:
        if not experiment_path:
            return None

        summary_path = experiment_path / "experiment_summary.txt"
        if not summary_path.exists():
            return None

        try:
            content = summary_path.read_text(encoding="utf-8")
        except OSError:
            return None

        match = re.search(
            r"Elapsed time:\s*(?P<duration>\d{2}:\d{2}:\d{2}(?:\.\d{3})?)",
            content,
            re.IGNORECASE,
        )
        if not match:
            return None

        duration_str = match.group("duration")
        parts = re.match(
            r"(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})(?:\.(?P<millis>\d{3}))?",
            duration_str,
        )
        if not parts:
            return None

        try:
            hours = int(parts.group("hours"))
            minutes = int(parts.group("minutes"))
            seconds = int(parts.group("seconds"))
            millis = int(parts.group("millis") or 0)
        except (TypeError, ValueError):
            return None

        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

    def _log_time_candidates(self, experiment_path: Optional[Path]) -> list[float]:
        if not experiment_path:
            return []

        log_path = experiment_path / "session.log"
        if not log_path.exists():
            return []

        timestamp_match = re.search(r"(\d{8}_\d{6})", experiment_path.name)
        base_dt = None
        if timestamp_match:
            try:
                base_dt = datetime.strptime(timestamp_match.group(1), "%Y%m%d_%H%M%S")
            except ValueError:
                base_dt = None

        times: list[float] = []
        try:
            with log_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    match = re.search(r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\]", line)
                    if not match:
                        continue
                    try:
                        time_part = datetime.strptime(
                            match.group(1), "%H:%M:%S.%f"
                        ).time()
                    except ValueError:
                        continue

                    if base_dt is None:
                        continue

                    candidate_dt = datetime.combine(base_dt.date(), time_part)
                    if candidate_dt < base_dt:
                        candidate_dt += timedelta(days=1)
                    times.append(candidate_dt.timestamp())
        except Exception:
            return []

        return times

    def _pcap_time_candidates(self, experiment_path: Optional[Path]) -> list[float]:
        if not experiment_path:
            return []

        pcap_dir = experiment_path / "csi_captures"
        if not pcap_dir.exists():
            return []

        times: list[float] = []
        for pcap in pcap_dir.rglob("*.pcap"):
            try:
                times.append(pcap.stat().st_mtime)
            except OSError:
                continue
        return times

    def _update_experiment_metadata_display(self) -> None:
        metadata = self._extract_experiment_metadata(self._current_experiment_path())
        self.experiment_datetime_label.setText(metadata.get("datetime", "—"))
        self.experiment_id_label.setText(metadata.get("experiment_id", "—"))
        self.participant_id_label.setText(metadata.get("participant_id", "—"))
        self.experiment_duration_label.setText(metadata.get("duration", "—"))

    def _refresh_experiments(self) -> None:
        self.experiment_combo.clear()
        self._capture_manifest_entries = []
        if not self.results_root.exists():
            self._set_status(f"Results directory not found: {self.results_root}")
            self._update_experiment_metadata_display()
            return

        experiments = sorted(
            [p for p in self.results_root.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for exp in experiments:
            self.experiment_combo.addItem(exp.name, str(exp))

        if experiments:
            self.experiment_combo.setCurrentIndex(0)
            self._populate_access_points()
        else:
            self._set_status("No experiments found in results directory.")
            self._update_experiment_metadata_display()

    def _populate_access_points(self) -> None:
        self.ap_combo.clear()
        experiment_path = self._current_experiment_path()
        if not experiment_path:
            self._set_status("Select an experiment to continue.")
            return

        self._capture_manifest = self._load_capture_manifest(experiment_path)

        ap_root = experiment_path / "csi_captures"
        if not ap_root.exists():
            self._set_status("No csi_captures directory for this experiment.")
            self._update_experiment_metadata_display()
            return

        ap_dirs = sorted([p for p in ap_root.iterdir() if p.is_dir()], key=lambda p: p.name)
        for ap_dir in ap_dirs:
            self.ap_combo.addItem(ap_dir.name, str(ap_dir))

        if ap_dirs:
            self._populate_trials()
        else:
            self._set_status("No access point captures found for this experiment.")

        self._update_experiment_metadata_display()

    def _populate_trials(self) -> None:
        self.trial_combo.clear()
        ap_path = self._current_ap_path()
        if not ap_path:
            self._set_status("Select an access point to continue.")
            return

        trial_dirs = sorted([p for p in ap_path.iterdir() if p.is_dir()], key=lambda p: p.name)
        for trial_dir in trial_dirs:
            self.trial_combo.addItem(trial_dir.name, str(trial_dir))

        if trial_dirs:
            self._populate_pcaps()
        else:
            self._set_status("No trials found for this access point.")

    def _populate_pcaps(self) -> None:
        self.pcap_combo.clear()
        trial_path = self._current_trial_path()
        if not trial_path:
            self._set_status("Select a trial to continue.")
            return

        pcap_files = sorted(trial_path.glob("*.pcap"))
        for pcap in pcap_files:
            self.pcap_combo.addItem(pcap.name, str(pcap))

        if not pcap_files:
            self._set_status("No PCAP files found in the selected trial.")
        else:
            self._set_status("")
        self._on_pcap_selection_changed()

    def _apply_initial_selection(self) -> None:
        if not self._initial_selection:
            return

        experiment = self._initial_selection.get("experiment")
        ap = self._initial_selection.get("ap")
        trial = self._initial_selection.get("trial")
        pcap = self._initial_selection.get("pcap")

        if experiment and self._set_combo_to_data(self.experiment_combo, experiment):
            self._populate_access_points()
        if ap and self._set_combo_to_data(self.ap_combo, ap):
            self._populate_trials()
        if trial and self._set_combo_to_data(self.trial_combo, trial):
            self._populate_pcaps()
        if pcap:
            self._set_combo_to_data(self.pcap_combo, pcap)

        self._on_pcap_selection_changed()

        if self._auto_plot and self.pcap_combo.currentData():
            self._load_and_plot()

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def _apply_hampel_filter(self, values: np.ndarray) -> np.ndarray:
        if hampel is None:
            self._set_status("Hampel filter unavailable (missing dependency). Proceeding without filtering.")
            return values

        try:
            filtered = hampel(values, window_size=8)
        except Exception as exc:  # pragma: no cover - runtime visualization helper
            self._set_status(f"Hampel filter failed: {exc}")
            return values

        if hasattr(filtered, "filtered_data"):
            return np.asarray(filtered.filtered_data)
        if isinstance(filtered, tuple):
            return np.asarray(filtered[0])
        return np.asarray(filtered)

    def _apply_butterworth_filter(self, values: np.ndarray) -> np.ndarray:
        if values.size < 5:
            return values

        try:
            sos = signal.butter(4, 0.2, btype="low", output="sos")
            return signal.sosfiltfilt(sos, values)
        except Exception as exc:  # pragma: no cover - runtime visualization helper
            self._set_status(f"Butterworth filter failed: {exc}")
            return values

    def _populate_mac_combo(self, mac_timestamp: dict[str, list[float]]) -> None:
        self.mac_combo.blockSignals(True)
        self.mac_combo.clear()
        for mac in sorted(mac_timestamp.keys()):
            self.mac_combo.addItem(mac, mac)
        self.mac_combo.setEnabled(bool(mac_timestamp))
        self.mac_combo.blockSignals(False)

    def _plot_mac_histogram(self, selected_mac: str) -> None:
        if self.histogram_ax is None:
            return
        self.histogram_ax.clear()
        diffs = np.diff(self.mac_timestamp.get(selected_mac, []))
        plot_histogram_with_annotations(
            diffs,
            self.histogram_ax,
            # title=f"Sampling Rate Histogram ({selected_mac})" if selected_mac else "Sampling Rate Histogram",
            title="",
            xlabel="Interval (s) - " + f"Sampling Rate Histogram ({selected_mac})" if selected_mac else "Sampling Rate Histogram" ,
        )
        self.canvas.draw_idle()

    def _load_and_plot(self) -> None:
        pcap_path = self._current_pcap_path()
        if not pcap_path:
            self.packet_count_label.setText("Packets: -")
            self._set_status("Please select a PCAP file to visualize.")
            return

        bandwidth = self.bandwidth_spin.value()
        framework = self.framework_combo.currentData() or self._default_framework_for_pcap(
            pcap_path
        )

        try:
            if framework == "ubilocate":
                csi_data, time_pkts, mac_addrs, nfft = self._load_ubilocate_csi(
                    pcap_path, bandwidth
                )
            else:
                csi_data, time_pkts, mac_addrs, nfft = self._load_nexmon_csi(
                    pcap_path, bandwidth
                )
        except Exception as exc:  # pragma: no cover - runtime visualization helper
            self.packet_count_label.setText("Packets: -")
            self._set_status(f"Failed to read PCAP: {exc}")
            return

        if csi_data.size == 0:
            self.packet_count_label.setText("Packets: 0")
            self._set_status("No CSI data found in the selected PCAP.")
            return

        time_vals = np.asarray(time_pkts)
        max_points = min(len(time_vals), csi_data.shape[0])
        time_vals = time_vals[:max_points]
        mac_addrs = mac_addrs[:max_points]
        self.packet_count_label.setText(f"Packets: {max_points}")

        if self.phase_processing_checkbox.isChecked():
            try:
                flat_csi = csi_data.reshape(csi_data.shape[0], -1)
                processed = ant_processing(flat_csi)
                csi_data = processed.reshape(csi_data.shape)
            except Exception as exc:  # pragma: no cover - runtime visualization helper
                self._set_status(f"Phase processing failed: {exc}")
                return

        try:
            self.figure.clear()
            rx_count = csi_data.shape[2] if csi_data.ndim >= 3 else 1
            tx_count = csi_data.shape[3] if csi_data.ndim >= 4 else 1
            total_streams = rx_count * tx_count
            self._populate_stream_checkboxes(rx_count, tx_count)
            selected_streams = [idx for idx in self._selected_stream_indices() if idx < total_streams]
            streams_to_plot = min(len(selected_streams), len(self.subcarrier_spins))
            if streams_to_plot == 0:
                raise ValueError("Select at least one RX/TX stream to plot.")

            grid = self.figure.add_gridspec(
                streams_to_plot + 1,
                3,
                height_ratios=[1] * streams_to_plot + [0.8],
                width_ratios=[1, 1, 1],
                wspace=0.3,
            )

            # Expand the canvas height based on the number of plots so the scroll area
            # shows scrollbars when multiple streams are displayed.
            plot_height_inches = max(8, (streams_to_plot + 1) * 2.6)
            self.figure.set_size_inches(10, plot_height_inches)
            self.canvas.setMinimumHeight(int(plot_height_inches * self.figure.get_dpi()))

            axes_mag = [self.figure.add_subplot(grid[i, 0]) for i in range(streams_to_plot)]
            axes_phase = [
                self.figure.add_subplot(grid[i, 1], sharex=axes_mag[0])
                for i in range(streams_to_plot)
            ]
            axes_ratio_phase = [
                self.figure.add_subplot(grid[i, 2], sharex=axes_mag[0])
                for i in range(streams_to_plot)
            ]
            histogram_ax = self.figure.add_subplot(grid[streams_to_plot, :])

            for ant, spin in enumerate(self.subcarrier_spins[:streams_to_plot]):
                stream_idx = selected_streams[ant]
                sub_idx = spin.value() - 1
                if sub_idx >= nfft:
                    raise ValueError(
                        f"Subcarrier {sub_idx + 1} exceeds available carriers ({nfft})."
                    )
                rx_idx = stream_idx % rx_count
                tx_idx = stream_idx // rx_count
                stream_slice = csi_data[:max_points, sub_idx, rx_idx, tx_idx]
                magnitudes = np.abs(stream_slice)
                phases = np.angle(stream_slice)
                ratio_phase_label = (
                    f"Stream {stream_idx + 1} Ratio Phase "
                    f"(SC {sub_idx + 1}, RX {rx_idx + 1}/TX {tx_idx + 1})"
                )
                if tx_count > 1:
                    ref_tx_idx = self._farthest_circular_index(tx_idx, tx_count)
                    if ref_tx_idx is None:
                        ratio_slice = np.full_like(stream_slice, np.nan)
                        ratio_phase_label += " (No alternate TX)"
                    else:
                        ref_slice = csi_data[:max_points, sub_idx, rx_idx, ref_tx_idx]
                        ratio_slice = stream_slice / (1e-6 + ref_slice)
                        ratio_phase_label = (
                            f"Stream {stream_idx + 1} Ratio Phase "
                            f"(SC {sub_idx + 1}, RX {rx_idx + 1}, TX {tx_idx + 1}/TX {ref_tx_idx + 1})"
                        )
                elif rx_count > 1:
                    ref_rx_idx = self._farthest_circular_index(rx_idx, rx_count)
                    if ref_rx_idx is None:
                        ratio_slice = np.full_like(stream_slice, np.nan)
                        ratio_phase_label += " (No alternate RX)"
                    else:
                        ref_slice = csi_data[:max_points, sub_idx, ref_rx_idx, tx_idx]
                        ratio_slice = stream_slice / ref_slice
                        ratio_phase_label = (
                            f"Stream {stream_idx + 1} Ratio Phase "
                            f"(SC {sub_idx + 1}, TX {tx_idx + 1}, RX {rx_idx + 1}/RX {ref_rx_idx + 1})"
                        )
                else:
                    ratio_slice = np.full_like(stream_slice, np.nan)
                    ratio_phase_label += " (Unavailable)"
                ratio_phases = np.angle(ratio_slice)
                if self.hampel_checkbox.isChecked():
                    magnitudes = self._apply_hampel_filter(magnitudes)
                    phases = self._apply_hampel_filter(phases)
                    ratio_phases = self._apply_hampel_filter(ratio_phases)
                if self.butterworth_checkbox.isChecked():
                    magnitudes = self._apply_butterworth_filter(magnitudes)
                    phases = self._apply_butterworth_filter(phases)
                    ratio_phases = self._apply_butterworth_filter(ratio_phases)

                mag_line, = axes_mag[ant].plot(
                    time_vals,
                    magnitudes,
                    label=(
                        f"Stream {stream_idx + 1} |CSI| "
                        f"(SC {sub_idx + 1}, RX {rx_idx + 1}/TX {tx_idx + 1})"
                    ),
                    color="tab:blue",
                )
                phase_line, = axes_phase[ant].plot(
                    time_vals,
                    phases,
                    label=(
                        f"Stream {stream_idx + 1} Phase "
                        f"(SC {sub_idx + 1}, RX {rx_idx + 1}/TX {tx_idx + 1})"
                    ),
                    color="tab:orange",
                )
                ratio_line, = axes_ratio_phase[ant].plot(
                    time_vals,
                    ratio_phases,
                    label=ratio_phase_label,
                    color="tab:green",
                )
                axes_mag[ant].set_ylabel("|CSI|")
                axes_phase[ant].set_ylabel("Phase (rad)")
                axes_ratio_phase[ant].set_ylabel("Ratio Phase (rad)")
                axes_mag[ant].grid(True)
                axes_phase[ant].grid(True)
                axes_ratio_phase[ant].grid(True)
                handles = [mag_line, phase_line, ratio_line]
                labels = [h.get_label() for h in handles]
                axes_phase[ant].legend(handles, labels, loc="center")

            axes_mag[-1].set_xlabel("Time (s)")
            axes_phase[-1].set_xlabel("Time (s)")
            axes_ratio_phase[-1].set_xlabel("Time (s)")

            mac_timestamp: dict[str, list[float]] = {}
            for mac, ts in zip(mac_addrs, time_vals):
                mac_timestamp.setdefault(mac, []).append(ts)

            self.mac_timestamp = mac_timestamp
            self.histogram_ax = histogram_ax
            self._populate_mac_combo(mac_timestamp)
            if mac_timestamp:
                self._plot_mac_histogram(self.mac_combo.currentData() or next(iter(mac_timestamp)))
            else:
                plot_histogram_with_annotations([], self.histogram_ax)
                self.mac_combo.clear()
                self.mac_combo.setEnabled(False)
                self.canvas.draw_idle()
            applied_filters = []
            if self.hampel_checkbox.isChecked():
                applied_filters.append("Hampel")
            if self.butterworth_checkbox.isChecked():
                applied_filters.append("Butterworth")
            if self.phase_processing_checkbox.isChecked():
                applied_filters.append("Phase processing")
            filter_note = f" with {' and '.join(applied_filters)} filter" if applied_filters else ""
            self._set_status(
                f"Loaded {pcap_path.name} with {max_points} packets (bandwidth {bandwidth} MHz, {framework} framework){filter_note}."
            )
        except Exception as exc:  # pragma: no cover - runtime visualization helper
            self._set_status(f"Unable to plot CSI data: {exc}")


def launch_viewer(results_root: Path | str = "results") -> None:
    """Launch the standalone PCAP CSI explorer window."""

    app = QApplication.instance() or QApplication([])
    window = PCAPExplorerWindow(results_root=results_root)
    window.show()
    app.exec_()


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    launch_viewer()
