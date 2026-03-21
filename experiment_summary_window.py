"""Experiment summary window with interactive plots and metadata."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtWidgets import QAbstractItemView, QHeaderView
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from password_manager import is_password_required, verify_password

_SUMMARY_WINDOWS: List[QWidget] = []


def register_summary_window(window: QWidget) -> None:
    """Keep the summary window alive until the user closes it."""

    _SUMMARY_WINDOWS.append(window)

    def _cleanup(*_args, win=window):
        try:
            _SUMMARY_WINDOWS.remove(win)
        except ValueError:
            pass

    window.destroyed.connect(_cleanup)


class ExperimentSummaryWindow(QWidget):
    """Displays captured signals and metadata after the experiment."""

    def __init__(
        self,
        *,
        results_dir: Optional[Path],
        subject_info: Optional[Dict[str, Any]] = None,
        experiment_info: Optional[Dict[str, Any]] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        actions_profile_name: Optional[str] = None,
        actions_list: Optional[List[str]] = None,
        elapsed_time: float = 0.0,
        expected_duration: float = 0.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.results_dir = Path(results_dir) if results_dir else None
        self.subject_info = subject_info or {}
        self.experiment_info = experiment_info or {}
        self.environment_info = environment_info or {}
        self.actions_profile_name = actions_profile_name or ""
        self.actions_list = actions_list or []
        self.elapsed_time = max(0.0, float(elapsed_time or 0.0))
        self.expected_duration = max(0.0, float(expected_duration or 0.0))
        self.capture_entries = self._load_capture_manifest()

        self.setWindowTitle("WIRLab - Experiment Summary")
        self.resize(1100, 850)

        self._summary_locked = is_password_required()
        self._content_ready = False
        self._info_text = self._build_info_text()
        self._persist_summary_file(self._info_text)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        lock_row = QHBoxLayout()
        self.lock_status_label = QLabel(
            "Summary locked. Enter the password to view.", self
        )
        lock_row.addWidget(self.lock_status_label, 1)
        self.btn_toggle_lock = QPushButton("Unlock Summary", self)
        self.btn_toggle_lock.clicked.connect(self._on_toggle_lock)
        lock_row.addWidget(self.btn_toggle_lock)
        root_layout.addLayout(lock_row)

        self.content_container = QWidget()
        self.content_container.setVisible(False)
        self.content_layout = QVBoxLayout(self.content_container)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(10)
        root_layout.addWidget(self.content_container, 1)

        if not is_password_required():
            self._ensure_content_built()
            self._set_lock_state(False)
        else:
            self._set_lock_state(True)

    # ------------------------------------------------------------------
    # Lock/unlock helpers
    # ------------------------------------------------------------------
    def _set_lock_state(self, locked: bool) -> None:
        self._summary_locked = locked
        if hasattr(self, "content_container"):
            self.content_container.setVisible(not locked and self._content_ready)

        status_text = (
            "Summary locked. Enter the password to view."
            if locked
            else "Summary unlocked. Experiment details are visible."
        )
        if hasattr(self, "lock_status_label"):
            self.lock_status_label.setText(status_text)
        if hasattr(self, "btn_toggle_lock"):
            self.btn_toggle_lock.setText(
                "Unlock Summary" if locked else "Lock Summary"
            )

    def _ensure_content_built(self) -> None:
        if self._content_ready:
            return

        info_label = QLabel(self._info_text)
        info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        info_label.setWordWrap(True)

        captures_table = self._build_capture_table()

        summary_row = QWidget()
        summary_layout = QHBoxLayout(summary_row)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(12)
        summary_layout.addWidget(info_label, 1)
        summary_layout.addWidget(captures_table, 1)

        self.content_layout.addWidget(summary_row)

        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        figure_container = QWidget()
        figure_layout = QVBoxLayout(figure_container)
        figure_layout.setContentsMargins(0, 0, 0, 0)
        figure_layout.setSpacing(4)
        figure_layout.addWidget(self.toolbar)
        figure_layout.addWidget(self.canvas)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(figure_container)
        self.content_layout.addWidget(scroll_area, 1)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)

        pcap_button = QPushButton("Open PCAP Viewer")
        pcap_button.clicked.connect(self._open_pcap_viewer)
        button_layout.addWidget(pcap_button, alignment=Qt.AlignRight)

        close_button = QPushButton("Close and Exit")
        close_button.clicked.connect(self._close_application)
        close_button.setDefault(True)
        button_layout.addWidget(close_button, alignment=Qt.AlignRight)
        button_row.setLayout(button_layout)
        self.content_layout.addWidget(button_row, alignment=Qt.AlignRight)

        self._plot_data()
        self._content_ready = True
        self.content_container.setVisible(not self._summary_locked)

    def _on_toggle_lock(self) -> None:
        if not self._summary_locked:
            self._set_lock_state(True)
            return

        if not is_password_required():
            self._ensure_content_built()
            self._set_lock_state(False)
            QMessageBox.information(
                self,
                "Summary unlocked",
                "Admin mode enabled. Password bypassed.",
            )
            return

        password, ok = QInputDialog.getText(
            self,
            "Unlock summary",
            "Enter password to view the experiment summary:",
            QLineEdit.Password,
        )
        if not ok:
            return

        if verify_password(password):
            self._ensure_content_built()
            self._set_lock_state(False)
            QMessageBox.information(
                self,
                "Summary unlocked",
                "Password accepted. The summary details are now visible.",
            )
        else:
            QMessageBox.warning(
                self,
                "Incorrect password",
                "The provided password is incorrect. Summary remains hidden.",
            )

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _build_info_text(self) -> str:
        experiment_name = self.experiment_info.get("name", "Experiment")

        elapsed = self._format_duration(self.elapsed_time)
        expected = self._format_duration(self.expected_duration)

        subject_lines = []
        fields = [
            ("Name", "name"),
            ("Participant ID", "participant_id"),
            ("Experiment ID", "experiment_id"),
            ("Profile", "profile"),
            ("Age group", "age_group"),
            ("Gender", "gender"),
            ("Height (cm)", "height_cm"),
            ("Weight (kg)", "weight_kg"),
            ("Dominant hand", "dominant_hand"),
            ("Description", "description"),
        ]
        handled_keys = set()
        for label, key in fields:
            value = self.subject_info.get(key)
            handled_keys.add(key)
            if value in (None, ""):
                continue
            if key in {"height_cm", "weight_kg"}:
                try:
                    value = f"{float(value):.1f}"
                except (TypeError, ValueError):
                    value = str(value)
            subject_lines.append(f"{label}: {value}")

        has_second = bool(self.subject_info.get("has_second_participant"))
        second_lines: List[str] = []
        second_fields = [
            ("Second participant name", "second_name"),
            ("Second participant ID", "second_participant_id"),
            ("Second age group", "second_age_group"),
            ("Second gender", "second_gender"),
            ("Second dominant hand", "second_dominant_hand"),
            ("Second height (cm)", "second_height_cm"),
            ("Second weight (kg)", "second_weight_kg"),
            ("Second description", "second_description"),
        ]
        for label, key in second_fields:
            handled_keys.add(key)
            if not has_second:
                continue
            value = self.subject_info.get(key)
            if value in (None, ""):
                continue
            if key in {"second_height_cm", "second_weight_kg"}:
                try:
                    value = f"{float(value):.1f}"
                except (TypeError, ValueError):
                    value = str(value)
            second_lines.append(f"{label}: {value}")

        # Include any extra metadata keys not covered above
        for key, value in self.subject_info.items():
            if key in handled_keys or value in (None, ""):
                continue
            subject_lines.append(f"{key}: {value}")

        if not subject_lines:
            subject_lines.append("No participant metadata provided")
        if has_second:
            if not second_lines:
                second_lines.append("Second participant metadata provided but empty")
            subject_lines.extend(["", "Second participant details:", *second_lines])

        environment_lines = []
        env_fields = [
            ("Room length (m)", "length_m"),
            ("Room width (m)", "width_m"),
            ("Room height (m)", "height_m"),
            ("Description", "description"),
        ]
        env_handled = set()
        for label, key in env_fields:
            value = self.environment_info.get(key)
            env_handled.add(key)
            if value in (None, ""):
                continue
            if key in {"length_m", "width_m", "height_m"}:
                try:
                    value = f"{float(value):.2f}"
                except (TypeError, ValueError):
                    value = str(value)
            environment_lines.append(f"{label}: {value}")
        for key, value in self.environment_info.items():
            if key in env_handled or value in (None, ""):
                continue
            environment_lines.append(f"{key}: {value}")
        if not environment_lines:
            environment_lines.append("No environment metadata provided")

        info_lines = [
            "Participant details:",
            *subject_lines,
            "",
            "Environment details:",
            *environment_lines,
            "",
            f"Experiment: {experiment_name}",
            f"Elapsed time: {elapsed} / Expected duration: {expected}",
        ]
        return "\n".join(info_lines)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0.0, float(seconds or 0.0))
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

    def _close_application(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.quit()
        else:
            self.close()

    def _build_capture_table(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel("Captured CSI files (size shown after access point):")
        title.setAlignment(Qt.AlignLeft)
        layout.addWidget(title)

        captures = self.capture_entries
        if not captures:
            empty_label = QLabel("No captured CSI files were found.")
            empty_label.setAlignment(Qt.AlignLeft)
            layout.addWidget(empty_label)
            return container

        table = QTableWidget(len(captures), 5)
        table.setHorizontalHeaderLabels(
            ["Index", "Access point", "File size", "Transmitter MACs", "Capture file"]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        for row, entry in enumerate(captures):
            index_display = str(entry.get("capture_index") or "") or "—"

            size_bytes_raw = entry.get("file_size_bytes") or entry.get("file_size")
            try:
                size_bytes = int(size_bytes_raw)
            except (TypeError, ValueError):
                size_bytes = None
            size_display = self._format_file_size(size_bytes)

            ap_name = entry.get("ap_name") or "Unknown access point"
            tx_macs_raw = entry.get("transmitter_macs")
            if isinstance(tx_macs_raw, list):
                tx_macs_list = [str(mac).strip() for mac in tx_macs_raw if str(mac).strip()]
            else:
                tx_macs_list = [mac.strip() for mac in str(tx_macs_raw or "").split(",") if mac.strip()]
            tx_macs = ", ".join(tx_macs_list) if tx_macs_list else "Unknown"
            file_name = Path(entry.get("file_path") or "").name or "Unknown file"

            table.setItem(row, 0, QTableWidgetItem(index_display))
            table.setItem(row, 1, QTableWidgetItem(ap_name))
            table.setItem(row, 2, QTableWidgetItem(size_display))
            table.setItem(row, 3, QTableWidgetItem(tx_macs))
            table.setItem(row, 4, QTableWidgetItem(file_name))

        layout.addWidget(table)
        return container

    @staticmethod
    def _format_file_size(size_bytes: Optional[int]) -> str:
        if size_bytes is None or size_bytes < 0:
            return "Unknown size"

        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(size_bytes)
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                break
            size /= 1024.0
        return f"{size:.2f} {unit}"

    def _load_capture_manifest(self) -> List[Dict[str, str]]:
        if not self.results_dir:
            return []

        manifest_path = self.results_dir / "csi_captures" / "captures.csv"
        if not manifest_path.exists():
            return []

        try:
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                captures = list(reader)
        except Exception:
            return []

        return sorted(
            captures, key=lambda entry: (self._capture_index(entry), entry.get("ap_name") or "")
        )

    def _default_capture_entry(self) -> Optional[Dict[str, str]]:
        for entry in self.capture_entries:
            file_path = entry.get("file_path")
            if file_path and Path(file_path).exists():
                return entry
        return self.capture_entries[0] if self.capture_entries else None

    def _initial_pcap_selection(self) -> Optional[dict[str, Path]]:
        entry = self._default_capture_entry()
        if not entry:
            return None

        file_path_raw = entry.get("file_path")
        if not file_path_raw:
            return None

        pcap_path = Path(file_path_raw)
        if not pcap_path.exists():
            return None

        capture_dir = pcap_path.parent
        ap_dir = capture_dir.parent
        experiment_dir = ap_dir.parent.parent
        results_root = experiment_dir.parent

        return {
            "results_root": results_root,
            "experiment": experiment_dir,
            "ap": ap_dir,
            "trial": capture_dir,
            "pcap": pcap_path,
        }

    def _open_pcap_viewer(self) -> None:
        selection = self._initial_pcap_selection()
        if selection is None:
            QMessageBox.information(
                self,
                "PCAP viewer unavailable",
                "No CSI capture files were found to display.",
            )
            return

        from pcap_reader_ui import PCAPExplorerWindow

        self._pcap_viewer = PCAPExplorerWindow(
            results_root=selection["results_root"],
            initial_selection={
                "experiment": selection["experiment"],
                "ap": selection["ap"],
                "trial": selection["trial"],
                "pcap": selection["pcap"],
            },
            auto_plot=True,
            parent=None,
        )
        self._pcap_viewer.setAttribute(Qt.WA_DeleteOnClose, True)
        self._pcap_viewer.show()
        self._pcap_viewer.raise_()
        self._pcap_viewer.activateWindow()

    @staticmethod
    def _capture_index(entry: Dict[str, str]) -> int:
        try:
            raw_index = entry.get("capture_index")
            return int(raw_index) if raw_index not in (None, "") else int(1e9)
        except (TypeError, ValueError):
            return int(1e9)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def _plot_data(self) -> None:
        measurements: Tuple[Tuple[str, Optional[int]], ...] = (
            ("Wrist X", 3),
            ("Wrist Y", 4),
            ("Wrist Z", 5),
            ("Hand in frame", 6),
        )

        action_signal = self._load_array("action_signal.npy", expected_cols=2)
        csi_signal = self._load_array("csi_capture_signal.npy", expected_cols=2)
        hand_data = self._load_array("hand_landmarks.npy", expected_cols=7)

        presence_samples = hand_data[hand_data[:, 2] < 0]

        base_ts = self._compute_base_timestamp(action_signal, hand_data, csi_signal)
        normalize = lambda ts: (ts.astype(np.float64) - base_ts) / 1e9 if ts.size else ts

        ordered_action = (
            action_signal[np.argsort(action_signal[:, 0])] if action_signal.size else action_signal
        )
        normalized_action_ts = (
            normalize(ordered_action[:, 0]) if ordered_action.size else np.array([])
        )

        ordered_csi = csi_signal[np.argsort(csi_signal[:, 0])] if csi_signal.size else csi_signal
        normalized_csi_ts = normalize(ordered_csi[:, 0]) if ordered_csi.size else np.array([])

        def action_segments() -> List[Tuple[float, float]]:
            if not ordered_action.size:
                return []

            values = ordered_action[:, 1] > 0.5
            segments: List[Tuple[float, float]] = []
            start_ts: Optional[float] = None

            for ts, active in zip(normalized_action_ts, values):
                if active and start_ts is None:
                    start_ts = float(ts)
                elif not active and start_ts is not None:
                    segments.append((start_ts, float(ts)))
                    start_ts = None

            if start_ts is not None:
                segments.append((start_ts, float(normalized_action_ts[-1])))

            return segments

        action_ranges = action_segments()

        def shade_action_windows(ax: Any) -> None:
            for start, end in action_ranges:
                ax.axvspan(start, end, color="red", alpha=0.08, linewidth=0)

        def presence_segments(
            presence_rows: np.ndarray, label_value: float
        ) -> List[Tuple[float, float]]:
            if presence_rows.size == 0:
                return []

            filtered = presence_rows[np.isclose(presence_rows[:, 1], label_value)]
            if filtered.size == 0:
                return []

            ordered = filtered[np.argsort(filtered[:, 0])]
            timestamps = ordered[:, 0]
            flags = ordered[:, 6] > 0.5

            segments: List[Tuple[float, float]] = []
            start_ts: Optional[float] = None

            for ts, present in zip(timestamps, flags):
                if present and start_ts is None:
                    start_ts = float(ts)
                elif not present and start_ts is not None:
                    segments.append((start_ts, float(ts)))
                    start_ts = None

            if start_ts is not None:
                segments.append((start_ts, float(timestamps[-1])))

            return segments

        self.figure.clear()
        hand_rows = len(measurements)
        height_ratios = [1.2, 0.9] + [1.0] * hand_rows
        grid = self.figure.add_gridspec(hand_rows + 2, 2, height_ratios=height_ratios)

        hands: Tuple[Tuple[float, str, str], ...] = (
            (0.0, "Left hand", "tab:blue"),
            (1.0, "Right hand", "tab:orange"),
        )
        wrist_index = 0.0
        hand_subsets = {
            label: hand_data[
                (hand_data[:, 1] == label) & (np.isclose(hand_data[:, 2], wrist_index))
            ]
            for label, _, _ in hands
        }

        action_axes: List[Any] = []
        for col_index, (_hand_label, title, _color) in enumerate(hands):
            ax = self.figure.add_subplot(grid[0, col_index])
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylabel("Action mode")
            ax.set_title(f"{title} — Action mode")
            shade_action_windows(ax)

            if ordered_action.size:
                timestamps = normalized_action_ts
                ax.step(timestamps, ordered_action[:, 1], where="post", linewidth=2.0, color="black")
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
            action_axes.append(ax)

        for col_index, (_hand_label, title, color) in enumerate(hands):
            csi_ax = self.figure.add_subplot(grid[1, col_index], sharex=action_axes[col_index])
            csi_ax.grid(True, linestyle="--", alpha=0.4)
            if col_index == 0:
                csi_ax.set_ylabel("CSI capture")
            csi_ax.set_title(f"{title} — CSI capture activity")
            shade_action_windows(csi_ax)
            csi_ax.tick_params(labelbottom=False)

            if ordered_csi.size:
                csi_ax.step(
                    normalized_csi_ts,
                    ordered_csi[:, 1],
                    where="post",
                    linewidth=1.8,
                    color=color,
                )
                csi_ax.set_ylim(-0.1, 1.1)
                csi_ax.set_yticks([0, 1])
                csi_ax.set_yticklabels(["Idle", "Recording"])
            else:
                csi_ax.text(
                    0.5,
                    0.5,
                    "CSI capture signal unavailable",
                    ha="center",
                    va="center",
                    transform=csi_ax.transAxes,
                    fontsize=10,
                )

        last_row_index = hand_rows + 1
        for row_offset, (label, value_index) in enumerate(measurements, start=2):
            for col_index, (hand_label, title, color) in enumerate(hands):
                ax = self.figure.add_subplot(
                    grid[row_offset, col_index], sharex=action_axes[col_index]
                )

                ax.grid(True, linestyle="--", alpha=0.4)
                ax.set_ylabel(label)
                ax.set_title(f"{title} — {label}")

                subset = hand_subsets.get(hand_label)
                shade_action_windows(ax)

                if value_index == 6:
                    if presence_samples.size:
                        ordered = presence_samples[
                            np.argsort(presence_samples[:, 0])
                        ]
                        ordered = ordered[np.isclose(ordered[:, 1], hand_label)]
                        if ordered.size:
                            timestamps = normalize(ordered[:, 0])
                            ax.step(
                                timestamps,
                                ordered[:, value_index],
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
                                fontsize=9,
                            )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No hand detections recorded",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=9,
                        )
                else:
                    visible_segments = presence_segments(presence_samples, hand_label)
                if subset is not None and subset.size and value_index != 6:
                    plotted = False
                    for start_ts, end_ts in visible_segments:
                        window = subset[
                            (subset[:, 0] >= start_ts) & (subset[:, 0] <= end_ts)
                        ]
                        if not window.size:
                            continue
                        timestamps = normalize(window[:, 0])
                        ax.plot(
                            timestamps,
                            window[:, value_index],
                            linewidth=1.5,
                            color=color,
                            alpha=0.95,
                        )
                        plotted = True

                    if not plotted:
                        ax.text(
                            0.5,
                            0.5,
                            f"No {title.lower()} data while visible",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=9,
                        )
                elif value_index != 6:
                    ax.text(
                        0.5,
                        0.5,
                        f"No {title.lower()} data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                    )

                if row_offset == last_row_index:
                    ax.set_xlabel("Time since start (s)")
                else:
                    ax.tick_params(labelbottom=False)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _persist_summary_file(self, info_text: str) -> None:
        if not self.results_dir:
            return

        summary_path = self.results_dir / "experiment_summary.txt"
        lines = [
            "Experiment summary",
            "==================",
            info_text,
            "",
            "Subject metadata:",
        ]
        if self.subject_info:
            for key, value in self.subject_info.items():
                lines.append(f"  - {key}: {value}")
        else:
            lines.append("  (no subject metadata provided)")

        lines.append("")
        lines.append("Experiment metadata:")
        if self.experiment_info:
            for key, value in self.experiment_info.items():
                lines.append(f"  - {key}: {value}")
        else:
            lines.append("  (no experiment metadata provided)")

        lines.append("")
        lines.append("Environment metadata:")
        if self.environment_info:
            for key, value in self.environment_info.items():
                lines.append(f"  - {key}: {value}")
        else:
            lines.append("  (no environment metadata provided)")

        lines.append("")
        lines.append("Actions:")
        if self.actions_profile_name:
            lines.append(f"  Selected profile: {self.actions_profile_name}")
        else:
            lines.append("  Selected profile: (not provided)")
        if self.actions_list:
            lines.append("  Actions performed:")
            for action in self.actions_list:
                lines.append(f"    - {action}")
        else:
            lines.append("  Actions performed: (none provided)")

        try:
            summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        except OSError:
            pass

    def _compute_base_timestamp(
        self, action_signal: np.ndarray, hand_data: np.ndarray, csi_signal: np.ndarray
    ) -> float:
        candidates: List[float] = []
        if action_signal.size:
            candidates.append(float(np.nanmin(action_signal[:, 0])))
        if hand_data.size:
            candidates.append(float(np.nanmin(hand_data[:, 0])))
        if csi_signal.size:
            candidates.append(float(np.nanmin(csi_signal[:, 0])))
        return candidates[0] if candidates else 0.0

    def _load_array(self, filename: str, expected_cols: int) -> np.ndarray:
        if not self.results_dir:
            return np.empty((0, expected_cols), dtype=np.float64)
        path = self.results_dir / filename
        if not path.exists():
            return np.empty((0, expected_cols), dtype=np.float64)
        try:
            data = np.load(path)
        except Exception:
            return np.empty((0, expected_cols), dtype=np.float64)
        if data.ndim != 2 or data.shape[1] < expected_cols:
            return np.empty((0, expected_cols), dtype=np.float64)
        return data[:, :expected_cols].astype(np.float64, copy=False)
