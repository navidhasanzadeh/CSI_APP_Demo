"""Realtime Wi-Fi CSI demo window."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from math import ceil

import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.widgets import Button as MatplotlibButton
from mpl_toolkits.mplot3d import proj3d
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QDialog,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)
from PyQt5.QtCore import Qt, QTimer, QUrl

from wifi_csi_manager import WiFiCSIManager
from demo_plot_calculations import DemoPlotCalculator
from demo_plot_renderers import DemoPlotRenderer


try:  # pragma: no cover - optional dependency at runtime
    from hampel import hampel
except ImportError:  # pragma: no cover - optional dependency at runtime
    hampel = None

DEFAULT_SUBPLOT_SETTINGS = {
    "csi_ratio_magnitude": {
        "visible": True,
        "title": "CSI Ratio Magnitude",
        "xlabel": "Time (s)",
        "ylabel": "|Ratio|",
        "info": "Shows the CSI magnitude ratio between two TX antennas for each RX antenna stream.",
    },
    "csi_ratio_phase": {
        "visible": True,
        "title": "CSI Ratio Phase",
        "xlabel": "Time (s)",
        "ylabel": "Phase (rad)",
        "info": "Shows the CSI phase ratio between two TX antennas for each RX antenna stream.",
    },
    "doppler_music": {
        "visible": True,
        "title": "Doppler MUSIC Projection",
        "xlabel": "Time (s)",
        "ylabel": "Norm. Doppler",
        "info": "Shows Doppler projections extracted with a Root-MUSIC style estimator.",
    },
    "dorf_loss": {
        "visible": True,
        "title": "DoRF DTW Loss",
        "xlabel": "Iteration",
        "ylabel": "Loss",
        "info": "Shows optimization loss across DoRF DTW iterations.",
    },
    "dorf_velocity": {
        "visible": True,
        "title": "Estimated Velocity Components",
        "xlabel": "Time index",
        "ylabel": "Velocity",
        "info": "Shows estimated 3D velocity components over time.",
    },
    "dorf_energy": {
        "visible": True,
        "title": "Energy Envelope",
        "xlabel": "Time index",
        "ylabel": "Energy",
        "info": "Shows energy envelope computed from estimated velocity magnitude squared.",
    },
    "dorf_projection_map": {
        "visible": True,
        "title": "Average DoRF Projection Map",
        "xlabel": "Longitude bins",
        "ylabel": "Latitude bins",
        "info": "Shows the average DoRF spatial projection map.",
    },
    "dorf_cluster_fit": {
        "visible": True,
        "title": "Cluster Fit",
        "xlabel": "Time index",
        "ylabel": "Amplitude",
        "info": "Compares observed and predicted Doppler signals for each DoRF cluster.",
    },
    "dorf_vmf_clusters": {
        "visible": True,
        "title": "vMF Clusters",
        "xlabel": "",
        "ylabel": "",
        "info": "Shows unit-direction cluster assignments on a sphere (vMF clustering).",
    },
}

DEFAULT_DORF_PLOT_ORDER = [
    "dorf_loss",
    "dorf_velocity",
    "dorf_energy",
    "dorf_projection_map",
    "dorf_cluster_fit",
    "dorf_vmf_clusters",
]


class CSICaptureGuidanceDialog(QDialog):
    start_capture_requested = pyqtSignal()

    def __init__(
        self,
        *,
        parent=None,
        title: str = "CSI Capture Guidance",
        message: str = "Please perform one of these gestures.",
        left_label: str = "Gesture 1",
        left_video_path: str = "",
        right_label: str = "Gesture 2",
        right_video_path: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(980, 560)
        self._accepted = False
        self._capture_started = False
        self._players: list[QMediaPlayer] = []
        self._video_paths = [left_video_path, right_video_path]
        self._video_widgets: list[QVideoWidget] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        message_label = QLabel(message, self)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #111827;")
        root.addWidget(message_label)

        videos_row = QHBoxLayout()
        videos_row.setSpacing(10)
        root.addLayout(videos_row, stretch=1)

        for idx, (label_text, video_path) in enumerate(
            [(left_label, left_video_path), (right_label, right_video_path)]
        ):
            col = QVBoxLayout()
            col.setSpacing(6)
            title_label = QLabel(label_text, self)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-size: 13px; font-weight: 700; color: #1f2937;")
            col.addWidget(title_label)

            holder = QWidget(self)
            holder_layout = QVBoxLayout(holder)
            holder_layout.setContentsMargins(0, 0, 0, 0)
            holder_layout.setSpacing(0)

            video_widget = QVideoWidget(holder)
            video_widget.setMinimumSize(400, 280)
            holder_layout.addWidget(video_widget)

            fallback_label = QLabel("", holder)
            fallback_label.setAlignment(Qt.AlignCenter)
            fallback_label.setWordWrap(True)
            fallback_label.setStyleSheet(
                "QLabel {border: 1px dashed #94a3b8; border-radius: 8px; background: #f8fafc;"
                "color: #475569; font-size: 12px; padding: 8px;}"
            )
            holder_layout.addWidget(fallback_label)
            col.addWidget(holder, stretch=1)

            media_player = QMediaPlayer(self)
            media_player.setVideoOutput(video_widget)
            media_player.mediaStatusChanged.connect(
                lambda status, i=idx: self._on_media_status_changed(i, status)
            )
            self._players.append(media_player)
            self._video_widgets.append(video_widget)

            resolved = self._resolve_video_path(video_path)
            if resolved:
                video_widget.show()
                fallback_label.hide()
                media_player.setMedia(QMediaContent(QUrl.fromLocalFile(resolved)))
                media_player.play()
            else:
                video_widget.hide()
                fallback_label.show()
                fallback_label.setText(
                    "Video not found.\nSet path in Demo profile guidance video fields."
                )
            videos_row.addLayout(col, stretch=1)

        self.capture_progress = QProgressBar(self)
        self.capture_progress.setRange(0, 100)
        self.capture_progress.setValue(0)
        self.capture_progress.setFormat("Capture: 0%")
        self.capture_progress.setStyleSheet("QProgressBar::chunk { background-color: #16a34a; }")
        self.capture_progress.hide()
        root.addWidget(self.capture_progress)

        self.elapsed_label = QLabel("Elapsed: 0.0s", self)
        self.elapsed_label.setAlignment(Qt.AlignCenter)
        self.elapsed_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #166534;")
        self.elapsed_label.hide()
        root.addWidget(self.elapsed_label)

        self.transfer_progress = QProgressBar(self)
        self.transfer_progress.setRange(0, 1)
        self.transfer_progress.setValue(0)
        self.transfer_progress.setFormat("Transfer: waiting")
        self.transfer_progress.setStyleSheet("QProgressBar::chunk { background-color: #f59e0b; }")
        self.transfer_progress.hide()
        root.addWidget(self.transfer_progress)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.btn_start = QPushButton("Start Capture", self)
        self.btn_start.setStyleSheet(
            "QPushButton {background-color: #16a34a; color: white; font-weight: 700; "
            "padding: 6px 12px; border-radius: 8px;}"
            "QPushButton:hover {background-color: #15803d;}"
        )
        self.btn_start.clicked.connect(self._on_start_clicked)
        button_row.addWidget(self.btn_start)

        self.btn_close = QPushButton("Close", self)
        self.btn_close.clicked.connect(self.reject)
        button_row.addWidget(self.btn_close)
        root.addLayout(button_row)

    def _resolve_video_path(self, raw_path: str) -> str:
        candidate = str(raw_path or "").strip()
        if not candidate:
            return ""
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parent / path).resolve()
        return str(path) if path.exists() else ""

    def _on_media_status_changed(self, index: int, status: QMediaPlayer.MediaStatus) -> None:
        if status != QMediaPlayer.EndOfMedia:
            return
        if 0 <= index < len(self._players):
            player = self._players[index]
            player.setPosition(0)
            player.play()

    def _on_start_clicked(self) -> None:
        if self._capture_started:
            return
        self._capture_started = True
        self._accepted = True
        self.btn_start.setEnabled(False)
        self.btn_close.setEnabled(False)
        self.capture_progress.show()
        self.elapsed_label.show()
        self.transfer_progress.hide()
        self.start_capture_requested.emit()

    def update_capture_progress(self, elapsed: float, duration: float) -> None:
        safe_duration = max(float(duration), 0.1)
        percent = int(min(100, (max(0.0, elapsed) / safe_duration) * 100))
        self.capture_progress.setValue(percent)
        self.capture_progress.setFormat(f"Capture: {percent}%")
        self.elapsed_label.setText(f"Elapsed: {max(0.0, elapsed):.1f}s")

    def start_transfer_progress(self, total_files: int) -> None:
        total = max(int(total_files), 1)
        self.transfer_progress.setRange(0, total)
        self.transfer_progress.setValue(0)
        self.transfer_progress.setFormat(f"Transfer: 0/{total} PCAPs")
        self.transfer_progress.show()

    def update_transfer_progress(self, completed: int, total_files: int, status: str = "") -> None:
        total = max(int(total_files), 1)
        done = min(max(int(completed), 0), total)
        self.transfer_progress.setRange(0, total)
        self.transfer_progress.setValue(done)
        suffix = f" - {status}" if status else ""
        self.transfer_progress.setFormat(f"Transfer: {done}/{total} PCAPs{suffix}")

    def finish_capture(self) -> None:
        self.btn_close.setEnabled(True)
        self.accept()

    def closeEvent(self, event):  # pragma: no cover - UI lifecycle
        for player in self._players:
            player.stop()
        super().closeEvent(event)


class DemoWindow(QWidget):
    capture_finished = pyqtSignal(bool, str)
    plot_requested = pyqtSignal(str, int)
    doppler_ready = pyqtSignal(object)
    dorf_ready = pyqtSignal(object)
    background_plot_failed = pyqtSignal(str)
    capture_progress_updated = pyqtSignal(float, float)
    transfer_progress_started = pyqtSignal(int)
    transfer_progress_updated = pyqtSignal(int, int, str)
    capture_ui_finished = pyqtSignal()

    def __init__(
        self,
        *,
        wifi_profile_name: str,
        wifi_profile: dict,
        demo_profile: dict,
        routers_info: list[dict],
        results_dir: Path,
        parent=None,
    ):
        super().__init__(parent)
        self.wifi_profile_name = wifi_profile_name
        self.wifi_profile = wifi_profile or {}
        self.demo_profile = demo_profile or {}
        self.routers_info = routers_info or []
        self.results_dir = Path(results_dir)
        self.wifi_manager = WiFiCSIManager(self.wifi_profile)
        self._capture_thread = None
        self._capture_started_at = 0.0
        self._capture_guidance_dialog: CSICaptureGuidanceDialog | None = None
        self._clock_timer: QTimer | None = None
        self._figure_maximize_buttons: dict[Figure, list[tuple[object, MatplotlibButton]]] = {}
        self._plot_detail_windows: list[QWidget] = []
        self.plot_calculator = DemoPlotCalculator()
        self.plot_renderer = DemoPlotRenderer(self)

        self.capture_finished.connect(self._on_capture_finished)
        self.plot_requested.connect(self._on_plot_requested)
        self.doppler_ready.connect(self._on_doppler_ready)
        self.dorf_ready.connect(self._on_dorf_ready)
        self.background_plot_failed.connect(self._on_background_plot_failed)
        self.capture_progress_updated.connect(self._on_capture_progress_updated)
        self.transfer_progress_started.connect(self._on_transfer_progress_started)
        self.transfer_progress_updated.connect(self._on_transfer_progress_updated)
        self.capture_ui_finished.connect(self._on_capture_ui_finished)
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("Demo Window")
        self.resize(1280, 840)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(10)
        header_left_col = QVBoxLayout()
        header_left_col.setSpacing(1)
        self.icassp_logo_label = QLabel(self._icassp_title_text(), self)
        self.icassp_logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.icassp_logo_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #1e3a8a;")
        header_left_col.addWidget(self.icassp_logo_label)

        self.authors_label = QLabel(self._authors_text(), self)
        self.authors_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.authors_label.setWordWrap(True)
        self.authors_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #111827;")
        header_left_col.addWidget(self.authors_label)

        self.university_label = QLabel(self._university_text(), self)
        self.university_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.university_label.setWordWrap(True)
        self.university_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #111827;")
        header_left_col.addWidget(self.university_label)
        header_row.addLayout(header_left_col, stretch=2)

        self.title_label = QLabel(self._demo_title_text(), self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-size: 22px; font-weight: 700; color: #0b1f3a;")
        header_row.addWidget(self.title_label, stretch=4)

        logo_col = QVBoxLayout()
        logo_col.setSpacing(2)
        logo_col.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.qr_placeholder = QLabel(self)
        self.qr_placeholder.setFixedSize(160, 160)
        self.qr_placeholder.setAlignment(Qt.AlignCenter)
        self.qr_placeholder.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.qr_placeholder.setStyleSheet(
            "QLabel {border: 2px dashed #94a3b8; border-radius: 10px; background: #f8fafc; "
            "color: #475569; font-size: 14px; font-weight: 600;}"
        )
        self._update_qr_placeholder()
        logo_col.addWidget(self.qr_placeholder, alignment=Qt.AlignRight)

        self.wirlab_logo_label = QLabel("WIRLab", self)
        self.wirlab_logo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.wirlab_logo_label.setStyleSheet("font-size: 20px; font-weight: 800; color: #0f5e2b;")
        logo_col.addWidget(self.wirlab_logo_label, alignment=Qt.AlignRight)
        self.qr_website_label = QLabel(self._qr_website_text(), self)
        self.qr_website_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.qr_website_label.setStyleSheet("font-size: 12px; font-weight: 700; color: #2563eb;")
        self.qr_website_label.setWordWrap(False)
        self.qr_website_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        logo_col.addWidget(self.qr_website_label, alignment=Qt.AlignRight)
        header_row.addLayout(logo_col, stretch=2)
        root.addLayout(header_row)

        self.status_label = QLabel("Ready for demo capture.", self)
        self.status_label.setStyleSheet("font-size: 12px; color: #1f2937;")
        root.addWidget(self.status_label)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_scroll = QScrollArea(self)
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setWidget(self.canvas)
        self.plot_scroll.setStyleSheet(
            "QScrollArea {border: 1px solid #cfd8e3; border-radius: 10px; background: #ffffff;}"
        )

        self.demo_tabs = QTabWidget(self)
        csi_tab = QWidget(self.demo_tabs)
        csi_layout = QVBoxLayout(csi_tab)
        self.chk_hampel_ratio_phase = QCheckBox(
            "Apply Hampel filter to CSI ratio phase", csi_tab
        )
        self.chk_hampel_ratio_phase.setChecked(
            bool(self.demo_profile.get("apply_hampel_to_ratio_phase", False))
        )
        self.chk_hampel_ratio_magnitude = QCheckBox(
            "Apply Hampel filter to CSI ratio magnitude", csi_tab
        )
        self.chk_hampel_ratio_magnitude.setChecked(
            bool(self.demo_profile.get("apply_hampel_to_ratio_magnitude", False))
        )
        csi_layout.addWidget(self.chk_hampel_ratio_phase)
        csi_layout.addWidget(self.chk_hampel_ratio_magnitude)
        csi_layout.addWidget(self.plot_scroll, stretch=1)
        self.demo_tabs.addTab(csi_tab, "1. CSI Magnitude and Phase")

        doppler_proj_tab = QWidget(self.demo_tabs)
        doppler_proj_layout = QVBoxLayout(doppler_proj_tab)
        self.doppler_figure = Figure(figsize=(10, 6), dpi=100)
        self.doppler_canvas = FigureCanvas(self.doppler_figure)
        self.doppler_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.doppler_scroll = QScrollArea(doppler_proj_tab)
        self.doppler_scroll.setWidgetResizable(True)
        self.doppler_scroll.setWidget(self.doppler_canvas)
        self.doppler_scroll.setStyleSheet(
            "QScrollArea {border: 1px solid #cfd8e3; border-radius: 10px; background: #ffffff;}"
        )
        doppler_proj_layout.addWidget(self.doppler_scroll, stretch=1)
        self.demo_tabs.addTab(doppler_proj_tab, "2. Doppler Projections")

        dorf_tab = QWidget(self.demo_tabs)
        dorf_layout = QVBoxLayout(dorf_tab)
        self.chk_dorf_visualize = QCheckBox("Visualize DoRF DTW plots", dorf_tab)
        self.chk_dorf_visualize.setChecked(bool(self.demo_profile.get("dorf_visualize", False)))
        dorf_layout.addWidget(self.chk_dorf_visualize)

        self.dorf_figure = Figure(figsize=(10, 6), dpi=100)
        self.dorf_canvas = FigureCanvas(self.dorf_figure)
        self.dorf_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dorf_scroll = QScrollArea(dorf_tab)
        self.dorf_scroll.setWidgetResizable(True)
        self.dorf_scroll.setWidget(self.dorf_canvas)
        self.dorf_scroll.setStyleSheet(
            "QScrollArea {border: 1px solid #cfd8e3; border-radius: 10px; background: #ffffff;}"
        )
        dorf_layout.addWidget(self.dorf_scroll, stretch=1)
        self.demo_tabs.addTab(dorf_tab, "3. Doppler Radiance Fields (DoRF)")

        har_tab = QWidget(self.demo_tabs)
        har_layout = QVBoxLayout(har_tab)
        self.har_figure = Figure(figsize=(10, 5), dpi=100)
        self.har_canvas = FigureCanvas(self.har_figure)
        self.har_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.har_scroll = QScrollArea(har_tab)
        self.har_scroll.setWidgetResizable(True)
        self.har_scroll.setWidget(self.har_canvas)
        self.har_scroll.setStyleSheet(
            "QScrollArea {border: 1px solid #cfd8e3; border-radius: 10px; background: #ffffff;}"
        )
        har_layout.addWidget(self.har_scroll, stretch=1)
        self.demo_tabs.addTab(har_tab, "4. Human Activity Recognition")

        content_row = QHBoxLayout()
        content_row.addWidget(self.demo_tabs, stretch=4)

        info_pane = QFrame(self)
        info_pane.setMaximumWidth(260)
        info_pane.setFrameShape(QFrame.StyledPanel)
        info_pane.setStyleSheet(
            "QFrame {border: 1px solid #cfd8e3; border-radius: 8px; background: #f8fafc;}"
            "QLabel {color: #1f2937; font-size: 12px;}"
        )
        info_layout = QFormLayout(info_pane)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setVerticalSpacing(4)
        info_layout.setLabelAlignment(Qt.AlignLeft)
        self.packet_count_label = QLabel("-", info_pane)
        self.sampling_rate_label = QLabel("- pkt/s", info_pane)
        self.datetime_label = QLabel("-", info_pane)
        info_layout.addRow("Received packets:", self.packet_count_label)
        info_layout.addRow("Sampling rate:", self.sampling_rate_label)
        info_layout.addRow("Current date & time:", self.datetime_label)
        content_row.addWidget(info_pane, stretch=1)
        root.addLayout(content_row, stretch=1)
        self._start_clock_updates()

        bottom_row = QVBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(6)
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)
        self.btn_capture = QPushButton("CSI Capture", self)
        self.btn_capture.clicked.connect(self._on_capture_clicked)
        self.btn_capture.setStyleSheet(
            "QPushButton {background-color: #16a34a; color: white; font-size: 14px; font-weight: 700; "
            "padding: 6px 12px; border-radius: 8px;}"
            "QPushButton:hover {background-color: #15803d;}"
            "QPushButton:disabled {background-color: #86efac; color: #f8fafc;}"
        )
        button_row.addWidget(self.btn_capture)

        self.btn_close = QPushButton("Close Window", self)
        self.btn_close.clicked.connect(self.close)
        self.btn_close.setStyleSheet(
            "QPushButton {background-color: #2563eb; color: white; font-size: 13px; font-weight: 600; "
            "padding: 6px 12px; border-radius: 8px;}"
            "QPushButton:hover {background-color: #1d4ed8;}"
        )
        button_row.addWidget(self.btn_close)
        button_row.addStretch(1)
        bottom_row.addLayout(button_row)

        root.addLayout(bottom_row)
        self._set_tab_processing_state(allow_primary_only=True)

    def _set_tab_processing_state(self, allow_primary_only: bool = False) -> None:
        tab_bar = self.demo_tabs.tabBar()
        tab_bar.setTabEnabled(0, True)
        tab_bar.setTabEnabled(1, not allow_primary_only)
        tab_bar.setTabEnabled(2, not allow_primary_only)
        tab_bar.setTabEnabled(3, not allow_primary_only)

    def _mark_tab_ready(self, tab_idx: int) -> None:
        self.demo_tabs.tabBar().setTabEnabled(tab_idx, True)

    def _demo_title_text(self) -> str:
        text = str(self.demo_profile.get("demo_title_text") or "").strip()
        return text or (
            "Doppler Radiance Fields (DoRF) for Robust Wi-Fi Sensing and Human Activity Recognition"
        )

    def _qr_image_path(self) -> str:
        return str(self.demo_profile.get("qr_code_image_path") or "").strip()

    def _qr_website_text(self) -> str:
        text = str(self.demo_profile.get("qr_website_url") or "").strip()
        return text or "https://dorf.navidhasanzadeh.com"

    def _icassp_title_text(self) -> str:
        text = str(self.demo_profile.get("icassp_title_text") or "").strip()
        return text or "IEEE ICASSP 2026"

    def _authors_text(self) -> str:
        text = str(self.demo_profile.get("authors_text") or "").strip()
        return text or "Authors: Navid Hasanzadeh, Shahrokh Valaee"

    def _university_text(self) -> str:
        text = str(self.demo_profile.get("university_text") or "").strip()
        return text or "University of Toronto"

    def _start_clock_updates(self) -> None:
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_datetime_label)
        self._clock_timer.start(1000)
        self._update_datetime_label()

    def _update_datetime_label(self) -> None:
        self.datetime_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _update_qr_placeholder(self) -> None:
        path = self._qr_image_path()
        if path and Path(path).exists():
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                target_size = self.qr_placeholder.size()
                self.qr_placeholder.setPixmap(
                    pixmap.scaled(
                        target_size,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
                self.qr_placeholder.setText("")
                return
        if path:
            self.qr_placeholder.setText("QR image not found\nSet a valid image path\nin Demo profile")
        else:
            self.qr_placeholder.setText("QR code placeholder\nSet image path\nin Demo profile")
        self.qr_placeholder.setPixmap(QPixmap())

    def _capture_duration(self) -> float:
        return max(float(self.demo_profile.get("capture_duration_seconds", 5.0)), 1.0)

    def _effective_capture_samples(self) -> int:
        try:
            value = int(self.demo_profile.get("effective_capture_samples", 0))
        except (TypeError, ValueError):
            return 0
        return max(0, value)

    def _activity_class_names(self) -> list[str]:
        value = self.demo_profile.get("activity_class_names", [])
        if isinstance(value, str):
            names = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, list):
            names = [str(part).strip() for part in value if str(part).strip()]
        else:
            names = []
        return names

    def _crop_to_effective_window(
        self, csi_data: np.ndarray, time_vals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        packet_count = int(min(csi_data.shape[0], time_vals.size if time_vals.size else csi_data.shape[0]))
        if packet_count <= 0:
            return csi_data, time_vals
        csi_data = csi_data[:packet_count]
        time_vals = time_vals[:packet_count] if time_vals.size else np.array([])

        effective_samples = self._effective_capture_samples()
        if effective_samples <= 0 or effective_samples >= packet_count:
            return csi_data, time_vals

        start_idx = (packet_count - effective_samples) // 2
        end_idx = start_idx + effective_samples
        cropped_time = time_vals[start_idx:end_idx] if time_vals.size else time_vals
        return csi_data[start_idx:end_idx], cropped_time

    def _subplot_setting(self, category: str) -> dict:
        settings = dict(DEFAULT_SUBPLOT_SETTINGS.get(category, {}))
        profile_settings = self.demo_profile.get("subplot_settings", {})
        if isinstance(profile_settings, dict):
            custom = profile_settings.get(category, {})
            if isinstance(custom, dict):
                settings.update(custom)
        return settings

    def _subplot_visible(self, category: str) -> bool:
        return bool(self._subplot_setting(category).get("visible", True))

    def _subplot_text(self, category: str, key: str, fallback: str) -> str:
        value = str(self._subplot_setting(category).get(key, fallback)).strip()
        return value or fallback

    def _apply_subplot_labels(
        self,
        ax,
        *,
        category: str,
        default_title: str,
        default_xlabel: str,
        default_ylabel: str,
    ) -> None:
        ax.set_title(self._subplot_text(category, "title", default_title))
        xlabel = self._subplot_text(category, "xlabel", default_xlabel)
        ylabel = self._subplot_text(category, "ylabel", default_ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax._subplot_category = category

    def _subplot_info_text(self, category: str, fallback_title: str) -> tuple[str, str]:
        title = self._subplot_text(category, "title", fallback_title)
        info = self._subplot_text(
            category,
            "info",
            "No additional information configured for this subplot.",
        )
        return title, info

    def _dorf_plot_order(self) -> list[str]:
        value = self.demo_profile.get("dorf_plot_order", DEFAULT_DORF_PLOT_ORDER)
        if isinstance(value, str):
            requested = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, list):
            requested = [str(item).strip() for item in value if str(item).strip()]
        else:
            requested = []
        valid = [item for item in requested if item in DEFAULT_DORF_PLOT_ORDER]
        for item in DEFAULT_DORF_PLOT_ORDER:
            if item not in valid:
                valid.append(item)
        return valid

    def _on_capture_clicked(self):
        if self._capture_thread and self._capture_thread.is_alive():
            QMessageBox.information(self, "Capture Running", "A demo capture is already running.")
            return
        if not self.routers_info:
            QMessageBox.warning(self, "No Routers", "No connected routers are available for demo capture.")
            return
        self._show_capture_guidance_dialog()

    def _show_capture_guidance_dialog(self) -> None:
        dlg = CSICaptureGuidanceDialog(
            parent=self,
            title=str(
                self.demo_profile.get("capture_guidance_title", "CSI Capture Guidance")
            ).strip()
            or "CSI Capture Guidance",
            message=str(
                self.demo_profile.get(
                    "capture_guidance_message",
                    "Please perform one of these gestures.",
                )
            ).strip()
            or "Please perform one of these gestures.",
            left_label=str(
                self.demo_profile.get("capture_guidance_video_left_label", "Gesture 1")
            ).strip()
            or "Gesture 1",
            left_video_path=str(
                self.demo_profile.get("capture_guidance_video_left_path", "")
            ).strip(),
            right_label=str(
                self.demo_profile.get("capture_guidance_video_right_label", "Gesture 2")
            ).strip()
            or "Gesture 2",
            right_video_path=str(
                self.demo_profile.get("capture_guidance_video_right_path", "")
            ).strip(),
        )
        dlg.start_capture_requested.connect(self._begin_capture_cycle)
        dlg.rejected.connect(self._on_capture_guidance_closed)
        self._capture_guidance_dialog = dlg
        dlg.show()

    def _begin_capture_cycle(self) -> None:
        self.btn_capture.setEnabled(False)
        self.status_label.setText("Capturing CSI... Please perform the target activity now.")
        self._set_tab_processing_state(allow_primary_only=True)
        capture_duration = self._capture_duration()
        self._capture_started_at = time.monotonic()
        self.capture_progress_updated.emit(0.0, capture_duration)
        self._capture_thread = threading.Thread(target=self._run_capture_cycle, daemon=True)
        self._capture_thread.start()

    def _on_capture_guidance_closed(self) -> None:
        self._capture_guidance_dialog = None

    def _run_capture_cycle(self):
        try:
            capture_duration = self._capture_duration()
            capture_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            downloaded = []
            capture_errors: list[str] = []

            # Match Scenario 2 behavior: start transmitters before sniffers so
            # sniffers receive packets while tcpdump is active.
            ordered_routers = sorted(
                self.routers_info,
                key=lambda item: 1
                if self.wifi_manager.is_sniffer(
                    (dict(item.get("run_info") or {})).get("ap", {})
                )
                else 0,
            )

            capture_threads: list[threading.Thread] = []

            def _capture_router(router_obj, *, duration: float, remote_directory: str, exp_name: str):
                try:
                    self.wifi_manager.start_csi_capture(
                        router_obj,
                        duration=duration,
                        remote_directory=remote_directory,
                        exp_name=exp_name,
                        delete_prev_pcap=False,
                    )
                except Exception as exc:  # pragma: no cover - runtime/network behavior
                    capture_errors.append(f"{exp_name}: {exc}")

            for router_entry in ordered_routers:
                run_info = dict(router_entry.get("run_info") or {})
                ap = run_info.get("ap", {})
                router = router_entry.get("router")
                if router is None:
                    continue
                ap_name = ap.get("name") or ap.get("ssid") or "ap"
                exp_name = f"demo_{self.wifi_manager.sanitize_ap_name(ap_name)}_{capture_tag}"
                remote_dir = run_info.get("remote_dir", "/mnt/CSI_USB/")

                run_info["exp_name"] = exp_name
                router_entry["run_info"] = run_info
                thread = threading.Thread(
                    target=_capture_router,
                    args=(router,),
                    kwargs={
                        "duration": capture_duration,
                        "remote_directory": remote_dir,
                        "exp_name": exp_name,
                    },
                    daemon=True,
                )
                thread.start()
                capture_threads.append(thread)

            while any(thread.is_alive() for thread in capture_threads):
                elapsed = max(0.0, time.monotonic() - self._capture_started_at)
                self.capture_progress_updated.emit(elapsed, capture_duration)
                time.sleep(0.1)
            self.capture_progress_updated.emit(capture_duration, capture_duration)

            sniffers = [
                entry
                for entry in ordered_routers
                if self.wifi_manager.is_sniffer((dict(entry.get("run_info") or {})).get("ap", {}))
            ]
            total_sniffers = len(sniffers)
            self.transfer_progress_started.emit(total_sniffers)
            transfer_completed = 0
            for router_entry in ordered_routers:
                run_info = dict(router_entry.get("run_info") or {})
                ap = run_info.get("ap", {})
                if not self.wifi_manager.is_sniffer(ap):
                    continue
                router = router_entry.get("router")
                if router is None:
                    continue
                ap_name = ap.get("name") or ap.get("ssid") or "ap"
                remote_dir = run_info.get("remote_dir", "/mnt/CSI_USB/")
                exp_name = run_info.get("exp_name") or f"demo_{self.wifi_manager.sanitize_ap_name(ap_name)}_{capture_tag}"
                remote_listing = router.send_command(f"ls {remote_dir}")
                pcap_files = [line.strip() for line in remote_listing]
                candidates = self.wifi_manager.filter_matching_pcaps(pcap_files, exp_name)
                target_file = self.wifi_manager.latest_pcap_filename(candidates)
                if not target_file:
                    transfer_completed += 1
                    self.transfer_progress_updated.emit(
                        transfer_completed, total_sniffers, f"{ap_name}: no matching file"
                    )
                    continue

                capture_dir = self.results_dir / "demo_csi_captures" / self.wifi_manager.sanitize_ap_name(ap_name)
                local_path, _ = self.wifi_manager.download_capture(
                    router,
                    remote_dir=remote_dir,
                    target_file=target_file,
                    local_directory=capture_dir,
                    use_ftp=str(ap.get("download_mode", "SFTP")).strip().upper() != "SFTP",
                )
                downloaded.append((local_path, int(ap.get("bandwidth", "80").replace("MHz", "") or "80")))
                transfer_completed += 1
                self.transfer_progress_updated.emit(
                    transfer_completed, total_sniffers, f"{ap_name}: downloaded {target_file}"
                )

            if not downloaded:
                error_suffix = f" Errors: {'; '.join(capture_errors)}" if capture_errors else ""
                self.capture_finished.emit(False, f"No PCAP file was downloaded from sniffers.{error_suffix}")
                return

            first_capture, first_bw = downloaded[0]
            if not self._has_payload_packets(first_capture):
                self.capture_finished.emit(
                    False,
                    (
                        f"Downloaded PCAP is empty ({first_capture.name}). "
                        "No CSI packets were captured. Please verify transmitter is running before sniffer capture."
                    ),
                )
                return

            self.plot_requested.emit(str(first_capture), int(first_bw))
            self.capture_finished.emit(True, f"Capture complete: {first_capture.name}")
        except Exception as exc:  # pragma: no cover - network/runtime behavior
            self.capture_finished.emit(False, f"Demo capture failed: {exc}")
        finally:
            self.capture_ui_finished.emit()

    @staticmethod
    def _has_payload_packets(pcap_path: Path) -> bool:
        return DemoPlotCalculator.has_payload_packets(pcap_path)

    def _apply_hampel_filter(self, values: np.ndarray) -> np.ndarray:
        if hampel is None:
            self.status_label.setText(
                "Hampel filter is unavailable (missing dependency). Plotting unfiltered CSI traces."
            )
            return values
        try:
            filtered = hampel(values, window_size=8)
        except Exception as exc:
            self.status_label.setText(f"Hampel filter failed ({exc}). Plotting unfiltered ratio phase.")
            return values
        if hasattr(filtered, "filtered_data"):
            return np.asarray(filtered.filtered_data)
        if isinstance(filtered, tuple):
            return np.asarray(filtered[0])
        return np.asarray(filtered)

    def _plot_ratio(self, pcap_path: Path, bandwidth_mhz: int):
        if not self._has_payload_packets(pcap_path):
            self.status_label.setText(
                f"Cannot plot {pcap_path.name}: PCAP is empty (header only, no CSI packets)."
            )
            self.packet_count_label.setText("0")
            self.sampling_rate_label.setText("0.00 pkt/s")
            return
        csi_data, time_pkts, nfft = self.plot_calculator.load_csi_capture(pcap_path, bandwidth_mhz)

        if csi_data is None or time_pkts is None or csi_data.size == 0:
            self.status_label.setText(f"Unable to load CSI data from {pcap_path.name}.")
            self.packet_count_label.setText("0")
            self.sampling_rate_label.setText("0.00 pkt/s")
            return

        csi_data, time_vals = self._crop_to_effective_window(
            np.asarray(csi_data), np.asarray(time_pkts, dtype=float)
        )
        ratio_payload = self.plot_calculator.compute_ratio_payload(csi_data, time_vals, nfft)
        packet_count = int(ratio_payload["packet_count"])
        csi_data = ratio_payload["csi_data"]
        time_vals = ratio_payload["time_vals"]
        tx_pairs = ratio_payload["tx_pairs"]
        self.packet_count_label.setText(str(packet_count))
        self.sampling_rate_label.setText(f"{float(ratio_payload['sampling_rate']):.2f} pkt/s")
        if not tx_pairs:
            self.status_label.setText(
                f"Unable to compute CSI ratio from {pcap_path.name}: at least 2 TX antennas are required."
            )
            return

        self.plot_renderer.plot_ratio(
            ratio_payload,
            apply_hampel_phase=self.chk_hampel_ratio_phase.isChecked(),
            apply_hampel_magnitude=self.chk_hampel_ratio_magnitude.isChecked(),
        )
        self._set_tab_processing_state(allow_primary_only=True)
        self.status_label.setText("CSI magnitude/phase plotted. Processing Doppler and DoRF in background...")
        worker = threading.Thread(
            target=self._run_background_plot_pipeline,
            args=(csi_data, time_vals, packet_count, tx_pairs),
            daemon=True,
        )
        worker.start()

    def _on_plot_requested(self, pcap_path: str, bandwidth_mhz: int):
        self._plot_ratio(Path(pcap_path), int(bandwidth_mhz))

    def _run_background_plot_pipeline(
        self, csi_data: np.ndarray, time_vals: np.ndarray, packet_count: int, tx_pairs: list[tuple[int, int]]
    ) -> None:
        try:
            doppler_payload = self.plot_calculator.compute_doppler_payload(
                csi_data, time_vals, packet_count, tx_pairs
            )
            self.doppler_ready.emit(doppler_payload)
            dorf_payload = self.plot_calculator.compute_dorf_payload(
                doppler_payload["dopplers"], dorf_visualize=self.chk_dorf_visualize.isChecked()
            )
            self.dorf_ready.emit(dorf_payload)
        except Exception as exc:
            self.background_plot_failed.emit(str(exc))

    def _on_doppler_ready(self, payload: dict) -> None:
        self.plot_renderer.plot_doppler(payload)
        self._mark_tab_ready(1)
        self.status_label.setText("Doppler tab ready. DoRF is still processing in background...")

    def _on_dorf_ready(self, payload: dict) -> None:
        self.plot_renderer.plot_dorf(payload)
        har_payload = self.plot_calculator.compute_har_payload(
            payload,
            class_names=self._activity_class_names(),
        )
        self.plot_renderer.plot_har(har_payload)
        self._mark_tab_ready(2)
        self._mark_tab_ready(3)
        self.status_label.setText("All demo tabs are ready.")

    def _on_background_plot_failed(self, message: str) -> None:
        self.status_label.setText(f"Background plotting failed: {message}")

    def _plot_doppler_from_payload(self, payload: dict) -> None:
        series = payload.get("series", [])
        total_pairs = len(series)
        self.doppler_figure.clear()
        if total_pairs == 0:
            self.doppler_canvas.draw_idle()
            return
        ncols = 2
        nrows = int(ceil(total_pairs / ncols))
        grid = self.doppler_figure.add_gridspec(nrows, ncols, hspace=0.6, wspace=0.25)
        fig_height = max(8, nrows * 2.4)
        self.doppler_figure.set_size_inches(11, fig_height)
        self.doppler_canvas.setMinimumHeight(int(fig_height * self.doppler_figure.get_dpi()))

        for row_idx, item in enumerate(series):
            rx_idx = int(item["rx_idx"])
            tx_pair = tuple(item["tx_pair"])
            music_output = np.asarray(item["music_output"], dtype=float)
            x_trim = np.asarray(item["x"], dtype=float)
            x_label = str(item["x_label"])
            ax = self.doppler_figure.add_subplot(grid[row_idx // ncols, row_idx % ncols])
            if music_output.size:
                ax.plot(x_trim, music_output, color="tab:purple", linewidth=0.9)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient valid subcarriers for MUSIC.",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            category = "doppler_music"
            if self._subplot_visible(category):
                self._apply_subplot_labels(
                    ax,
                    category=category,
                    default_title=f"RX {rx_idx + 1}: TX {tx_pair[0] + 1}/TX {tx_pair[1] + 1} MUSIC",
                    default_xlabel=x_label,
                    default_ylabel="Norm. Doppler",
                )
                ax.grid(True)
            else:
                ax.set_axis_off()
        if total_pairs % ncols:
            spare_ax = self.doppler_figure.add_subplot(grid[-1, -1])
            spare_ax.set_axis_off()
        self.doppler_figure.subplots_adjust(left=0.08, right=0.95, top=0.96, bottom=0.08)
        self._install_subplot_maximize_buttons(self.doppler_figure, self.doppler_canvas)
        self.doppler_canvas.draw_idle()

    def _plot_dorf_from_payload(self, payload: dict) -> None:
        self.dorf_figure.clear()
        if payload.get("status") != "ok":
            ax = self.dorf_figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                str(payload.get("message", "Need 24 Doppler projections for DoRF velocity estimation.")),
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            self.dorf_canvas.draw_idle()
            return

        doppler_matrix = np.asarray(payload["doppler_matrix"], dtype=float)
        best_v = np.asarray(payload["best_v"], dtype=float)
        best_r = np.asarray(payload["best_r"], dtype=float)
        best_mask = np.asarray(payload["best_mask"], dtype=bool)
        best_loss = float(payload["best_loss"])
        loss_hist = np.asarray(payload["loss_hist"], dtype=float)
        proj_images = np.asarray(payload["proj_images"], dtype=float)
        dorf_meta = dict(payload.get("dorf_meta") or {})

        cluster_stats = dorf_meta.get("cluster_stats", [])
        kept_ids = dorf_meta.get("kept_ids", np.where(best_mask)[0])
        kept_dirs = dorf_meta.get("kept_dirs", best_r[kept_ids] if kept_ids.size else np.empty((0, 3)))
        labels = dorf_meta.get("labels", np.zeros(kept_dirs.shape[0], dtype=int))

        ordered_panels: list[tuple[str, object]] = []
        for key in self._dorf_plot_order():
            if key == "dorf_cluster_fit":
                ordered_panels.extend([(key, stat) for stat in cluster_stats])
            else:
                ordered_panels.append((key, None))
        visible_panels = [panel for panel in ordered_panels if self._subplot_visible(panel[0])]
        panel_slots = 0
        for category, _ in visible_panels:
            panel_slots += 4 if category == "dorf_vmf_clusters" else 1
        panel_count = max(panel_slots, 1)
        ncols = 2
        nrows = int(ceil(panel_count / ncols))
        grid = self.dorf_figure.add_gridspec(nrows, ncols, hspace=0.6, wspace=0.28)
        fig_height = max(10, nrows * 2.7)
        self.dorf_figure.set_size_inches(12, fig_height)
        self.dorf_canvas.setMinimumHeight(int(fig_height * self.dorf_figure.get_dpi()))

        panel_idx = 0
        for category, payload in visible_panels:
            if category == "dorf_loss":
                ax_loss = self.dorf_figure.add_subplot(grid[panel_idx // ncols, panel_idx % ncols])
                panel_idx += 1
                ax_loss.plot(loss_hist, marker="o", color="tab:blue", linewidth=1.0)
                self._apply_subplot_labels(
                    ax_loss,
                    category=category,
                    default_title=f"DoRF DTW loss (best={best_loss:.4f})",
                    default_xlabel="Iteration",
                    default_ylabel="Loss",
                )
                ax_loss.grid(True)
            elif category == "dorf_velocity":
                ax_vel = self.dorf_figure.add_subplot(grid[panel_idx // ncols, panel_idx % ncols])
                panel_idx += 1
                for dim, label in enumerate(("v_x", "v_y", "v_z")):
                    ax_vel.plot(best_v[:, dim], label=label, linewidth=0.95)
                self._apply_subplot_labels(
                    ax_vel,
                    category=category,
                    default_title="Estimated velocity components",
                    default_xlabel="Time index",
                    default_ylabel="Velocity",
                )
                ax_vel.legend(loc="upper right")
                ax_vel.grid(True)
            elif category == "dorf_energy":
                ax_energy = self.dorf_figure.add_subplot(grid[panel_idx // ncols, panel_idx % ncols])
                panel_idx += 1
                ax_energy.plot((best_v ** 2).sum(axis=1), label="‖v_est‖²", color="tab:green", linewidth=1.0)
                self._apply_subplot_labels(
                    ax_energy,
                    category=category,
                    default_title="Energy envelope",
                    default_xlabel="Time index",
                    default_ylabel="Energy",
                )
                ax_energy.legend(loc="upper right")
                ax_energy.grid(True)
            elif category == "dorf_projection_map":
                ax_proj = self.dorf_figure.add_subplot(grid[panel_idx // ncols, panel_idx % ncols])
                panel_idx += 1
                projection_map = proj_images.mean(axis=0)
                im = ax_proj.imshow(projection_map, cmap="seismic", aspect="auto", origin="upper")
                kept = int(np.sum(best_mask)) if best_mask is not None else 0
                self._apply_subplot_labels(
                    ax_proj,
                    category=category,
                    default_title=f"Average DoRF projection map ({kept}/{doppler_matrix.shape[1]} kept)",
                    default_xlabel="Longitude bins",
                    default_ylabel="Latitude bins",
                )
                self.dorf_figure.colorbar(
                    im, ax=ax_proj, orientation="vertical", fraction=0.045, pad=0.02
                )
            elif category == "dorf_cluster_fit" and payload is not None:
                cid, _, _, ap, perc, idxs = payload
                ax_cluster = self.dorf_figure.add_subplot(grid[panel_idx // ncols, panel_idx % ncols])
                panel_idx += 1
                obs = doppler_matrix[:, kept_ids[idxs]].mean(axis=1)
                pred = (best_v @ best_r[kept_ids[idxs]].T).mean(axis=1)
                ax_cluster.plot(obs, label="obs", linewidth=1.0)
                ax_cluster.plot(pred, label="pred", linewidth=1.0)
                self._apply_subplot_labels(
                    ax_cluster,
                    category=category,
                    default_title=f"Cluster {cid} (dom Ant{ap}, {perc:.0f}%)",
                    default_xlabel="Time index",
                    default_ylabel="Amplitude",
                )
                ax_cluster.grid(True)
                ax_cluster.legend(loc="upper right")
            elif category == "dorf_vmf_clusters":
                if panel_idx % ncols != 0:
                    spacer_ax = self.dorf_figure.add_subplot(grid[panel_idx // ncols, panel_idx % ncols])
                    spacer_ax.set_axis_off()
                    panel_idx += 1
                row_start = panel_idx // ncols
                ax_sphere = self.dorf_figure.add_subplot(
                    grid[row_start : row_start + 2, 0:ncols], projection="3d"
                )
                panel_idx += 2 * ncols
                u, vang = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 30j]
                ax_sphere.plot_surface(
                    np.cos(u) * np.sin(vang),
                    np.sin(u) * np.sin(vang),
                    np.cos(vang),
                    alpha=0.1,
                    color="gray",
                    linewidth=0,
                )
                if kept_dirs.size:
                    ax_sphere.scatter(
                        kept_dirs[:, 0],
                        kept_dirs[:, 1],
                        kept_dirs[:, 2],
                        c=cm.tab10(labels % 10),
                        s=30,
                    )
                    kappa_max = max(stat[2] for stat in cluster_stats) + 1e-9 if cluster_stats else 1.0
                    for cid, mu, kappa, ap, perc, _ in cluster_stats:
                        ax_sphere.quiver(
                            0, 0, 0, *mu, length=1, color="k", linewidth=2 + 4 * kappa / kappa_max
                        )
                        ax_sphere.text(*(1.08 * mu), f"κ={kappa:.1f}\nAnt{ap} {perc:.0f}%", ha="center")
                ax_sphere.set_title(self._subplot_text(category, "title", "vMF clusters"))
                ax_sphere._subplot_category = category
                ax_sphere.set_xlim([-1.2, 1.2])
                ax_sphere.set_ylim([-1.2, 1.2])
                ax_sphere.set_zlim([-1.2, 1.2])
                ax_sphere._vmf_plot_payload = {
                    "kept_dirs": np.asarray(kept_dirs),
                    "cluster_stats": list(cluster_stats),
                    "doppler_vectors": np.asarray(doppler_matrix[:, kept_ids]).T if kept_ids.size else np.empty((0, 0)),
                }

        while panel_idx < nrows * ncols:
            empty_ax = self.dorf_figure.add_subplot(grid[panel_idx // ncols, panel_idx % ncols])
            empty_ax.set_axis_off()
            panel_idx += 1

        self.dorf_figure.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.07)
        self._install_subplot_maximize_buttons(self.dorf_figure, self.dorf_canvas)
        self.dorf_canvas.draw_idle()

    def _install_subplot_maximize_buttons(self, figure: Figure, canvas: FigureCanvas) -> None:
        for button_ax, button in self._figure_maximize_buttons.get(figure, []):
            try:
                button.disconnect_events()
            except Exception:
                pass
            try:
                button_ax.remove()
            except Exception:
                pass

        button_refs: list[tuple[object, MatplotlibButton]] = []
        figure_px_w = max(float(figure.get_figwidth() * figure.dpi), 1.0)
        figure_px_h = max(float(figure.get_figheight() * figure.dpi), 1.0)
        button_px = 14.0
        btn_w = button_px / figure_px_w
        btn_h = button_px / figure_px_h
        gap_w = (2.0 / figure_px_w)
        for ax in figure.axes:
            if not ax.get_visible() or ax.get_label() == "<colorbar>":
                continue
            bbox = ax.get_position()
            x0 = min(max(bbox.x1 - btn_w, 0.002), 0.998 - btn_w)
            y0 = min(max(bbox.y1 - btn_h, 0.002), 0.998 - btn_h)

            button_ax = figure.add_axes([x0, y0, btn_w, btn_h])
            button = MatplotlibButton(button_ax, "⤢")
            button.label.set_fontsize(8)
            button.on_clicked(lambda _evt, source_ax=ax: self._open_subplot_window(source_ax))
            button_refs.append((button_ax, button))

            info_x0 = min(max(x0 - btn_w - gap_w, 0.002), 0.998 - btn_w)
            info_button_ax = figure.add_axes([info_x0, y0, btn_w, btn_h])
            info_button = MatplotlibButton(info_button_ax, "?")
            info_button.label.set_fontsize(8)
            info_button.on_clicked(lambda _evt, source_ax=ax: self._show_subplot_info(source_ax))
            button_refs.append((info_button_ax, info_button))

        self._figure_maximize_buttons[figure] = button_refs
        canvas.draw_idle()

    def _show_subplot_info(self, source_ax) -> None:
        category = getattr(source_ax, "_subplot_category", "")
        fallback_title = source_ax.get_title() or "Subplot"
        title, info = self._subplot_info_text(category, fallback_title)
        QMessageBox.information(self, f"Plot Info: {title}", info)

    def _open_subplot_window(self, source_ax) -> None:
        window = QWidget(self, Qt.Window)
        window.setWindowTitle(f"Plot Detail - {source_ax.get_title() or 'Subplot'}")
        window.resize(980, 620)
        layout = QVBoxLayout(window)
        detail_figure = Figure(figsize=(9, 5), dpi=100)
        detail_canvas = FigureCanvas(detail_figure)
        toolbar = NavigationToolbar(detail_canvas, window)
        layout.addWidget(toolbar)
        layout.addWidget(detail_canvas, stretch=1)

        vmf_payload = getattr(source_ax, "_vmf_plot_payload", None)
        if vmf_payload is not None and source_ax.name == "3d":
            left_ax = detail_figure.add_subplot(121, projection="3d")
            self._copy_axis_contents(source_ax, left_ax)
            right_ax = detail_figure.add_subplot(122)
            right_ax.set_title("Doppler projection vector")
            right_ax.set_xlabel("Time index")
            right_ax.set_ylabel("Amplitude")
            right_ax.grid(True)
            right_ax.text(
                0.5,
                0.5,
                "Click a point on the sphere",
                transform=right_ax.transAxes,
                ha="center",
                va="center",
                color="gray",
            )
            self._connect_vmf_detail_click(detail_figure, detail_canvas, left_ax, right_ax, vmf_payload)
        else:
            detail_ax = detail_figure.add_subplot(111, projection=source_ax.name)
            self._copy_axis_contents(source_ax, detail_ax)
        detail_figure.tight_layout()
        detail_canvas.draw_idle()

        self._plot_detail_windows.append(window)
        window.destroyed.connect(lambda *_: self._cleanup_detail_window(window))
        window.show()

    def _cleanup_detail_window(self, window: QWidget) -> None:
        if window in self._plot_detail_windows:
            self._plot_detail_windows.remove(window)

    @staticmethod
    def _connect_vmf_detail_click(detail_figure, detail_canvas, sphere_ax, vector_ax, vmf_payload) -> None:
        kept_dirs = np.asarray(vmf_payload.get("kept_dirs", np.empty((0, 3))), dtype=float)
        doppler_vectors = np.asarray(vmf_payload.get("doppler_vectors", np.empty((0, 0))), dtype=float)
        if kept_dirs.size == 0 or doppler_vectors.size == 0:
            return

        def _on_click(event):
            if event.inaxes != sphere_ax:
                return
            x2d, y2d, _ = proj3d.proj_transform(
                kept_dirs[:, 0], kept_dirs[:, 1], kept_dirs[:, 2], sphere_ax.get_proj()
            )
            projected_xy = sphere_ax.transData.transform(np.column_stack([x2d, y2d]))
            click_xy = np.array([event.x, event.y], dtype=float)
            distances = np.linalg.norm(projected_xy - click_xy, axis=1)
            nearest_idx = int(np.argmin(distances))
            if distances[nearest_idx] > 18.0:
                return

            vector_ax.clear()
            vector_ax.plot(doppler_vectors[nearest_idx], color="tab:purple", linewidth=1.2)
            direction = kept_dirs[nearest_idx]
            vector_ax.set_title(
                "Doppler projection vector\n"
                f"dir=({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})"
            )
            vector_ax.set_xlabel("Time index")
            vector_ax.set_ylabel("Amplitude")
            vector_ax.grid(True)
            detail_figure.tight_layout()
            detail_canvas.draw_idle()

        detail_canvas.mpl_connect("button_press_event", _on_click)

    @staticmethod
    def _copy_axis_contents(source_ax, target_ax) -> None:
        vmf_payload = getattr(source_ax, "_vmf_plot_payload", None)
        if vmf_payload is not None and source_ax.name == "3d" and target_ax.name == "3d":
            u, vang = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 30j]
            target_ax.plot_surface(
                np.cos(u) * np.sin(vang),
                np.sin(u) * np.sin(vang),
                np.cos(vang),
                alpha=0.1,
                color="gray",
                linewidth=0,
            )
            kept_dirs = np.asarray(vmf_payload.get("kept_dirs", np.empty((0, 3))))
            cluster_stats = vmf_payload.get("cluster_stats", [])
            if kept_dirs.size:
                labels = np.zeros(kept_dirs.shape[0], dtype=int)
                for cid, _, _, _, _, idxs in cluster_stats:
                    labels[np.asarray(idxs, dtype=int)] = int(cid)
                target_ax.scatter(
                    kept_dirs[:, 0],
                    kept_dirs[:, 1],
                    kept_dirs[:, 2],
                    c=cm.tab10(labels % 10),
                    s=30,
                )
                kappa_max = max(stat[2] for stat in cluster_stats) + 1e-9 if cluster_stats else 1.0
                for _cid, mu, kappa, ap, perc, _ in cluster_stats:
                    target_ax.quiver(0, 0, 0, *mu, length=1, color="k", linewidth=2 + 4 * kappa / kappa_max)
                    target_ax.text(*(1.08 * mu), f"κ={kappa:.1f}\nAnt{ap} {perc:.0f}%", ha="center")

        for line in source_ax.get_lines():
            target_ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=line.get_color(),
                linewidth=line.get_linewidth(),
                linestyle=line.get_linestyle(),
                marker=line.get_marker(),
                label=line.get_label(),
            )

        for image in source_ax.images:
            arr = image.get_array()
            target_ax.imshow(
                arr,
                cmap=image.get_cmap(),
                aspect=image.get_aspect(),
                origin=image.origin,
                extent=image.get_extent(),
                vmin=image.get_clim()[0],
                vmax=image.get_clim()[1],
            )

        for text in source_ax.texts:
            common_kwargs = {
                "ha": text.get_ha(),
                "va": text.get_va(),
                "fontsize": text.get_fontsize(),
                "color": text.get_color(),
            }
            if hasattr(source_ax, "get_zlim") and hasattr(target_ax, "set_zlim") and hasattr(text, "get_position_3d"):
                x, y, z = text.get_position_3d()
                target_ax.text(x, y, z, text.get_text(), **common_kwargs)
            else:
                transform = target_ax.transAxes if text.get_transform() == source_ax.transAxes else target_ax.transData
                target_ax.text(
                    text.get_position()[0],
                    text.get_position()[1],
                    text.get_text(),
                    transform=transform,
                    **common_kwargs,
                )

        target_ax.set_title(source_ax.get_title())
        target_ax.set_xlabel(source_ax.get_xlabel())
        target_ax.set_ylabel(source_ax.get_ylabel())
        grid_on = any(line.get_visible() for line in source_ax.get_xgridlines() + source_ax.get_ygridlines())
        target_ax.grid(grid_on)

        xlim = source_ax.get_xlim()
        ylim = source_ax.get_ylim()
        target_ax.set_xlim(*xlim)
        target_ax.set_ylim(*ylim)
        if hasattr(source_ax, "get_zlim") and hasattr(target_ax, "set_zlim"):
            try:
                target_ax.set_zlim(*source_ax.get_zlim())
            except Exception:
                pass

        legend = source_ax.get_legend()
        if legend is not None and target_ax.get_lines():
            target_ax.legend(loc=legend._loc if hasattr(legend, "_loc") else "best")

    def _on_capture_finished(self, success: bool, message: str):
        self.btn_capture.setEnabled(True)
        self.status_label.setText(message)
        if not success:
            QMessageBox.warning(self, "Demo Capture", message)

    def _on_capture_progress_updated(self, elapsed: float, duration: float) -> None:
        if self._capture_guidance_dialog is not None:
            self._capture_guidance_dialog.update_capture_progress(elapsed, duration)

    def _on_transfer_progress_started(self, total_files: int) -> None:
        if self._capture_guidance_dialog is not None:
            self._capture_guidance_dialog.start_transfer_progress(total_files)

    def _on_transfer_progress_updated(self, completed: int, total_files: int, status: str) -> None:
        if self._capture_guidance_dialog is not None:
            self._capture_guidance_dialog.update_transfer_progress(completed, total_files, status)

    def _on_capture_ui_finished(self) -> None:
        if self._capture_guidance_dialog is not None:
            self._capture_guidance_dialog.finish_capture()
            self._capture_guidance_dialog = None

    def closeEvent(self, event):
        if self._clock_timer is not None:
            self._clock_timer.stop()
        for entry in self.routers_info:
            router = entry.get("router")
            if router is None:
                continue
            try:
                router.close()
            except Exception:
                pass
        super().closeEvent(event)
