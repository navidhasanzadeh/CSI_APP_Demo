"""Realtime Wi-Fi CSI demo window."""

from __future__ import annotations

import threading
import time
import sys
from datetime import datetime
from pathlib import Path
from itertools import combinations
from math import ceil

import numpy as np
import scipy.linalg as LA
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
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)
from PyQt5.QtCore import Qt, QTimer

from pcap_reader import ProcessPcap
from pcap_reader_ui import _read_ubilocate_csi
from wifi_csi_manager import WiFiCSIManager

DORF_PATH = Path(__file__).resolve().parent / "DoRF"
if str(DORF_PATH) not in sys.path:
    sys.path.append(str(DORF_PATH))
import doatools.estimation as estimation
from nerfs2 import estimate_velocity_from_radial_old_dtw

try:  # pragma: no cover - optional dependency at runtime
    from hampel import hampel
except ImportError:  # pragma: no cover - optional dependency at runtime
    hampel = None


class DemoWindow(QWidget):
    capture_finished = pyqtSignal(bool, str)
    plot_requested = pyqtSignal(str, int)

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
        self._capture_progress_dialog: QProgressDialog | None = None
        self._capture_progress_timer: QTimer | None = None
        self._clock_timer: QTimer | None = None
        self._figure_maximize_buttons: dict[Figure, list[tuple[object, MatplotlibButton]]] = {}
        self._plot_detail_windows: list[QWidget] = []

        self.capture_finished.connect(self._on_capture_finished)
        self.plot_requested.connect(self._on_plot_requested)
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
        self.demo_tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #cfd8e3; border-radius: 8px; background: #ffffff; }"
            "QTabBar::tab { padding: 5px 9px; font-size: 12px; }"
        )
        csi_tab = QWidget(self.demo_tabs)
        csi_layout = QVBoxLayout(csi_tab)
        self.chk_hampel_ratio_phase = QCheckBox(
            "Apply Hampel filter to CSI ratio phase", csi_tab
        )
        self.chk_hampel_ratio_phase.setChecked(
            bool(self.demo_profile.get("apply_hampel_to_ratio_phase", False))
        )
        csi_layout.addWidget(self.chk_hampel_ratio_phase)
        csi_layout.addWidget(self.plot_scroll, stretch=1)
        self.demo_tabs.addTab(csi_tab, "CSI Magnitude and Phase")

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
        self.demo_tabs.addTab(doppler_proj_tab, "Doppler Projections")

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
        self.demo_tabs.addTab(dorf_tab, "Doppler Radiance Fields (DoRF)")

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
        self.qr_website_label = QLabel(self._qr_website_text(), self)
        self.qr_website_label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self.qr_website_label.setStyleSheet("font-size: 12px; font-weight: 700; color: #2563eb;")
        self.qr_website_label.setWordWrap(True)
        self.qr_website_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bottom_row.addWidget(self.qr_website_label, alignment=Qt.AlignLeft | Qt.AlignBottom)

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

    def _on_capture_clicked(self):
        if self._capture_thread and self._capture_thread.is_alive():
            QMessageBox.information(self, "Capture Running", "A demo capture is already running.")
            return
        if not self.routers_info:
            QMessageBox.warning(self, "No Routers", "No connected routers are available for demo capture.")
            return

        self.btn_capture.setEnabled(False)
        capture_duration = self._capture_duration()
        self.status_label.setText("Capturing CSI... Please perform the target activity now.")
        self._start_capture_progress(capture_duration)
        self._capture_thread = threading.Thread(target=self._run_capture_cycle, daemon=True)
        self._capture_thread.start()

    def _start_capture_progress(self, capture_duration: float) -> None:
        self._capture_started_at = time.monotonic()
        self._capture_progress_dialog = QProgressDialog(
            "Capture in progress.\nPlease perform the target activity now.",
            None,
            0,
            100,
            self,
        )
        self._capture_progress_dialog.setWindowTitle("Demo CSI Capture")
        self._capture_progress_dialog.setWindowModality(Qt.WindowModal)
        self._capture_progress_dialog.setCancelButton(None)
        self._capture_progress_dialog.setMinimumDuration(0)
        self._capture_progress_dialog.setValue(0)
        self._capture_progress_dialog.show()

        self._capture_progress_timer = QTimer(self)

        def _update_progress() -> None:
            if not self._capture_progress_dialog:
                return
            elapsed = max(0.0, time.monotonic() - self._capture_started_at)
            percent = int(min(99, (elapsed / max(capture_duration, 0.1)) * 100))
            self._capture_progress_dialog.setValue(percent)

        self._capture_progress_timer.timeout.connect(_update_progress)
        self._capture_progress_timer.start(100)
        _update_progress()

    def _stop_capture_progress(self) -> None:
        if self._capture_progress_timer is not None:
            self._capture_progress_timer.stop()
            self._capture_progress_timer.deleteLater()
            self._capture_progress_timer = None
        if self._capture_progress_dialog is not None:
            self._capture_progress_dialog.setValue(100)
            self._capture_progress_dialog.close()
            self._capture_progress_dialog.deleteLater()
            self._capture_progress_dialog = None

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

            for thread in capture_threads:
                thread.join()

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

    @staticmethod
    def _has_payload_packets(pcap_path: Path) -> bool:
        try:
            # 24 bytes is a global pcap header-only file with zero packets.
            return pcap_path.exists() and pcap_path.stat().st_size > 24
        except Exception:
            return False

    def _apply_hampel_filter(self, values: np.ndarray) -> np.ndarray:
        if hampel is None:
            self.status_label.setText(
                "Hampel filter is unavailable (missing dependency). Plotting unfiltered ratio phase."
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
        csi_data: np.ndarray | None = None
        time_pkts: np.ndarray | None = None
        nfft = int(3.2 * bandwidth_mhz)

        try:
            # Match PCAP reader behavior for demo plotting by using the
            # UbiLocate parser first.
            csi_data, time_pkts, _ = _read_ubilocate_csi(
                str(pcap_path), bw=bandwidth_mhz, is_4ss=True
            )
            if csi_data.size == 0:
                raise ValueError("No UbiLocate CSI packets found")
        except Exception:
            # Fallback to Nexmon processing so existing captures continue to
            # render even if they are not in UbiLocate format.
            processor = ProcessPcap(str(pcap_path.parent), bw=bandwidth_mhz, tx_loc=[0, 0])
            csi_data_raw, time_raw, _, _ = processor.process_pcap(str(pcap_path), bw=bandwidth_mhz)
            rx_count = getattr(processor, "num_cores", 1) or 1
            nfft = processor.nfft
            csi_data = csi_data_raw.reshape((-1, nfft, rx_count, 1))
            time_pkts = np.asarray(time_raw)

        if csi_data is None or time_pkts is None or csi_data.size == 0:
            self.status_label.setText(f"Unable to load CSI data from {pcap_path.name}.")
            self.packet_count_label.setText("0")
            self.sampling_rate_label.setText("0.00 pkt/s")
            return

        time_vals = np.asarray(time_pkts, dtype=float)
        packet_count = int(min(csi_data.shape[0], time_vals.size if time_vals.size else csi_data.shape[0]))
        csi_data = csi_data[:packet_count]
        time_vals = time_vals[:packet_count] if time_vals.size else np.array([])
        self.packet_count_label.setText(str(packet_count))

        sampling_rate = 0.0
        if packet_count > 1 and time_vals.size == packet_count:
            duration = float(time_vals[-1] - time_vals[0])
            if duration > 0:
                sampling_rate = float((packet_count - 1) / duration)
        self.sampling_rate_label.setText(f"{sampling_rate:.2f} pkt/s")

        subcarrier_idx = min(23, nfft - 1)
        rx_count = csi_data.shape[2] if csi_data.ndim >= 3 else 1
        tx_count = csi_data.shape[3] if csi_data.ndim >= 4 else 1
        tx_pairs = list(combinations(range(tx_count), 2))
        if not tx_pairs:
            self.status_label.setText(
                f"Unable to compute CSI ratio from {pcap_path.name}: at least 2 TX antennas are required."
            )
            return

        total_pairs = rx_count * len(tx_pairs)
        if time_vals.size == packet_count:
            x = time_vals
            x_label = "Time (s)"
        else:
            x = np.arange(packet_count)
            x_label = "Packet index"

        self.figure.clear()
        grid = self.figure.add_gridspec(
            total_pairs,
            2,
            width_ratios=[1, 1],
            wspace=0.65,
            hspace=1.05,
        )
        figure_height = max(8, total_pairs * 2.1)
        self.figure.set_size_inches(12, figure_height)
        self.canvas.setMinimumHeight(int(figure_height * self.figure.get_dpi()))

        for row_idx, (rx_idx, (tx_num, tx_den)) in enumerate(
            (rx_tx for rx_tx in ((r, pair) for r in range(rx_count) for pair in tx_pairs))
        ):
            numerator = csi_data[:, subcarrier_idx, rx_idx, tx_num]
            denominator = csi_data[:, subcarrier_idx, rx_idx, tx_den]
            ratio = numerator / (denominator + 1e-12)
            ratio_mag = np.abs(ratio)
            ratio_phase = np.angle(ratio)
            if self.chk_hampel_ratio_phase.isChecked():
                ratio_phase = self._apply_hampel_filter(ratio_phase)

            ax_mag = self.figure.add_subplot(grid[row_idx, 0])
            ax_phase = self.figure.add_subplot(grid[row_idx, 1], sharex=ax_mag)

            ax_mag.plot(x, ratio_mag, color="tab:blue", linewidth=0.9)
            ax_mag.margins(x=0.08, y=0.25)
            ax_mag.set_ylabel("|Ratio|")
            ax_mag.set_title(f"RX {rx_idx + 1}: TX {tx_num + 1}/TX {tx_den + 1} Magnitude")
            ax_mag.grid(True)

            ax_phase.plot(x, ratio_phase, color="tab:green", linewidth=0.9)
            ax_phase.margins(x=0.08, y=0.25)
            ax_phase.set_ylabel("Phase (rad)")
            ax_phase.set_title(f"RX {rx_idx + 1}: TX {tx_num + 1}/TX {tx_den + 1} Phase")
            ax_phase.grid(True)

            if row_idx == total_pairs - 1:
                ax_mag.set_xlabel(x_label)
                ax_phase.set_xlabel(x_label)

        self.figure.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.07)
        self._install_subplot_maximize_buttons(self.figure, self.canvas)
        self.canvas.draw_idle()
        self._plot_doppler_music(csi_data, time_vals, packet_count, tx_pairs)

    def _on_plot_requested(self, pcap_path: str, bandwidth_mhz: int):
        self._plot_ratio(Path(pcap_path), int(bandwidth_mhz))

    def _root_music_csi_like(
        self, sample_data: np.ndarray, do_cov_processing: bool = False, L: int = 1
    ) -> np.ndarray:
        """Use original DoRF Root_MUSIC_CSI routine for Doppler projection."""
        _ = do_cov_processing
        if sample_data.ndim != 2 or sample_data.shape[1] < 32:
            return np.array([], dtype=float)
        n_sc, n_t = sample_data.shape
        sig_padded = np.zeros((n_sc, n_t + 100), dtype=np.complex128)
        sig_padded[:, 50:-50] = sample_data
        sig_padded[:, :50] = sample_data[:, 50:0:-1]
        sig_padded[:, -50:] = sample_data[:, -1:-51:-1]

        doppler_vector = []
        for w in range(50, n_t + 50):
            sig_window = sig_padded[:, w - 16 : w + 16]
            h = sig_window.T
            covariance = h @ h.conj().T
            covariance = np.nan_to_num(covariance)
            eigvals, eigvecs = LA.eig(covariance)
            eigvals = np.abs(eigvals)
            _ = (eigvals, eigvecs)
            estimator = estimation.RootMUSIC1D(1.0)
            _, estimates = estimator.estimate(covariance, L)
            doppler_vector.append(estimates.locations)
        return np.asarray(doppler_vector, dtype=float).reshape(-1)

    def _extract_csi_ratio_for_stream(
        self, csi_data: np.ndarray, rx_idx: int, tx_pair: tuple[int, int]
    ) -> np.ndarray:
        """Apply DoRF CSI-ratio preprocessing for a single RX/TX stream pair."""
        tx_num, tx_den = tx_pair
        ratio_chunks: list[np.ndarray] = []
        max_subants = min(4, csi_data.shape[1] // 64)
        for subant in range(max_subants):
            csi_1 = csi_data[:, :, rx_idx, tx_num]
            csi_1_1 = csi_1[:, 64 * subant : 64 * (1 + subant)]
            csi_1_1 = csi_1_1[:, 6:-6]
            valid_idx = [i for i in range(csi_1_1.shape[1]) if i not in (19, 46)]
            csi_1_1 = csi_1_1[:, np.array(valid_idx, dtype=int)]

            csi_2 = csi_data[:, :, rx_idx, tx_den]
            csi_2_1 = csi_2[:, 64 * subant : 64 * (1 + subant)]
            csi_2_1 = csi_2_1[:, 6:-6]
            valid_idx = [i for i in range(csi_2_1.shape[1]) if i not in (19, 46)]
            csi_2_1 = csi_2_1[:, np.array(valid_idx, dtype=int)]

            csi_21 = csi_2_1 / (1e-6 + csi_1_1)
            ratio_chunks.append(csi_21)

        if not ratio_chunks:
            return np.array([], dtype=np.complex128)
        csi_ratio = np.concatenate(ratio_chunks, axis=1)
        if csi_ratio.shape[1] < 2:
            return csi_ratio

        good_subcarriers = []
        for iii in range(csi_ratio.shape[1] - 1):
            corr = np.corrcoef(np.angle(csi_ratio[:, iii]), np.angle(csi_ratio[:, iii + 1]))[0][1]
            good_subcarriers.append(corr)
        good_subcarriers = np.abs(np.nan_to_num(np.asarray(good_subcarriers)))
        good_subcarriers = np.where(good_subcarriers > 0.6)[0]
        if good_subcarriers.size == 0:
            return np.array([], dtype=np.complex128)
        return csi_ratio[:, good_subcarriers]

    def _plot_doppler_music(
        self,
        csi_data: np.ndarray,
        time_vals: np.ndarray,
        packet_count: int,
        tx_pairs: list[tuple[int, int]],
    ) -> None:
        rx_count = csi_data.shape[2] if csi_data.ndim >= 3 else 1
        total_pairs = rx_count * len(tx_pairs)
        self.doppler_figure.clear()
        if total_pairs == 0:
            self.doppler_canvas.draw_idle()
            return
        if time_vals.size == packet_count:
            x = time_vals
            x_label = "Time (s)"
        else:
            x = np.arange(packet_count)
            x_label = "Packet index"

        ncols = 2
        nrows = int(ceil(total_pairs / ncols))
        grid = self.doppler_figure.add_gridspec(nrows, ncols, hspace=0.6, wspace=0.25)
        fig_height = max(8, nrows * 2.4)
        self.doppler_figure.set_size_inches(11, fig_height)
        self.doppler_canvas.setMinimumHeight(int(fig_height * self.doppler_figure.get_dpi()))
        dopplers: list[np.ndarray] = []

        for row_idx, (rx_idx, tx_pair) in enumerate(
            (rx_tx for rx_tx in ((r, pair) for r in range(rx_count) for pair in tx_pairs))
        ):
            csi_ratio = self._extract_csi_ratio_for_stream(csi_data, rx_idx, tx_pair)
            music_output = self._root_music_csi_like(csi_ratio.T) if csi_ratio.size else np.array([])
            dopplers.append(np.asarray(music_output, dtype=float))
            ax = self.doppler_figure.add_subplot(grid[row_idx // ncols, row_idx % ncols])
            if music_output.size:
                x_trim = x[: music_output.size]
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
            ax.set_ylabel("Norm. Doppler")
            ax.set_title(f"RX {rx_idx + 1}: TX {tx_pair[0] + 1}/TX {tx_pair[1] + 1} MUSIC")
            ax.grid(True)
            if row_idx >= total_pairs - ncols:
                ax.set_xlabel(x_label)
        if total_pairs % ncols:
            spare_ax = self.doppler_figure.add_subplot(grid[-1, -1])
            spare_ax.set_axis_off()
        self.doppler_figure.subplots_adjust(left=0.08, right=0.95, top=0.96, bottom=0.08)
        self._install_subplot_maximize_buttons(self.doppler_figure, self.doppler_canvas)
        self.doppler_canvas.draw_idle()
        self._plot_dorf_from_dopplers(dopplers)

    def _plot_dorf_from_dopplers(self, dopplers: list[np.ndarray]) -> None:
        self.dorf_figure.clear()
        if len(dopplers) < 24:
            ax = self.dorf_figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Need 24 Doppler projections for DoRF velocity estimation.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            self.dorf_canvas.draw_idle()
            return

        selected = [np.asarray(v, dtype=float).reshape(-1) for v in dopplers[:24]]
        min_len = min((v.size for v in selected), default=0)
        if min_len == 0:
            ax = self.dorf_figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "DoRF estimation skipped: at least one Doppler projection is empty.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            self.dorf_canvas.draw_idle()
            return

        doppler_matrix = np.vstack([v[:min_len] for v in selected]).T
        for i in range(doppler_matrix.shape[1]):
            doppler_matrix[:, i] = doppler_matrix[:, i] - np.mean(doppler_matrix[:, i])

        t = np.arange(doppler_matrix.shape[0])
        best_v, best_r, best_mask, best_loss, loss_hist, proj_images, _, dorf_meta = estimate_velocity_from_radial_old_dtw(
            doppler_matrix[:, :],
            subset_fraction=1.0,
            outer_iterations=10,
            mean_zero_velocity=False,
            true_v=None,
            time_axis=t,
            camera_numbers=list(range(doppler_matrix.shape[1])),
            dtw_window=8,
            use_support_dtw=False,
            visualise=self.chk_dorf_visualize.isChecked(),
            grid_res=6,
            max_clusters=2,
            return_metadata=True,
        )

        cluster_stats = dorf_meta.get("cluster_stats", [])
        kept_ids = dorf_meta.get("kept_ids", np.where(best_mask)[0])
        kept_dirs = dorf_meta.get("kept_dirs", best_r[kept_ids] if kept_ids.size else np.empty((0, 3)))
        labels = dorf_meta.get("labels", np.zeros(kept_dirs.shape[0], dtype=int))

        panel_count = 4 + len(cluster_stats) + 1
        ncols = 2
        nrows = int(ceil(panel_count / ncols))
        grid = self.dorf_figure.add_gridspec(nrows, ncols, hspace=0.6, wspace=0.28)
        fig_height = max(10, nrows * 2.7)
        self.dorf_figure.set_size_inches(12, fig_height)
        self.dorf_canvas.setMinimumHeight(int(fig_height * self.dorf_figure.get_dpi()))

        panel_idx = 0
        ax_loss = self.dorf_figure.add_subplot(grid[panel_idx // ncols, panel_idx % ncols])
        panel_idx += 1
        ax_loss.plot(loss_hist, marker="o", color="tab:blue", linewidth=1.0)
        ax_loss.set_title(f"DoRF DTW loss (best={best_loss:.4f})")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_xlabel("Iteration")
        ax_loss.grid(True)

        ax_vel = self.dorf_figure.add_subplot(grid[panel_idx // ncols, panel_idx % ncols])
        panel_idx += 1
        for dim, label in enumerate(("v_x", "v_y", "v_z")):
            ax_vel.plot(best_v[:, dim], label=label, linewidth=0.95)
        ax_vel.set_title("Estimated velocity components")
        ax_vel.set_xlabel("Time index")
        ax_vel.set_ylabel("Velocity")
        ax_vel.legend(loc="upper right")
        ax_vel.grid(True)

        ax_energy = self.dorf_figure.add_subplot(grid[panel_idx // ncols, panel_idx % ncols])
        panel_idx += 1
        ax_energy.plot((best_v ** 2).sum(axis=1), label="‖v_est‖²", color="tab:green", linewidth=1.0)
        ax_energy.set_title("Energy envelope")
        ax_energy.set_xlabel("Time index")
        ax_energy.set_ylabel("Energy")
        ax_energy.legend(loc="upper right")
        ax_energy.grid(True)

        ax_proj = self.dorf_figure.add_subplot(grid[panel_idx // ncols, panel_idx % ncols])
        panel_idx += 1
        projection_map = proj_images.mean(axis=0)
        im = ax_proj.imshow(projection_map, cmap="seismic", aspect="auto", origin="upper")
        kept = int(np.sum(best_mask)) if best_mask is not None else 0
        ax_proj.set_title(f"Average DoRF projection map ({kept}/{doppler_matrix.shape[1]} kept)")
        ax_proj.set_xlabel("Longitude bins")
        ax_proj.set_ylabel("Latitude bins")
        self.dorf_figure.colorbar(im, ax=ax_proj, orientation="vertical", fraction=0.045, pad=0.02)

        for cid, _, _, ap, perc, idxs in cluster_stats:
            ax_cluster = self.dorf_figure.add_subplot(grid[panel_idx // ncols, panel_idx % ncols])
            panel_idx += 1
            obs = doppler_matrix[:, kept_ids[idxs]].mean(axis=1)
            pred = (best_v @ best_r[kept_ids[idxs]].T).mean(axis=1)
            ax_cluster.plot(obs, label="obs", linewidth=1.0)
            ax_cluster.plot(pred, label="pred", linewidth=1.0)
            ax_cluster.set_title(f"Cluster {cid} (dom Ant{ap}, {perc:.0f}%)")
            ax_cluster.set_xlabel("Time index")
            ax_cluster.grid(True)
            ax_cluster.legend(loc="upper right")

        ax_sphere = self.dorf_figure.add_subplot(
            grid[panel_idx // ncols, panel_idx % ncols], projection="3d"
        )
        panel_idx += 1
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
                ax_sphere.quiver(0, 0, 0, *mu, length=1, color="k", linewidth=2 + 4 * kappa / kappa_max)
                ax_sphere.text(*(1.08 * mu), f"κ={kappa:.1f}\nAnt{ap} {perc:.0f}%", ha="center")
        ax_sphere.set_title("vMF clusters")
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

        self._figure_maximize_buttons[figure] = button_refs
        canvas.draw_idle()

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
        self._stop_capture_progress()
        self.btn_capture.setEnabled(True)
        self.status_label.setText(message)
        if not success:
            QMessageBox.warning(self, "Demo Capture", message)

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
