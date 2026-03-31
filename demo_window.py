"""Realtime Wi-Fi CSI demo window."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from itertools import combinations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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

        self.capture_finished.connect(self._on_capture_finished)
        self.plot_requested.connect(self._on_plot_requested)
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("Demo Window")
        self.resize(1280, 840)
        root = QVBoxLayout(self)
        root.setSpacing(12)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(16)
        header_left_col = QVBoxLayout()
        header_left_col.setSpacing(2)
        self.icassp_logo_label = QLabel(self._icassp_title_text(), self)
        self.icassp_logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.icassp_logo_label.setStyleSheet("font-size: 24px; font-weight: 700; color: #1e3a8a;")
        header_left_col.addWidget(self.icassp_logo_label)

        self.authors_label = QLabel(self._authors_text(), self)
        self.authors_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.authors_label.setWordWrap(True)
        self.authors_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #111827;")
        header_left_col.addWidget(self.authors_label)

        self.university_label = QLabel(self._university_text(), self)
        self.university_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.university_label.setWordWrap(True)
        self.university_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #111827;")
        header_left_col.addWidget(self.university_label)
        header_row.addLayout(header_left_col, stretch=2)

        self.title_label = QLabel(self._demo_title_text(), self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-size: 30px; font-weight: 700; color: #0b1f3a;")
        header_row.addWidget(self.title_label, stretch=4)

        logo_col = QVBoxLayout()
        logo_col.setSpacing(6)
        logo_col.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.qr_placeholder = QLabel(self)
        self.qr_placeholder.setFixedSize(180, 180)
        self.qr_placeholder.setAlignment(Qt.AlignCenter)
        self.qr_placeholder.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.qr_placeholder.setStyleSheet(
            "QLabel {border: 2px dashed #94a3b8; border-radius: 10px; background: #f8fafc; "
            "color: #475569; font-size: 14px; font-weight: 600;}"
        )
        self._update_qr_placeholder()
        logo_col.addWidget(self.qr_placeholder, alignment=Qt.AlignRight)

        self.qr_website_label = QLabel(self._qr_website_text(), self)
        self.qr_website_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.qr_website_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #2563eb;")
        self.qr_website_label.setWordWrap(True)
        logo_col.addWidget(self.qr_website_label, alignment=Qt.AlignRight)

        self.wirlab_logo_label = QLabel("WIRLab", self)
        self.wirlab_logo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.wirlab_logo_label.setStyleSheet("font-size: 28px; font-weight: 800; color: #0f5e2b;")
        logo_col.addWidget(self.wirlab_logo_label, alignment=Qt.AlignRight)
        header_row.addLayout(logo_col, stretch=2)
        root.addLayout(header_row)

        self.status_label = QLabel("Ready for demo capture.", self)
        self.status_label.setStyleSheet("font-size: 15px; color: #1f2937;")
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
            "QTabBar::tab { padding: 8px 12px; }"
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
        doppler_proj_layout.addWidget(QLabel("Doppler Projections (coming soon).", doppler_proj_tab))
        doppler_proj_layout.addStretch(1)
        self.demo_tabs.addTab(doppler_proj_tab, "Doppler Projections")

        dorf_tab = QWidget(self.demo_tabs)
        dorf_layout = QVBoxLayout(dorf_tab)
        dorf_layout.addWidget(QLabel("Doppler Radiance Fields (DoRF) (coming soon).", dorf_tab))
        dorf_layout.addStretch(1)
        self.demo_tabs.addTab(dorf_tab, "Doppler Radiance Fields (DoRF)")

        content_row = QHBoxLayout()
        content_row.addWidget(self.demo_tabs, stretch=4)

        info_pane = QFrame(self)
        info_pane.setFrameShape(QFrame.StyledPanel)
        info_pane.setStyleSheet(
            "QFrame {border: 1px solid #cfd8e3; border-radius: 8px; background: #f8fafc;}"
            "QLabel {color: #1f2937;}"
        )
        info_layout = QFormLayout(info_pane)
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

        bottom_row = QHBoxLayout()
        button_row = QHBoxLayout()
        self.btn_capture = QPushButton("CSI Capture", self)
        self.btn_capture.clicked.connect(self._on_capture_clicked)
        self.btn_capture.setStyleSheet(
            "QPushButton {background-color: #16a34a; color: white; font-size: 16px; font-weight: 700; "
            "padding: 10px 18px; border-radius: 8px;}"
            "QPushButton:hover {background-color: #15803d;}"
            "QPushButton:disabled {background-color: #86efac; color: #f8fafc;}"
        )
        button_row.addWidget(self.btn_capture)

        self.btn_close = QPushButton("Close Window", self)
        self.btn_close.clicked.connect(self.close)
        self.btn_close.setStyleSheet(
            "QPushButton {background-color: #2563eb; color: white; font-size: 15px; font-weight: 600; "
            "padding: 10px 18px; border-radius: 8px;}"
            "QPushButton:hover {background-color: #1d4ed8;}"
        )
        button_row.addWidget(self.btn_close)
        button_row.addStretch(1)

        bottom_row.addLayout(button_row, stretch=3)

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
                self.qr_placeholder.setPixmap(
                    pixmap.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
        self.canvas.draw_idle()

    def _on_plot_requested(self, pcap_path: str, bandwidth_mhz: int):
        self._plot_ratio(Path(pcap_path), int(bandwidth_mhz))

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
