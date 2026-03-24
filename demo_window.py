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
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
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

        self.capture_finished.connect(self._on_capture_finished)
        self.plot_requested.connect(self._on_plot_requested)
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("Demo Window")
        self.resize(1200, 800)
        root = QVBoxLayout(self)

        self.status_label = QLabel("Ready for demo capture.", self)
        root.addWidget(self.status_label)

        stats_row = QFormLayout()
        self.packet_count_label = QLabel("Packets: -", self)
        self.sampling_rate_label = QLabel("Sampling rate: - pkt/s", self)
        stats_row.addRow("Received packets:", self.packet_count_label)
        stats_row.addRow("Sampling rate:", self.sampling_rate_label)
        root.addLayout(stats_row)

        self.chk_hampel_ratio_phase = QCheckBox(
            "Apply Hampel filter to CSI ratio phase", self
        )
        self.chk_hampel_ratio_phase.setChecked(
            bool(self.demo_profile.get("apply_hampel_to_ratio_phase", False))
        )
        root.addWidget(self.chk_hampel_ratio_phase)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_scroll = QScrollArea(self)
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setWidget(self.canvas)
        root.addWidget(self.plot_scroll, stretch=1)

        button_row = QHBoxLayout()
        self.btn_capture = QPushButton("CSI Capture", self)
        self.btn_capture.clicked.connect(self._on_capture_clicked)
        button_row.addWidget(self.btn_capture)

        self.btn_close = QPushButton("Close Window", self)
        self.btn_close.clicked.connect(self.close)
        button_row.addWidget(self.btn_close)
        button_row.addStretch(1)
        root.addLayout(button_row)

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
            self.packet_count_label.setText("Packets: 0")
            self.sampling_rate_label.setText("Sampling rate: 0.00 pkt/s")
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
            self.packet_count_label.setText("Packets: 0")
            self.sampling_rate_label.setText("Sampling rate: 0.00 pkt/s")
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
            wspace=0.35,
            hspace=0.6,
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
            ax_mag.set_ylabel("|Ratio|")
            ax_mag.set_title(f"RX {rx_idx + 1}: TX {tx_num + 1}/TX {tx_den + 1} Magnitude")
            ax_mag.grid(True)

            ax_phase.plot(x, ratio_phase, color="tab:green", linewidth=0.9)
            ax_phase.set_ylabel("Phase (rad)")
            ax_phase.set_title(f"RX {rx_idx + 1}: TX {tx_num + 1}/TX {tx_den + 1} Phase")
            ax_phase.grid(True)

            if row_idx == total_pairs - 1:
                ax_mag.set_xlabel(x_label)
                ax_phase.set_xlabel(x_label)

        self.figure.tight_layout()
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
        for entry in self.routers_info:
            router = entry.get("router")
            if router is None:
                continue
            try:
                router.close()
            except Exception:
                pass
        super().closeEvent(event)
