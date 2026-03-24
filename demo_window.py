"""Realtime Wi-Fi CSI demo window."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pcap_reader import ProcessPcap
from wifi_csi_manager import WiFiCSIManager


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

        self.capture_finished.connect(self._on_capture_finished)
        self.plot_requested.connect(self._on_plot_requested)
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("Demo Window")
        self.resize(1200, 800)
        root = QVBoxLayout(self)

        self.status_label = QLabel("Ready for demo capture.", self)
        root.addWidget(self.status_label)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        root.addWidget(self.canvas, stretch=1)

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
        self.status_label.setText("Capturing CSI...")
        self._capture_thread = threading.Thread(target=self._run_capture_cycle, daemon=True)
        self._capture_thread.start()

    def _run_capture_cycle(self):
        try:
            capture_duration = self._capture_duration()
            capture_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            downloaded = []

            for router_entry in self.routers_info:
                run_info = dict(router_entry.get("run_info") or {})
                ap = run_info.get("ap", {})
                if not self.wifi_manager.is_sniffer(ap):
                    continue
                router = router_entry.get("router")
                if router is None:
                    continue
                ap_name = ap.get("name") or ap.get("ssid") or "ap"
                exp_name = f"demo_{self.wifi_manager.sanitize_ap_name(ap_name)}_{capture_tag}"
                remote_dir = run_info.get("remote_dir", "/mnt/CSI_USB/")

                self.wifi_manager.start_csi_capture(
                    router,
                    duration=capture_duration,
                    remote_directory=remote_dir,
                    exp_name=exp_name,
                    delete_prev_pcap=False,
                )

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
                self.capture_finished.emit(False, "No PCAP file was downloaded from sniffers.")
                return

            self.plot_requested.emit(str(downloaded[0][0]), int(downloaded[0][1]))
            self.capture_finished.emit(True, f"Capture complete: {downloaded[0][0].name}")
        except Exception as exc:  # pragma: no cover - network/runtime behavior
            self.capture_finished.emit(False, f"Demo capture failed: {exc}")

    def _plot_ratio(self, pcap_path: Path, bandwidth_mhz: int):
        processor = ProcessPcap(str(pcap_path.parent), bw=bandwidth_mhz, tx_loc=[0, 0])
        csi_data, time_pkts, _, _ = processor.process_pcap(str(pcap_path), bw=bandwidth_mhz)
        rx_count = getattr(processor, "num_cores", 1) or 1
        nfft = processor.nfft
        csi_data = csi_data.reshape((-1, nfft, rx_count, 1))

        stream = csi_data[:, min(23, nfft - 1), 0, 0]
        magnitude = np.abs(stream)
        ratio_phase = np.angle(stream / (np.mean(stream) + 1e-6))
        time_vals = np.asarray(time_pkts)
        if time_vals.size == magnitude.size:
            x = time_vals
            x_label = "Time (s)"
        else:
            x = np.arange(magnitude.size)
            x_label = "Packet index"

        self.figure.clear()
        ax_mag = self.figure.add_subplot(2, 1, 1)
        ax_ratio = self.figure.add_subplot(2, 1, 2, sharex=ax_mag)
        ax_mag.plot(x, magnitude, color="tab:blue")
        ax_mag.set_title(f"CSI Magnitude ({pcap_path.name})")
        ax_mag.set_ylabel("|CSI|")
        ax_mag.grid(True)

        ax_ratio.plot(x, ratio_phase, color="tab:green")
        ax_ratio.set_title("CSI Ratio-like Phase (real-time demo)")
        ax_ratio.set_ylabel("Phase (rad)")
        ax_ratio.set_xlabel(x_label)
        ax_ratio.grid(True)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _on_plot_requested(self, pcap_path: str, bandwidth_mhz: int):
        self._plot_ratio(Path(pcap_path), int(bandwidth_mhz))

    def _on_capture_finished(self, success: bool, message: str):
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
