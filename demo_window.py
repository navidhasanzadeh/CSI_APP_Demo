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
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pcap_reader import ProcessPcap
from wifi_csi_manager import WiFiCSIManager, WiFiRouter


class DemoWindow(QMainWindow):
    capture_completed = pyqtSignal(str, str)
    capture_failed = pyqtSignal(str)

    def __init__(self, *, subject: dict, wifi_profile: dict, demo_profile: dict, results_dir: Path):
        super().__init__()
        self.subject = subject or {}
        self.wifi_profile = wifi_profile or {}
        self.demo_profile = demo_profile or {}
        self.results_dir = Path(results_dir)
        self.wifi_manager = WiFiCSIManager(self.wifi_profile)
        self.routers: dict[str, object] = {}
        self._capture_running = False

        self.setWindowTitle("Demo Window")
        self.resize(1200, 800)
        self._build_ui()

        self.capture_completed.connect(self._on_capture_completed)
        self.capture_failed.connect(self._on_capture_failed)

    def _build_ui(self):
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        self.status_label = QLabel("Ready for real-time CSI demo capture.", self)
        top.addWidget(self.status_label, stretch=1)

        self.capture_button = QPushButton("CSI Capture", self)
        self.capture_button.clicked.connect(self._on_capture_clicked)
        top.addWidget(self.capture_button)

        self.close_button = QPushButton("Close Window", self)
        self.close_button.clicked.connect(self.close)
        top.addWidget(self.close_button)
        layout.addLayout(top)

        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        self._plot_placeholder()

    def _plot_placeholder(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Run CSI Capture to visualize CSI ratio phase.", ha="center", va="center")
        ax.axis("off")
        self.canvas.draw_idle()

    def _on_capture_clicked(self):
        if self._capture_running:
            return
        access_points = self.wifi_profile.get("access_points", []) if isinstance(self.wifi_profile, dict) else []
        if not access_points:
            QMessageBox.warning(self, "No Access Points", "No access points configured in current Wi-Fi profile.")
            return

        self._capture_running = True
        self.capture_button.setEnabled(False)
        self.status_label.setText("Capturing CSI...")
        threading.Thread(target=self._capture_flow, daemon=True).start()

    def _capture_flow(self):
        try:
            duration = float(self.demo_profile.get("capture_duration", 5.0))
            delete_prev = bool(self.wifi_profile.get("delete_prev_pcap", False))
            ordered_aps = self.wifi_manager.prioritize_transmitters_first(
                self.wifi_profile.get("access_points", [])
            )
            local_capture_dir = self.results_dir / "csi_captures" / "demo"
            local_capture_dir.mkdir(parents=True, exist_ok=True)

            first_local_pcap = None
            first_ap_name = ""
            for index, ap in enumerate(ordered_aps, start=1):
                ap_name = ap.get("name") or ap.get("ssid") or f"ap_{index}"
                self.status_label.setText(f"Capturing from {ap_name}...")
                ap_key = self.wifi_manager.sanitize_ap_name(ap_name)
                router = self.routers.get(ap_key)
                if router is None:
                    router = self.wifi_manager.connect_router(ap, WiFiRouter)
                    self.routers[ap_key] = router

                channel_bw = self.wifi_manager.format_channel_bandwidth(
                    ap.get("channel", ""), ap.get("bandwidth", "")
                )
                macs = self.wifi_manager.parse_mac_addresses(ap.get("transmitter_macs", ""))
                self.wifi_manager.ensure_router_ready(
                    router,
                    channel_bw=channel_bw,
                    mac_addresses=macs,
                    skip_setup=False,
                )
                prefix = (
                    f"demo_{self.wifi_manager.sanitize_ap_name(self.subject.get('participant_id', 'participant'))}"
                    f"_{self.wifi_manager.sanitize_ap_name(ap_name)}"
                )
                remote_dir = self.wifi_manager.get_capture_remote_dir(ap, wifi_profile=self.wifi_profile)
                self.wifi_manager.start_csi_capture(
                    router,
                    duration=duration,
                    remote_directory=remote_dir,
                    exp_name=prefix,
                    delete_prev_pcap=delete_prev,
                )
                remote_files = [f.strip() for f in router.send_command(f"ls {remote_dir}")]
                target_candidates = self.wifi_manager.filter_matching_pcaps(remote_files, prefix)
                target_file = self.wifi_manager.latest_pcap_filename(target_candidates) or self.wifi_manager.latest_pcap_filename(remote_files)
                if not target_file:
                    continue
                use_ftp = str(ap.get("download_mode", "SFTP")).strip().upper() == "FTP"
                local_path, _ = self.wifi_manager.download_capture(
                    router,
                    remote_dir=remote_dir,
                    target_file=target_file,
                    local_directory=local_capture_dir,
                    use_ftp=use_ftp,
                )
                if first_local_pcap is None:
                    first_local_pcap = str(local_path)
                    first_ap_name = ap_name

            if first_local_pcap is None:
                self.capture_failed.emit("No pcap file was captured/downloaded.")
                return
            self.capture_completed.emit(first_local_pcap, first_ap_name)
        except Exception as exc:
            self.capture_failed.emit(str(exc))

    def _infer_bandwidth(self, ap: dict) -> int:
        text = str(ap.get("bandwidth", "80MHz")).lower()
        if "20" in text:
            return 20
        if "40" in text:
            return 40
        return 80

    def _on_capture_completed(self, local_pcap_path: str, ap_name: str):
        try:
            bw = 80
            for ap in self.wifi_profile.get("access_points", []):
                if (ap.get("name") or ap.get("ssid") or "") == ap_name:
                    bw = self._infer_bandwidth(ap)
                    break
            processor = ProcessPcap(str(Path(local_pcap_path).parent), bw=bw, tx_loc=[0, 0])
            csi_data, time_pkts, _, _ = processor.process_pcap(local_pcap_path, bw=bw)
            if csi_data.size == 0:
                raise RuntimeError("Captured pcap is empty or contains no CSI frames.")

            nfft = int(3.2 * bw)
            stream_a = csi_data[:, :nfft]
            stream_b = csi_data[:, nfft : 2 * nfft] if csi_data.shape[1] >= 2 * nfft else stream_a
            sc_idx = nfft // 2
            mag = np.abs(stream_a[:, sc_idx])
            ratio_phase = np.angle(stream_b[:, sc_idx] / (1e-6 + stream_a[:, sc_idx]))

            self.figure.clear()
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212)
            ax1.plot(time_pkts[: len(mag)], mag)
            ax1.set_ylabel("Magnitude")
            ax1.set_title(f"Demo CSI Magnitude ({Path(local_pcap_path).name})")
            ax1.grid(True)

            ax2.plot(time_pkts[: len(ratio_phase)], ratio_phase)
            ax2.set_ylabel("Ratio Phase (rad)")
            ax2.set_xlabel("Time (s)")
            ax2.set_title("CSI Ratio Phase")
            ax2.grid(True)
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self.status_label.setText(
                f"Capture complete ({datetime.now().strftime('%H:%M:%S')}). Loaded: {Path(local_pcap_path).name}"
            )
        except Exception as exc:
            self.capture_failed.emit(str(exc))
            return
        finally:
            self._capture_running = False
            self.capture_button.setEnabled(True)

    def _on_capture_failed(self, message: str):
        self._capture_running = False
        self.capture_button.setEnabled(True)
        self.status_label.setText("Capture failed.")
        QMessageBox.warning(self, "Demo Capture Error", message)

    def closeEvent(self, event):
        for router in self.routers.values():
            try:
                router.close()
            except Exception:
                pass
        self.routers.clear()
        super().closeEvent(event)
