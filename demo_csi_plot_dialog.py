"""Dialog helpers for plotting CSI immediately after demo-mode capture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout

from UbiLocate_pcap_loader import read_csi_data
from pcap_reader import ProcessPcap


def load_csi_for_framework(
    pcap_path: str | Path,
    *,
    framework: str,
    bandwidth: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Load CSI + packet time arrays in a common shape for plotting."""
    path = Path(pcap_path)
    framework_name = (framework or "nexmon").strip().lower()
    if framework_name == "ubilocate":
        csi_data, timestamps = read_csi_data(str(path), bw=bandwidth)
        return np.asarray(csi_data), np.asarray(timestamps)

    processor = ProcessPcap(str(path.parent), bw=bandwidth, tx_loc=[0, 0])
    csi_data, time_pkts, _, _ = processor.process_pcap(str(path), bw=bandwidth)
    csi_data = np.asarray(csi_data)
    if csi_data.ndim == 2:
        csi_data = csi_data[:, :, np.newaxis, np.newaxis]
    return csi_data, np.asarray(time_pkts)


class DemoCSIPlotDialog(QDialog):
    """Small plotting dialog used by demo mode to display latest capture."""

    def __init__(
        self,
        *,
        csi_data: np.ndarray,
        time_values: np.ndarray,
        title: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 700)

        layout = QVBoxLayout(self)
        figure = Figure(figsize=(10, 6))
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

        axes = figure.subplots(2, 1, sharex=True)
        safe_time = np.asarray(time_values).flatten()
        if safe_time.size == 0:
            safe_time = np.arange(csi_data.shape[0], dtype=float)

        stream = csi_data[:, 0, 0, 0]
        axes[0].plot(safe_time[: len(stream)], np.abs(stream), color="tab:blue")
        axes[0].set_ylabel("|CSI|")
        axes[0].set_title("Demo capture magnitude (SC1, RX1/TX1)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(safe_time[: len(stream)], np.angle(stream), color="tab:orange")
        axes[1].set_ylabel("Phase (rad)")
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(True, alpha=0.3)

        figure.tight_layout()
        canvas.draw_idle()

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
