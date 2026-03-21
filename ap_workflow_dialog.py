"""Shared dialog utilities for access point workflows."""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Iterable, Optional

import sip
from PyQt5.QtCore import QObject, QThread
from PyQt5.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from wifi_csi_manager import WiFiCSIManager

CSI_MANAGER = WiFiCSIManager()


def _format_ap_title(ap: dict) -> str:
    return CSI_MANAGER.build_ap_title(ap)


class AccessPointStatusWidget(QWidget):
    STATUS_COLORS = {
        "red": "#d9534f",
        "yellow": "#f0ad4e",
        "green": "#5cb85c",
    }

    def __init__(self, ap: dict, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_name = QLabel(_format_ap_title(ap), self)
        self.lbl_name.setWordWrap(True)
        layout.addWidget(self.lbl_name)
        layout.addStretch(1)

        self.indicators = {}
        for status in ("red", "yellow", "green"):
            lbl = QLabel(self)
            lbl.setFixedSize(16, 16)
            lbl.setStyleSheet(self._circle_style(status == "red", status))
            self.indicators[status] = lbl
            layout.addWidget(lbl)

    def _circle_style(self, active: bool, status: str) -> str:
        color = self.STATUS_COLORS.get(status, "#d3d3d3")
        background = color if active else "#f8f9fa"
        return (
            "border: 2px solid {color}; border-radius: 8px; "
            "background: {background};"
        ).format(color=color, background=background)

    def set_status(self, status: str, tooltip: str = ""):
        for key, lbl in self.indicators.items():
            lbl.setStyleSheet(self._circle_style(key == status, key))
            if tooltip:
                lbl.setToolTip(tooltip)


class BaseAccessPointDialog(QDialog):
    error_title = "Operation Failed"
    stop_tooltip = "Stopped by user"

    def __init__(
        self,
        *,
        profile_name: str,
        access_points: Iterable[dict],
        heading_template: str,
        window_title: str,
        start_message: str,
        parent=None,
        log_file_path: Optional[str | Path] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"WIRLab - {window_title}")
        self.resize(700, 500)
        self.log_entries: list[str] = []
        self._log_file_path = Path(log_file_path) if log_file_path else None
        self._log_file_handle = None
        if self._log_file_path is not None:
            self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file_handle = self._log_file_path.open("a", encoding="utf-8")

        self.sorted_access_points = sorted(access_points, key=lambda ap: ap.get("order", 0))

        main_layout = QVBoxLayout(self)
        heading = QLabel(heading_template.format(profile_name=profile_name), self)
        heading.setWordWrap(True)
        main_layout.addWidget(heading)

        ap_group = QGroupBox("Access Points", self)
        ap_layout = QVBoxLayout(ap_group)
        self.ap_status_widgets: list[AccessPointStatusWidget] = []
        for ap in self.sorted_access_points:
            widget = AccessPointStatusWidget(ap)
            widget.set_status("red", "Waiting in queue")
            self.ap_status_widgets.append(widget)
            ap_layout.addWidget(widget)
        ap_layout.addStretch(1)
        main_layout.addWidget(ap_group)

        log_group = QGroupBox("Log", self)
        log_layout = QVBoxLayout(log_group)
        self.txt_log = QTextEdit(log_group)
        self.txt_log.setReadOnly(True)
        log_layout.addWidget(self.txt_log)
        main_layout.addWidget(log_group)

        button_row = QHBoxLayout()
        self.btn_stop = QPushButton("Stop", self)
        self.btn_stop.setStyleSheet("background-color: #d9534f; color: white;")
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        button_row.addWidget(self.btn_stop)

        button_row.addStretch(1)
        self.btn_close = QPushButton("Close", self)
        self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.close)
        button_row.addWidget(self.btn_close)
        main_layout.addLayout(button_row)

        self.worker_thread = QThread(self)
        self.worker = self._create_worker()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)

        self.worker.status_changed.connect(self._on_status_changed)
        self.worker.log_message.connect(self._append_log)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.worker_thread.quit)

        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self._stop_requested = False
        self._close_requested = False

        self._append_log(start_message)
        self.worker_thread.start()

    # Template hooks -----------------------------------------------------
    def _create_worker(self) -> QObject:  # pragma: no cover - abstract hook
        raise NotImplementedError

    def _handle_finished(self, success: bool, payload):
        del success, payload

    def _request_stop(self):
        request_stop = getattr(self.worker, "request_stop", None)
        if callable(request_stop):
            request_stop()

    # Signal handlers ----------------------------------------------------
    def _on_status_changed(self, idx: int, status: str, tooltip: str):
        if 0 <= idx < len(self.ap_status_widgets):
            self.ap_status_widgets[idx].set_status(status, tooltip)

    def _append_log(self, message: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.log_entries.append(line)
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.write(line + "\n")
                self._log_file_handle.flush()
            except Exception:
                pass
        elif self._log_file_path is not None:
            try:
                self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_file_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception:
                pass
        self.txt_log.append(line)

    def _on_error(self, message: str):
        self._append_log(message)
        QMessageBox.critical(self, self.error_title, message)
        self.btn_close.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._request_stop()
        self._close_log_file()

    def _on_finished(self, success: bool, payload):
        self._handle_finished(success, payload)
        self.btn_close.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._close_log_file()
        if self._close_requested:
            self.close()

    def _on_stop_clicked(self):
        self._request_stop_with_ui()
        self._close_requested = True
        self.close()

    def _request_stop_with_ui(self):
        if self._stop_requested:
            return
        self._stop_requested = True
        self.btn_stop.setEnabled(False)
        self._append_log("Stop requested by user. Attempting to cancel...")
        for widget in self.ap_status_widgets:
            widget.set_status("red", self.stop_tooltip)
        self._request_stop()

    def _close_log_file(self):
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None

    # Qt events ---------------------------------------------------------
    def closeEvent(self, event):
        if self._is_worker_thread_running():
            self._close_requested = True
            self._request_stop_with_ui()
            event.ignore()
            return
        self._request_stop()
        self._close_log_file()
        super().closeEvent(event)

    def _is_worker_thread_running(self) -> bool:
        thread = self.worker_thread
        if thread is None:
            return False
        try:
            if sip.isdeleted(thread):
                return False
        except Exception:
            return False
        try:
            return thread.isRunning()
        except RuntimeError:
            return False
