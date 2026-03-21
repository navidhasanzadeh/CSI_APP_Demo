"""Dialog that displays progress while CSI files are transferred."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class TransferringFilesDialog(QDialog):
    """Simple dialog to track CSI file transfers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WIRLab - Transferring files")
        self.setModal(False)
        self.resize(720, 480)

        self._rows = {}
        self._total_files = 0
        self._downloaded_files = 0

        layout = QVBoxLayout(self)
        self.overall_progress = QProgressBar(self)
        self.overall_progress.setRange(0, 1)
        self.overall_progress.setValue(0)
        self.overall_progress.setFormat("0/0 PCAPs downloaded")
        self.overall_progress.setTextVisible(True)
        self.overall_progress.setStyleSheet("QProgressBar::chunk { background-color: #2ecc71; }")
        layout.addWidget(self.overall_progress)

        intro = QLabel("Downloading CSI captures via SCP…", self)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_contents = QWidget(self.scroll_area)
        self.scroll_layout = QVBoxLayout(self.scroll_contents)
        self.scroll_layout.addStretch(1)
        self.scroll_area.setWidget(self.scroll_contents)
        layout.addWidget(self.scroll_area)

    def _format_size(self, size_bytes: int | None) -> str:
        if not size_bytes:
            return "Size unknown"
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.2f} MB"

    def _update_label_text(self, filename: str, row: dict):
        size_text = self._format_size(row.get("size"))
        row["label"].setText(f"{filename} — {size_text}")

    def reset(self):
        for row in self._rows.values():
            row["label"].deleteLater()
            row["progress"].deleteLater()
        self._rows = {}
        self._downloaded_files = 0
        self._update_overall_progress()
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.scroll_layout.addStretch(1)

    def set_total_files(self, total_files: int):
        self._total_files = max(int(total_files), 0)
        self._downloaded_files = 0
        self._update_overall_progress()

    def _update_overall_progress(self):
        total = self._total_files
        completed = min(self._downloaded_files, total) if total else 0
        if total <= 0:
            self.overall_progress.setRange(0, 1)
            self.overall_progress.setValue(0)
            self.overall_progress.setFormat("0/0 PCAPs downloaded")
            return
        self.overall_progress.setRange(0, total)
        self.overall_progress.setValue(completed)
        self.overall_progress.setFormat(f"{completed}/{total} PCAPs downloaded")

    def handle_transfer_started(self, filename: str, size_bytes: int | None):
        if filename in self._rows:
            return

        label = QLabel(filename, self.scroll_contents)
        label.setAlignment(Qt.AlignLeft)
        progress = QProgressBar(self.scroll_contents)
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setFormat("0%")
        progress.setTextVisible(True)

        self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, label)
        self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, progress)

        self._rows[filename] = {
            "label": label,
            "progress": progress,
            "size": size_bytes or 0,
            "completed": False,
        }
        self._update_label_text(filename, self._rows[filename])

    def handle_transfer_progress(self, filename: str, sent: int, total: int):
        row = self._rows.get(filename)
        if row is None:
            self.handle_transfer_started(filename, total)
            row = self._rows.get(filename)
        if row is None:
            return

        total = total or row.get("size", 0)
        row["size"] = total
        if total <= 0:
            row["progress"].setRange(0, 0)
            row["progress"].setFormat("Calculating size…")
            return

        percent = max(0, min(int((sent / total) * 100), 100))
        row["progress"].setRange(0, 100)
        row["progress"].setValue(percent)
        row["progress"].setFormat(
            f"{percent}% ({self._format_size(sent)} / {self._format_size(total)})"
        )
        self._update_label_text(filename, row)

    def handle_transfer_finished(self, filename: str, success: bool):
        row = self._rows.get(filename)
        if row is None:
            return
        progress = row["progress"]
        progress.setRange(0, 100)
        progress.setValue(100 if success else progress.value())
        progress.setFormat("100%" if success else f"{progress.value()}%")
        if success and not row.get("completed"):
            row["completed"] = True
            self._downloaded_files += 1
            self._update_overall_progress()

    def finish_all(self):
        for filename, row in list(self._rows.items()):
            self.handle_transfer_finished(filename, bool(row.get("completed")))
