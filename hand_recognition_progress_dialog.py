"""Dialog showing progress for post-experiment hand recognition."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout


class HandRecognitionProgressDialog(QDialog):
    """Display progress updates while processing recorded webcam footage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WIRLab - Processing hand recognition")
        self.setModal(False)
        self.resize(480, 180)

        layout = QVBoxLayout(self)
        self._intro = QLabel(
            "Processing recorded webcam video to extract hand landmarks…", self
        )
        self._intro.setWordWrap(True)
        layout.addWidget(self._intro)

        self._progress = QProgressBar(self)
        self._progress.setAlignment(Qt.AlignCenter)
        self._progress.setRange(0, 0)
        layout.addWidget(self._progress)

        self._details = QLabel("Preparing…", self)
        self._details.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._details)

        self._total_frames = 0

    def reset(self):
        self._total_frames = 0
        self._progress.setRange(0, 0)
        self._progress.setValue(0)
        self._progress.setFormat("")
        self._details.setText("Preparing…")

    def handle_progress(self, processed_frames: int, total_frames: int):
        self._total_frames = total_frames
        if total_frames <= 0:
            self._progress.setRange(0, 0)
            self._progress.setFormat("Processing frames…")
            self._details.setText(f"Processed {processed_frames} frames")
            return

        percent = max(0, min(int((processed_frames / total_frames) * 100), 100))
        self._progress.setRange(0, 100)
        self._progress.setValue(percent)
        self._progress.setFormat(f"{percent}%")
        self._details.setText(
            f"Processed {processed_frames} / {total_frames} frames"
        )

    def handle_finished(self, success: bool):
        if success:
            self._progress.setRange(0, 100)
            self._progress.setValue(100)
            self._progress.setFormat("100%")
            if self._total_frames > 0:
                self._details.setText(
                    f"Completed processing {self._total_frames} frames"
                )
            else:
                self._details.setText("Finished processing recorded video")
        else:
            self._progress.setRange(0, 100)
            self._progress.setValue(self._progress.value())
            self._progress.setFormat(f"{self._progress.value()}%")
            self._details.setText("Hand recognition failed; results may be missing")
