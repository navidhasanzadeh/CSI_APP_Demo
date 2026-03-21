"""Utilities for playing spoken prompts using gTTS."""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from PyQt5.QtCore import QObject, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer

try:  # pragma: no cover - optional dependency
    from gtts import gTTS
except ImportError:  # pragma: no cover - optional dependency
    gTTS = None


class GTTSVoiceAssistant(QObject):
    """Play short synthesized speech snippets using gTTS."""

    def __init__(self, language: str = "en", parent=None):
        if gTTS is None:
            raise RuntimeError("gTTS is not available")
        super().__init__(parent)
        self.language = language or "en"
        self.player = QMediaPlayer(parent)
        self.player.mediaStatusChanged.connect(self._handle_media_status)
        self._temp_file = None

    # ------------------------------------------------------------------
    def speak(self, message: str):
        if not message:
            return
        self.stop()
        try:
            tts = gTTS(text=message, lang=self.language)
            temp_path = Path(tempfile.gettempdir()) / f"csi_voice_{uuid.uuid4().hex}.mp3"
            tts.save(str(temp_path))
            self._temp_file = temp_path
            url = QUrl.fromLocalFile(str(temp_path))
            self.player.setMedia(QMediaContent(url))
            self.player.play()
        except Exception:
            self._cleanup_temp_file()
            raise

    # ------------------------------------------------------------------
    def stop(self):
        if self.player.state() != QMediaPlayer.StoppedState:
            self.player.stop()
        self._cleanup_temp_file()

    # ------------------------------------------------------------------
    def _handle_media_status(self, status):
        if status in (QMediaPlayer.EndOfMedia, QMediaPlayer.InvalidMedia):
            self._cleanup_temp_file()

    # ------------------------------------------------------------------
    def _cleanup_temp_file(self):
        if self._temp_file and self._temp_file.exists():
            try:
                os.remove(str(self._temp_file))
            except OSError:
                pass
        self._temp_file = None

    # ------------------------------------------------------------------
    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.stop()
        except Exception:
            pass
