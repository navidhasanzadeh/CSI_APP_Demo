"""Action preview dialog used by the configuration window."""

import subprocess
import sys
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QStackedLayout,
    QWidget,
    QSizePolicy,
)
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QImage, QPixmap

from voice_assistant import GTTSVoiceAssistant
from hand_recognition import HandRecognitionEngine, HandRecognitionError

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


class ActionPreviewDialog(QDialog):
    """Simple dialog to preview each configured action before starting."""

    def __init__(
        self,
        actions: dict,
        action_time=None,
        stop_time=None,
        preview_stop_time=None,
        voice_profile=None,
        camera_profile=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("WIRLab - Preview Actions")
        self.resize(640, 480)
        self.setModal(True)

        self.actions = list(actions.items()) if actions else []
        self.current_index = 0
        self.webcam_capture = None
        self.webcam_timer = None
        self._webcam_preview_initialised = False
        self.camera_profile = camera_profile or {}
        self.use_webcam = bool(self.camera_profile.get("use_webcam", False))
        self.camera_device_source = self._camera_device_source(self.camera_profile)
        self.hand_recognition_enabled = self._should_use_hand_recognition(
            self.camera_profile
        )
        if not self.use_webcam:
            self.hand_recognition_enabled = False
        self.hand_recognition_engine = None
        self._last_beep_time = 0.0
        self._beep_cooldown = 0.5
        self.action_time_seconds = self._coerce_positive_float(action_time)
        if preview_stop_time is None:
            preview_stop_time = stop_time if stop_time is not None else 2.0
        self.preview_pause_seconds = self._coerce_positive_float(preview_stop_time)
        self.voice_profile = voice_profile if isinstance(voice_profile, dict) else {}
        self.voice_enabled = bool(
            self.voice_profile.get("use_voice_assistant")
        ) and bool(self.voice_profile.get("preview_actions_voice", True))
        self.voice_assistant = None
        self._intro_active = True
        self._progress_start_time = None
        self._progress_duration = None
        self._progress_mode = None
        self._default_instruction_html = (
            "<b>Perform the activity only after you hear the beep.</b>"
        )
        self._stop_instruction_html = (
            "<b>This is the stop-move time between actions. Please remain still and do not move.</b>"
        )
        self._intro_text = ""

        layout = QVBoxLayout(self)

        self.lbl_action_name = QLabel("", self)
        self.lbl_action_name.setAlignment(Qt.AlignCenter)
        self.lbl_action_name.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.lbl_action_name)

        self.lbl_position = QLabel("", self)
        self.lbl_position.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_position)

        display_row = QHBoxLayout()
        self.video_container = QWidget(self)
        self.video_stack = QStackedLayout(self.video_container)
        self.video_widget = QVideoWidget(self.video_container)
        self.blackout_label = QLabel("", self.video_container)
        self.blackout_label.setAlignment(Qt.AlignCenter)
        self.blackout_label.setStyleSheet("background-color: black;")
        self.video_stack.addWidget(self.video_widget)
        self.video_stack.addWidget(self.blackout_label)
        self.video_stack.setCurrentWidget(self.video_widget)
        display_row.addWidget(self.video_container, stretch=3)

        self.webcam_label = QLabel("Webcam preview disabled", self)
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setMinimumSize(240, 180)
        self.webcam_label.setStyleSheet(
            "background-color: black; color: white; border: 1px solid white;"
        )
        display_row.addWidget(self.webcam_label, stretch=2)
        layout.addLayout(display_row, stretch=1)

        self.intro_image_label = QLabel("", self)
        self.intro_image_label.setAlignment(Qt.AlignCenter)
        self.intro_image_label.setMinimumSize(420, 260)
        self.intro_image_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.intro_image_container = QWidget(self)
        image_layout = QHBoxLayout(self.intro_image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addStretch(1)
        image_layout.addWidget(self.intro_image_label, stretch=0, alignment=Qt.AlignCenter)
        image_layout.addStretch(1)
        layout.addWidget(self.intro_image_container)
        self.intro_image_container.hide()
        self.intro_image_label.hide()
        self._intro_pixmap = self._load_intro_pixmap()

        self.lbl_instruction = QLabel(self._default_instruction_html, self)
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_instruction.setWordWrap(True)
        self.lbl_instruction.setStyleSheet("font-size: 18px;")
        layout.addWidget(self.lbl_instruction)

        controls = QHBoxLayout()
        self.btn_start = QPushButton("Start Experiment", self)
        self.btn_start.clicked.connect(self._handle_start)
        controls.addWidget(self.btn_start)

        self.btn_return = QPushButton("Return to Config", self)
        self.btn_return.clicked.connect(self._handle_return_to_config)
        controls.addWidget(self.btn_return)

        controls.addStretch(1)

        self.btn_prev = QPushButton("Previous", self)
        self.btn_prev.clicked.connect(self.show_previous)
        controls.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next", self)
        self.btn_next.clicked.connect(self.show_next)
        controls.addWidget(self.btn_next)

        layout.addLayout(controls)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.media_player = QMediaPlayer(self)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.positionChanged.connect(self._handle_position_changed)
        self.media_player.mediaStatusChanged.connect(
            self._handle_media_status_changed
        )

        self.progress_timer = QTimer(self)
        self.progress_timer.setInterval(50)
        self.progress_timer.timeout.connect(self._update_progress_bar)

        self.blackout_timer = QTimer(self)
        self.blackout_timer.setSingleShot(True)
        self.blackout_timer.timeout.connect(self._restart_action_video)

        self._init_voice_assistant()
        self._init_hand_recognition()
        self._show_intro_message()

    # ------------------------------------------------------------------
    def _show_action(self, index: int):
        self._cancel_blackout_timer()
        self._hide_black_screen()
        self._reset_progress_bar()

        if not self.actions:
            self.lbl_action_name.setText("No actions configured")
            self.lbl_position.setText("")
            self.media_player.stop()
            self._update_navigation_buttons()
            return

        index = max(0, min(index, len(self.actions) - 1))
        self.current_index = index

        name, video = self.actions[index]
        display_name = name or f"Action {index + 1}"
        self.lbl_action_name.setText(display_name)
        self.lbl_position.setText(f"Action {index + 1} of {len(self.actions)}")

        self._set_default_instruction()
        self._announce_action(display_name)

        if video:
            try:
                abs_path = str(Path(video).expanduser().resolve())
                url = QUrl.fromLocalFile(abs_path)
                self.media_player.setMedia(QMediaContent(url))
                self.media_player.play()
            except Exception:
                self.media_player.stop()
        else:
            self.media_player.stop()

        self._update_navigation_buttons()

    def _show_intro_message(self):
        self._intro_active = True
        self._intro_text = self._build_intro_text()
        self.lbl_action_name.setText("Preview instructions")
        self.lbl_position.setText("")
        self.lbl_instruction.setText(self._intro_text)
        self.video_container.hide()
        self.webcam_label.hide()
        self.progress_bar.hide()
        self._set_intro_image_visible(True)
        self._speak_message(self._intro_text)
        self._update_navigation_buttons()

    def _exit_intro_mode(self):
        if not self._intro_active:
            return

        self._intro_active = False
        self.video_container.show()
        self.webcam_label.show()
        if self.action_time_seconds or self.preview_pause_seconds:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()

        self._set_intro_image_visible(False)
        self._set_default_instruction()
        self._init_webcam_preview()
        self._show_action(self.current_index)
        self._update_navigation_buttons()

    def _update_navigation_buttons(self):
        if self._intro_active:
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(True)
            return

        has_actions = bool(self.actions)
        self.btn_prev.setEnabled(has_actions and self.current_index > 0)
        self.btn_next.setEnabled(
            has_actions and self.current_index < len(self.actions) - 1
        )

    def show_previous(self):
        if self._intro_active:
            return
        if self.actions and self.current_index > 0:
            self._show_action(self.current_index - 1)
            self._update_navigation_buttons()

    def show_next(self):
        if self._intro_active:
            self._exit_intro_mode()
            return
        if self.actions and self.current_index < len(self.actions) - 1:
            self._show_action(self.current_index + 1)
            self._update_navigation_buttons()

    def _handle_start(self):
        self.accept()

    def _handle_return_to_config(self):
        self.reject()

    def _stop_video(self):
        try:
            self.media_player.stop()
        except Exception:
            pass

    def _handle_position_changed(self, position_ms: int):
        if position_ms is None or position_ms > 250:
            return
        if self._intro_active:
            return
        if self.media_player.state() != QMediaPlayer.PlayingState:
            return
        now = time.monotonic()
        if now - self._last_beep_time >= self._beep_cooldown:
            self._trigger_beep()

    def _trigger_beep(self):
        self._play_beep()
        self._last_beep_time = time.monotonic()
        self._start_action_progress()

    def _play_beep(self):
        duration = 0.2
        frequency = 880
        cmd = None
        if sys.platform.startswith("darwin"):
            cmd = [
                "play",
                "-nq",
                "-t",
                "coreaudio",
                "synth",
                str(duration),
                "sine",
                str(frequency),
            ]
        elif sys.platform.startswith("linux"):
            cmd = [
                "play",
                "-nq",
                "-t",
                "alsa",
                "synth",
                str(duration),
                "sine",
                str(frequency),
            ]

        if not cmd:
            return

        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def _handle_media_status_changed(self, status):
        if status != QMediaPlayer.EndOfMedia or self._intro_active:
            return
        self._on_action_video_finished()

    def _on_action_video_finished(self):
        if self.preview_pause_seconds:
            self._reset_progress_bar()
            self._show_black_screen()
            self._set_stop_time_instruction()
            self._start_stop_progress()
            self.blackout_timer.start(int(self.preview_pause_seconds * 1000))
        else:
            self._restart_action_video()

    def _restart_action_video(self):
        if self._intro_active:
            return
        self._hide_black_screen()
        try:
            self.media_player.setPosition(0)
            self.media_player.play()
        except Exception:
            pass

    def _show_black_screen(self):
        if self.video_stack is not None:
            self.video_stack.setCurrentWidget(self.blackout_label)

    def _hide_black_screen(self):
        if self.video_stack is not None:
            self.video_stack.setCurrentWidget(self.video_widget)
        self._set_default_instruction()

    def _cancel_blackout_timer(self):
        if self.blackout_timer.isActive():
            self.blackout_timer.stop()

    def _start_action_progress(self):
        if self.action_time_seconds is None or self._intro_active:
            return
        self._start_progress(self.action_time_seconds, "action")

    def _start_stop_progress(self):
        if self.preview_pause_seconds is None or self._intro_active:
            return
        self._start_progress(self.preview_pause_seconds, "stop")

    def _start_progress(self, duration: float, mode: str):
        if duration <= 0:
            return
        self._progress_duration = duration
        self._progress_mode = mode
        self._progress_start_time = time.monotonic()
        self.progress_bar.setValue(0)
        self.progress_timer.start()
        self._update_progress_bar()

    def _reset_progress_bar(self):
        self._progress_start_time = None
        self._progress_duration = None
        self._progress_mode = None
        if self.progress_timer.isActive():
            self.progress_timer.stop()
        self.progress_bar.setValue(0)

    def _update_progress_bar(self):
        if self._progress_start_time is None or not self._progress_duration:
            return
        elapsed = max(0.0, time.monotonic() - self._progress_start_time)
        ratio = min(1.0, elapsed / self._progress_duration)
        self.progress_bar.setValue(int(ratio * 100))
        if ratio >= 1.0 and self.progress_timer.isActive():
            self.progress_timer.stop()
            if self._progress_mode == "stop":
                self._set_default_instruction()

    def _set_default_instruction(self):
        self.lbl_instruction.setText(self._default_instruction_html)

    def _set_stop_time_instruction(self):
        self.lbl_instruction.setText(self._stop_instruction_html)

    def _build_intro_text(self):
        segments = [
            "In this experiment the participant should perform several gestures or activities.",
            "They should perform the activity only once after the beep.",
        ]
        if self.action_time_seconds:
            segments.append(
                f"For each activity they have {self.action_time_seconds:g} seconds of action time."
            )
        else:
            segments.append("For each activity they have a limited amount of time.")
        segments.append(
            "Between the actions or baselines they should remain seated and try not to move."
        )
        segments.append("Please click Next to preview the activities.")
        return " ".join(segments)

    def _load_intro_pixmap(self):
        image_path = Path(__file__).resolve().parent / "images" / "wifisensing.png"
        if not image_path.exists():
            return None
        try:
            return QPixmap(str(image_path))
        except Exception:
            return None

    def _set_intro_image_visible(self, visible: bool):
        label = getattr(self, "intro_image_label", None)
        container = getattr(self, "intro_image_container", None)
        if label is None or container is None:
            return
        if visible:
            if self._intro_pixmap:
                self._update_intro_image()
                label.setText("")
            else:
                label.setText("WiFi sensing preview")
            label.show()
            container.show()
        else:
            label.hide()
            container.hide()

    def _update_intro_image(self):
        label = getattr(self, "intro_image_label", None)
        if label is None or not self._intro_pixmap:
            return
        rect = label.contentsRect()
        target_width = max(1, rect.width())
        target_height = max(1, rect.height())
        scaled = self._intro_pixmap.scaled(
            target_width,
            target_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setPixmap(scaled)

    def _init_voice_assistant(self):
        if not self.voice_enabled:
            return
        try:
            language = (self.voice_profile.get("language") or "en").strip() or "en"
            self.voice_assistant = GTTSVoiceAssistant(language=language, parent=self)
        except Exception:
            self.voice_assistant = None
            self.voice_enabled = False

    def _speak_message(self, message: str):
        if (
            not message
            or self.voice_assistant is None
            or not self.voice_enabled
        ):
            return
        try:
            self.voice_assistant.stop()
            self.voice_assistant.speak(message)
        except Exception:
            self.voice_assistant = None
            self.voice_enabled = False

    def _announce_action(self, display_name: str):
        if not display_name:
            return
        announcement = (
            display_name
            if display_name.endswith((".", "!", "?"))
            else f"{display_name}."
        )
        self._speak_message(announcement)

    @staticmethod
    def _coerce_positive_float(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @staticmethod
    def _camera_device_source(profile: dict):
        device = profile.get("camera_device", 0)
        if isinstance(device, str):
            device = device.strip()
            if not device:
                return 0
            try:
                return int(device)
            except ValueError:
                return device
        try:
            return int(device)
        except (TypeError, ValueError):
            return 0

    def _init_webcam_preview(self):
        if self._webcam_preview_initialised:
            return
        self._webcam_preview_initialised = True

        if not hasattr(self, "webcam_label") or self.webcam_label is None:
            return

        if not self.use_webcam:
            self.webcam_label.setText("Webcam disabled in camera settings")
            return

        if cv2 is None:
            self.webcam_label.setText("OpenCV is required for the webcam preview")
            return

        self.webcam_capture = cv2.VideoCapture(self.camera_device_source)
        if not self.webcam_capture or not self.webcam_capture.isOpened():
            self.webcam_label.setText("Webcam unavailable")
            self.webcam_capture = None
            return

        fps = self.webcam_capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        interval_ms = max(15, int(1000 / min(max(fps, 1.0), 60.0)))
        self.webcam_timer = QTimer(self)
        self.webcam_timer.timeout.connect(self._capture_webcam_frame)
        self.webcam_timer.start(interval_ms)
        self._capture_webcam_frame()

    def _capture_webcam_frame(self):
        if self.webcam_capture is None:
            return
        ret, frame = self.webcam_capture.read()
        if not ret or frame is None:
            return
        display_frame = self._process_hand_recognition(frame.copy())
        self._update_webcam_label(display_frame)

    def _update_webcam_label(self, frame):
        if self.webcam_label is None or cv2 is None:
            return
        try:
            mirrored = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
        except Exception:
            return

        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        scaled = image.scaled(
            self.webcam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.webcam_label.setPixmap(QPixmap.fromImage(scaled))
        self.webcam_label.setText("")

    def _should_use_hand_recognition(self, profile):
        return bool(profile.get("use_hand_recognition", False))

    def _init_hand_recognition(self):
        if not self.hand_recognition_enabled:
            return
        try:
            model_complexity = (
                self.camera_profile.get("hand_model_complexity") or "light"
            ).strip()
            left_color = self.camera_profile.get("hand_left_wrist_color")
            right_color = self.camera_profile.get("hand_right_wrist_color")
            wrist_radius = self.camera_profile.get("hand_wrist_circle_radius")
            self.hand_recognition_engine = HandRecognitionEngine(
                max_num_hands=2,
                model_complexity=model_complexity,
                left_wrist_color=left_color,
                right_wrist_color=right_color,
                wrist_circle_radius=wrist_radius,
            )
            self.hand_recognition_engine.start()
        except HandRecognitionError:
            self.hand_recognition_enabled = False
            self.hand_recognition_engine = None
            if self.webcam_label is not None:
                self.webcam_label.setText("Hand recognition unavailable")
        except Exception:
            self.hand_recognition_enabled = False
            self.hand_recognition_engine = None

    def _process_hand_recognition(self, frame):
        if not self.hand_recognition_enabled:
            return frame
        if self.hand_recognition_engine is None:
            return frame
        return self.hand_recognition_engine.process_frame(frame)

    def _shutdown_webcam(self):
        if self.webcam_timer is not None:
            try:
                self.webcam_timer.stop()
            except Exception:
                pass
            self.webcam_timer.deleteLater()
            self.webcam_timer = None
        if self.webcam_capture is not None:
            try:
                self.webcam_capture.release()
            except Exception:
                pass
            self.webcam_capture = None

    def _shutdown_hand_recognition(self):
        if self.hand_recognition_engine is None:
            return
        try:
            self.hand_recognition_engine.shutdown()
        except Exception:
            pass
        self.hand_recognition_engine = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_intro_image()

    def _cleanup(self):
        self._stop_video()
        self._cancel_blackout_timer()
        if self.progress_timer.isActive():
            self.progress_timer.stop()
        self._shutdown_webcam()
        self._shutdown_hand_recognition()
        if self.voice_assistant is not None:
            try:
                self.voice_assistant.stop()
            except Exception:
                pass

    def accept(self):
        self._cleanup()
        super().accept()

    def reject(self):
        self._cleanup()
        super().reject()

    def closeEvent(self, event):
        self._cleanup()
        super().closeEvent(event)
