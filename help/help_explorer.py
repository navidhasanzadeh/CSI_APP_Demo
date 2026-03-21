from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QTextBrowser,
    QVBoxLayout,
)


@dataclass(frozen=True)
class HelpTopic:
    title: str
    filename: str
    keywords: tuple[str, ...] = ()


class HelpExplorerDialog(QDialog):
    def __init__(self, parent=None, *, initial_topic: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("WIRLab - Help Explorer")
        self.resize(980, 620)

        self._initial_topic = initial_topic
        self._topics = self._load_topics()
        self._topic_items: list[QListWidgetItem] = []

        main_layout = QVBoxLayout(self)
        header_layout = QHBoxLayout()
        header_label = QLabel("Search help topics:", self)
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText(
            "Type to filter topics (profiles, experiment, PCAP, export, ...)")
        self.search_input.textChanged.connect(self._filter_topics)
        header_layout.addWidget(header_label)
        header_layout.addWidget(self.search_input, stretch=1)
        main_layout.addLayout(header_layout)

        content_layout = QHBoxLayout()
        self.topic_list = QListWidget(self)
        self.topic_list.setMinimumWidth(260)
        self.topic_list.currentItemChanged.connect(self._on_topic_selected)
        content_layout.addWidget(self.topic_list)

        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(True)
        content_layout.addWidget(self.browser, stretch=1)
        main_layout.addLayout(content_layout, stretch=1)

        self._populate_topics()
        if self._initial_topic:
            self._select_topic_by_filename(self._initial_topic)
        if self.topic_list.currentItem() is None and self._topic_items:
            self.topic_list.setCurrentRow(0)

    def _help_root(self) -> Path:
        return Path(__file__).resolve().parent

    def _load_topics(self) -> list[HelpTopic]:
        return [
            HelpTopic(
                title="Configuration Window Overview",
                filename="config_overview.html",
                keywords=("config", "tabs", "workflow"),
            ),
            HelpTopic(
                title="Profiles & Parameters (General)",
                filename="profiles_parameters.html",
                keywords=("profile", "parameter", "saving", "duplicate"),
            ),
            HelpTopic(
                title="Participant Profile",
                filename="participant_profile.html",
                keywords=("participant", "subject", "demographics"),
            ),
            HelpTopic(
                title="Experiment Profile",
                filename="experiment_profile.html",
                keywords=("experiment", "timing", "baseline", "example"),
            ),
            HelpTopic(
                title="Actions, Scripts, and Repetitions",
                filename="actions_profile.html",
                keywords=("actions", "scripts", "gesture", "sequence"),
            ),
            HelpTopic(
                title="Wi-Fi Capture & PCAP Management",
                filename="wifi_pcap_capture.html",
                keywords=("wifi", "pcap", "capture", "router"),
            ),
            HelpTopic(
                title="PCAP Reader & Plotting",
                filename="pcap_reader_plot.html",
                keywords=("pcap", "plot", "reader", "csi"),
            ),
            HelpTopic(
                title="Exporting Datasets",
                filename="exporting_datasets.html",
                keywords=("export", "dataset", "csv", "archive"),
            ),
            HelpTopic(
                title="UI, Camera, Voice, and Time Settings",
                filename="ui_camera_voice_time.html",
                keywords=("ui", "camera", "voice", "time", "clock"),
            ),
            HelpTopic(
                title="Depth Camera (RealSense D455)",
                filename="depth_camera.html",
                keywords=("depth", "camera", "realsense", "d455", "api", "recording"),
            ),
            HelpTopic(
                title="Nexmon Framework & CSI Collection",
                filename="nexmon_csi.html",
                keywords=("nexmon", "csi", "framework", "pcap", "sniffer"),
            ),
            HelpTopic(
                title="FAQ",
                filename="faq.html",
                keywords=("faq", "issues", "troubleshooting"),
            ),
            HelpTopic(
                title="About Us",
                filename="about_us.html",
                keywords=("about", "contact", "credits"),
            ),
        ]

    def _populate_topics(self) -> None:
        self.topic_list.clear()
        self._topic_items = []
        for topic in self._topics:
            item = QListWidgetItem(topic.title, self.topic_list)
            item.setData(Qt.UserRole, topic)
            self._topic_items.append(item)

    def _filter_topics(self, text: str) -> None:
        query = text.strip().lower()
        for item in self._topic_items:
            topic: HelpTopic = item.data(Qt.UserRole)
            haystack = " ".join((topic.title, *topic.keywords)).lower()
            item.setHidden(bool(query) and query not in haystack)

        if self.topic_list.currentItem() is None or self.topic_list.currentItem().isHidden():
            for index in range(self.topic_list.count()):
                candidate = self.topic_list.item(index)
                if not candidate.isHidden():
                    self.topic_list.setCurrentItem(candidate)
                    break

    def _select_topic_by_filename(self, filename: str) -> None:
        for item in self._topic_items:
            topic: HelpTopic = item.data(Qt.UserRole)
            if topic.filename == filename:
                self.topic_list.setCurrentItem(item)
                break

    def show_topic(self, filename: str) -> None:
        self._select_topic_by_filename(filename)

    def _on_topic_selected(self, current: QListWidgetItem, _previous: QListWidgetItem) -> None:
        if not current:
            self.browser.clear()
            return
        topic: HelpTopic = current.data(Qt.UserRole)
        doc_path = self._help_root() / topic.filename
        if doc_path.exists():
            self.browser.setSource(QUrl.fromLocalFile(str(doc_path)))
        else:
            self.browser.setHtml(
                f"<h2>{topic.title}</h2><p>Help document not found: {doc_path}</p>")
