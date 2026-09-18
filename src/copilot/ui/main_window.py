"""The main desktop window.

Layout, top to bottom: provider status, listening controls, live transcript,
the detected question, and the streaming answer with its actions.  Compact mode
hides everything except the question and the answer and pins the window on top,
which is what you actually want during a call.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QFont, QGuiApplication, QKeySequence, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import messages as msg
from ..config import Settings
from ..core.pipeline import IDLE, LISTENING, PAUSED, CopilotPipeline
from ..credentials import CredentialStore
from ..documents.extract import DocumentError, ExtractedDocument, extract_text
from ..documents.profile import CandidateProfile
from ..llm.models import get_spec
from ..llm.openai_client import AnswerResult
from .bridge import PipelineBridge
from .dev_panel import DevPanel
from .settings_dialog import SettingsDialog
from .widgets import StatusIndicator

log = logging.getLogger(__name__)

STYLESHEET = """
QMainWindow, QWidget#Root { background-color: #14161b; }
QLabel { color: #e6e8ee; }
QLabel#SectionTitle { color: #8d94a5; font-size: 11px; letter-spacing: 1px; }
QLabel#Question {
    color: #ffffff; font-size: 17px; font-weight: 600;
    padding: 10px 12px; background-color: #1d222c; border-radius: 8px;
}
QTextEdit {
    background-color: #1a1d24; color: #e6e8ee; border: 1px solid #262b35;
    border-radius: 8px; padding: 10px; selection-background-color: #2f6feb;
}
QPushButton {
    background-color: #242935; color: #e6e8ee; border: 1px solid #333a49;
    border-radius: 6px; padding: 7px 14px; font-weight: 500;
}
QPushButton:hover:enabled { background-color: #2e3542; }
QPushButton:disabled { color: #5a6172; background-color: #1c2029; }
QPushButton#Primary { background-color: #2f6feb; border-color: #2f6feb; color: white; }
QPushButton#Primary:hover:enabled { background-color: #3b7bf5; }
QPushButton#Danger { background-color: #3a2226; border-color: #5d2b33; }
QFrame#Card { background-color: #181b22; border: 1px solid #242a35; border-radius: 10px; }
QProgressBar { background-color: #1a1d24; border: 1px solid #262b35;
    border-radius: 4px; height: 6px; text-align: center; }
QProgressBar::chunk { background-color: #3fb950; border-radius: 3px; }
"""


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, credentials: CredentialStore | None = None) -> None:
        super().__init__()
        self.settings = settings
        self.credentials = credentials or CredentialStore()

        self.pipeline = CopilotPipeline(settings, credentials=self.credentials)
        self.pipeline.start()
        self.bridge = PipelineBridge(self.pipeline, self)

        self._answer_buffer: list[str] = []
        self._answer_text = ""
        self._partial_text = ""
        self._cv: ExtractedDocument | None = None
        self._jd: ExtractedDocument | None = None
        self._dev_panel: DevPanel | None = None
        self._normal_geometry = None

        self.setWindowTitle("AZ Interview Copilot")
        self.setStyleSheet(STYLESHEET)
        self.resize(1040, 760)
        self._build_ui()
        self._connect_signals()
        self._restore_documents()
        self._refresh_provider_labels()
        self._apply_always_on_top(self.settings.ui.always_on_top)
        if self.settings.ui.compact_mode:
            self.set_compact(True)

        # Coalesce streaming deltas so a fast stream does not repaint per token.
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(60)
        self._flush_timer.timeout.connect(self._flush_answer_buffer)
        self._flush_timer.start()

        self._level_decay = QTimer(self)
        self._level_decay.setInterval(120)
        self._level_decay.timeout.connect(self._decay_level)
        self._level_decay.start()

    # =====================================================================
    # Construction
    # =====================================================================
    def _build_ui(self) -> None:
        root = QWidget(objectName="Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        outer.addWidget(self._build_header())
        outer.addWidget(self._build_controls())

        self.body_splitter = QSplitter(Qt.Vertical)
        self.body_splitter.setChildrenCollapsible(False)
        self.transcript_card = self._build_transcript()
        self.body_splitter.addWidget(self.transcript_card)
        self.body_splitter.addWidget(self._build_answer())
        self.body_splitter.setSizes([260, 420])
        outer.addWidget(self.body_splitter, 1)

        self.status_label = QLabel(msg.STATUS_IDLE)
        self.status_label.setObjectName("SectionTitle")
        outer.addWidget(self.status_label)

        self._build_menu()

    def _build_header(self) -> QWidget:
        card = QFrame(objectName="Card")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(24)

        stt_box = QVBoxLayout()
        stt_box.setSpacing(2)
        title = QLabel("Speech Recognition")
        title.setObjectName("SectionTitle")
        self.stt_indicator = StatusIndicator("ElevenLabs Scribe v2 Realtime")
        stt_box.addWidget(title)
        stt_box.addWidget(self.stt_indicator)
        layout.addLayout(stt_box)

        llm_box = QVBoxLayout()
        llm_box.setSpacing(2)
        title2 = QLabel("Answer Generation")
        title2.setObjectName("SectionTitle")
        self.llm_indicator = StatusIndicator("GPT-4.1 Nano")
        llm_box.addWidget(title2)
        llm_box.addWidget(self.llm_indicator)
        layout.addLayout(llm_box)

        layout.addStretch(1)

        doc_box = QVBoxLayout()
        doc_box.setSpacing(2)
        doc_title = QLabel("Sənədlər")
        doc_title.setObjectName("SectionTitle")
        self.documents_label = QLabel("Sənəd yüklənməyib")
        self.documents_label.setWordWrap(True)
        self.documents_label.setMaximumWidth(340)
        doc_box.addWidget(doc_title)
        doc_box.addWidget(self.documents_label)
        layout.addLayout(doc_box)
        self.header_card = card
        return card

    def _build_controls(self) -> QWidget:
        card = QFrame(objectName="Card")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(8)

        self.start_button = QPushButton("Start Listening", objectName="Primary")
        self.pause_button = QPushButton("Pause Listening")
        self.stop_button = QPushButton("Stop Listening", objectName="Danger")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        for button in (self.start_button, self.pause_button, self.stop_button):
            layout.addWidget(button)

        layout.addSpacing(12)
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedWidth(120)
        self.level_bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(QLabel("Səs"))
        layout.addWidget(self.level_bar)
        layout.addStretch(1)

        self.settings_button = QPushButton("Settings")
        self.compact_button = QPushButton("Compact")
        self.compact_button.setCheckable(True)
        layout.addWidget(self.settings_button)
        layout.addWidget(self.compact_button)
        self.controls_card = card
        return card

    def _build_transcript(self) -> QWidget:
        card = QFrame(objectName="Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(6)
        title = QLabel("CANLI TRANSKRİPT (AZƏRBAYCAN DİLİ)")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setFont(QFont("Segoe UI", 11))
        layout.addWidget(self.transcript_view, 1)
        return card

    def _build_answer(self) -> QWidget:
        card = QFrame(objectName="Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(8)

        title = QLabel("AŞKARLANAN SUAL")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        self.question_label = QLabel("—")
        self.question_label.setObjectName("Question")
        self.question_label.setWordWrap(True)
        layout.addWidget(self.question_label)

        answer_title = QLabel("CAVAB")
        answer_title.setObjectName("SectionTitle")
        layout.addWidget(answer_title)
        self.answer_view = QTextEdit()
        self.answer_view.setReadOnly(True)
        self.answer_view.setFont(QFont("Segoe UI", 13))
        layout.addWidget(self.answer_view, 1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.generate_button = QPushButton("Generate Answer", objectName="Primary")
        self.regenerate_button = QPushButton("Regenerate")
        self.expand_button = QPushButton("Expand Answer")
        self.copy_button = QPushButton("Copy Answer")
        for button in (
            self.generate_button, self.regenerate_button,
            self.expand_button, self.copy_button,
        ):
            row.addWidget(button)
        row.addStretch(1)
        self.answer_meta = QLabel("")
        self.answer_meta.setObjectName("SectionTitle")
        row.addWidget(self.answer_meta)
        layout.addLayout(row)
        self._set_answer_actions_enabled(False)
        return card

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&Fayl")
        cv_action = QAction("CV yüklə…", self)
        cv_action.triggered.connect(lambda: self._upload_document("cv"))
        jd_action = QAction("Vakansiya təsviri yüklə…", self)
        jd_action.triggered.connect(lambda: self._upload_document("job_description"))
        clear_docs = QAction("Sənədləri təmizlə", self)
        clear_docs.triggered.connect(self._clear_documents)
        quit_action = QAction("Çıxış", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        for action in (cv_action, jd_action, clear_docs):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&Görünüş")
        self.compact_action = QAction("Compact rejim", self, checkable=True)
        self.compact_action.setShortcut("Ctrl+Shift+C")
        self.compact_action.triggered.connect(self.set_compact)
        self.on_top_action = QAction("Həmişə üstdə", self, checkable=True)
        self.on_top_action.setChecked(self.settings.ui.always_on_top)
        self.on_top_action.triggered.connect(self._toggle_always_on_top)
        dev_action = QAction("Developer paneli", self)
        dev_action.setShortcut("Ctrl+Shift+D")
        dev_action.triggered.connect(self.show_dev_panel)
        settings_action = QAction("Settings…", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.open_settings)
        for action in (self.compact_action, self.on_top_action):
            view_menu.addAction(action)
        view_menu.addSeparator()
        view_menu.addAction(dev_action)
        view_menu.addAction(settings_action)

    # =====================================================================
    # Wiring
    # =====================================================================
    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self.start_listening)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_listening)
        self.settings_button.clicked.connect(self.open_settings)
        self.compact_button.toggled.connect(self.set_compact)

        self.generate_button.clicked.connect(lambda: self.pipeline.generate_answer())
        self.regenerate_button.clicked.connect(self.pipeline.regenerate)
        self.expand_button.clicked.connect(self.pipeline.expand)
        self.copy_button.clicked.connect(self.copy_answer)

        self.bridge.state_changed.connect(self.on_state_changed)
        self.bridge.provider_state_changed.connect(self.on_provider_state)
        self.bridge.partial_transcript.connect(self.on_partial)
        self.bridge.committed_transcript.connect(self.on_committed)
        self.bridge.question_detected.connect(self.on_question)
        self.bridge.answer_started.connect(self.on_answer_started)
        self.bridge.answer_delta.connect(self.on_answer_delta)
        self.bridge.answer_finished.connect(self.on_answer_finished)
        self.bridge.error_raised.connect(self.on_error)
        self.bridge.audio_level.connect(self.on_audio_level)

    # =====================================================================
    # Listening
    # =====================================================================
    @Slot()
    def start_listening(self) -> None:
        missing = [
            label
            for provider, label in (("elevenlabs", msg.KEY_MISSING_ELEVENLABS),
                                    ("openai", msg.KEY_MISSING_OPENAI))
            if not self.credentials.get(provider)  # type: ignore[arg-type]
        ]
        if missing:
            QMessageBox.warning(
                self, "API açarı yoxdur",
                "\n".join(missing) + "\n\nSettings bölməsindən açarları əlavə edin.",
            )
            self.open_settings()
            return
        self.transcript_view.clear()
        self._partial_text = ""
        self.pipeline.start_listening()

    @Slot()
    def toggle_pause(self) -> None:
        if self.pipeline.session.state == PAUSED:
            self.pipeline.resume_listening()
        else:
            self.pipeline.pause_listening()

    @Slot()
    def stop_listening(self) -> None:
        self.pipeline.stop_listening()

    @Slot(str)
    def on_state_changed(self, state: str) -> None:
        listening = state == LISTENING
        paused = state == PAUSED
        self.start_button.setEnabled(not listening and not paused)
        self.pause_button.setEnabled(listening or paused)
        self.stop_button.setEnabled(listening or paused)
        self.pause_button.setText("Resume Listening" if paused else "Pause Listening")
        self.status_label.setText(
            {
                LISTENING: msg.STATUS_LISTENING,
                PAUSED: msg.STATUS_PAUSED,
                IDLE: msg.STATUS_IDLE,
            }.get(state, state)
        )
        if not listening:
            self.level_bar.setValue(0)

    @Slot(str, str)
    def on_provider_state(self, provider: str, state: str) -> None:
        indicator = self.stt_indicator if provider == "elevenlabs" else self.llm_indicator
        indicator.set_state(state)

    # =====================================================================
    # Transcript
    # =====================================================================
    @Slot(float)
    def on_audio_level(self, level: float) -> None:
        value = min(100, int(level * 400))
        if value > self.level_bar.value():
            self.level_bar.setValue(value)

    def _decay_level(self) -> None:
        current = self.level_bar.value()
        if current:
            self.level_bar.setValue(max(0, current - 12))

    @Slot(str)
    def on_partial(self, text: str) -> None:
        self._partial_text = text
        self._render_transcript()

    @Slot(str)
    def on_committed(self, text: str) -> None:
        self._partial_text = ""
        self._render_transcript()

    def _render_transcript(self) -> None:
        """Committed text in full colour, the in-flight partial greyed out."""
        lines = self.pipeline.session.committed_lines[-60:]
        body = "<br>".join(_escape(line) for line in lines)
        if self._partial_text:
            partial = _escape(self._partial_text)
            body += (
                f"<br><span style='color:#8d94a5;font-style:italic'>{partial}</span>"
                if body else
                f"<span style='color:#8d94a5;font-style:italic'>{partial}</span>"
            )
        self.transcript_view.setHtml(
            f"<div style='color:#e6e8ee;line-height:1.5'>{body}</div>"
        )
        scrollbar = self.transcript_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # =====================================================================
    # Question and answer
    # =====================================================================
    @Slot(str)
    def on_question(self, question: str) -> None:
        self.question_label.setText(question)
        self._set_answer_actions_enabled(True)

    @Slot(str)
    def on_answer_started(self, question: str) -> None:
        self.question_label.setText(question)
        self._answer_buffer.clear()
        self._answer_text = ""
        self.answer_view.clear()
        self.answer_meta.setText("")
        self.status_label.setText(msg.STATUS_GENERATING)
        self._set_answer_actions_enabled(True)

    @Slot(str)
    def on_answer_delta(self, delta: str) -> None:
        self._answer_buffer.append(delta)

    def _flush_answer_buffer(self) -> None:
        if not self._answer_buffer:
            return
        chunk = "".join(self._answer_buffer)
        self._answer_buffer.clear()
        self._answer_text += chunk
        cursor = self.answer_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(chunk)
        self.answer_view.setTextCursor(cursor)
        self.answer_view.ensureCursorVisible()

    @Slot(object)
    def on_answer_finished(self, result: AnswerResult) -> None:
        self._flush_answer_buffer()
        # The stream is authoritative; replace to guard against a dropped delta.
        if result.text != self._answer_text:
            self.answer_view.setPlainText(result.text)
            self._answer_text = result.text

        parts = []
        if result.first_token_ms is not None:
            parts.append(f"ilk token {result.first_token_ms:.0f} ms")
        parts.append(f"tam {result.total_ms:.0f} ms")
        parts.append(f"{result.char_count} simvol")
        if result.context_labels:
            parts.append("kontekst: " + ", ".join(result.context_labels[:3]))
        if result.truncated:
            parts.append("⚠ kəsildi")
        self.answer_meta.setText(" · ".join(parts))
        self.status_label.setText(
            msg.STATUS_LISTENING
            if self.pipeline.session.state == LISTENING
            else msg.STATUS_IDLE
        )

    @Slot()
    def copy_answer(self) -> None:
        text = self._answer_text or self.answer_view.toPlainText()
        if text:
            QGuiApplication.clipboard().setText(text)
            self.status_label.setText("Cavab kopyalandı")

    def _set_answer_actions_enabled(self, enabled: bool) -> None:
        for button in (
            self.generate_button, self.regenerate_button,
            self.expand_button, self.copy_button,
        ):
            button.setEnabled(enabled)

    # =====================================================================
    # Errors
    # =====================================================================
    @Slot(str, str)
    def on_error(self, provider: str, message: str) -> None:
        self.status_label.setText(f"⚠ {message}")
        if provider == "elevenlabs":
            self.stt_indicator.set_state("error", message)
        elif provider == "openai":
            self.llm_indicator.set_state("error", message)
        if self._dev_panel is not None:
            self._dev_panel.append_log(f"[{provider}] {message}")

    # =====================================================================
    # Documents
    # =====================================================================
    def _upload_document(self, kind: str) -> None:
        title = "CV seçin" if kind == "cv" else "Vakansiya təsvirini seçin"
        path, _ = QFileDialog.getOpenFileName(
            self, title, "", "Sənədlər (*.pdf *.docx *.txt *.md);;Bütün fayllar (*)"
        )
        if not path:
            return
        self._load_document(path, kind, interactive=True)

    def _load_document(self, path: str, kind: str, interactive: bool = False) -> None:
        try:
            document = extract_text(path, kind)
        except DocumentError as exc:
            if interactive:
                QMessageBox.warning(self, "Sənəd oxunmadı", str(exc))
            else:
                log.warning("Could not reload %s: %s", path, exc)
            return
        if kind == "cv":
            self._cv = document
            self.settings.documents.cv_path = str(path)
        else:
            self._jd = document
            self.settings.documents.job_description_path = str(path)
        self.settings.save()
        self._rebuild_profile()

    def _clear_documents(self) -> None:
        self._cv = self._jd = None
        self.settings.documents.cv_path = ""
        self.settings.documents.job_description_path = ""
        self.settings.save()
        self._rebuild_profile()

    def _rebuild_profile(self) -> None:
        profile = CandidateProfile.build(self._cv, self._jd)
        self.pipeline.set_profile(profile)
        self.documents_label.setText(profile.describe())

    def _restore_documents(self) -> None:
        for path, kind in (
            (self.settings.documents.cv_path, "cv"),
            (self.settings.documents.job_description_path, "job_description"),
        ):
            if path and Path(path).exists():
                self._load_document(path, kind)
        self._rebuild_profile()

    # =====================================================================
    # Settings, dev panel, window modes
    # =====================================================================
    @Slot()
    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self.credentials, self.bridge, self)
        if dialog.exec():
            self.settings.save()
            self.pipeline.settings = self.settings
            self._refresh_provider_labels()

    @Slot()
    def show_dev_panel(self) -> None:
        if self._dev_panel is None:
            self._dev_panel = DevPanel(self.pipeline, self.bridge, self.credentials, self)
            self.bridge.log_line.connect(self._dev_panel.append_log)
        self._dev_panel.show()
        self._dev_panel.raise_()
        self._dev_panel.activateWindow()

    def _refresh_provider_labels(self) -> None:
        self.stt_indicator.set_title(f"ElevenLabs {_pretty_model(self.settings.stt.model_id)}")
        self.llm_indicator.set_title(get_spec(self.settings.llm.model).label)

    @Slot(bool)
    def set_compact(self, compact: bool) -> None:
        """Compact mode: question + answer only, pinned above other windows."""
        if compact and self._normal_geometry is None:
            self._normal_geometry = self.saveGeometry()

        self.transcript_card.setVisible(not compact)
        self.header_card.setVisible(not compact)
        self.menuBar().setVisible(not compact)
        self.settings_button.setVisible(not compact)
        self.level_bar.setVisible(not compact)

        for widget, action in ((self.compact_button, None), (None, self.compact_action)):
            if widget is not None and widget.isChecked() != compact:
                widget.blockSignals(True)
                widget.setChecked(compact)
                widget.blockSignals(False)
            if action is not None and action.isChecked() != compact:
                action.setChecked(compact)

        self.settings.ui.compact_mode = compact
        if compact:
            self._apply_always_on_top(True)
            self.resize(520, 420)
        else:
            self._apply_always_on_top(self.settings.ui.always_on_top)
            if self._normal_geometry is not None:
                self.restoreGeometry(self._normal_geometry)
                self._normal_geometry = None
        self.settings.save()

    @Slot(bool)
    def _toggle_always_on_top(self, enabled: bool) -> None:
        self.settings.ui.always_on_top = enabled
        self.settings.save()
        self._apply_always_on_top(enabled)

    def _apply_always_on_top(self, enabled: bool) -> None:
        if bool(self.windowFlags() & Qt.WindowStaysOnTopHint) == enabled:
            return
        self.setWindowFlag(Qt.WindowStaysOnTopHint, enabled)
        self.show()

    # =====================================================================
    # Shutdown
    # =====================================================================
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        try:
            self.settings.save()
        except Exception:
            log.exception("Could not save settings on exit")
        try:
            self.pipeline.shutdown()
        except Exception:
            log.exception("Pipeline shutdown failed")
        super().closeEvent(event)


def _escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _pretty_model(model_id: str) -> str:
    return model_id.replace("_", " ").replace("scribe", "Scribe").title().replace("V2", "v2")


def run() -> int:
    """Application entry point."""
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("AZ Interview Copilot")
    settings = Settings.load()
    window = MainWindow(settings)
    window.show()
    return app.exec()
