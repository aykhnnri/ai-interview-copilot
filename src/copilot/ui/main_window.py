"""The main desktop window.

The answer is the product, so it gets the space and the largest type; every
other element is deliberately quiet.  Top to bottom: a slim status bar, the
detected question, the streaming answer, an optional transcript strip, and a
slim control bar.  Compact mode drops everything except question and answer and
pins the window above the call.

The window can also be made translucent, which is what makes it usable *over* a
meeting rather than beside it - see `_apply_opacity`.
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import messages as msg
from ..config import OPACITY_RANGE, Settings
from ..core.pipeline import IDLE, LISTENING, PAUSED, CopilotPipeline
from ..credentials import CredentialStore
from ..documents.extract import DocumentError, ExtractedDocument, extract_text
from ..documents.profile import CandidateProfile
from ..llm.models import get_spec
from ..llm.openai_client import AnswerResult
from . import theme
from .bridge import PipelineBridge
from .dev_panel import DevPanel
from .settings_dialog import SettingsDialog
from .widgets import StatusIndicator

log = logging.getLogger(__name__)

#: How far Ctrl+Shift+Up/Down moves the opacity per press.
OPACITY_STEP = 0.05

QUESTION_PLACEHOLDER = "Sual gözlənilir…"


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
        self.apply_theme()
        self.resize(880, 680)
        self._build_ui()
        self._connect_signals()
        self._restore_documents()
        self._refresh_provider_labels()
        self._apply_always_on_top(self.settings.ui.always_on_top)
        self._apply_opacity(self.settings.ui.effective_opacity())
        self.set_transcript_visible(self.settings.ui.show_transcript, persist=False)
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
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        outer.addWidget(self._build_status_bar())
        outer.addWidget(self._build_answer(), 1)
        outer.addWidget(self._build_transcript())
        outer.addWidget(self._build_controls())

        self._build_menu()

    def _build_status_bar(self) -> QWidget:
        """Two dots, the document state and three quiet buttons - one line."""
        bar = QFrame(objectName="Bar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 7, 8, 7)
        layout.setSpacing(14)

        self.stt_indicator = StatusIndicator("ElevenLabs Scribe v2 Realtime", compact=True)
        self.llm_indicator = StatusIndicator("GPT-5.6 Luna", compact=True)
        layout.addWidget(self.stt_indicator)
        layout.addWidget(self.llm_indicator)
        layout.addStretch(1)

        self.documents_label = QLabel("Sənəd yüklənməyib")
        self.documents_label.setObjectName("Hint")
        self.documents_label.setMaximumWidth(280)
        layout.addWidget(self.documents_label)

        self.transcript_button = QPushButton("Transkript", objectName="Ghost")
        self.transcript_button.setCheckable(True)
        self.transcript_button.setToolTip("Transkripti göstər / gizlət  (Ctrl+T)")
        self.settings_button = QPushButton("Settings", objectName="Ghost")
        self.compact_button = QPushButton("Compact", objectName="Ghost")
        self.compact_button.setCheckable(True)
        self.compact_button.setToolTip("Yalnız sual və cavab  (Ctrl+Shift+C)")
        for button in (self.transcript_button, self.settings_button, self.compact_button):
            layout.addWidget(button)

        self.header_card = bar
        return bar

    def _build_answer(self) -> QWidget:
        """Question and answer in one card, with the answer given the room."""
        card = QFrame(objectName="Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        caption = QLabel("SUAL")
        caption.setObjectName("SectionTitle")
        layout.addWidget(caption)

        self.question_label = QLabel(QUESTION_PLACEHOLDER)
        self.question_label.setObjectName("QuestionEmpty")
        self.question_label.setWordWrap(True)
        self.question_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.question_label)

        # No caption above the answer: it is the biggest text in the window and
        # sits under the question, so labelling it only adds furniture.
        self.answer_view = QTextEdit()
        self.answer_view.setObjectName("Answer")
        self.answer_view.setReadOnly(True)
        self.answer_view.setFont(QFont("Segoe UI", 13))
        self.answer_view.setPlaceholderText(
            "Cavab burada görünəcək.\n\n"
            "Müsahib danışmağı bitirdikdə cavab avtomatik yazılmağa başlayır."
        )
        layout.addWidget(self.answer_view, 1)

        row = QHBoxLayout()
        row.setSpacing(6)
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
        self.answer_meta.setObjectName("Meta")
        row.addWidget(self.answer_meta)
        layout.addLayout(row)

        self._set_answer_actions_enabled(False)
        self.answer_card = card
        return card

    def _build_transcript(self) -> QWidget:
        """A short strip, not a panel: enough to see what was heard."""
        card = QFrame(objectName="Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 10, 18, 10)
        layout.setSpacing(4)

        title = QLabel("CANLI TRANSKRİPT")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.transcript_view = QTextEdit()
        self.transcript_view.setObjectName("Transcript")
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setFont(QFont("Segoe UI", 9))
        self.transcript_view.setFixedHeight(76)
        layout.addWidget(self.transcript_view)

        self.transcript_card = card
        return card

    def _build_controls(self) -> QWidget:
        bar = QFrame(objectName="Bar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        self.start_button = QPushButton("Start Listening", objectName="Primary")
        self.pause_button = QPushButton("Pause Listening")
        self.stop_button = QPushButton("Stop Listening", objectName="Danger")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        for button in (self.start_button, self.pause_button, self.stop_button):
            layout.addWidget(button)

        layout.addSpacing(10)
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedWidth(90)
        self.level_bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.level_bar.setToolTip("Sistem səsinin səviyyəsi")
        layout.addWidget(self.level_bar)

        layout.addStretch(1)
        self.status_label = QLabel(msg.STATUS_IDLE)
        self.status_label.setObjectName("Meta")
        layout.addWidget(self.status_label)

        self.controls_card = bar
        return bar

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
        self.transcript_action = QAction("Transkripti göstər", self, checkable=True)
        self.transcript_action.setShortcut("Ctrl+T")
        self.transcript_action.setChecked(self.settings.ui.show_transcript)
        self.transcript_action.triggered.connect(self.set_transcript_visible)
        self.on_top_action = QAction("Həmişə üstdə", self, checkable=True)
        self.on_top_action.setChecked(self.settings.ui.always_on_top)
        self.on_top_action.triggered.connect(self._toggle_always_on_top)

        # Opacity is adjusted mid-call, so it needs keys as well as a slider.
        less = QAction("Daha şəffaf", self)
        less.setShortcut("Ctrl+Shift+Down")
        less.triggered.connect(lambda: self.nudge_opacity(-OPACITY_STEP))
        more = QAction("Daha tünd", self)
        more.setShortcut("Ctrl+Shift+Up")
        more.triggered.connect(lambda: self.nudge_opacity(OPACITY_STEP))

        dev_action = QAction("Developer paneli", self)
        dev_action.setShortcut("Ctrl+Shift+D")
        dev_action.triggered.connect(self.show_dev_panel)
        settings_action = QAction("Settings…", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.open_settings)

        for action in (self.compact_action, self.transcript_action, self.on_top_action):
            view_menu.addAction(action)
        view_menu.addSeparator()
        for action in (less, more):
            view_menu.addAction(action)
        view_menu.addSeparator()
        view_menu.addAction(dev_action)
        view_menu.addAction(settings_action)

    # =====================================================================
    # Appearance
    # =====================================================================
    def apply_theme(self) -> None:
        """(Re)build the stylesheet, honouring the configured font scale."""
        self.setStyleSheet(theme.stylesheet(self.settings.ui.font_scale))

    def _apply_opacity(self, value: float) -> None:
        low, high = OPACITY_RANGE
        value = min(high, max(low, float(value)))
        self.setWindowOpacity(value)
        return value

    @Slot(float)
    def set_opacity(self, value: float, persist: bool = True) -> None:
        """Set window translucency, clamped so the window can never vanish."""
        applied = self._apply_opacity(value)
        self.settings.ui.opacity = applied
        if persist:
            self.settings.save()
        if applied < 1.0:
            self.status_label.setText(f"Şəffaflıq {applied * 100:.0f}%")

    def nudge_opacity(self, delta: float) -> None:
        self.set_opacity(self.settings.ui.effective_opacity() + delta)

    @Slot(bool)
    def set_transcript_visible(self, visible: bool, persist: bool = True) -> None:
        self.settings.ui.show_transcript = bool(visible)
        # In compact mode the transcript is hidden regardless; the preference is
        # remembered so leaving compact restores what the user actually chose.
        self.transcript_card.setVisible(bool(visible) and not self.settings.ui.compact_mode)
        for control in (self.transcript_button, self.transcript_action):
            if control.isChecked() != bool(visible):
                control.blockSignals(True)
                control.setChecked(bool(visible))
                control.blockSignals(False)
        if persist:
            self.settings.save()

    # =====================================================================
    # Wiring
    # =====================================================================
    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self.start_listening)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_listening)
        self.settings_button.clicked.connect(self.open_settings)
        self.compact_button.toggled.connect(self.set_compact)
        self.transcript_button.toggled.connect(self.set_transcript_visible)

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
            span = (
                f"<span style='color:{theme.TEXT_FAINT};font-style:italic'>{partial}</span>"
            )
            body = f"{body}<br>{span}" if body else span
        self.transcript_view.setHtml(
            f"<div style='color:{theme.TEXT_DIM};line-height:1.5'>{body}</div>"
        )
        scrollbar = self.transcript_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # =====================================================================
    # Question and answer
    # =====================================================================
    def _set_question_text(self, question: str) -> None:
        """Swap between the placeholder and real-question styling."""
        real = bool((question or "").strip())
        self.question_label.setText(question if real else QUESTION_PLACEHOLDER)
        name = "Question" if real else "QuestionEmpty"
        if self.question_label.objectName() != name:
            self.question_label.setObjectName(name)
            # Qt does not restyle on an objectName change without a repolish.
            self.question_label.style().unpolish(self.question_label)
            self.question_label.style().polish(self.question_label)

    @Slot(str)
    def on_question(self, question: str) -> None:
        self._set_question_text(question)
        self._set_answer_actions_enabled(True)

    @Slot(str)
    def on_answer_started(self, question: str) -> None:
        self._set_question_text(question)
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
        self.documents_label.setToolTip(profile.describe())

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
        before = (self.settings.ui.opacity, self.settings.ui.font_scale)
        dialog = SettingsDialog(self.settings, self.credentials, self.bridge, self)
        # Live preview: dragging the slider should change the window behind it.
        dialog.opacity_changed.connect(lambda v: self._apply_opacity(v))
        if dialog.exec():
            self.settings.save()
            self.pipeline.settings = self.settings
            self._refresh_provider_labels()
            if self.settings.ui.font_scale != before[1]:
                self.apply_theme()
            self._apply_opacity(self.settings.ui.effective_opacity())
            self.set_transcript_visible(self.settings.ui.show_transcript, persist=False)
            self._apply_always_on_top(self.settings.ui.always_on_top)
            self.on_top_action.setChecked(self.settings.ui.always_on_top)
        else:
            # Cancelled: undo whatever the live preview did.
            self.settings.ui.opacity = before[0]
            self._apply_opacity(self.settings.ui.effective_opacity())

    @Slot()
    def show_dev_panel(self) -> None:
        if self._dev_panel is None:
            self._dev_panel = DevPanel(self.pipeline, self.bridge, self.credentials, self)
            self.bridge.log_line.connect(self._dev_panel.append_log)
        self._dev_panel.show()
        self._dev_panel.raise_()
        self._dev_panel.activateWindow()

    def _refresh_provider_labels(self) -> None:
        # Provider name in the bar, exact model in the tooltip: the model id is
        # long, rarely consulted, and was a third of the old header's width.
        self.stt_indicator.set_title("ElevenLabs")
        self.stt_indicator.set_detail(_pretty_model(self.settings.stt.model_id))
        spec = get_spec(self.settings.llm.model)
        self.llm_indicator.set_title(spec.label)
        self.llm_indicator.set_detail(f"{spec.id} · {self.settings.llm.service_tier}")

    @Slot(bool)
    def set_compact(self, compact: bool) -> None:
        """Compact mode: question + answer only, pinned above other windows."""
        if compact and self._normal_geometry is None:
            self._normal_geometry = self.saveGeometry()

        self.settings.ui.compact_mode = compact
        self.header_card.setVisible(not compact)
        self.transcript_card.setVisible(not compact and self.settings.ui.show_transcript)
        self.menuBar().setVisible(not compact)
        self.level_bar.setVisible(not compact)
        self.answer_meta.setVisible(not compact)
        # Regenerate and Expand are refinements; Pause is something you press
        # once a session, if ever.  In compact mode the room goes to the answer.
        self.regenerate_button.setVisible(not compact)
        self.expand_button.setVisible(not compact)
        self.pause_button.setVisible(not compact)

        if self.compact_button.isChecked() != compact:
            self.compact_button.blockSignals(True)
            self.compact_button.setChecked(compact)
            self.compact_button.blockSignals(False)
        if self.compact_action.isChecked() != compact:
            self.compact_action.setChecked(compact)

        if compact:
            self._apply_always_on_top(True)
            self.resize(480, 380)
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
        # Changing a window flag re-creates the native window, which drops the
        # opacity Windows had applied to it.
        self.show()
        self._apply_opacity(self.settings.ui.effective_opacity())

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
