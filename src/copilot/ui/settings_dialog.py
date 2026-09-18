"""Settings: two independent API keys, audio, recognition and generation.

The two providers get separate sections, separate storage entries and separate
connection tests.  An ElevenLabs key is never used for OpenAI, or the reverse.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..audio.capture import AudioCaptureError, list_loopback_devices
from ..audio.processing import SUPPORTED_PCM_RATES
from ..config import (
    DEFAULT_KEYTERMS,
    LlmSettings,
    KEYTERM_MAX_COUNT,
    KEYTERM_MAX_LENGTH,
    SERVICE_TIERS,
    VAD_SILENCE_RANGE,
    Settings,
)
from ..credentials import PROVIDERS, CredentialStore, Provider, mask
from ..llm.models import selectable_models
from ..stt import elevenlabs_client
from ..llm import openai_client
from .bridge import PipelineBridge

log = logging.getLogger(__name__)


def _hint(text: str) -> QLabel:
    """A small grey explanatory row under a settings field."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color:#6e7681; font-size:10px;")
    return label


class ProviderKeySection(QGroupBox):
    """Key entry, test, save and remove for exactly one provider."""

    def __init__(
        self,
        provider: Provider,
        credentials: CredentialStore,
        bridge: PipelineBridge,
        parent: QWidget | None = None,
        llm_settings: LlmSettings | None = None,
    ) -> None:
        spec = PROVIDERS[provider]
        super().__init__(f"{spec.label.upper()} API", parent)
        self.provider = provider
        self.credentials = credentials
        self.bridge = bridge
        # Test the combination the app will actually use, not a generic default.
        self.llm_settings = llm_settings or LlmSettings()

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText(f"{spec.label} API Key")
        layout.addWidget(self.key_edit)

        row = QHBoxLayout()
        self.test_button = QPushButton("Test Connection")
        self.save_button = QPushButton("Save Key")
        self.remove_button = QPushButton("Remove Key")
        for button in (self.test_button, self.save_button, self.remove_button):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        hint = QLabel(
            f"Açar Windows Credential Manager-də saxlanılır. "
            f"Alternativ olaraq {spec.env_var} mühit dəyişəni istifadə edilə bilər."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6e7681; font-size:10px;")
        layout.addWidget(hint)

        self.test_button.clicked.connect(self.test_connection)
        self.save_button.clicked.connect(self.save_key)
        self.remove_button.clicked.connect(self.remove_key)
        self.bridge.task_finished.connect(self._on_task_finished)

        self.refresh()

    # -- state ------------------------------------------------------------
    @property
    def _tag(self) -> str:
        return f"test:{self.provider}"

    def refresh(self) -> None:
        stored = self.credentials.get(self.provider)
        source = self.credentials.source_of(self.provider)
        if stored:
            origin = {
                "credential-manager": "Credential Manager",
                "environment": PROVIDERS[self.provider].env_var,
            }.get(source or "", "naməlum mənbə")
            self._set_status(f"Saxlanılıb: {mask(stored)} ({origin})", "#8d94a5")
        else:
            self._set_status("Açar təyin edilməyib.", "#d29922")
        self.remove_button.setEnabled(
            self.credentials.source_of(self.provider) == "credential-manager"
        )

    def _set_status(self, text: str, colour: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color:{colour}; font-size:11px;")

    def _current_key(self) -> str | None:
        typed = self.key_edit.text().strip()
        return typed or self.credentials.get(self.provider)

    # -- actions ----------------------------------------------------------
    @Slot()
    def save_key(self) -> None:
        value = self.key_edit.text().strip()
        if not value:
            QMessageBox.warning(self, "Açar boşdur", "Əvvəlcə API açarını daxil edin.")
            return
        try:
            self.credentials.set(self.provider, value)
        except Exception as exc:
            QMessageBox.critical(self, "Saxlanmadı", str(exc))
            return
        self.key_edit.clear()
        self.refresh()
        self._set_status("Açar saxlanıldı.", "#3fb950")

    @Slot()
    def remove_key(self) -> None:
        if self.credentials.delete(self.provider):
            self.key_edit.clear()
            self.refresh()
            self._set_status("Açar silindi.", "#8d94a5")
        else:
            QMessageBox.information(
                self, "Silinmədi",
                "Credential Manager-də bu provayder üçün saxlanmış açar yoxdur.",
            )

    @Slot()
    def test_connection(self) -> None:
        key = self._current_key()
        if not key:
            self._set_status("Açar təyin edilməyib.", "#f85149")
            return
        self.test_button.setEnabled(False)
        self._set_status("Yoxlanılır…", "#d29922")
        if self.provider == "elevenlabs":
            coro = elevenlabs_client.test_api_key(key)
        else:
            coro = openai_client.test_api_key(
                key,
                model=self.llm_settings.model,
                service_tier=self.llm_settings.service_tier,
            )
        self.bridge.run_task(self._tag, coro)

    @Slot(str, bool, object)
    def _on_task_finished(self, tag: str, ok: bool, payload: object) -> None:
        if tag != self._tag:
            return
        self.test_button.setEnabled(True)
        if not ok:
            self._set_status(f"Xəta: {payload}", "#f85149")
            return
        success, detail = payload  # type: ignore[misc]
        if success:
            self._set_status(f"Bağlantı uğurludur — {detail}", "#3fb950")
        else:
            self._set_status(detail, "#f85149")


class SettingsDialog(QDialog):
    def __init__(
        self,
        settings: Settings,
        credentials: CredentialStore,
        bridge: PipelineBridge,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.credentials = credentials
        self.bridge = bridge
        self.setWindowTitle("Settings")
        self.setMinimumWidth(620)
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._build_keys_tab(), "API açarları")
        tabs.addTab(self._build_audio_tab(), "Səs")
        tabs.addTab(self._build_stt_tab(), "Tanınma")
        tabs.addTab(self._build_llm_tab(), "Cavab")
        layout.addWidget(tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # -- tabs -------------------------------------------------------------
    def _build_keys_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        self.elevenlabs_section = ProviderKeySection(
            "elevenlabs", self.credentials, self.bridge
        )
        self.openai_section = ProviderKeySection(
            "openai", self.credentials, self.bridge, llm_settings=self.settings.llm
        )
        layout.addWidget(self.elevenlabs_section)
        layout.addWidget(self.openai_section)

        note = QLabel(
            "Hər iki provayder müstəqildir: ElevenLabs yalnız nitq tanıma, "
            "OpenAI yalnız cavab generasiyası üçün istifadə olunur. "
            f"Saxlama: {self.credentials.backend_name}"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#6e7681; font-size:10px;")
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    def _build_audio_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Defolt (sistem səsi)", None)
        try:
            for device in list_loopback_devices():
                self.device_combo.addItem(device.label, device.index)
        except AudioCaptureError as exc:
            self.device_combo.addItem(f"Cihaz siyahısı alınmadı: {exc}", None)
        index = self.device_combo.findData(self.settings.audio.device_index)
        self.device_combo.setCurrentIndex(max(0, index))
        form.addRow("Loopback cihazı", self.device_combo)

        self.rate_combo = QComboBox()
        for rate in SUPPORTED_PCM_RATES:
            self.rate_combo.addItem(f"{rate} Hz", rate)
        rate_index = self.rate_combo.findData(self.settings.audio.target_sample_rate)
        self.rate_combo.setCurrentIndex(max(0, rate_index))
        form.addRow("Göndərilən sample rate", self.rate_combo)

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(20, 500)
        self.chunk_spin.setSingleStep(10)
        self.chunk_spin.setSuffix(" ms")
        self.chunk_spin.setValue(self.settings.audio.chunk_ms)
        form.addRow("Audio chunk ölçüsü", self.chunk_spin)

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 8.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setValue(self.settings.audio.input_gain)
        form.addRow("Giriş gücləndirməsi", self.gain_spin)

        hint = QLabel(
            "Sistem səsi WASAPI loopback vasitəsilə tutulur — yəni Windows-un "
            "səsləndirdiyi audio (Teams, Zoom, Meet) yazılır, mikrofon deyil."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6e7681; font-size:10px;")
        form.addRow(hint)
        return page

    def _build_stt_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.model_edit = QLineEdit(self.settings.stt.model_id)
        form.addRow("Model ID", self.model_edit)

        self.language_edit = QLineEdit(self.settings.stt.language_code)
        form.addRow("Əsas dil (ISO 639-3)", self.language_edit)

        self.secondary_edit = QLineEdit(", ".join(self.settings.stt.secondary_languages))
        form.addRow("İkinci dillər", self.secondary_edit)
        form.addRow(_hint(
            "Boş saxlayın. Ölçmələrə görə 'eng' əlavə edildikdə Scribe cümləni "
            "ingilis dilinə tərcümə edir və Azərbaycan mətni itir. İngilis texniki "
            "terminlər onsuz da qorunur."
        ))

        self.commit_combo = QComboBox()
        self.commit_combo.addItems(["vad", "manual"])
        self.commit_combo.setCurrentText(self.settings.stt.commit_strategy)
        form.addRow("Commit strategiyası", self.commit_combo)

        self.vad_threshold = QDoubleSpinBox()
        self.vad_threshold.setRange(0.0, 1.0)
        self.vad_threshold.setSingleStep(0.05)
        self.vad_threshold.setValue(self.settings.stt.vad_threshold)
        form.addRow("VAD həssaslığı", self.vad_threshold)

        self.vad_silence = QDoubleSpinBox()
        self.vad_silence.setRange(*VAD_SILENCE_RANGE)   # the API rejects anything else
        self.vad_silence.setSingleStep(0.1)
        self.vad_silence.setSuffix(" san")
        self.vad_silence.setValue(self.settings.stt.vad_silence_threshold_secs)
        form.addRow("Sükut həddi", self.vad_silence)

        self.min_speech = QSpinBox()
        self.min_speech.setRange(0, 2000)
        self.min_speech.setSuffix(" ms")
        self.min_speech.setValue(self.settings.stt.min_speech_duration_ms)
        form.addRow("Min. nitq müddəti", self.min_speech)

        self.min_silence = QSpinBox()
        self.min_silence.setRange(0, 5000)
        self.min_silence.setSuffix(" ms")
        self.min_silence.setValue(self.settings.stt.min_silence_duration_ms)
        form.addRow("Min. sükut müddəti", self.min_silence)

        self.sensitivity_combo = QComboBox()
        for level, caption in (
            ("broad", "broad - müsahibin hər cümləsi (defolt)"),
            ("normal", "normal - yalnız suallar və istəklər"),
            ("strict", "strict - yalnız qrammatik suallar"),
        ):
            self.sensitivity_combo.addItem(caption, level)
        index = self.sensitivity_combo.findData(self.settings.question.sensitivity)
        self.sensitivity_combo.setCurrentIndex(max(0, index))
        form.addRow("Aşkarlama həssaslığı", self.sensitivity_combo)
        form.addRow(_hint(
            "Defolt 'broad': musahibin dediyi hər cümləyə cavab hazırlanır - "
            "sualı aşkarlamaq üçün heç bir süzgəc yoxdur. Təsdiq sözləri "
            "(«Bəli», «Aydındır», «Maraqlıdır») heç vaxt cavab yaratmır. "
            "Müsahib cümləsini davam etdirirsə, cavab yenidən - bu dəfə tam "
            "cümlə ilə - hazırlanır. Çox cavab gəlirsə 'normal' seçin."
        ))

        self.merge_window = QSpinBox()
        self.merge_window.setRange(0, 3000)
        self.merge_window.setSuffix(" ms")
        self.merge_window.setValue(self.settings.question.merge_window_ms)
        form.addRow("Sual birləşdirmə pəncərəsi", self.merge_window)

        self.no_verbatim = QCheckBox("Doldurucu sözləri təmizlə")
        self.no_verbatim.setChecked(self.settings.stt.no_verbatim)
        form.addRow(self.no_verbatim)
        form.addRow(_hint(
            "Sualı nəqli cümləyə çevirdiyi üçün defolt olaraq sönülüdür - "
            "bu, sual aşkarlanmasını çətinləşdirir."
        ))

        self.filter_background = QCheckBox("Fon səsini filtrlə")
        self.filter_background.setChecked(self.settings.stt.filter_background_audio)
        form.addRow(self.filter_background)

        self.keyterms_edit = QLineEdit(", ".join(self.settings.stt.keyterms))
        form.addRow(
            f"Keyterms (maks. {KEYTERM_MAX_COUNT}, hər biri ≤{KEYTERM_MAX_LENGTH} simvol)",
            self.keyterms_edit,
        )
        form.addRow(_hint(
            "Defolt olaraq boşdur. Ölçmələrə görə keyterms siyahısı (hətta 3 söz) "
            "cümləni vergüllə ayrılmış siyahıya çevirir və Azərbaycan şəkilçilərini "
            "pozur. Ətraflı: docs/stt-tuning.md"
        ))
        reset_keyterms = QPushButton("Tövsiyə olunan keyterms siyahısını doldur")
        reset_keyterms.clicked.connect(
            lambda: self.keyterms_edit.setText(", ".join(DEFAULT_KEYTERMS))
        )
        clear_keyterms = QPushButton("Keyterms-i təmizlə")
        clear_keyterms.clicked.connect(lambda: self.keyterms_edit.clear())
        keyterm_row = QHBoxLayout()
        keyterm_row.addWidget(reset_keyterms)
        keyterm_row.addWidget(clear_keyterms)
        keyterm_row.addStretch(1)
        keyterm_holder = QWidget()
        keyterm_holder.setLayout(keyterm_row)
        form.addRow(keyterm_holder)
        return page

    def _build_llm_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.llm_model_combo = QComboBox()
        self.llm_model_combo.setEditable(True)
        for spec in selectable_models():
            self.llm_model_combo.addItem(f"{spec.label}  ({spec.id})", spec.id)
        index = self.llm_model_combo.findData(self.settings.llm.model)
        if index >= 0:
            self.llm_model_combo.setCurrentIndex(index)
        else:
            self.llm_model_combo.setCurrentText(self.settings.llm.model)
        form.addRow("Model", self.llm_model_combo)

        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(100, 4000)
        self.max_tokens.setSingleStep(50)
        self.max_tokens.setValue(self.settings.llm.max_output_tokens)
        form.addRow("Maks. çıxış token", self.max_tokens)

        self.expand_tokens = QSpinBox()
        self.expand_tokens.setRange(200, 8000)
        self.expand_tokens.setSingleStep(100)
        self.expand_tokens.setValue(self.settings.llm.expand_max_output_tokens)
        form.addRow("Genişləndirilmiş cavab", self.expand_tokens)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(self.settings.llm.temperature)
        form.addRow("Temperature", self.temperature)

        self.history_turns = QSpinBox()
        self.history_turns.setRange(0, 10)
        self.history_turns.setValue(self.settings.llm.history_turns)
        form.addRow("Saxlanan dialoq addımı", self.history_turns)

        self.service_tier_combo = QComboBox()
        for tier in SERVICE_TIERS:
            self.service_tier_combo.addItem(tier, tier)
        index = self.service_tier_combo.findData(self.settings.llm.service_tier)
        self.service_tier_combo.setCurrentIndex(max(0, index))
        form.addRow("Service tier", self.service_tier_combo)
        form.addRow(_hint(
            "'fast' ölçmələrdə ən sürətlisi oldu (GPT-5.6 Luna: ilk token 682 ms, "
            "tam 2512 ms; 'default' ilə 852 ms / 5614 ms). Hesabınız bu tier-i "
            "qəbul etməsə, proqram defolta keçir və bunu sizə bildirir."
        ))

        self.auto_generate = QCheckBox("Sual aşkarlananda avtomatik cavab yarat")
        self.auto_generate.setChecked(self.settings.llm.auto_generate)
        form.addRow(self.auto_generate)

        note = QLabel(
            "CV və vakansiya təsviri tam şəkildə modelin təlimat qatına yüklənir — "
            "sual gələndə sənəd üzrə axtarış aparılmır, model lazımi təcrübəni "
            "artıq bilir.\n\n"
            "Model heç vaxt avtomatik dəyişmir. Benchmark yalnız tövsiyə verir — "
            "seçim burada edilir. Developer panelindən benchmark işə salın."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#6e7681; font-size:10px;")
        form.addRow(note)
        return page

    # -- persistence ------------------------------------------------------
    def accept(self) -> None:  # noqa: D102
        audio = self.settings.audio
        audio.device_index = self.device_combo.currentData()
        audio.device_name = self.device_combo.currentText()
        audio.target_sample_rate = int(self.rate_combo.currentData())
        audio.chunk_ms = self.chunk_spin.value()
        audio.input_gain = self.gain_spin.value()

        stt = self.settings.stt
        stt.model_id = self.model_edit.text().strip() or "scribe_v2_realtime"
        stt.language_code = self.language_edit.text().strip() or "aze"
        stt.secondary_languages = [
            part.strip() for part in self.secondary_edit.text().split(",") if part.strip()
        ]
        stt.commit_strategy = self.commit_combo.currentText()
        stt.vad_threshold = self.vad_threshold.value()
        stt.vad_silence_threshold_secs = self.vad_silence.value()
        stt.min_speech_duration_ms = self.min_speech.value()
        stt.min_silence_duration_ms = self.min_silence.value()
        stt.no_verbatim = self.no_verbatim.isChecked()
        stt.filter_background_audio = self.filter_background.isChecked()
        stt.keyterms = [
            part.strip() for part in self.keyterms_edit.text().split(",") if part.strip()
        ]

        self.settings.question.merge_window_ms = self.merge_window.value()
        self.settings.question.sensitivity = (
            self.sensitivity_combo.currentData() or "normal"
        )

        llm = self.settings.llm
        data = self.llm_model_combo.currentData()
        llm.model = data or self.llm_model_combo.currentText().strip()
        llm.max_output_tokens = self.max_tokens.value()
        llm.expand_max_output_tokens = self.expand_tokens.value()
        llm.temperature = self.temperature.value()
        llm.history_turns = self.history_turns.value()
        llm.service_tier = self.service_tier_combo.currentData() or "fast"
        llm.auto_generate = self.auto_generate.isChecked()

        super().accept()
