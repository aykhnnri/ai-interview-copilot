"""GUI smoke tests: the window builds, renders state, and shuts down cleanly.

Qt needs a display; on a headless machine these run under the offscreen
platform plugin, which is set up in the fixture below.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gui

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from copilot.config import Settings  # noqa: E402
from copilot.credentials import CredentialStore  # noqa: E402
from copilot.llm.openai_client import AnswerResult  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class StubCredentials(CredentialStore):
    def __init__(self, keys=None):
        self._values = keys or {}
        self._keyring = None
        self._keyring_error = None
        self.service_name = "test"

    def get(self, provider):
        return self._values.get(provider)

    def source_of(self, provider):
        return "credential-manager" if self._values.get(provider) else None


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("AZCOPILOT_DATA_DIR", str(tmp_path))
    from copilot.ui.main_window import MainWindow

    win = MainWindow(Settings(), StubCredentials({"elevenlabs": "a", "openai": "b"}))
    yield win
    win.pipeline.shutdown()
    win.deleteLater()


def test_window_builds_with_both_provider_indicators(window):
    assert window.stt_indicator is not None
    assert window.llm_indicator is not None
    assert "ElevenLabs" in window.stt_indicator._title.text()
    assert "GPT-5.6 Luna" in window.llm_indicator._title.text()


def test_all_required_controls_exist(window):
    assert window.start_button.text() == "Start Listening"
    assert window.pause_button.text() == "Pause Listening"
    assert window.stop_button.text() == "Stop Listening"
    assert window.generate_button.text() == "Generate Answer"
    assert window.regenerate_button.text() == "Regenerate"
    assert window.expand_button.text() == "Expand Answer"
    assert window.copy_button.text() == "Copy Answer"


def test_provider_state_updates_the_indicator(window):
    window.on_provider_state("elevenlabs", "connected")
    assert window.stt_indicator._state.text() == "Qoşulub"
    window.on_provider_state("openai", "error")
    assert window.llm_indicator._state.text() == "Xəta"


def test_listening_state_toggles_the_buttons(window):
    from copilot.core.pipeline import IDLE, LISTENING, PAUSED

    window.on_state_changed(LISTENING)
    assert not window.start_button.isEnabled()
    assert window.stop_button.isEnabled()

    window.on_state_changed(PAUSED)
    assert window.pause_button.text() == "Resume Listening"

    window.on_state_changed(IDLE)
    assert window.start_button.isEnabled()
    assert not window.stop_button.isEnabled()


def test_partial_transcript_is_shown_distinctly(window):
    window.pipeline.session.committed_lines.append("RAG nədir?")
    window.on_partial("Vector database necə")
    html = window.transcript_view.toHtml()
    assert "RAG nədir?" in html
    assert "Vector database necə" in html
    assert "italic" in html          # the in-flight partial is styled apart


def test_committed_transcript_clears_the_partial(window):
    window.on_partial("yarımçıq mətn")
    window.pipeline.session.committed_lines.append("Tam cümlə.")
    window.on_committed("Tam cümlə.")
    assert "yarımçıq mətn" not in window.transcript_view.toPlainText()


def test_streaming_answer_accumulates(window):
    window.on_answer_started("RAG nədir?")
    for delta in ("RAG ", "retrieval ", "deməkdir."):
        window.on_answer_delta(delta)
    window._flush_answer_buffer()
    assert window.answer_view.toPlainText() == "RAG retrieval deməkdir."


def test_finished_answer_shows_its_timings(window):
    window.on_answer_started("RAG nədir?")
    window.on_answer_delta("cavab")
    window.on_answer_finished(
        AnswerResult(question="RAG nədir?", text="cavab", model="gpt-4.1-nano",
                     first_token_ms=250.0, total_ms=1200.0)
    )
    meta = window.answer_meta.text()
    assert "ilk token 250 ms" in meta
    assert "tam 1200 ms" in meta
    assert "5 simvol" in meta


def test_a_dropped_delta_is_repaired_by_the_final_result(window):
    """The final result is authoritative, so the panel can never end up short."""
    window.on_answer_started("RAG nədir?")
    window.on_answer_delta("yarım")
    window.on_answer_finished(
        AnswerResult(question="RAG nədir?", text="yarım tam cavab",
                     model="gpt-4.1-nano", total_ms=10.0)
    )
    assert window.answer_view.toPlainText() == "yarım tam cavab"


def test_question_display(window):
    window.on_question("Vector database necə işləyir?")
    assert window.question_label.text() == "Vector database necə işləyir?"
    assert window.generate_button.isEnabled()


def test_errors_are_surfaced_in_the_status_bar(window):
    from copilot import messages as msg

    window.on_error("elevenlabs", msg.STT_DISCONNECTED)
    assert msg.STT_DISCONNECTED in window.status_label.text()
    assert window.stt_indicator._state.text() == "Xəta"


def test_compact_mode_hides_the_transcript_and_pins_on_top(window):
    from PySide6.QtCore import Qt

    window.set_compact(True)
    assert not window.transcript_card.isVisible()
    assert not window.header_card.isVisible()
    assert bool(window.windowFlags() & Qt.WindowStaysOnTopHint)
    assert window.settings.ui.compact_mode

    window.set_compact(False)
    assert window.transcript_card.isVisible()
    assert not window.settings.ui.compact_mode


def test_copy_answer_puts_text_on_the_clipboard(window, qapp):
    from PySide6.QtGui import QGuiApplication

    window.on_answer_started("q")
    window.on_answer_delta("kopyalanacaq mətn")
    window._flush_answer_buffer()
    window.copy_answer()
    assert QGuiApplication.clipboard().text() == "kopyalanacaq mətn"


def test_documents_can_be_loaded_and_cleared(window, tmp_path):
    cv = tmp_path / "cv.txt"
    cv.write_text("Aysel Məmmədova\nAI Engineer\n\nTƏCRÜBƏ\nLangGraph, RAG", encoding="utf-8")

    window._load_document(str(cv), "cv")
    assert "cv.txt" in window.documents_label.text()
    assert window.pipeline.profile.has_documents

    window._clear_documents()
    assert not window.pipeline.profile.has_documents


def test_settings_dialog_builds_with_both_provider_sections(window, qapp):
    from copilot.ui.settings_dialog import SettingsDialog

    dialog = SettingsDialog(window.settings, window.credentials, window.bridge, window)
    try:
        assert dialog.elevenlabs_section.provider == "elevenlabs"
        assert dialog.openai_section.provider == "openai"
        assert dialog.elevenlabs_section.test_button.text() == "Test Connection"
        assert dialog.openai_section.save_button.text() == "Save Key"
        assert dialog.openai_section.remove_button.text() == "Remove Key"
    finally:
        dialog.deleteLater()


def test_settings_dialog_writes_back_changes(window, qapp):
    from copilot.ui.settings_dialog import SettingsDialog

    dialog = SettingsDialog(window.settings, window.credentials, window.bridge, window)
    try:
        dialog.vad_threshold.setValue(0.75)
        dialog.merge_window.setValue(900)
        dialog.llm_model_combo.setCurrentIndex(
            dialog.llm_model_combo.findData("gpt-4.1-nano")
        )
        dialog.service_tier_combo.setCurrentIndex(
            dialog.service_tier_combo.findData("priority")
        )
        dialog.accept()
    finally:
        dialog.deleteLater()

    assert window.settings.stt.vad_threshold == 0.75
    assert window.settings.question.merge_window_ms == 900
    assert window.settings.llm.model == "gpt-4.1-nano"
    assert window.settings.llm.service_tier == "priority"


def test_dev_panel_builds_and_reports_metrics(window, qapp):
    window.show_dev_panel()
    panel = window._dev_panel
    assert panel is not None
    panel.refresh_metrics()
    panel.append_log("test line")
    assert "test line" in panel.log_view.toPlainText()
    panel.close()


def test_the_openai_connection_test_uses_the_configured_model_and_tier(window, qapp, monkeypatch):
    """Testing a generic default would pass while the real configuration fails."""
    from copilot.ui.settings_dialog import SettingsDialog

    window.settings.llm.model = "gpt-5.6-luna"
    window.settings.llm.service_tier = "fast"
    captured: dict = {}

    async def fake_test(key, model=None, timeout=20.0, service_tier=None):
        captured.update(key=key, model=model, service_tier=service_tier)
        return True, "ok"

    from copilot.llm import openai_client

    monkeypatch.setattr(openai_client, "test_api_key", fake_test)

    dialog = SettingsDialog(window.settings, window.credentials, window.bridge, window)
    try:
        dialog.openai_section.test_connection()
        qapp.processEvents()
    finally:
        dialog.deleteLater()

    assert captured.get("model") == "gpt-5.6-luna"
    assert captured.get("service_tier") == "fast"


# =========================================================================
# Appearance: translucency, focus and the palette
# =========================================================================
#: Qt stores window opacity as an 8-bit value, so a round-trip lands within
#: 1/255 of what was asked for.  The stored *setting* is exact; the window is not.
QUANTISED = dict(abs=1 / 255)


def test_opacity_is_clamped_so_the_window_can_never_vanish(window):
    """A stored or fat-fingered 0.02 would leave an invisible, unfindable window."""
    from copilot.config import OPACITY_RANGE

    low, high = OPACITY_RANGE

    window.set_opacity(0.02)
    assert window.windowOpacity() == pytest.approx(low, **QUANTISED)
    assert window.settings.ui.opacity == pytest.approx(low)

    window.set_opacity(5.0)
    assert window.windowOpacity() == pytest.approx(high, **QUANTISED)

    window.set_opacity(0.7)
    assert window.windowOpacity() == pytest.approx(0.7, **QUANTISED)


def test_effective_opacity_repairs_a_corrupt_settings_value():
    from copilot.config import OPACITY_RANGE, UiSettings

    assert UiSettings(opacity=0.01).effective_opacity() == pytest.approx(OPACITY_RANGE[0])
    assert UiSettings(opacity=9.0).effective_opacity() == pytest.approx(OPACITY_RANGE[1])
    assert UiSettings(opacity="nonsense").effective_opacity() == pytest.approx(1.0)
    assert UiSettings(opacity=0.8).effective_opacity() == pytest.approx(0.8)


def test_nudging_opacity_steps_and_stops_at_the_floor(window):
    from copilot.config import OPACITY_RANGE
    from copilot.ui.main_window import OPACITY_STEP

    window.set_opacity(1.0)
    window.nudge_opacity(-OPACITY_STEP)
    assert window.windowOpacity() == pytest.approx(1.0 - OPACITY_STEP, **QUANTISED)

    for _ in range(50):
        window.nudge_opacity(-OPACITY_STEP)
    assert window.windowOpacity() == pytest.approx(OPACITY_RANGE[0], **QUANTISED)


def test_opacity_survives_toggling_always_on_top(window):
    """Changing a window flag re-creates the native window and drops opacity."""
    window.set_opacity(0.6)
    window._toggle_always_on_top(True)
    assert window.windowOpacity() == pytest.approx(0.6, **QUANTISED)
    window._toggle_always_on_top(False)
    assert window.windowOpacity() == pytest.approx(0.6, **QUANTISED)


def test_the_transcript_can_be_hidden_and_the_choice_is_remembered(window):
    window.show()   # isVisible() on a child is False until the window itself is
    window.set_transcript_visible(False)
    assert not window.transcript_card.isVisible()
    assert not window.settings.ui.show_transcript
    assert not window.transcript_button.isChecked()

    window.set_transcript_visible(True)
    assert window.transcript_card.isVisible()
    assert window.settings.ui.show_transcript


def test_leaving_compact_restores_the_transcript_preference_not_a_default(window):
    """Compact hides the transcript; it must not silently re-enable it after."""
    window.show()
    window.set_transcript_visible(False)
    window.set_compact(True)
    assert not window.transcript_card.isVisible()

    window.set_compact(False)
    assert not window.transcript_card.isVisible(), "compact resurrected a hidden strip"
    assert not window.settings.ui.show_transcript


def test_compact_mode_drops_the_secondary_controls(window):
    window.set_compact(True)
    assert not window.pause_button.isVisible()
    assert not window.regenerate_button.isVisible()
    assert not window.expand_button.isVisible()
    assert window.generate_button.isVisible()
    assert window.copy_button.isVisible()
    assert window.stop_button.isVisible()

    window.set_compact(False)
    assert window.pause_button.isVisible()
    assert window.regenerate_button.isVisible()


def test_the_question_placeholder_is_styled_apart_from_a_real_question(window):
    from copilot.ui.main_window import QUESTION_PLACEHOLDER

    assert window.question_label.text() == QUESTION_PLACEHOLDER
    assert window.question_label.objectName() == "QuestionEmpty"

    window.on_question("RAG nədir?")
    assert window.question_label.objectName() == "Question"
    assert window.question_label.text() == "RAG nədir?"


def test_the_model_id_lives_in_the_tooltip_not_the_bar(window):
    """The bar names the provider; the exact model is looked up, not stared at."""
    assert window.stt_indicator._title.text() == "ElevenLabs"
    assert "Scribe" in window.stt_indicator.toolTip()
    assert window.settings.llm.service_tier in window.llm_indicator.toolTip()


# -- the palette ----------------------------------------------------------
def _spread(hex_colour: str) -> int:
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return max(r, g, b) - min(r, g, b)


def test_the_palette_is_greyscale_apart_from_the_error_tone():
    """A blue 'primary' creeping back in is exactly the regression to catch."""
    from copilot.ui import theme

    neutral = {
        name: value
        for name, value in vars(theme).items()
        if name.isupper() and isinstance(value, str) and value.startswith("#")
        and name != "ERROR"
    }
    assert neutral, "no colours found - did the theme move?"
    offenders = {n: v for n, v in neutral.items() if _spread(v) > 12}
    assert not offenders, f"non-grey colours in the palette: {offenders}"
    assert _spread(theme.ERROR) > 40, "the error tone must stay distinguishable"


def test_the_answer_is_the_largest_text_in_the_window():
    """The layout's whole premise: the answer outranks everything else."""
    import re

    from copilot.ui import theme

    sheet = theme.stylesheet()

    def size_of(selector: str) -> int:
        block = sheet.split(selector, 1)[1].split("}", 1)[0]
        return int(re.search(r"font-size:\s*(\d+)px", block).group(1))

    answer = size_of("QTextEdit#Answer")
    question = size_of("QLabel#Question ")
    transcript = size_of("QTextEdit#Transcript")
    assert answer > question > transcript


def test_font_scale_scales_every_size_together():
    from copilot.ui import theme

    normal = theme.stylesheet(1.0)
    large = theme.stylesheet(1.5)
    assert "font-size: 16px" in normal          # the answer at 1.0
    assert "font-size: 24px" in large           # and at 1.5
    assert normal != large


def test_settings_dialog_writes_the_appearance_tab_back(window, qapp):
    from copilot.ui.settings_dialog import SettingsDialog

    dialog = SettingsDialog(window.settings, window.credentials, window.bridge, window)
    try:
        dialog.opacity_slider.setValue(70)
        dialog.show_transcript.setChecked(False)
        dialog.always_on_top.setChecked(True)
        dialog.font_scale.setValue(1.2)
        dialog.accept()
    finally:
        dialog.deleteLater()

    assert window.settings.ui.opacity == pytest.approx(0.70)
    assert window.settings.ui.show_transcript is False
    assert window.settings.ui.always_on_top is True
    assert window.settings.ui.font_scale == pytest.approx(1.2)


def test_the_opacity_slider_previews_live(window, qapp):
    """Picking a translucency you cannot see is guesswork."""
    from copilot.ui.settings_dialog import SettingsDialog

    seen: list[float] = []
    dialog = SettingsDialog(window.settings, window.credentials, window.bridge, window)
    try:
        dialog.opacity_changed.connect(seen.append)
        dialog.opacity_slider.setValue(55)
        assert seen and seen[-1] == pytest.approx(0.55)
        assert dialog.opacity_value.text() == "55%"
    finally:
        dialog.deleteLater()
