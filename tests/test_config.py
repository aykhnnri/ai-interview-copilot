"""Settings persistence and defaults."""
from __future__ import annotations

import json

from copilot.config import (
    DEFAULT_KEYTERMS,
    KEYTERM_MAX_COUNT,
    SCHEMA_VERSION,
    Settings,
    SttSettings,
    migrate,
)


def test_defaults_match_the_specified_architecture():
    settings = Settings()
    assert settings.stt.model_id == "scribe_v2_realtime"
    assert settings.stt.language_code == "aze"
    assert settings.llm.model == "gpt-5.6-luna"
    assert settings.llm.service_tier == "fast"
    assert settings.audio.target_sample_rate == 16000


def test_secondary_language_is_opt_in(monkeypatch):
    """Measured against the live service: enabling it made Scribe translate the
    Azerbaijani sentence into English.  It stays available, just not default."""
    assert Settings().stt.secondary_languages == []
    assert SttSettings(secondary_languages=["eng"]).secondary_languages == ["eng"]


def test_the_keyterm_vocabulary_is_still_offered_in_the_ui():
    """The curated list is what the Settings 'default keyterms' button fills in,
    even though the shipped default sends none."""
    for term in ("RAG", "LLM", "FastAPI", "Python", "LangGraph", "LangChain",
                 "Embeddings", "Vector Database", "Fine-tuning", "Machine Learning",
                 "Docker", "Azure", "OpenAI"):
        assert term in DEFAULT_KEYTERMS


def test_the_offered_keyterms_fit_the_api_limits():
    assert len(SttSettings(keyterms=list(DEFAULT_KEYTERMS)).effective_keyterms()) <= KEYTERM_MAX_COUNT


def test_round_trip_through_disk(tmp_path):
    settings = Settings()
    settings.llm.model = "gpt-5.6-luna"
    settings.stt.vad_threshold = 0.77
    settings.question.merge_window_ms = 900
    settings.ui.compact_mode = True

    path = settings.save(tmp_path / "settings.json")
    reloaded = Settings.load(path)

    assert reloaded.to_dict() == settings.to_dict()
    assert reloaded.llm.model == "gpt-5.6-luna"
    assert reloaded.stt.vad_threshold == 0.77


def test_missing_file_yields_defaults(tmp_path):
    assert Settings.load(tmp_path / "absent.json").llm.model == "gpt-5.6-luna"


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert Settings.load(path).stt.model_id == "scribe_v2_realtime"


def test_unknown_keys_from_a_newer_version_are_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"llm": {"model": "gpt-4.1-mini", "future_option": 1}, "unknown": {}}),
        encoding="utf-8",
    )
    settings = Settings.load(path)
    assert settings.llm.model == "gpt-4.1-mini"
    assert settings.stt.language_code == "aze"     # untouched sections keep defaults


def test_partial_settings_keep_other_defaults():
    settings = Settings.from_dict({"stt": {"vad_threshold": 0.9}})
    assert settings.stt.vad_threshold == 0.9
    assert settings.stt.model_id == "scribe_v2_realtime"
    assert settings.audio.chunk_ms == 100


def test_settings_have_no_field_that_could_hold_a_credential(tmp_path):
    """API keys live in Credential Manager; the settings file must not have a
    slot for one at all, so a future edit cannot quietly start persisting keys."""
    data = Settings().to_dict()

    # "keyterms" is speech-recognition vocabulary, not a credential.
    allowed = {"keyterms"}
    forbidden = ("api_key", "apikey", "_key", "key_", "secret", "password", "credential")

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = key.lower()
                if lowered not in allowed:
                    for marker in forbidden:
                        assert marker not in lowered, f"credential-shaped field: {path}{key}"
                    assert lowered != "key", f"credential-shaped field: {path}{key}"
                walk(value, f"{path}{key}.")

    walk(data)


def test_saved_settings_contain_no_credential_shaped_values(tmp_path):
    from copilot.logging_setup import redact

    path = Settings().save(tmp_path / "settings.json")
    text = path.read_text(encoding="utf-8")
    assert redact(text) == text, "settings file contains something key-shaped"


# =========================================================================
# Upgrading a settings file written by an older version
# =========================================================================
#: What the app shipped with before the defaults were re-measured.  A file like
#: this on disk pinned every one of those old values forever, so none of the
#: improvements reached anyone who had already run the app once.
V1_SETTINGS = {
    "question": {"sensitivity": "normal", "merge_window_ms": 700},
    "stt": {
        "vad_silence_threshold_secs": 0.8,
        "min_silence_duration_ms": 500,
        "no_verbatim": True,
        "secondary_languages": ["eng"],
        "keyterms": list(DEFAULT_KEYTERMS),
    },
    "llm": {"model": "gpt-4.1-nano"},
}


def test_an_old_settings_file_is_upgraded(tmp_path):
    import json

    path = tmp_path / "settings.json"
    path.write_text(json.dumps(V1_SETTINGS), encoding="utf-8")

    settings = Settings.load(path)
    assert settings.question.sensitivity == "broad"
    assert settings.question.merge_window_ms == 4000
    assert settings.stt.vad_silence_threshold_secs == 0.5
    assert settings.stt.min_silence_duration_ms == 300
    assert settings.stt.no_verbatim is False
    assert settings.stt.secondary_languages == []
    assert settings.stt.keyterms == []
    assert settings.llm.model == "gpt-5.6-luna"
    assert settings.schema_version == SCHEMA_VERSION


def test_a_deliberate_choice_survives_the_upgrade():
    """Only values still at the old default are moved."""
    migrated = migrate({
        "schema_version": 1,
        "question": {"sensitivity": "strict", "merge_window_ms": 1234},
        "llm": {"model": "gpt-4.1-mini"},
    })
    assert migrated["question"]["sensitivity"] == "strict"
    assert migrated["question"]["merge_window_ms"] == 1234
    assert migrated["llm"]["model"] == "gpt-4.1-mini"


def test_migration_is_idempotent():
    once = migrate(dict(V1_SETTINGS, schema_version=1))
    twice = migrate(dict(once))
    assert once == twice
    assert twice["question"]["sensitivity"] == "broad"


def test_a_current_file_is_left_alone():
    current = {
        "schema_version": SCHEMA_VERSION,
        "question": {"sensitivity": "normal"},   # chosen after the migration
    }
    assert migrate(current)["question"]["sensitivity"] == "normal"


def test_saved_settings_record_the_schema_version(tmp_path):
    import json

    path = Settings().save(tmp_path / "settings.json")
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == SCHEMA_VERSION


def test_missing_sections_do_not_break_the_upgrade():
    assert migrate({"schema_version": 1})["schema_version"] == SCHEMA_VERSION
    assert migrate({})["schema_version"] == SCHEMA_VERSION
