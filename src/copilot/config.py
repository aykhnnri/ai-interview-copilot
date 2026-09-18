"""Application settings: typed dataclasses persisted as JSON.

Nothing secret is ever stored here - API keys live in `credentials.py`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .paths import settings_path

log = logging.getLogger(__name__)

# Technical vocabulary biased into the speech recogniser.  ElevenLabs accepts at
# most 50 keyterms of <= 20 characters each.
DEFAULT_KEYTERMS: list[str] = [
    "RAG", "LLM", "FastAPI", "Python", "LangGraph", "LangChain",
    "Embeddings", "Vector Database", "Fine-tuning", "Machine Learning",
    "Docker", "Azure", "OpenAI", "Kubernetes", "PostgreSQL", "Redis",
    "Transformer", "Prompt", "Hallucination", "Retrieval", "Chunking",
    "Reranking", "Agent", "Multi-agent", "Latency", "Throughput",
    "Inference", "Deployment", "Pipeline", "Async", "API", "GPU",
    "Quantization", "Evaluation", "Benchmark", "Guardrails",
]

KEYTERM_MAX_COUNT = 50
KEYTERM_MAX_LENGTH = 20

#: Server-enforced bounds.  Sending 0.25 is answered with
#: `invalid_request: vad_silence_threshold_secs must be between 0.3 and 3.0`
#: and the socket is closed, so the value is clamped before it is ever sent.
#: Bumped whenever a shipped default changes.  A settings file written before
#: the change keeps the old value forever otherwise - a saved value always wins
#: over a default - so improvements silently never reach anyone who has already
#: run the app once.
SCHEMA_VERSION = 2

#: (section, field, old default, new default) applied when upgrading from v1.
#: Only values still equal to the *old* default are moved: anything the user
#: deliberately changed is left alone.
_MIGRATIONS_V2: tuple[tuple[str, str, Any, Any], ...] = (
    ("question", "sensitivity", "normal", "broad"),
    ("question", "merge_window_ms", 700, 4000),
    ("stt", "vad_silence_threshold_secs", 0.8, 0.5),
    ("stt", "min_silence_duration_ms", 500, 300),
    ("stt", "no_verbatim", True, False),
    ("stt", "secondary_languages", ["eng"], []),
    ("llm", "model", "gpt-4.1-nano", "gpt-5.6-luna"),
)

VAD_SILENCE_RANGE = (0.3, 3.0)
VAD_THRESHOLD_RANGE = (0.0, 1.0)


@dataclass
class AudioSettings:
    """Windows loopback capture."""

    device_index: int | None = None          # None -> default render device
    device_name: str = ""                    # informational, for the UI
    target_sample_rate: int = 16000          # what we stream to ElevenLabs
    chunk_ms: int = 100                      # audio per websocket frame
    input_gain: float = 1.0


@dataclass
class SttSettings:
    """ElevenLabs Scribe v2 Realtime.

    Three defaults here were chosen from measurements against the live service
    rather than from the parameter list, because the obvious settings made the
    Azerbaijani transcript worse.  See `docs/stt-tuning.md`:

    * `secondary_languages` is **empty**.  Setting it to `eng` made Scribe
      return the sentence translated into English - exactly what this app must
      never do.  English technical terms still survive without it.
    * `keyterms` is **empty**.  Any keyterm list, even three entries, turned a
      spoken sentence into a comma-separated list of the keyterms and mangled
      the Azerbaijani suffixes.
    * `no_verbatim` is **off**.  It rewrote the question as a nominalisation,
      which removes the grammatical cues question detection relies on.

    All three remain configurable in Settings; the defaults are just what
    measured best.
    """

    model_id: str = "scribe_v2_realtime"
    language_code: str = "aze"
    secondary_languages: list[str] = field(default_factory=list)
    base_url: str = "wss://api.elevenlabs.io"
    commit_strategy: str = "vad"             # "vad" | "manual"
    vad_threshold: float = 0.5
    #: 0.5 measured ~0.34s faster to commit than 0.8 with no extra
    #: fragmentation (0.95s vs 1.29s after the speaker stops).
    vad_silence_threshold_secs: float = 0.5
    min_speech_duration_ms: int = 150
    min_silence_duration_ms: int = 300
    include_timestamps: bool = False
    include_language_detection: bool = False
    no_verbatim: bool = False                # see the note above
    filter_background_audio: bool = True     # measured harmless, helps on calls
    send_sample_rate_field: bool = True
    keyterms: list[str] = field(default_factory=list)

    def effective_vad_silence(self) -> float:
        """Silence threshold clamped to the range the API accepts."""
        low, high = VAD_SILENCE_RANGE
        return min(max(float(self.vad_silence_threshold_secs), low), high)

    def effective_vad_threshold(self) -> float:
        low, high = VAD_THRESHOLD_RANGE
        return min(max(float(self.vad_threshold), low), high)

    def effective_keyterms(self) -> list[str]:
        """Keyterms trimmed to the documented API limits, order preserved."""
        seen: set[str] = set()
        out: list[str] = []
        for raw in self.keyterms:
            term = (raw or "").strip()
            if not term or len(term) > KEYTERM_MAX_LENGTH:
                continue
            low = term.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(term)
            if len(out) >= KEYTERM_MAX_COUNT:
                break
        return out


#: OpenAI service tiers.  "fast" measured best here: on gpt-5.6-luna it gave
#: TTFT 682 ms / total 2512 ms against 852 ms / 5614 ms for "default".
SERVICE_TIERS: tuple[str, ...] = ("fast", "priority", "auto", "default", "flex")


@dataclass
class LlmSettings:
    """OpenAI answer generation."""

    model: str = "gpt-5.6-luna"
    base_url: str | None = None
    service_tier: str = "fast"
    max_output_tokens: int = 700
    expand_max_output_tokens: int = 1400
    temperature: float = 0.4
    request_timeout_s: float = 60.0
    history_turns: int = 3                   # prior Q/A pairs kept as context
    auto_generate: bool = True               # answer as soon as a question is final


@dataclass
class QuestionSettings:
    """Turning committed transcript segments into things worth answering."""

    #: "strict"  - grammatical questions only
    #: "normal"  - questions plus requests ("bəhs edin", "gəlin danışaq")
    #: "broad"   - every interviewer turn that is not an acknowledgement
    #:
    #: Defaults to "broad": in a real interview almost everything the other
    #: person says is addressed to you, and a missed question costs far more
    #: than an answer you ignore.  Narrow it if it proves too chatty.
    sensitivity: str = "broad"

    #: Silence between commits that means the interviewer finished their turn.
    #: Below this, a new commit extends the same turn and re-answers the whole
    #: thing; above it, a new turn starts.
    turn_gap_ms: int = 6000

    #: How long to hold a fragment that is not yet answerable, waiting for the
    #: rest of the sentence.  This must outlast the gap between real commits.
    #: Measured end to end: a speaker hesitating for 1.6s mid-sentence produced
    #: commits 3.3s apart, because the recogniser adds its own silence
    #: threshold plus transcription time on top of the human pause.  A window
    #: shorter than that judges the sentence on half of itself.
    #:
    #: Holding is close to free: text that already reads as a prompt fires
    #: immediately and never waits, so this only delays material that would
    #: otherwise have been thrown away.
    merge_window_ms: int = 4000
    max_buffer_chars: int = 600              # stop coalescing runaway speech
    min_question_chars: int = 8
    duplicate_similarity: float = 0.86
    duplicate_memory: int = 12


@dataclass
class UiSettings:
    always_on_top: bool = False
    compact_mode: bool = False
    font_scale: float = 1.0
    show_dev_panel: bool = False
    window_geometry: str = ""


@dataclass
class DocumentSettings:
    cv_path: str = ""
    job_description_path: str = ""


@dataclass
class Settings:
    schema_version: int = SCHEMA_VERSION
    audio: AudioSettings = field(default_factory=AudioSettings)
    stt: SttSettings = field(default_factory=SttSettings)
    llm: LlmSettings = field(default_factory=LlmSettings)
    question: QuestionSettings = field(default_factory=QuestionSettings)
    ui: UiSettings = field(default_factory=UiSettings)
    documents: DocumentSettings = field(default_factory=DocumentSettings)

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        return _build(cls, data or {})

    def save(self, path: Path | None = None) -> Path:
        target = path or settings_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        target = path or settings_path()
        if not target.exists():
            return cls()
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read settings (%s); using defaults", exc)
            return cls()

        data = migrate(data)
        settings = cls.from_dict(data)
        if settings.schema_version != SCHEMA_VERSION:
            settings.schema_version = SCHEMA_VERSION
        return settings


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Bring a settings file written by an older version up to date.

    Without this, every default improved after someone first ran the app is
    dead on their machine: `Settings.load` reads their stored value and the new
    default never applies.
    """
    data = dict(data or {})
    version = data.get("schema_version")
    version = int(version) if isinstance(version, int) else 1

    if version < 2:
        changed: list[str] = []
        for section, field_name, old_default, new_default in _MIGRATIONS_V2:
            block = data.get(section)
            if not isinstance(block, dict) or field_name not in block:
                continue
            if block[field_name] == old_default:
                block[field_name] = new_default
                changed.append(f"{section}.{field_name}={new_default!r}")
        # The old curated keyterm list measured worse than sending none.
        stt = data.get("stt")
        if isinstance(stt, dict) and stt.get("keyterms") == DEFAULT_KEYTERMS:
            stt["keyterms"] = []
            changed.append("stt.keyterms=[]")
        if changed:
            log.info("Upgraded settings to schema v2: %s", ", ".join(changed))

    data["schema_version"] = SCHEMA_VERSION
    return data


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Rebuild a nested dataclass, ignoring unknown keys.

    Field types are compared against the *default instance* rather than the
    annotation, because `from __future__ import annotations` makes every
    annotation a string.
    """
    obj = cls()
    known = {f.name for f in fields(cls)}
    for name, value in data.items():
        if name not in known:
            continue                      # settings from a newer version
        current = getattr(obj, name)
        if is_dataclass(current) and isinstance(value, dict):
            setattr(obj, name, _build(type(current), value))
        else:
            try:
                setattr(obj, name, value)
            except Exception:  # pragma: no cover - dataclasses accept anything
                pass
    return obj
