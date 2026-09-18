"""Heuristic scoring of a generated Azerbaijani answer.

Three independent scores, each in 0..1:

* **Azerbaijani quality** - does the text read as Azerbaijani rather than
  Turkish?  The decisive signal is the letter "ə", which Azerbaijani uses
  constantly and Turkish does not have at all, backed up by a list of Turkish
  function words whose Azerbaijani equivalents differ.
* **English term preservation** - technical terms must stay in English.
* **Coverage** - how many of the concepts a correct answer should mention are
  actually present.

These are deterministic proxies, not a substitute for human review; the
benchmark report says so.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..nlp.question_detector import az_lower

#: Letters that only Azerbaijani (of the two) uses.
AZ_ONLY_LETTERS = "əƏ"

#: Turkish function words whose Azerbaijani forms are different.  Finding these
#: is the strongest evidence the model drifted into Turkish.
TURKISH_MARKERS: tuple[tuple[str, str], ...] = (
    ("ve", "və"), ("ile", "ilə"), ("için", "üçün"), ("değil", "deyil"),
    ("nasıl", "necə"), ("şey", "şey"), ("gibi", "kimi"), ("çok", "çox"),
    ("daha sonra", "sonra"), ("yapılır", "edilir"), ("kullanılır", "istifadə olunur"),
    ("olarak", "olaraq"), ("sağlar", "təmin edir"), ("gerekir", "lazımdır"),
    ("bulunur", "olur"), ("veri", "məlumat"), ("öğrenme", "öyrənmə"),
    ("değerlendirme", "qiymətləndirmə"), ("yaklaşım", "yanaşma"),
    ("şekilde", "şəkildə"), ("edilebilir", "edilə bilər"), ("olabilir", "ola bilər"),
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True)
class QualityScores:
    azerbaijani: float
    english_preservation: float
    coverage: float
    turkish_hits: tuple[str, ...] = ()
    missing_terms: tuple[str, ...] = ()
    missing_english: tuple[str, ...] = ()

    @property
    def overall(self) -> float:
        """Weighted blend, language quality counting double."""
        return round(
            (2 * self.azerbaijani + self.english_preservation + 2 * self.coverage) / 5.0, 3
        )


def score_azerbaijani(text: str) -> tuple[float, tuple[str, ...]]:
    """Score 0..1 for "this reads as Azerbaijani, not Turkish"."""
    if not text or not text.strip():
        return 0.0, ()
    normalized = unicodedata.normalize("NFC", text)
    lowered = az_lower(normalized)
    words = _WORD_RE.findall(lowered)
    if not words:
        return 0.0, ()

    # 1. Density of the Azerbaijani-only letter across the text.
    az_letters = sum(normalized.count(ch) for ch in AZ_ONLY_LETTERS)
    letters = sum(1 for ch in normalized if ch.isalpha())
    density = az_letters / letters if letters else 0.0
    # Natural Azerbaijani prose runs around 6-10% "ə"; treat 4% as full marks.
    density_score = min(1.0, density / 0.04)

    # 2. Penalty for Turkish function words.
    word_set = set(words)
    hits = tuple(
        turkish for turkish, azeri in TURKISH_MARKERS
        if (turkish in word_set or f" {turkish} " in f" {lowered} ") and azeri not in word_set
    )
    penalty = min(1.0, len(hits) / 6.0)

    return round(max(0.0, density_score * (1.0 - penalty)), 3), hits


def score_english_preservation(text: str, english_terms: tuple[str, ...]) -> tuple[float, tuple[str, ...]]:
    """Fraction of required English terms that survived untranslated."""
    if not english_terms:
        return 1.0, ()
    lowered = az_lower(text or "")
    missing = tuple(term for term in english_terms if az_lower(term) not in lowered)
    return round(1.0 - len(missing) / len(english_terms), 3), missing


def score_coverage(
    text: str, expected_terms: tuple[tuple[str, ...], ...]
) -> tuple[float, tuple[str, ...]]:
    """Fraction of expected concept groups mentioned at least once."""
    if not expected_terms:
        return 1.0, ()
    lowered = az_lower(text or "")
    missing: list[str] = []
    hit = 0
    for group in expected_terms:
        if any(az_lower(option) in lowered for option in group):
            hit += 1
        else:
            missing.append("/".join(group))
    return round(hit / len(expected_terms), 3), tuple(missing)


def score_answer(
    text: str,
    expected_terms: tuple[tuple[str, ...], ...] = (),
    english_terms: tuple[str, ...] = (),
) -> QualityScores:
    azerbaijani, turkish_hits = score_azerbaijani(text)
    english, missing_english = score_english_preservation(text, english_terms)
    coverage, missing_terms = score_coverage(text, expected_terms)
    return QualityScores(
        azerbaijani=azerbaijani,
        english_preservation=english,
        coverage=coverage,
        turkish_hits=turkish_hits,
        missing_terms=missing_terms,
        missing_english=missing_english,
    )


def preserves_azerbaijani_characters(original: str, transcript: str) -> bool:
    """Every Azerbaijani-specific character in `original` survived transcription."""
    specials = set("əƏıİğĞşŞçÇöÖüÜ")
    needed = {ch for ch in original if ch in specials}
    present = {ch for ch in transcript if ch in specials}
    return needed.issubset(present)
