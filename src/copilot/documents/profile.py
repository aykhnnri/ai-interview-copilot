"""The candidate CV and job description, as the model sees them.

`render_full()` is what the answer path uses: the complete documents go into the
model's standing instructions, so it already knows the candidate and never has
to search for the relevant experience while a question is waiting.  The block is
byte-identical on every request, which also makes it a cacheable prompt prefix.

The sectioning and IDF scoring below still exist and are still tested - they
drive the developer panel, and `render_full()` falls back on nothing, so the
retrieval behaviour can be restored without rebuilding it.

Nothing here invents content: every character is a verbatim slice of an uploaded
document.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from ..nlp.question_detector import az_lower, normalize
from .extract import ExtractedDocument

log = logging.getLogger(__name__)

CV = "cv"
JOB_DESCRIPTION = "job_description"

#: Backstop against a pathological upload filling the context window.  A normal
#: CV plus job advert is a few thousand characters, so this should never bite.
PROFILE_MAX_CHARS = 30_000

#: Headings that start a new section, in Azerbaijani, Turkish-adjacent and English.
_HEADING_WORDS = (
    "təhsil", "education", "təcrübə", "iş təcrübəsi", "experience",
    "work experience", "professional experience", "bacarıqlar", "skills",
    "technical skills", "layihələr", "projects", "sertifikat", "certificates",
    "certifications", "dillər", "languages", "haqqımda", "about", "summary",
    "profile", "profil", "xülasə", "əlaqə", "contact", "publications",
    "nailiyyətlər", "achievements", "responsibilities", "requirements",
    "tələblər", "vəzifə", "öhdəliklər", "qualifications", "about the role",
    "what you will do", "what we offer", "təklif edirik",
)

_TOKEN_RE = re.compile(r"[\w\-\+#\.]{2,}", re.UNICODE)

_STOPWORDS = frozenset(
    {
        # Azerbaijani
        "və", "ilə", "üçün", "bir", "bu", "o", "da", "də", "ki", "olan", "olaraq",
        "kimi", "daha", "çox", "az", "amma", "ancaq", "həm", "nə", "necə", "hansı",
        "var", "yox", "edir", "etmək", "olub", "idi", "dir", "dır", "mən", "siz",
        "biz", "onlar", "öz", "hər", "bütün", "sonra", "əvvəl", "zaman", "vaxt",
        "arasında", "üzrə", "görə", "haqqında", "necədir", "nədir", "edərdiniz",
        "edirsiniz", "olardı", "sizin", "mənim",
        # English
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "is", "are", "was", "were", "be", "been", "as", "at", "by", "from",
        "that", "this", "it", "its", "we", "you", "your", "our", "i", "my",
        "how", "what", "why", "which", "would", "could", "can", "do", "does",
        "have", "has", "will", "about", "into", "than", "then", "there",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase content tokens, Azerbaijani-aware, stopwords removed."""
    lowered = az_lower(text or "")
    tokens = [t.strip(".-+#") for t in _TOKEN_RE.findall(lowered)]
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) >= 2]


@dataclass
class Section:
    """One headed slice of an uploaded document."""

    source: str                      # CV | JOB_DESCRIPTION
    heading: str
    text: str
    tokens: frozenset[str] = field(default_factory=frozenset)

    @property
    def label(self) -> str:
        return f"{'CV' if self.source == CV else 'Vakansiya'} / {self.heading}"

    def render(self, max_chars: int = 900) -> str:
        body = self.text if len(self.text) <= max_chars else self.text[:max_chars].rsplit(" ", 1)[0] + "…"
        return f"[{self.label}]\n{body}"


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    # "Dillər: Python, SQL" is a key/value line, not a section heading.
    head, sep, tail = stripped.partition(":")
    if sep and len(tail.strip()) > 2:
        return False
    lowered = az_lower(stripped).strip(":").strip()
    if lowered in _HEADING_WORDS:
        return True
    if any(lowered.startswith(word) for word in _HEADING_WORDS) and len(lowered) <= 32:
        return True
    # A short ALL-CAPS line is a heading in most CV templates.
    letters = [c for c in stripped if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters) and len(stripped) <= 40


def split_sections(text: str, source: str, max_chars: int = 1200) -> list[Section]:
    """Split a document into headed sections, then cap oversized ones."""
    lines = (text or "").split("\n")
    blocks: list[tuple[str, list[str]]] = []
    heading = "Ümumi"
    body: list[str] = []

    for line in lines:
        if _is_heading(line):
            if any(part.strip() for part in body):
                blocks.append((heading, body))
            heading = line.strip().strip(":") or "Ümumi"
            body = []
        else:
            body.append(line)
    if any(part.strip() for part in body):
        blocks.append((heading, body))

    sections: list[Section] = []
    for block_heading, block_lines in blocks:
        joined = "\n".join(block_lines).strip()
        if not joined:
            continue
        for piece in _chunk_text(joined, max_chars):
            sections.append(
                Section(
                    source=source,
                    heading=block_heading,
                    text=piece,
                    tokens=frozenset(tokenize(f"{block_heading} {piece}")),
                )
            )
    return sections


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries, keeping pieces under `max_chars`."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if size + len(paragraph) > max_chars and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


@dataclass
class CandidateProfile:
    """The candidate documents, indexed for per-question retrieval."""

    cv: ExtractedDocument | None = None
    job_description: ExtractedDocument | None = None
    sections: list[Section] = field(default_factory=list)
    truncated: bool = False
    _idf: dict[str, float] = field(default_factory=dict, repr=False)

    # -- construction -----------------------------------------------------
    @classmethod
    def build(
        cls,
        cv: ExtractedDocument | None = None,
        job_description: ExtractedDocument | None = None,
    ) -> "CandidateProfile":
        sections: list[Section] = []
        if cv is not None:
            sections.extend(split_sections(cv.text, CV))
        if job_description is not None:
            sections.extend(split_sections(job_description.text, JOB_DESCRIPTION))
        profile = cls(cv=cv, job_description=job_description, sections=sections)
        profile._build_idf()
        return profile

    def _build_idf(self) -> None:
        total = len(self.sections)
        self._idf = {}
        if not total:
            return
        counts: dict[str, int] = {}
        for section in self.sections:
            for token in section.tokens:
                counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            self._idf[token] = math.log((total + 1) / (count + 0.5)) + 1.0

    # -- introspection ----------------------------------------------------
    @property
    def has_documents(self) -> bool:
        return bool(self.sections)

    @property
    def headline(self) -> str:
        """First couple of CV lines - name and title - always worth including."""
        if self.cv is None:
            return ""
        lines = [line.strip() for line in self.cv.text.split("\n") if line.strip()]
        return " · ".join(lines[:3])[:200]

    def describe(self) -> str:
        parts = []
        if self.cv is not None:
            parts.append(f"CV: {self.cv.name} ({self.cv.char_count:,} simvol)")
        if self.job_description is not None:
            parts.append(
                f"Vakansiya: {self.job_description.name} "
                f"({self.job_description.char_count:,} simvol)"
            )
        return " | ".join(parts) if parts else "Sənəd yüklənməyib"

    # -- retrieval --------------------------------------------------------
    def score_section(self, section: Section, query_tokens: list[str]) -> float:
        if not query_tokens or not section.tokens:
            return 0.0
        score = 0.0
        seen: set[str] = set()
        for token in query_tokens:
            if token in seen:
                continue
            seen.add(token)
            if token in section.tokens:
                score += self._idf.get(token, 1.0)
            else:
                # Partial credit for shared stems, which matters for a heavily
                # suffixed language: "modelin" should still match "model".
                if len(token) >= 5:
                    for candidate in section.tokens:
                        if len(candidate) >= 5 and (
                            candidate.startswith(token[:5]) or token.startswith(candidate[:5])
                        ):
                            score += 0.4 * self._idf.get(candidate, 1.0)
                            break
        # Normalise so a long section does not win on length alone.
        return score / math.sqrt(len(section.tokens) + 8)

    def select_context(
        self, question: str, max_sections: int = 4, min_score: float = 0.05
    ) -> list[Section]:
        """The most relevant document sections for one question."""
        if not self.sections:
            return []
        query_tokens = tokenize(question)
        if not query_tokens:
            return []
        scored = [
            (self.score_section(section, query_tokens), index, section)
            for index, section in enumerate(self.sections)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = [section for score, _, section in scored if score > min_score][:max_sections]

        # Always keep at least one job-description section in view when one was
        # uploaded, so answers stay aimed at the role being interviewed for.
        if self.job_description is not None and not any(s.source == JOB_DESCRIPTION for s in chosen):
            for score, _, section in scored:
                if section.source == JOB_DESCRIPTION and score > 0:
                    chosen = chosen[: max(1, max_sections - 1)] + [section]
                    break
        return chosen

    def render_context(self, question: str, max_sections: int = 4) -> str:
        """The most relevant sections only.

        Retained for the developer panel and for anyone who wants the retrieval
        behaviour back; the answer path uses `render_full()` instead, so the
        model is given the whole profile up front rather than searching it per
        question.
        """
        sections = self.select_context(question, max_sections=max_sections)
        if not sections and not self.headline:
            return ""
        parts: list[str] = []
        if self.headline:
            parts.append(f"[Namizəd]\n{self.headline}")
        parts.extend(section.render() for section in sections)
        return "\n\n".join(parts)

    def render_full(self, max_chars: int = PROFILE_MAX_CHARS) -> str:
        """The complete CV and job description, for the instruction layer.

        The model is given everything once, as part of its standing
        instructions, so it already knows the candidate and never has to search
        for the relevant experience at question time.  Because the block is
        identical on every request it also sits in the cacheable prompt prefix.

        `max_chars` is a backstop against a pathological upload, not a working
        limit - a normal CV plus advert is a few thousand characters.  If it
        bites, it is logged and reported through `truncated`.
        """
        parts: list[str] = []
        if self.cv is not None:
            parts.append(f"=== NAMİZƏDİN CV-si ({self.cv.name}) ===\n{self.cv.text.strip()}")
        if self.job_description is not None:
            parts.append(
                f"=== VAKANSİYA TƏSVİRİ ({self.job_description.name}) ===\n"
                f"{self.job_description.text.strip()}"
            )
        text = "\n\n".join(parts)

        self.truncated = len(text) > max_chars
        if self.truncated:
            log.warning(
                "Candidate documents are %d characters; sending the first %d. "
                "Trim the CV if the tail matters.",
                len(text), max_chars,
            )
            text = text[:max_chars].rsplit("\n", 1)[0] + "\n…"
        return text

    @property
    def full_context_chars(self) -> int:
        return len(self.render_full())


def normalized_question_key(question: str) -> str:
    """Stable key for caching answers per question."""
    return normalize(question)
