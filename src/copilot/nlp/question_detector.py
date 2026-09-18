"""Turn committed transcript segments into interview questions.

Three jobs, in order:

1. **Coalesce.**  The recogniser commits on silence, so one spoken question can
   arrive as two or three segments.  A segment that does not end in terminal
   punctuation is held for a short window and joined to whatever follows.
2. **Classify.**  Decide whether the coalesced text deserves an answer.  That
   is a wider net than "is this a question": interviewers ask for things far
   more often than they ask questions - "Mənə layihənizdən bəhs edin",
   "Gəlin RAG-dan danışaq", "Təcrübənizi eşitmək istərdim".  Azerbaijani also
   marks real questions with interrogative words (nə, necə, hansı...), the
   enclitic particle -mı/-mi/-mu/-mü and conditional verb endings
   (-ardınız/-ərdiniz), not only with a question mark - which the recogniser
   often omits anyway.  `sensitivity` chooses how wide the net is.
3. **Deduplicate.**  Partial re-commits and interviewer repetitions must not
   trigger a second answer.

The class is driven by an explicit clock so the coalescing window can be tested
without sleeping.
"""
from __future__ import annotations

import re
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ..config import QuestionSettings

# -- Azerbaijani-aware normalisation --------------------------------------
# Azerbaijani has a dotted/dotless i distinction: I -> ı and İ -> i.
_AZ_LOWER = str.maketrans({"I": "ı", "İ": "i", "Ə": "ə", "Ğ": "ğ", "Ş": "ş", "Ç": "ç", "Ö": "ö", "Ü": "ü"})

_PUNCT_RE = re.compile(r"[^\w\s\-]", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)


def az_lower(text: str) -> str:
    """Lowercase that respects the Azerbaijani dotted/dotless i."""
    return text.translate(_AZ_LOWER).lower()


def normalize(text: str) -> str:
    """Casefolded, punctuation-free, whitespace-collapsed form for comparison."""
    text = unicodedata.normalize("NFC", text or "")
    text = az_lower(text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_english(text: str) -> str:
    """Like `normalize`, but with ordinary lowercasing.

    `az_lower` maps "I" to the dotless "ı", which is right for Azerbaijani and
    wrong for English - it silently breaks every phrase starting with "I".
    English cues are matched against this form instead.
    """
    text = unicodedata.normalize("NFC", text or "").lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


# -- Question cues ---------------------------------------------------------
AZ_INTERROGATIVES: frozenset[str] = frozenset(
    {
        "nə", "nədir", "nədən", "nəyə", "nəyi", "nəyin", "nələr", "nəylə",
        "necə", "niyə", "nəçün", "hansı", "hansını", "hansılar",
        "harada", "haradan", "hara", "haraya", "haradadır",
        "kim", "kimin", "kimə", "kimi̇", "kimlər",
        "neçə", "neçəyə", "nəqədər",
        "fərq", "fərqi", "fərqlər",
    }
)

#: Multi-word interrogatives / interview prompts.
AZ_PHRASES: tuple[str, ...] = (
    "nə üçün", "nə vaxt", "nə zaman", "nə qədər", "nə cür",
    "izah edin", "izah edərdiniz", "izah edə bilərsiniz",
    "danışın", "danışa bilərsiniz", "söyləyin", "deyin", "deyə bilərsiniz",
    "açıqlayın", "təsvir edin", "müqayisə edin", "nümunə verin",
    "misal çəkin", "sadalayın", "göstərin", "bölüşün", "izah et",
    "fikriniz nədir", "necə izah", "necə edərdiniz", "necə qurulur",
    "necə işləyir", "necə ölçər", "necə təmin", "necə azaldar",
    "təcrübəniz var", "işləmisiniz", "tanışsınız",
)

ENG_INTERROGATIVES: frozenset[str] = frozenset(
    {"what", "how", "why", "which", "where", "who", "whom", "when", "whose"}
)

ENG_PHRASES: tuple[str, ...] = (
    "can you", "could you", "would you", "do you", "did you", "have you",
    "are you", "tell me", "walk me through", "explain", "describe",
    "compare", "give an example", "what if", "how would",
)

#: Verb endings that in practice mark an Azerbaijani interview question:
#: the -mI question particle and the conditional "would you" forms.
_AZ_QUESTION_SUFFIX_RE = re.compile(
    r"(?:"
    r"m[ıiuü](?:sınız|siniz|sunuz|sünüz|dır|dir|dur|dür|ş)?"   # -mı particle
    r"|[ae]rdiniz|[ae]rdınız|ardınız|ərdiniz"                    # would you ...
    r"|[ıiuü]rsunuz|[ıiuü]rsünüz|ırsınız|irsiniz"               # do you ...
    r"|m[ıiuü]s[ıi]n[ıi]z"                                       # have you ...
    r")$",
    re.UNICODE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[?!])\s+", re.UNICODE)
TERMINAL_PUNCT = "?!."

#: Pleasantries that share an ending with real questions - "xoş gəlmisiniz"
#: (welcome) looks exactly like the -mIsInIz question form but never is one.
SMALL_TALK: tuple[str, ...] = (
    "xoş gəlmisiniz", "xoş gördük", "sağ olun", "çox sağ olun", "təşəkkür",
    "salam", "günaydın", "xoş gəldiniz", "buyurun", "welcome", "thank you",
    "thanks", "good morning", "nice to meet",
)

# -- sensitivity -----------------------------------------------------------
STRICT = "strict"
NORMAL = "normal"
BROAD = "broad"
SENSITIVITY_LEVELS: tuple[str, ...] = (STRICT, NORMAL, BROAD)

#: Invitations and requests that expect an answer without asking a question.
#: An interviewer says "tell me about X" far more often than "what is X?".
REQUEST_PHRASES: tuple[str, ...] = (
    # let's / we will
    "gəlin", "danışaq", "keçək", "keçə bilərik", "müzakirə edək",
    "söhbət edək", "baxaq", "başlayaq",
    # tell me / share
    "bəhs ed", "bəhs et", "paylaşın", "paylaşa bilərsiniz", "söyləyin",
    "danışın", "danışa bilərsiniz", "deyin", "deyə bilərsiniz",
    "izah ed", "açıqla", "təsvir ed", "müqayisə ed", "sadalayın",
    "nümunə ver", "misal çək", "göstərin", "məlumat ver", "məlumat verin",
    "qısaca", "təqdim ed", "danış",
    # I would like to hear / understand
    "istərdim", "istəyirəm", "maraqlanıram", "maraqlıdır",
    "eşitmək", "bilmək istər", "anlamaq istər", "öyrənmək istər",
    # framing an item
    "növbəti sual", "növbəti mövzu", "son sual", "bir sual", "ikinci sual",
    "sualım", "mövzusuna", "haqqında danış", "barədə danış",
    # opinion
    "sizcə", "fikriniz", "fikirlərinizi", "nə düşünürsünüz",
)

ENG_REQUEST_PHRASES: tuple[str, ...] = (
    "tell me", "walk me through", "let us talk", "let's talk", "talk about",
    "i would like", "i'd like", "i want to hear", "curious about",
    "next question", "next topic", "your thoughts", "share your",
    "give me an example", "describe", "explain", "compare", "elaborate",
    "move on to", "touch on",
)

#: Acknowledgements and filler the interviewer says between real prompts.
#: These must never trigger an answer, at any sensitivity.
BACKCHANNEL: frozenset[str] = frozenset(
    {
        "bəli", "hə", "yox", "xeyr", "yaxşı", "aydındır", "aydın", "başa düşdüm",
        "doğrudur", "düzdür", "razıyam", "maraqlı", "maraqlıdır", "maraqlıdı",
        "əla", "mükəmməl", "təşəkkür", "anladım", "oldu", "bəlli",
        "təşəkkürlər", "sağol", "davam", "davam edin", "davam edək", "bir dəqiqə",
        "hmm", "ok", "okay", "yes", "no", "right", "great", "perfect", "i see",
        "got it", "understood", "sure", "thanks", "mhm", "uh", "um",
    }
)


@dataclass(frozen=True)
class DetectedQuestion:
    """A finalised interview question, ready to be answered."""

    text: str
    detected_at: float
    is_explicit: bool          # ended with an explicit question mark
    reason: str                # which cue fired, for the developer panel

    @property
    def normalized(self) -> str:
        return normalize(self.text)


def is_backchannel(text: str) -> bool:
    """Acknowledgements and filler that never deserve an answer."""
    norm = normalize(text)
    if not norm:
        return True
    words = norm.split()
    if len(words) > 5:
        return False
    # "Bəli, aydındır" - every word is an acknowledgement.
    return all(word in BACKCHANNEL for word in words)


#: Words that carry no topic of their own.  "Let us move to the next subject"
#: is a transition, not something to answer; "let us talk about RAG" is.
_GENERIC_WORDS: frozenset[str] = frozenset(
    {
        # Azerbaijani filler and pure transitions.  Deliberately narrow: words
        # like "fikirlərinizi" (your thoughts) are the *subject* of a request,
        # not filler, so they must stay out of this set.
        "indi", "bir", "az", "da", "də", "isə", "ki", "amma", "lakin", "ancaq",
        "həm", "sonra", "əvvəl", "yaxşı", "bəli", "olaraq", "üzrə",
        "növbəti", "mövzu", "mövzuya", "mövzusuna", "mövzunu", "sual", "sualı",
        "suala", "ikinci", "son", "birinci", "keçək", "gəlin", "danışaq",
        "baxaq", "başlayaq", "edək", "haqqında", "barədə", "üçün", "sizin",
        "mənə", "ondan", "bundan", "onu", "bunu",
        "hissə", "hissəyə", "hissəsinə", "bölmə", "bölməyə", "mərhələ",
        "mərhələyə", "texniki", "part", "section", "stage",
        # English equivalents
        "now", "next", "topic", "subject", "question", "second", "last",
        "first", "let", "lets", "us", "about", "the", "and", "to", "of",
        "on", "in", "for", "with", "this", "that", "it", "is", "are",
        "move", "then", "okay", "well", "so",
    }
)


def _has_topic(text: str) -> bool:
    """Does the segment name anything beyond the request itself?

    "Gəlin RAG-dan danışaq" has a topic; "Yaxşı, növbəti mövzuya keçək" does
    not, and answering the latter would just produce noise.
    """
    for token in normalize(text).split():
        cleaned = token.strip("-")
        if len(cleaned) >= 3 and cleaned not in _GENERIC_WORDS:
            return True
    return False


def request_reason(text: str) -> str | None:
    """Cue that makes `text` an *invitation to answer* rather than a question.

    Interviewers ask for things far more often than they ask questions:
    "tell me about X", "let's talk about Y", "I'd like to hear about Z".
    A request with no topic in it is a transition, not a prompt.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    norm = normalize(raw)
    english = normalize_english(raw)
    if not norm:
        return None

    matched = next((p for p in REQUEST_PHRASES if p in norm), None)
    kind = "az"
    if matched is None:
        matched = next((p for p in ENG_REQUEST_PHRASES if p in english), None)
        kind = "en"
    if matched is None:
        return None
    if not _has_topic(raw):
        return None
    return f"{kind} request: {matched}"


def question_reason(text: str) -> str | None:
    """Return the cue that makes `text` a question, or None if it is not one."""
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.endswith("?"):
        return "question mark"

    norm = normalize(raw)
    if not norm:
        return None
    words = norm.split()

    for phrase in AZ_PHRASES:
        if phrase in norm:
            return f"az phrase: {phrase}"
    english = normalize_english(raw)
    for phrase in ENG_PHRASES:
        if phrase in english:
            return f"en phrase: {phrase}"

    for word in words:
        if word in AZ_INTERROGATIVES:
            return f"az interrogative: {word}"
    # English interrogatives only count at the start, so "what" inside a
    # statement ("that is what we did") does not trigger.
    english_words = english.split()
    if english_words and english_words[0] in ENG_INTERROGATIVES:
        return f"en interrogative: {english_words[0]}"

    # The -mI particle attaches to the final word of the clause.  This is the
    # weakest cue - the perfect tense shares the same ending ("xoş gəlmisiniz")
    # - so it only fires when nothing marks the text as a finished statement.
    if raw.rstrip().endswith("."):
        return None
    if any(phrase in norm for phrase in SMALL_TALK):
        return None
    if words and _AZ_QUESTION_SUFFIX_RE.search(words[-1]):
        return f"az verb ending: {words[-1]}"
    return None


def is_question(text: str) -> bool:
    return question_reason(text) is not None


def prompt_reason(text: str, sensitivity: str = NORMAL, min_broad_words: int = 4) -> str | None:
    """Should this transcript segment be answered, and why?

    Three levels, because the right trade-off depends on the interview:

    * ``strict`` - grammatical questions only.  Fewest false answers, but misses
      "Mənə layihənizdən bəhs edin".
    * ``normal`` (default) - questions *and* requests/invitations.  This is how
      interviewers actually speak.
    * ``broad`` - anything substantial the interviewer says that is not an
      acknowledgement.  Nothing is missed; expect answers to remarks too.

    Backchannel ("Bəli", "Aydındır", "Yaxşı, davam edək") is never answered at
    any level.
    """
    raw = (text or "").strip()
    if not raw or is_backchannel(raw):
        return None

    reason = question_reason(raw)
    if reason is not None:
        return reason
    if sensitivity == STRICT:
        return None

    reason = request_reason(raw)
    if reason is not None:
        return reason
    if sensitivity != BROAD:
        return None

    # Broad: answer whatever the interviewer said.  The only things filtered
    # out are acknowledgements (already handled above) and pure pleasantries,
    # because answering "Salam, xoş gəlmisiniz" helps nobody.
    words = normalize(raw).split()
    if len(words) < min_broad_words:
        return None
    if any(phrase in normalize(raw) for phrase in SMALL_TALK):
        return None
    return "broad: interviewer turn"


def looks_complete(text: str) -> bool:
    """Whether a committed segment already ends a sentence."""
    stripped = (text or "").rstrip()
    return bool(stripped) and stripped[-1] in TERMINAL_PUNCT


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


class QuestionDetector:
    """Stateful coalescing + classification + deduplication."""

    def __init__(self, settings: QuestionSettings | None = None) -> None:
        self.settings = settings or QuestionSettings()
        self._buffer: str = ""
        self._buffer_started_at: float | None = None
        self._deadline: float | None = None
        self._recent: deque[str] = deque(maxlen=max(1, self.settings.duplicate_memory))
        self.skipped_duplicates = 0
        self.skipped_backchannel = 0
        self._last_commit_at: float | None = None

    # -- state ------------------------------------------------------------
    @property
    def pending_text(self) -> str:
        return self._buffer

    @property
    def deadline(self) -> float | None:
        """Monotonic time at which the buffer should be flushed, if any."""
        return self._deadline

    def reset(self) -> None:
        self._buffer = ""
        self._buffer_started_at = None
        self._deadline = None
        self._last_commit_at = None

    def clear_history(self) -> None:
        """Forget everything - including the turn in progress.

        The retained turn buffer is part of what suppresses a repeat, so
        clearing the history without clearing it would leave repeats blocked.
        """
        self._recent.clear()
        self.skipped_duplicates = 0
        self.skipped_backchannel = 0
        self.reset()

    # -- driving ----------------------------------------------------------
    def add_committed(self, text: str, now: float | None = None) -> list[DetectedQuestion]:
        """Feed a committed transcript segment; returns prompts ready now.

        The decision is asymmetric, and that asymmetry is the whole point:

        * the buffer **already reads as a prompt** -> fire immediately, so a
          complete question costs no extra latency; but
        * it **does not** -> hold it, because the rest of the sentence may still
          be coming.  Real commits arrive seconds apart (the recogniser waits
          out its own silence threshold and then needs time to transcribe), so
          the hold has to outlast that gap or a sentence split by a pause is
          judged on half of itself and thrown away.
        """
        now = time.monotonic() if now is None else now
        segment = (text or "").strip()
        if not segment:
            return []
        # "Bəli." between two halves of a question must not end up glued into
        # it, and must not reset the hold on its own.
        if is_backchannel(segment):
            self.skipped_backchannel += 1
            return []

        # A long gap means the previous turn is over and this starts a new one.
        if (
            self._last_commit_at is not None
            and now - self._last_commit_at > self.settings.turn_gap_ms / 1000.0
        ):
            self._buffer = ""
            self._buffer_started_at = None
        self._last_commit_at = now

        if self._buffer:
            # The recogniser re-sends text it has already committed.  Appending
            # it would build "RAG nədir? RAG nədir?", which is not a duplicate
            # of anything and would therefore be answered again.
            if self._repeats_buffer(segment):
                self.skipped_duplicates += 1
                return []
            self._buffer = f"{self._buffer} {segment}".strip()
        else:
            self._buffer = segment
            self._buffer_started_at = now

        # Whatever we have so far is answerable: do not make the user wait.
        # The buffer is kept, not cleared, so that if the interviewer carries
        # on, the next commit re-emits the *whole* turn.  That longer version
        # supersedes the answer already streaming instead of starting a second
        # unrelated one - which is what makes "answer every utterance" usable.
        if prompt_reason(self._buffer, self.settings.sensitivity) is not None:
            keep = len(self._buffer) < self.settings.max_buffer_chars
            return self._emit(now, keep_buffer=keep)

        # Not answerable yet.  Hold it - but never let it grow without bound.
        if len(self._buffer) >= self.settings.max_buffer_chars:
            return self._emit(now)

        self._deadline = now + self.settings.merge_window_ms / 1000.0
        return []

    def _repeats_buffer(self, segment: str) -> bool:
        """Is `segment` something the buffer already ends with?"""
        norm_segment = normalize(segment)
        norm_buffer = normalize(self._buffer)
        if not norm_segment or not norm_buffer:
            return False
        if norm_buffer.endswith(norm_segment):
            return True
        tail = norm_buffer[-max(len(norm_segment), 1) :]
        return similarity(tail, norm_segment) >= self.settings.duplicate_similarity

    def flush_due(self, now: float | None = None) -> list[DetectedQuestion]:
        """Flush the buffer if its coalescing window has elapsed."""
        now = time.monotonic() if now is None else now
        if self._deadline is not None and now >= self._deadline:
            return self._flush(now)
        return []

    def force_flush(self, now: float | None = None) -> list[DetectedQuestion]:
        """Flush immediately - used when the user stops listening."""
        return self._flush(time.monotonic() if now is None else now)

    # -- internals --------------------------------------------------------
    def _flush(self, now: float) -> list[DetectedQuestion]:
        return self._emit(now)

    def _emit(self, now: float, keep_buffer: bool = False) -> list[DetectedQuestion]:
        """Turn the buffer into prompts.

        `keep_buffer` leaves the turn text in place so a continuation re-emits
        the whole turn; otherwise the buffer is cleared (end of turn, or giving
        up on text that never became answerable).
        """
        buffered = self._buffer.strip()
        started = self._buffer_started_at
        if not keep_buffer:
            self._buffer = ""
            self._buffer_started_at = None
        self._deadline = None
        if not buffered:
            return []

        out: list[DetectedQuestion] = []
        for sentence in self._split_sentences(buffered):
            candidate = sentence.strip()
            if len(candidate) < self.settings.min_question_chars:
                continue
            reason = prompt_reason(candidate, self.settings.sensitivity)
            if reason is None:
                continue
            if self._is_duplicate(candidate):
                self.skipped_duplicates += 1
                continue
            self._remember(candidate)
            out.append(
                DetectedQuestion(
                    text=candidate,
                    detected_at=started if started is not None else now,
                    is_explicit=candidate.endswith("?"),
                    reason=reason,
                )
            )
        return out

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split on '?'/'!' boundaries so two questions in one commit both fire."""
        parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
        return parts or [text]

    def _is_duplicate(self, candidate: str) -> bool:
        norm = normalize(candidate)
        if not norm:
            return True
        threshold = self.settings.duplicate_similarity
        for previous in self._recent:
            if norm == previous:
                return True
            # A re-commit of something already handled is a duplicate...
            if len(norm) >= 12 and norm in previous:
                return True
            # ...but text that *extends* what we answered is the interviewer
            # carrying on.  It must be allowed through so the fuller turn
            # supersedes the answer already streaming.
            if len(norm) >= 12 and previous in norm:
                continue
            if similarity(norm, previous) >= threshold:
                return True
        return False

    def _remember(self, candidate: str) -> None:
        self._recent.append(normalize(candidate))
