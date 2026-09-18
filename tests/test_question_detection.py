"""Question detection, coalescing of split segments, and duplicate prevention."""
from __future__ import annotations

import pytest

from copilot.config import QuestionSettings
from copilot.nlp.question_detector import (
    QuestionDetector,
    az_lower,
    is_question,
    looks_complete,
    normalize,
    question_reason,
)

#: The ten questions the specification requires to be handled.
SPEC_QUESTIONS = [
    "RAG nədir?",
    "Fine-tuning və RAG arasında əsas fərq nədir?",
    "Multi-agent sistemlərdə agentlər arasında kommunikasiya necə qurulur?",
    "Production mühitində LLM hallucination problemini necə azaldarsınız?",
    "Python-da async və await nədir?",
    "Vector database necə işləyir?",
    "LLM tətbiqlərində latency-ni necə optimizasiya edərdiniz?",
    "Modelin performansını hansı metriklərlə qiymətləndirərdiniz?",
    "Sisteminizdə məlumatların təhlükəsizliyini necə təmin edərdiniz?",
    "RAG pipeline-ın retrieval quality-sini necə ölçərdiniz?",
]


# =========================================================================
# Normalisation
# =========================================================================
def test_az_lower_handles_the_dotted_and_dotless_i():
    """Azerbaijani maps I->ı and İ->i, unlike the default Unicode lowercase."""
    assert az_lower("IŞIQ") == "ışıq"
    assert az_lower("İSTANBUL") == "istanbul"


def test_normalize_strips_punctuation_and_case():
    assert normalize("RAG  nədir?!") == "rag nədir"


def test_normalize_preserves_azerbaijani_letters():
    assert normalize("Çətinlik, şübhə və öyrənmə!") == "çətinlik şübhə və öyrənmə"


# =========================================================================
# Classification
# =========================================================================
@pytest.mark.parametrize("question", SPEC_QUESTIONS)
def test_all_specified_questions_are_detected(question):
    assert is_question(question), f"missed: {question}"


@pytest.mark.parametrize("question", [q.rstrip("?") for q in SPEC_QUESTIONS])
def test_specified_questions_are_detected_without_a_question_mark(question):
    """The recogniser often omits terminal punctuation, so cues must carry it."""
    assert is_question(question), f"missed without '?': {question}"


@pytest.mark.parametrize(
    "text",
    [
        "Özünüz haqqında danışın",
        "Bu layihədə hansı problemlərlə qarşılaşdınız",
        "Docker ilə təcrübəniz var",
        "Multi-agent sistemlərlə işləmisiniz?",
        "Komanda işini necə təşkil edərdiniz",
        "Tell me about your experience with LangGraph",
        "How would you reduce inference latency?",
        "What is a vector database",
    ],
)
def test_other_interview_prompts_are_detected(text):
    assert is_question(text)


@pytest.mark.parametrize(
    "text",
    [
        "Mən Bakıda yaşayıram.",
        "Bəli, düzdür.",
        "Salam, xoş gəlmisiniz.",
        "Biz FastAPI istifadə edirik.",
        "Yaxşı, davam edək.",
        "Təşəkkür edirəm.",
        "Good morning, thanks for joining.",
        "",
        "   ",
    ],
)
def test_statements_are_not_treated_as_questions(text):
    assert not is_question(text)


def test_question_reason_names_the_cue():
    assert question_reason("RAG nədir?") == "question mark"
    assert "interrogative" in (question_reason("Vector database nədir") or "")


def test_looks_complete():
    assert looks_complete("RAG nədir?")
    assert looks_complete("Bəli.")
    assert not looks_complete("Production mühitində RAG")
    assert not looks_complete("")


# =========================================================================
# Coalescing split questions
# =========================================================================
def test_a_split_question_is_rejoined_not_answered_twice():
    """The VAD commits on silence, so one question can arrive in two pieces."""
    detector = QuestionDetector(QuestionSettings(sensitivity="normal"))
    first = detector.add_committed("Production mühitində RAG pipeline-ın", now=0.0)
    assert first == [], "an incomplete fragment must not fire on its own"

    second = detector.add_committed(
        "retrieval quality-sini necə evaluate edərdiniz?", now=0.3
    )
    assert len(second) == 1
    assert second[0].text == (
        "Production mühitində RAG pipeline-ın retrieval quality-sini "
        "necə evaluate edərdiniz?"
    )


def test_a_complete_question_fires_immediately():
    """Latency matters: do not wait out the merge window when '?' already came."""
    detector = QuestionDetector()
    out = detector.add_committed("RAG nədir?", now=0.0)
    assert len(out) == 1
    assert detector.deadline is None


def test_the_merge_window_eventually_flushes():
    """A fragment that is not answerable alone is held, then given up on."""
    detector = QuestionDetector(QuestionSettings(merge_window_ms=700, sensitivity="normal"))
    detector.add_committed("Vector database", now=0.0)
    assert detector.flush_due(now=0.5) == []
    assert detector.flush_due(now=0.75) == []      # never became a prompt
    assert detector.pending_text == ""             # and was discarded


def test_an_answerable_segment_does_not_wait_for_the_window():
    detector = QuestionDetector(QuestionSettings(merge_window_ms=700, sensitivity="normal"))
    out = detector.add_committed("Vector database necə işləyir", now=0.0)
    assert [q.text for q in out] == ["Vector database necə işləyir"]
    assert detector.deadline is None


def test_force_flush_emits_a_held_prompt():
    """Stopping the session must not swallow a prompt still in the buffer."""
    detector = QuestionDetector(QuestionSettings(sensitivity="normal"))
    assert detector.add_committed("Python-da async", now=0.0) == []
    assert detector.add_committed("və await nədir", now=2.0)[0].text == (
        "Python-da async və await nədir"
    )

    detector2 = QuestionDetector(QuestionSettings(sensitivity="normal"))
    detector2.add_committed("Mənə deployment", now=0.0)
    detector2.add_committed("prosesinizdən bəhs", now=2.0)
    # Held, not yet answerable; stopping flushes what is there.
    assert detector2.pending_text
    detector2.force_flush(now=2.1)
    assert detector2.pending_text == ""


def test_two_questions_in_one_commit_both_fire():
    detector = QuestionDetector()
    out = detector.add_committed("RAG nədir? Vector database necə işləyir?", now=0.0)
    assert [q.text for q in out] == ["RAG nədir?", "Vector database necə işləyir?"]


def test_non_question_segments_are_discarded_at_normal_sensitivity():
    detector = QuestionDetector(QuestionSettings(sensitivity="normal"))
    assert detector.add_committed("Yaxşı, indi növbəti mövzuya keçək.", now=0.0) == []


def test_very_short_fragments_are_ignored():
    detector = QuestionDetector(QuestionSettings(min_question_chars=8))
    assert detector.add_committed("Nə?", now=0.0) == []


# =========================================================================
# Duplicate prevention
# =========================================================================
def test_an_identical_question_is_not_answered_twice():
    detector = QuestionDetector()
    assert len(detector.add_committed("RAG nədir?", now=0.0)) == 1
    assert detector.add_committed("RAG nədir?", now=5.0) == []
    assert detector.skipped_duplicates == 1


def test_near_identical_questions_are_deduplicated():
    """Re-transcription jitter must not trigger a second answer."""
    detector = QuestionDetector()
    detector.add_committed("Vector database necə işləyir?", now=0.0)
    assert detector.add_committed("Vector database necə işliyir?", now=4.0) == []


def test_a_recommitted_prefix_is_treated_as_the_same_question():
    detector = QuestionDetector()
    detector.add_committed("Multi-agent sistemlərdə kommunikasiya necə qurulur?", now=0.0)
    assert detector.add_committed("kommunikasiya necə qurulur?", now=2.0) == []


def test_a_genuinely_different_question_still_fires():
    detector = QuestionDetector()
    detector.add_committed("RAG nədir?", now=0.0)
    out = detector.add_committed("Fine-tuning nədir?", now=3.0)
    assert len(out) == 1


def test_duplicate_memory_is_bounded():
    detector = QuestionDetector(QuestionSettings(duplicate_memory=2))
    for text in ("RAG nədir?", "Docker nədir?", "Kubernetes nədir?"):
        detector.add_committed(text, now=0.0)
    # The first question has aged out of the window, so it can fire again.
    assert len(detector.add_committed("RAG nədir?", now=9.0)) == 1


def test_clear_history_allows_a_repeat():
    detector = QuestionDetector()
    detector.add_committed("RAG nədir?", now=0.0)
    detector.clear_history()
    assert len(detector.add_committed("RAG nədir?", now=1.0)) == 1


def test_detected_question_reports_its_cue_and_explicitness():
    detector = QuestionDetector()
    question = detector.add_committed("RAG nədir?", now=0.0)[0]
    assert question.is_explicit
    assert question.reason == "question mark"
    assert question.normalized == "rag nədir"


def test_azerbaijani_text_is_preserved_through_detection():
    detector = QuestionDetector()
    text = "Çətin şəraitdə işləyən sistemi necə qurardınız?"
    out = detector.add_committed(text, now=0.0)
    assert out[0].text == text


# =========================================================================
# Prompts that are not grammatical questions
# =========================================================================
INTERVIEW_REQUESTS = [
    "Gəlin RAG haqqında danışaq.",
    "Mənə LangGraph layihənizdən bəhs edin.",
    "Multi-agent təcrübənizi eşitmək istərdim.",
    "Növbəti sual: vector database.",
    "Sizin deployment prosesinizi anlamaq istəyirəm.",
    "İndi isə keçək fine-tuning mövzusuna.",
    "Bir az da monitorinqdən danışaq.",
    "Sizcə bu yanaşma düzgündür.",
    "Fikirlərinizi paylaşın.",
    "Bu barədə bir nümunə verin.",
    "Let us talk about Kubernetes.",
    "Walk me through your RAG pipeline.",
    "I would like to hear about your evaluation setup.",
]

BACKCHANNEL_NOISE = [
    "Bəli.", "Hə.", "Yaxşı, aydındır.", "Doğrudur.", "Təşəkkür edirəm.",
    "OK", "Great.", "I see.", "Salam, xoş gəlmisiniz.",
]


@pytest.mark.parametrize("text", INTERVIEW_REQUESTS)
def test_requests_are_answered_at_normal_sensitivity(text):
    """Interviewers ask for things more often than they ask questions."""
    from copilot.nlp.question_detector import NORMAL, prompt_reason

    assert prompt_reason(text, NORMAL), f"missed: {text}"


@pytest.mark.parametrize("text", INTERVIEW_REQUESTS)
def test_strict_sensitivity_keeps_only_real_questions(text):
    from copilot.nlp.question_detector import STRICT, prompt_reason

    # These are requests, not questions - strict mode is allowed to skip them,
    # and must skip the ones with no interrogative cue at all.
    if not is_question(text):
        assert prompt_reason(text, STRICT) is None


@pytest.mark.parametrize("text", BACKCHANNEL_NOISE)
@pytest.mark.parametrize("level", ["strict", "normal", "broad"])
def test_acknowledgements_never_trigger_an_answer(text, level):
    from copilot.nlp.question_detector import prompt_reason

    assert prompt_reason(text, level) is None, f"{level} fired on: {text}"


@pytest.mark.parametrize("text", SPEC_QUESTIONS)
@pytest.mark.parametrize("level", ["strict", "normal", "broad"])
def test_real_questions_fire_at_every_sensitivity(text, level):
    from copilot.nlp.question_detector import prompt_reason

    assert prompt_reason(text, level), f"{level} missed: {text}"


def test_broad_answers_a_plain_statement_that_normal_skips():
    from copilot.nlp.question_detector import BROAD, NORMAL, prompt_reason

    remark = "Sizin CV-nizdə LangGraph təcrübəsi yazılıb."
    assert prompt_reason(remark, NORMAL) is None
    assert prompt_reason(remark, BROAD)


def test_broad_still_ignores_very_short_fragments():
    from copilot.nlp.question_detector import BROAD, prompt_reason

    assert prompt_reason("Bir dəqiqə", BROAD) is None


def test_is_backchannel():
    from copilot.nlp.question_detector import is_backchannel

    assert is_backchannel("Bəli")
    assert is_backchannel("yaxşı, aydındır")
    assert not is_backchannel("Bəli, amma RAG pipeline-ı necə qurdunuz")


def test_detector_honours_the_configured_sensitivity():
    request = "Mənə multi-agent layihənizdən bəhs edin."

    strict = QuestionDetector(QuestionSettings(sensitivity="strict"))
    assert strict.add_committed(request, now=0.0) == []

    normal = QuestionDetector(QuestionSettings(sensitivity="normal"))
    fired = normal.add_committed(request, now=0.0)
    assert [q.text for q in fired] == [request]
    assert "request" in fired[0].reason


def test_default_sensitivity_is_broad():
    """A missed question costs more than an answer the user can ignore."""
    assert QuestionSettings().sensitivity == "broad"


def test_a_transition_with_no_topic_is_not_answered():
    """'Let us move to the next subject' expects nothing; answering is noise."""
    from copilot.nlp.question_detector import BROAD, NORMAL, prompt_reason

    for transition in (
        "Yaxşı, indi növbəti mövzuya keçək.",
        "Gəlin növbəti suala keçək.",
        "Let us move on to the next topic.",
    ):
        assert prompt_reason(transition, NORMAL) is None, transition


def test_the_same_phrase_with_a_topic_is_answered():
    from copilot.nlp.question_detector import NORMAL, prompt_reason

    assert prompt_reason("Gəlin RAG mövzusuna keçək.", NORMAL)
    assert prompt_reason("Let us move on to Kubernetes.", NORMAL)


def test_english_capital_i_is_not_mangled_into_dotless():
    """az_lower maps I -> ı, which would break every English 'I ...' phrase."""
    from copilot.nlp.question_detector import NORMAL, normalize_english, prompt_reason

    assert normalize_english("I would like") == "i would like"
    assert prompt_reason("I would like to hear about your RAG work.", NORMAL)


# =========================================================================
# A whole interview, end to end
# =========================================================================
#: (spoken line, should it be answered at "normal" sensitivity)
INTERVIEW_TRANSCRIPT = [
    ("Salam, xoş gəlmisiniz.", False),
    ("Bugün AI mühəndisliyi üzrə texniki müsahibəmizi keçirəcəyik.", False),
    ("Əvvəlcə özünüz haqqında qısaca danışın.", True),
    ("Yaxşı, aydındır.", False),
    ("İndi isə keçək texniki hissəyə.", False),
    ("Mənə RAG pipeline təcrübənizdən bəhs edin.", True),
    ("Bəli.", False),
    ("RAG pipeline-ın retrieval quality-sini necə ölçərdiniz?", True),
    ("Maraqlıdır.", False),
    ("Multi-agent sistemlərlə iş təcrübənizi eşitmək istərdim.", True),
    ("Təşəkkür edirəm.", False),
    ("Növbəti sual: Kubernetes deployment.", True),
    ("Hə, davam edin.", False),
    ("Son olaraq, komanda işi barədə fikirlərinizi paylaşın.", True),
    ("Çox sağ olun, müsahibəmiz bitdi.", False),
]


@pytest.mark.parametrize("line,should_answer", INTERVIEW_TRANSCRIPT)
def test_a_realistic_interview_is_classified_correctly(line, should_answer):
    from copilot.nlp.question_detector import NORMAL, prompt_reason

    fired = bool(prompt_reason(line, NORMAL))
    assert fired is should_answer, (
        f"{'fired on' if fired else 'missed'}: {line!r}"
    )


def test_sensitivity_widens_monotonically():
    """Anything strict answers, normal answers; anything normal answers, broad does."""
    from copilot.nlp.question_detector import BROAD, NORMAL, STRICT, prompt_reason

    lines = [line for line, _ in INTERVIEW_TRANSCRIPT]
    strict = {line for line in lines if prompt_reason(line, STRICT)}
    normal = {line for line in lines if prompt_reason(line, NORMAL)}
    broad = {line for line in lines if prompt_reason(line, BROAD)}

    assert strict <= normal <= broad
    assert len(strict) < len(normal) < len(broad)


# =========================================================================
# Coalescing at the cadence commits actually arrive
# =========================================================================
def test_a_sentence_split_by_a_real_pause_is_rejoined():
    """Measured: commits land ~1-2s apart, so a 700ms window could never work.

    The first half is not answerable on its own, so holding it long enough for
    the second half is the difference between an answer and silence.
    """
    detector = QuestionDetector(QuestionSettings(sensitivity="normal"))
    assert detector.add_committed("Mənə RAG pipeline", now=0.0) == []
    assert detector.flush_due(now=0.7) == [], "flushed before the rest could arrive"
    assert detector.flush_due(now=1.5) == []

    out = detector.add_committed("təcrübənizdən bəhs edin.", now=2.0)
    assert [q.text for q in out] == ["Mənə RAG pipeline təcrübənizdən bəhs edin."]


def test_a_three_way_split_is_rejoined():
    detector = QuestionDetector(QuestionSettings(sensitivity="normal"))
    assert detector.add_committed("Bir az da", now=0.0) == []
    assert detector.add_committed("multi-agent sistemlərdən", now=2.1) == []
    out = detector.add_committed("danışaq.", now=4.3)
    assert [q.text for q in out] == ["Bir az da multi-agent sistemlərdən danışaq."]


def test_an_answerable_prompt_is_never_delayed_by_the_merge_window():
    """The window only holds text that is not yet answerable."""
    detector = QuestionDetector(QuestionSettings())
    out = detector.add_committed("RAG nədir?", now=0.0)
    assert [q.text for q in out] == ["RAG nədir?"]
    assert detector.deadline is None


def test_a_request_fires_immediately_without_terminal_punctuation():
    detector = QuestionDetector(QuestionSettings(sensitivity="normal"))
    out = detector.add_committed("Mənə RAG təcrübənizdən bəhs edin", now=0.0)
    assert len(out) == 1
    assert detector.deadline is None


def test_the_buffer_does_not_grow_without_bound():
    detector = QuestionDetector(QuestionSettings(max_buffer_chars=80, sensitivity="normal"))
    for index in range(20):
        detector.add_committed(f"uzun bir cümlə parçası nömrə {index}", now=index * 2.0)
        assert len(detector.pending_text) <= 120, detector.pending_text


def test_unanswerable_text_is_still_discarded_after_the_window():
    settings = QuestionSettings(sensitivity="normal")
    detector = QuestionDetector(settings)
    detector.add_committed("Bugün hava çox gözəldir", now=0.0)

    just_inside = settings.merge_window_ms / 1000.0 - 0.1
    assert detector.flush_due(now=just_inside) == []
    assert detector.pending_text, "gave up before the window elapsed"

    assert detector.flush_due(now=settings.merge_window_ms / 1000.0 + 0.1) == []
    assert detector.pending_text == ""


def test_backchannel_between_halves_does_not_pollute_the_buffer():
    """'Bəli.' mid-sentence must not be glued into the question."""
    detector = QuestionDetector(QuestionSettings(sensitivity="normal"))
    detector.add_committed("Mənə RAG pipeline", now=0.0)
    assert detector.add_committed("Bəli.", now=1.0) == []
    assert "Bəli" not in detector.pending_text

    out = detector.add_committed("təcrübənizdən bəhs edin.", now=2.0)
    assert [q.text for q in out] == ["Mənə RAG pipeline təcrübənizdən bəhs edin."]
    assert detector.skipped_backchannel == 1


def test_the_window_covers_a_measured_real_world_gap():
    """A 1.6s human pause produced commits 3.3s apart in a live run."""
    detector = QuestionDetector(QuestionSettings(sensitivity="normal"))
    assert detector.add_committed("Mənə Rag Pipeline.", now=0.0) == []
    assert detector.flush_due(now=3.2) == [], "gave up before the rest arrived"
    out = detector.add_committed("Təcrübənizdən bəhs edin.", now=3.3)
    assert len(out) == 1
    assert out[0].text.startswith("Mənə Rag Pipeline.")


# =========================================================================
# Turn-based answering (the default)
# =========================================================================
def test_broad_answers_a_plain_statement_with_no_cue_at_all():
    """The point of the default: no classifier standing between the
    interviewer finishing a sentence and an answer appearing."""
    detector = QuestionDetector(QuestionSettings())
    out = detector.add_committed("Bizim komandada deployment prosesi belədir.", now=0.0)
    assert len(out) == 1
    assert out[0].reason.startswith("broad")


def test_a_continuation_supersedes_instead_of_starting_a_second_answer():
    detector = QuestionDetector(QuestionSettings())
    first = detector.add_committed("Sizin CV-nizdə LangGraph təcrübəsi var.", now=0.0)
    assert len(first) == 1

    second = detector.add_committed("Bu barədə eşitmək istərdim.", now=2.5)
    assert len(second) == 1
    # The whole turn, so the fuller version replaces the answer in flight.
    assert second[0].text.startswith("Sizin CV-nizdə")
    assert "eşitmək istərdim" in second[0].text


def test_a_long_gap_starts_a_new_turn():
    detector = QuestionDetector(QuestionSettings())
    detector.add_committed("RAG pipeline qurmusunuz.", now=0.0)
    out = detector.add_committed("Kubernetes ilə nə etmisiniz?", now=30.0)
    assert out[0].text == "Kubernetes ilə nə etmisiniz?"


def test_a_recommitted_segment_within_a_turn_is_not_re_answered():
    detector = QuestionDetector(QuestionSettings())
    assert len(detector.add_committed("RAG nədir?", now=0.0)) == 1
    assert detector.add_committed("RAG nədir?", now=2.0) == []
    assert detector.skipped_duplicates >= 1


def test_two_questions_in_one_turn_are_each_answered_once():
    detector = QuestionDetector(QuestionSettings())
    assert [q.text for q in detector.add_committed("RAG nədir?", now=0.0)] == ["RAG nədir?"]
    out = detector.add_committed("Bəs fine-tuning nədir?", now=2.0)
    assert [q.text for q in out] == ["Bəs fine-tuning nədir?"]


def test_acknowledgements_are_still_never_answered_at_the_default():
    detector = QuestionDetector(QuestionSettings())
    for noise in ("Bəli.", "Yaxşı, aydındır.", "Maraqlıdır.", "Təşəkkür edirəm."):
        assert detector.add_committed(noise, now=0.0) == [], noise
