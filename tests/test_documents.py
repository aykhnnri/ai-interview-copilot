"""CV / job-description extraction and per-question context selection."""
from __future__ import annotations

import pytest

from copilot.documents.extract import DocumentError, extract_text
from copilot.documents.profile import (
    CV,
    JOB_DESCRIPTION,
    CandidateProfile,
    split_sections,
    tokenize,
)


# =========================================================================
# Extraction
# =========================================================================
def test_txt_extraction_preserves_azerbaijani(cv_document):
    assert "Azərbaycan" in cv_document.text or "təcrübə" in cv_document.text
    for char in "əğşçöüı":
        assert char in cv_document.text.lower()


def test_extraction_records_metadata(cv_document):
    assert cv_document.kind == "cv"
    assert cv_document.name == "sample_cv.txt"
    assert cv_document.char_count == len(cv_document.text)


def test_unsupported_file_type_is_rejected(tmp_path):
    bad = tmp_path / "notes.pptx"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(DocumentError, match="Unsupported file type"):
        extract_text(bad)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(DocumentError, match="File not found"):
        extract_text(tmp_path / "nope.pdf")


def test_empty_document_is_rejected(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(DocumentError, match="No text could be extracted"):
        extract_text(empty)


def test_docx_extraction_includes_tables(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "cv.docx"
    document = docx.Document()
    document.add_paragraph("Aysel Məmmədova")
    document.add_paragraph("AI Engineer")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Bacarıqlar"
    table.cell(0, 1).text = "Python, LangGraph"
    document.save(path)

    extracted = extract_text(path, "cv")
    assert "Aysel Məmmədova" in extracted.text
    assert "LangGraph" in extracted.text        # table content is not dropped


def test_pdf_extraction(tmp_path):
    """A generated PDF must round-trip through pypdf."""
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "blank.pdf"
    with path.open("wb") as handle:
        writer.write(handle)

    # A blank page yields no text, and that must be a clear error rather than
    # silently sending an empty CV to the model.
    with pytest.raises(DocumentError, match="No text could be extracted"):
        extract_text(path, "cv")


def test_text_encoding_fallback(tmp_path):
    path = tmp_path / "cv.txt"
    path.write_bytes("Təcrübə: 5 il".encode("utf-8-sig"))
    assert "Təcrübə" in extract_text(path).text


# =========================================================================
# Sectioning
# =========================================================================
def test_sections_follow_cv_headings(cv_document):
    headings = {section.heading for section in split_sections(cv_document.text, CV)}
    assert "İŞ TƏCRÜBƏSİ" in headings
    assert "BACARIQLAR" in headings
    assert "TƏHSİL" in headings


def test_key_value_lines_are_not_mistaken_for_headings(cv_document):
    headings = {section.heading for section in split_sections(cv_document.text, CV)}
    assert not any(heading.startswith("Dillər:") for heading in headings)


def test_tokenize_drops_stopwords_and_keeps_technical_terms():
    tokens = tokenize("Bu, RAG pipeline-ın retrieval quality-si üçün vacibdir")
    assert "rag" in tokens and "retrieval" in tokens
    assert "bu" not in tokens and "üçün" not in tokens


# =========================================================================
# Context selection
# =========================================================================
def test_profile_reports_both_documents(profile):
    description = profile.describe()
    assert "sample_cv.txt" in description
    assert "sample_job.txt" in description
    assert profile.has_documents


def test_headline_carries_name_and_title(profile):
    assert "Aysel Məmmədova" in profile.headline
    assert "Engineer" in profile.headline


def test_cv_context_is_selected_for_an_experience_question(profile):
    labels = [s.label for s in profile.select_context("Multi-agent sistemlərlə işləmisiniz?")]
    assert any("LAYİHƏLƏR" in label or "TƏCRÜBƏSİ" in label for label in labels)


def test_education_question_selects_the_education_section(profile):
    sections = profile.select_context("Təhsiliniz haqqında danışın", max_sections=2)
    assert any(s.heading == "TƏHSİL" for s in sections)


def test_job_description_context_is_always_represented(profile):
    """Answers should stay aimed at the advertised role."""
    for question in (
        "RAG pipeline-ın retrieval quality-sini necə ölçərdiniz?",
        "Python-da async və await nədir?",
        "Docker təcrübəniz varmı?",
    ):
        sections = profile.select_context(question, max_sections=4)
        assert any(s.source == JOB_DESCRIPTION for s in sections), question


def test_only_relevant_sections_are_sent_not_the_whole_cv(profile, cv_document, jd_document):
    """Latency and cost both depend on not shipping the entire CV each time."""
    context = profile.render_context("Təhsiliniz haqqında danışın", max_sections=2)
    assert len(context) < (cv_document.char_count + jd_document.char_count) * 0.8
    assert "Bakı Dövlət Universiteti" in context


def test_context_is_labelled_by_source(profile):
    context = profile.render_context("Vector database təcrübəniz nədir?")
    assert "[CV /" in context or "[Vakansiya /" in context
    assert "[Namizəd]" in context


def test_context_selection_is_stable_for_the_same_question(profile):
    first = profile.render_context("RAG nədir?")
    second = profile.render_context("RAG nədir?")
    assert first == second


def test_empty_profile_yields_no_context():
    profile = CandidateProfile.build()
    assert not profile.has_documents
    assert profile.render_context("RAG nədir?") == ""
    assert profile.select_context("RAG nədir?") == []


def test_cv_only_profile_works(cv_document):
    profile = CandidateProfile.build(cv_document, None)
    sections = profile.select_context("Hansı vector database-lərdən istifadə etmisiniz?")
    assert sections and all(s.source == CV for s in sections)


def test_irrelevant_question_still_returns_the_headline(profile):
    """Identity grounding stays even when no section scores well."""
    context = profile.render_context("Bugünkü hava necədir?", max_sections=2)
    assert "Aysel Məmmədova" in context
