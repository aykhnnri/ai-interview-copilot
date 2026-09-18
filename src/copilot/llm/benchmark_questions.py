"""The Azerbaijani technical interview question bank used for benchmarking.

The first ten are the questions named in the specification; the rest broaden
coverage across retrieval, agents, serving, evaluation and Python internals.

`expected_terms` are concepts a technically correct answer should touch on.
They drive a coverage score - a coarse but model-independent proxy for
correctness that needs no second model to judge it.  `english_terms` are terms
that must survive in English rather than being translated into Azerbaijani.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    text: str
    expected_terms: tuple[tuple[str, ...], ...] = ()
    """Each inner tuple is a set of synonyms; the answer scores if any matches."""
    english_terms: tuple[str, ...] = ()
    category: str = "general"


QUESTIONS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        id="q01", category="rag", text="RAG nədir?",
        expected_terms=(("retrieval", "axtarış"), ("kontekst", "context"),
                        ("sənəd", "document", "bilik bazası"), ("generasiya", "generation", "model")),
        english_terms=("RAG",),
    ),
    BenchmarkQuestion(
        id="q02", category="rag",
        text="Fine-tuning və RAG arasında əsas fərq nədir?",
        expected_terms=(("çəki", "weight", "parametr"), ("məlumat", "bilik", "knowledge"),
                        ("yeniləmə", "update", "dəyişmə"), ("xərc", "cost", "maya")),
        english_terms=("fine-tuning", "RAG"),
    ),
    BenchmarkQuestion(
        id="q03", category="agents",
        text="Multi-agent sistemlərdə agentlər arasında kommunikasiya necə qurulur?",
        expected_terms=(("message", "mesaj"), ("state", "vəziyyət", "shared"),
                        ("orkestr", "orchestr", "koordin", "supervisor"), ("protokol", "interfeys", "tool")),
        english_terms=("agent",),
    ),
    BenchmarkQuestion(
        id="q04", category="reliability",
        text="Production mühitində LLM hallucination problemini necə azaldarsınız?",
        expected_terms=(("grounding", "istinad", "citation", "mənbə"),
                        ("retrieval", "RAG"), ("guardrail", "validasiya", "yoxlama"),
                        ("temperature", "evaluation", "monitor")),
        english_terms=("hallucination", "LLM"),
    ),
    BenchmarkQuestion(
        id="q05", category="python",
        text="Python-da async və await nədir?",
        expected_terms=(("event loop", "hadisə dövrü"), ("coroutine", "korutin"),
                        ("bloklama", "blocking", "I/O"), ("konkurrent", "paralel", "eyni vaxtda")),
        english_terms=("async", "await"),
    ),
    BenchmarkQuestion(
        id="q06", category="rag",
        text="Vector database necə işləyir?",
        expected_terms=(("embedding", "vektor"), ("oxşarlıq", "similarity", "cosine"),
                        ("index", "HNSW", "IVF"), ("axtarış", "search", "nearest")),
        english_terms=("vector database", "embedding"),
    ),
    BenchmarkQuestion(
        id="q07", category="performance",
        text="LLM tətbiqlərində latency-ni necə optimizasiya edərdiniz?",
        expected_terms=(("streaming", "axın"), ("cache", "keş"),
                        ("kiçik model", "smaller model", "routing", "model seçimi"),
                        ("token", "prompt", "batch")),
        english_terms=("latency", "LLM"),
    ),
    BenchmarkQuestion(
        id="q08", category="evaluation",
        text="Modelin performansını hansı metriklərlə qiymətləndirərdiniz?",
        expected_terms=(("precision", "dəqiqlik"), ("recall", "F1"),
                        ("latency", "gecikmə"), ("dataset", "test", "evaluation")),
        english_terms=("precision", "recall"),
    ),
    BenchmarkQuestion(
        id="q09", category="security",
        text="Sisteminizdə məlumatların təhlükəsizliyini necə təmin edərdiniz?",
        expected_terms=(("şifrələmə", "encryption", "TLS"), ("giriş", "access", "RBAC", "icazə"),
                        ("PII", "şəxsi məlumat", "anonim", "maskalama"), ("audit", "log", "GDPR")),
        english_terms=(),
    ),
    BenchmarkQuestion(
        id="q10", category="rag",
        text="RAG pipeline-ın retrieval quality-sini necə ölçərdiniz?",
        expected_terms=(("recall", "Recall@k"), ("precision", "MRR", "NDCG"),
                        ("qızıl", "gold", "ground truth", "etalon"), ("dataset", "annotasiya", "test")),
        english_terms=("retrieval", "RAG"),
    ),
    BenchmarkQuestion(
        id="q11", category="rag",
        text="Chunking strategiyasını necə seçərsiniz?",
        expected_terms=(("ölçü", "size", "token"), ("overlap", "kəsişmə"),
                        ("semantik", "semantic", "struktur"), ("test", "evaluation", "sınaq")),
        english_terms=("chunking",),
    ),
    BenchmarkQuestion(
        id="q12", category="rag",
        text="Hybrid search və reranking nə üçün lazımdır?",
        expected_terms=(("BM25", "keyword", "açar söz"), ("dense", "semantik", "embedding"),
                        ("rerank", "cross-encoder"), ("dəqiqlik", "precision", "keyfiyyət")),
        english_terms=("reranking",),
    ),
    BenchmarkQuestion(
        id="q13", category="agents",
        text="LangGraph ilə LangChain arasındakı fərq nədir?",
        expected_terms=(("graph", "qraf", "state machine"), ("state", "vəziyyət"),
                        ("dövr", "loop", "cycle", "şərti"), ("zəncir", "chain", "abstraksiya")),
        english_terms=("LangGraph", "LangChain"),
    ),
    BenchmarkQuestion(
        id="q14", category="performance",
        text="Prompt caching necə işləyir və nə vaxt istifadə edilməlidir?",
        expected_terms=(("prefix", "başlanğıc", "təkrar"), ("xərc", "cost", "qənaət"),
                        ("latency", "gecikmə"), ("sabit", "static", "dəyişməyən")),
        english_terms=("prompt caching",),
    ),
    BenchmarkQuestion(
        id="q15", category="serving",
        text="FastAPI ilə LLM inference API-ni necə qurarsınız?",
        expected_terms=(("async", "await"), ("streaming", "SSE", "axın"),
                        ("timeout", "retry", "xəta"), ("Docker", "deploy", "scale")),
        english_terms=("FastAPI",),
    ),
    BenchmarkQuestion(
        id="q16", category="evaluation",
        text="LLM cavablarını necə avtomatik qiymətləndirərsiniz?",
        expected_terms=(("LLM-as-judge", "judge", "hakim"), ("dataset", "test seti"),
                        ("rubrika", "rubric", "meyar"), ("regresiya", "regression", "CI")),
        english_terms=("LLM",),
    ),
    BenchmarkQuestion(
        id="q17", category="python",
        text="Python-da GIL nədir və necə təsir edir?",
        expected_terms=(("thread", "axın"), ("CPU", "hesablama"),
                        ("I/O", "giriş-çıxış"), ("multiprocessing", "proses")),
        english_terms=("GIL",),
    ),
    BenchmarkQuestion(
        id="q18", category="serving",
        text="Docker və Kubernetes ilə model deployment təcrübəniz necədir?",
        expected_terms=(("image", "konteyner", "container"), ("scale", "replika", "autoscal"),
                        ("health", "probe", "monitor"), ("resurs", "GPU", "limit")),
        english_terms=("Docker", "Kubernetes"),
    ),
    BenchmarkQuestion(
        id="q19", category="reliability",
        text="Prompt injection hücumundan necə qorunarsınız?",
        expected_terms=(("data", "məlumat", "təlimat deyil"), ("validasiya", "filter", "sanitiz"),
                        ("icazə", "permission", "least privilege", "sandbox"), ("test", "monitor", "audit")),
        english_terms=("prompt injection",),
    ),
    BenchmarkQuestion(
        id="q20", category="rag",
        text="Embedding modelini necə seçərsiniz?",
        expected_terms=(("dil", "language", "çoxdilli", "multilingual"), ("ölçü", "dimension", "size"),
                        ("MTEB", "benchmark", "evaluation"), ("xərc", "latency", "sürət")),
        english_terms=("embedding",),
    ),
    BenchmarkQuestion(
        id="q21", category="performance",
        text="Token xərclərini necə azaldarsınız?",
        expected_terms=(("prompt", "qısaltma", "compression"), ("cache", "keş"),
                        ("kiçik model", "routing", "model seçimi"), ("max_tokens", "limit", "çıxış")),
        english_terms=("token",),
    ),
    BenchmarkQuestion(
        id="q22", category="agents",
        text="Agent sistemində tool calling necə işləyir?",
        expected_terms=(("schema", "sxem", "funksiya"), ("model", "seçim", "çağırış"),
                        ("nəticə", "result", "geri"), ("xəta", "validasiya", "retry")),
        english_terms=("tool calling",),
    ),
    BenchmarkQuestion(
        id="q23", category="evaluation",
        text="A/B test ilə model dəyişikliyini necə yoxlayarsınız?",
        expected_terms=(("trafik", "traffic", "bölmə", "split"), ("metrik", "metric", "KPI"),
                        ("statistik", "significance", "əhəmiyyət"), ("rollback", "geri qaytar", "risk")),
        english_terms=("A/B",),
    ),
    BenchmarkQuestion(
        id="q24", category="general",
        text="Komanda ilə işləyərkən texniki qərarları necə sənədləşdirirsiniz?",
        expected_terms=(("ADR", "sənəd", "document"), ("review", "baxış", "müzakirə"),
                        ("səbəb", "rationale", "alternativ"), ("repo", "wiki", "versiya")),
        english_terms=(),
    ),
]


def question_by_id(question_id: str) -> BenchmarkQuestion | None:
    return next((q for q in QUESTIONS if q.id == question_id), None)


def categories() -> list[str]:
    return sorted({q.category for q in QUESTIONS})
