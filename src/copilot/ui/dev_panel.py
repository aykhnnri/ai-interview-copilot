"""Developer panel: live latency metrics, provider diagnostics and benchmarks.

The benchmark tab spends real money on the OpenAI API, so it states the cost
up front and only runs when the user presses the button.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.pipeline import CopilotPipeline
from ..credentials import CredentialStore
from ..llm.benchmark import BenchmarkReport, load_latest_report, run_benchmark
from ..llm.benchmark_questions import QUESTIONS
from ..llm.models import selectable_models
from .bridge import PipelineBridge

log = logging.getLogger(__name__)

BENCHMARK_TAG = "benchmark"


class DevPanel(QWidget):
    def __init__(
        self,
        pipeline: CopilotPipeline,
        bridge: PipelineBridge,
        credentials: CredentialStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.Window)
        self.pipeline = pipeline
        self.bridge = bridge
        self.credentials = credentials
        self._report: BenchmarkReport | None = None

        self.setWindowTitle("Developer paneli")
        self.resize(900, 680)
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._build_metrics_tab(), "Latency")
        tabs.addTab(self._build_benchmark_tab(), "Benchmark")
        tabs.addTab(self._build_log_tab(), "Jurnal")
        layout.addWidget(tabs)

        self._timer = QTimer(self)
        self._timer.setInterval(700)
        self._timer.timeout.connect(self.refresh_metrics)
        self._timer.start()

        self.bridge.task_finished.connect(self._on_task_finished)
        self._load_previous_report()

    # =====================================================================
    # Metrics
    # =====================================================================
    def _build_metrics_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.metrics_view = QPlainTextEdit()
        self.metrics_view.setReadOnly(True)
        self.metrics_view.setStyleSheet("font-family: Consolas, monospace;")
        layout.addWidget(self.metrics_view, 1)

        self.stt_view = QPlainTextEdit()
        self.stt_view.setReadOnly(True)
        self.stt_view.setMaximumHeight(170)
        self.stt_view.setStyleSheet("font-family: Consolas, monospace;")
        layout.addWidget(self.stt_view)

        row = QHBoxLayout()
        reset = QPushButton("Metrikləri sıfırla")
        reset.clicked.connect(self._reset_metrics)
        row.addWidget(reset)
        row.addStretch(1)
        layout.addLayout(row)
        return page

    @Slot()
    def refresh_metrics(self) -> None:
        if not self.isVisible():
            return
        self.metrics_view.setPlainText(self.pipeline.metrics.report())
        debug = self.pipeline.stt_debug()
        lines = [f"{key:<16} {value}" for key, value in debug.items()]
        lines.append("")
        lines.append(f"{'stt model':<16} {self.pipeline.settings.stt.model_id}")
        lines.append(f"{'stt language':<16} {self.pipeline.settings.stt.language_code}")
        lines.append(f"{'llm model':<16} {self.pipeline.settings.llm.model}")
        lines.append(f"{'service tier':<16} {self.pipeline.settings.llm.service_tier}")
        # The single most useful line when answers are not appearing: it shows
        # the value actually in force, not the one the current build defaults to.
        question = self.pipeline.settings.question
        lines.append(f"{'answer when':<16} {question.sensitivity}")
        lines.append(f"{'turn merge':<16} {question.merge_window_ms} ms")
        lines.append(f"{'settings v':<16} {self.pipeline.settings.schema_version}")
        lines.append(f"{'documents':<16} {self.pipeline.profile.describe()}")
        self.stt_view.setPlainText("\n".join(lines))

    def _reset_metrics(self) -> None:
        self.pipeline.metrics.reset()
        self.refresh_metrics()

    # =====================================================================
    # Benchmark
    # =====================================================================
    def _build_benchmark_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        form = QFormLayout()
        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(110)
        for spec in selectable_models():
            item = QListWidgetItem(f"{spec.label}  ({spec.id})")
            item.setData(Qt.UserRole, spec.id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if spec.id in ("gpt-4.1-nano", "gpt-5.6-luna") else Qt.Unchecked)
            self.model_list.addItem(item)
        form.addRow("Modellər", self.model_list)

        self.question_count = QSpinBox()
        self.question_count.setRange(1, len(QUESTIONS))
        self.question_count.setValue(len(QUESTIONS))
        form.addRow(f"Sual sayı (bank: {len(QUESTIONS)})", self.question_count)

        self.bench_tokens = QSpinBox()
        self.bench_tokens.setRange(100, 2000)
        self.bench_tokens.setSingleStep(50)
        self.bench_tokens.setValue(500)
        form.addRow("Hər cavab üçün maks. token", self.bench_tokens)

        self.use_profile = QCheckBox("CV və vakansiya kontekstindən istifadə et")
        form.addRow(self.use_profile)
        layout.addLayout(form)

        warning = QLabel(
            "⚠ Benchmark real OpenAI API sorğuları göndərir və xərc yaradır. "
            "ElevenLabs bu testdə istifadə olunmur."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color:#d29922; font-size:11px;")
        layout.addWidget(warning)

        row = QHBoxLayout()
        self.run_button = QPushButton("Benchmark-ı işə sal")
        self.run_button.clicked.connect(self.run_benchmark)
        self.save_button = QPushButton("Nəticəni saxla")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_report)
        row.addWidget(self.run_button)
        row.addWidget(self.save_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.benchmark_view = QPlainTextEdit()
        self.benchmark_view.setReadOnly(True)
        self.benchmark_view.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        layout.addWidget(self.benchmark_view, 1)
        return page

    def _selected_models(self) -> list[str]:
        out = []
        for row in range(self.model_list.count()):
            item = self.model_list.item(row)
            if item.checkState() == Qt.Checked:
                out.append(item.data(Qt.UserRole))
        return out

    @Slot()
    def run_benchmark(self) -> None:
        models = self._selected_models()
        if not models:
            QMessageBox.information(self, "Model seçilməyib", "Ən azı bir model seçin.")
            return
        api_key = self.credentials.get("openai")
        if not api_key:
            QMessageBox.warning(self, "OpenAI açarı yoxdur", "Settings-dən OpenAI açarını əlavə edin.")
            return

        count = self.question_count.value()
        total = count * len(models)
        confirmed = QMessageBox.question(
            self,
            "Benchmark",
            f"{len(models)} model × {count} sual = {total} real API sorğusu göndəriləcək.\n"
            "Davam edilsin?",
        )
        if confirmed != QMessageBox.Yes:
            return

        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress.setRange(0, total)
        self.progress.setValue(0)
        self.benchmark_view.setPlainText("Benchmark işləyir…")
        self._completed = 0
        self._total = total

        profile = self.pipeline.profile if self.use_profile.isChecked() else None

        def progress_hook(model_id: str, done: int, per_model_total: int) -> None:
            # Called on the pipeline loop; QProgressBar.setValue is queued by Qt
            # through the metaobject system only for signals, so keep it simple
            # and just count, letting the GUI timer paint it.
            self._completed += 1

        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(300)
        self._progress_timer.timeout.connect(
            lambda: self.progress.setValue(min(self._completed, self._total))
        )
        self._progress_timer.start()

        self.bridge.run_task(
            BENCHMARK_TAG,
            run_benchmark(
                api_key=api_key,
                models=models,
                questions=QUESTIONS[:count],
                settings=self.pipeline.settings.llm,
                profile=profile,
                max_output_tokens=self.bench_tokens.value(),
                progress=progress_hook,
            ),
        )

    @Slot(str, bool, object)
    def _on_task_finished(self, tag: str, ok: bool, payload: object) -> None:
        if tag != BENCHMARK_TAG:
            return
        if hasattr(self, "_progress_timer"):
            self._progress_timer.stop()
        self.run_button.setEnabled(True)
        if not ok:
            self.benchmark_view.setPlainText(f"Benchmark alınmadı:\n{payload}")
            return
        report: BenchmarkReport = payload  # type: ignore[assignment]
        self._report = report
        self.progress.setValue(self.progress.maximum())
        self.benchmark_view.setPlainText(report.to_text())
        self.save_button.setEnabled(True)
        try:
            path = report.save()
            self.append_log(f"Benchmark saxlanıldı: {path}")
        except OSError as exc:
            self.append_log(f"Benchmark saxlanmadı: {exc}")

    @Slot()
    def save_report(self) -> None:
        if self._report is None:
            return
        try:
            path = self._report.save()
            QMessageBox.information(self, "Saxlanıldı", str(path))
        except OSError as exc:
            QMessageBox.warning(self, "Saxlanmadı", str(exc))

    def _load_previous_report(self) -> None:
        data = load_latest_report()
        if not data:
            self.benchmark_view.setPlainText(
                "Hələ benchmark işlədilməyib.\n\n"
                f"Bank: {len(QUESTIONS)} Azərbaycan dilində texniki sual.\n"
                "Ölçülür: ilk token vaxtı, tam cavab vaxtı, cavab uzunluğu, "
                "Azərbaycan dili keyfiyyəti, texniki əhatə, təxmini xərc."
            )
            return
        summary = data.get("summary") or []
        lines = [f"Son benchmark: {data.get('finished_at', '?')}", ""]
        for row in summary:
            lines.append(
                f"{row.get('label')}: TTFT median {row.get('ttft_median_ms')} ms, "
                f"tam {row.get('total_median_ms')} ms, keyfiyyət {row.get('overall_quality')}"
            )
        recommended = data.get("recommended")
        if recommended:
            lines.append("")
            lines.append(f"Tövsiyə olunan model: {recommended}")
        self.benchmark_view.setPlainText("\n".join(lines))

    # =====================================================================
    # Log
    # =====================================================================
    def _build_log_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(self.log_view)
        return page

    @Slot(str)
    def append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)
