"""Run the model benchmark from the command line.

    python benchmark_cli.py --models gpt-4.1-nano gpt-5.6-luna

Sends real OpenAI requests and costs money, so it asks for confirmation unless
`--yes` is given.  Results are printed and saved as JSON next to the app data.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from copilot.config import Settings  # noqa: E402
from copilot.credentials import CredentialStore  # noqa: E402
from copilot.documents.extract import DocumentError, extract_text  # noqa: E402
from copilot.documents.profile import CandidateProfile  # noqa: E402
from copilot.llm.benchmark import run_benchmark  # noqa: E402
from copilot.env_file import load_env_file  # noqa: E402
from copilot.llm.benchmark_questions import QUESTIONS  # noqa: E402
from copilot.logging_setup import configure_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark answer-generation models")
    parser.add_argument(
        "--models", nargs="+", default=["gpt-4.1-nano", "gpt-5.6-luna"],
        help="model ids to compare (default: gpt-4.1-nano gpt-5.6-luna)",
    )
    parser.add_argument(
        "--questions", type=int, default=len(QUESTIONS),
        help=f"how many questions from the bank of {len(QUESTIONS)} to use",
    )
    parser.add_argument("--max-output-tokens", type=int, default=500)
    parser.add_argument("--cv", help="optional CV file, to benchmark with real context")
    parser.add_argument("--job", help="optional job-description file")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args(argv)

    configure_logging(to_file=False)
    load_env_file()

    api_key = CredentialStore().get("openai")
    if not api_key:
        print(
            "No OpenAI API key found.\n"
            "Set OPENAI_API_KEY, or save a key in the application Settings.",
            file=sys.stderr,
        )
        return 2

    count = max(1, min(args.questions, len(QUESTIONS)))
    total = count * len(args.models)
    print(f"{len(args.models)} model(s) x {count} question(s) = {total} real API requests.")
    if not args.yes:
        answer = input("This will be billed to your OpenAI account. Continue? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return 1

    profile = None
    if args.cv or args.job:
        try:
            cv = extract_text(args.cv, "cv") if args.cv else None
            job = extract_text(args.job, "job_description") if args.job else None
        except DocumentError as exc:
            print(f"Could not read a document: {exc}", file=sys.stderr)
            return 2
        profile = CandidateProfile.build(cv, job)
        print(f"Context: {profile.describe()}")

    settings = Settings.load()

    def progress(model_id: str, done: int, total_for_model: int) -> None:
        print(f"  {model_id}: {done}/{total_for_model}", end="\r", flush=True)

    report = asyncio.run(
        run_benchmark(
            api_key=api_key,
            models=args.models,
            questions=QUESTIONS[:count],
            settings=settings.llm,
            profile=profile,
            max_output_tokens=args.max_output_tokens,
            progress=progress,
        )
    )

    print("\n")
    print(report.to_text())
    path = report.save()
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
