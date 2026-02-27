"""CLI entry point for the BlindTest benchmark evaluation."""

import argparse
import os
import sys

from .config import DEFAULT_MODEL, TASKS, RESULTS_DIR
from .runner import run_benchmark
from .reporter import load_results, print_report, save_run, format_report


def main():
    parser = argparse.ArgumentParser(
        description="Run the BlindTest benchmark on VLMs via OpenRouter"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model ID to evaluate (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--task", nargs="*", default=None,
        choices=list(TASKS.keys()),
        help="Task(s) to run (default: all)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max images per task/prompt variant (0 = all)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing results (skip already-evaluated images)"
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Only generate report from existing results, don't run evaluation"
    )
    parser.add_argument(
        "--api-key", default="",
        help="OpenRouter API key (default: use OPENROUTER_API_KEY env var)"
    )

    args = parser.parse_args()

    if args.report_only:
        if not os.path.isdir(RESULTS_DIR):
            print(f"No results directory found at {RESULTS_DIR}")
            sys.exit(1)
        results = load_results(RESULTS_DIR)
        if not results:
            print("No results found.")
            sys.exit(1)
        print_report(results)
        return

    all_results, run_meta = run_benchmark(
        model=args.model,
        tasks=args.task,
        limit=args.limit,
        resume=args.resume,
        api_key=args.api_key,
    )

    if all_results:
        print_report(all_results, run_meta)

        eval_dir = os.path.dirname(os.path.abspath(__file__))
        run_dir = save_run(all_results, run_meta, eval_dir)
        print(f"\nRun archived to: {run_dir}")


if __name__ == "__main__":
    main()
