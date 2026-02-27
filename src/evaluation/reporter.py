"""Aggregate and display benchmark results, and archive runs."""

import json
import os
from collections import defaultdict

# Paper-reported accuracy for Claude 3.5 Sonnet (per-task averages)
PAPER_RESULTS = {
    "LineIntersection": 77.33,
    "TouchingCircle": 91.66,
    "NestedSquares": 87.50,
    "CountingRowsAndColumns": 74.26,
}


def load_results(results_dir: str) -> list[dict]:
    """Load all result JSON files from a directory."""
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json") and fname != "summary.json":
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
    return results


def _group_results(results: list[dict]) -> dict:
    """Group results by model → task → prompt."""
    by_model = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        by_model[r["model"]][r["task"]][r["prompt_key"]].append(r)
    return by_model


def build_summary(results: list[dict], run_meta: dict | None = None) -> dict:
    """Build a full summary dict from results."""
    by_model = _group_results(results)
    summary = {"run": run_meta or {}, "models": {}}

    for model, tasks in by_model.items():
        model_summary = {}
        for task, prompts in tasks.items():
            task_summary = {}
            for prompt_key, entries in prompts.items():
                correct = sum(1 for r in entries if r["correct"])
                total = len(entries)
                task_summary[prompt_key] = {
                    "correct": correct,
                    "total": total,
                    "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
                }
            # Task average
            all_correct = sum(v["correct"] for v in task_summary.values())
            all_total = sum(v["total"] for v in task_summary.values())
            task_summary["_average"] = {
                "correct": all_correct,
                "total": all_total,
                "accuracy": round(all_correct / all_total * 100, 1) if all_total > 0 else 0,
            }
            model_summary[task] = task_summary
        summary["models"][model] = model_summary

    return summary


def format_report(results: list[dict], run_meta: dict | None = None) -> str:
    """Format a full text report with comparison table and run statistics."""
    lines = []
    by_model = _group_results(results)

    for model, tasks in sorted(by_model.items()):
        lines.append(f"{'='*72}")
        lines.append(f"Model: {model}")
        lines.append(f"{'='*72}")
        lines.append("")

        # Comparison table
        lines.append(f"{'Task':<28} {'Prompt':<20} {'Ours':>8} {'Paper':>8}")
        lines.append(f"{'-'*28} {'-'*20} {'-'*8} {'-'*8}")

        model_correct = 0
        model_total = 0

        for task in sorted(tasks):
            prompts = tasks[task]
            paper_avg = PAPER_RESULTS.get(task)
            task_correct = 0
            task_total = 0

            for prompt_key in sorted(prompts):
                entries = prompts[prompt_key]
                correct = sum(1 for r in entries if r["correct"])
                total = len(entries)
                acc = correct / total * 100 if total > 0 else 0
                lines.append(f"{task:<28} {prompt_key:<20} {acc:>7.1f}% {'':>8}")
                task_correct += correct
                task_total += total

            task_avg = task_correct / task_total * 100 if task_total > 0 else 0
            paper_str = f"{paper_avg:.1f}%" if paper_avg is not None else "N/A"
            lines.append(f"{'':<28} {'average':<20} {task_avg:>7.1f}% {paper_str:>8}")
            lines.append("")
            model_correct += task_correct
            model_total += task_total

        overall = model_correct / model_total * 100 if model_total > 0 else 0
        lines.append(f"{'OVERALL':<28} {'':<20} {overall:>7.1f}% {'74.9%':>8}")

    # Run statistics
    if run_meta:
        lines.append("")
        lines.append(f"{'='*72}")
        lines.append("Run Statistics")
        lines.append(f"{'='*72}")
        lines.append(f"  Timestamp:      {run_meta.get('timestamp', 'N/A')}")
        lines.append(f"  Model:          {run_meta.get('model', 'N/A')}")
        lines.append(f"  Tasks:          {', '.join(run_meta.get('tasks', []))}")
        lines.append(f"  Limit/task:     {run_meta.get('limit', 0) or 'all'}")
        lines.append(f"  Total requests: {run_meta.get('total_requests', 0)}")
        lines.append(f"  Input tokens:   {run_meta.get('total_input_tokens', 0):,}")
        lines.append(f"  Output tokens:  {run_meta.get('total_output_tokens', 0):,}")
        lines.append(f"  Total tokens:   {run_meta.get('total_tokens', 0):,}")

        wall = run_meta.get("wall_clock_s", 0)
        minutes = int(wall // 60)
        seconds = int(wall % 60)
        lines.append(f"  Wall clock:     {minutes}m {seconds}s")

        reqs = run_meta.get("total_requests", 0)
        if reqs > 0:
            lines.append(f"  Avg latency:    {wall / reqs:.1f}s/request")

        # Cost estimate (Claude 3.5 Sonnet: $3/M input, $15/M output)
        input_cost = run_meta.get("total_input_tokens", 0) * 3.0 / 1_000_000
        output_cost = run_meta.get("total_output_tokens", 0) * 15.0 / 1_000_000
        total_cost = input_cost + output_cost
        lines.append(f"  Est. cost:      ${total_cost:.2f} "
                      f"(${input_cost:.2f} input + ${output_cost:.2f} output)")

        errors = run_meta.get("errors", 0)
        if errors:
            lines.append(f"  Errors:         {errors}")

    return "\n".join(lines)


def save_run(results: list[dict], run_meta: dict, base_dir: str) -> str:
    """Archive a complete run to a timestamped subfolder.

    Creates:
      runs/{timestamp}_{model_slug}/
        ├── results.json     # All 800 individual query results
        ├── summary.json     # Per-task accuracy + run metadata
        └── report.txt       # Human-readable comparison table

    Returns the path to the run directory.
    """
    timestamp = run_meta.get("timestamp", "unknown")
    model_slug = run_meta.get("model", "unknown").replace("/", "__")
    limit = run_meta.get("limit", 0)
    limit_str = f"_n{limit}" if limit else "_full"
    run_dir = os.path.join(base_dir, "runs", f"{timestamp}_{model_slug}{limit_str}")
    os.makedirs(run_dir, exist_ok=True)

    # Detailed results (all queries)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Summary with metadata
    summary = build_summary(results, run_meta)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable report
    report = format_report(results, run_meta)
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write(report + "\n")

    return run_dir


def print_report(results: list[dict], run_meta: dict | None = None):
    """Print the report to stdout."""
    print(format_report(results, run_meta))
