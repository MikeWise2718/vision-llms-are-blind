"""Aggregate and display benchmark results."""

import json
import os
from collections import defaultdict


def load_results(results_dir: str) -> list[dict]:
    """Load all result JSON files from a directory."""
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                results.extend(json.load(f))
    return results


def print_report(results: list[dict]):
    """Print accuracy summary grouped by model, task, and prompt variant."""
    # Group by model → task → prompt
    by_model = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        by_model[r["model"]][r["task"]][r["prompt_key"]].append(r["correct"])

    for model, tasks in sorted(by_model.items()):
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        model_correct = 0
        model_total = 0

        for task, prompts in sorted(tasks.items()):
            print(f"\n  {task}:")
            for prompt_key, scores in sorted(prompts.items()):
                correct = sum(scores)
                total = len(scores)
                acc = correct / total * 100 if total > 0 else 0
                print(f"    {prompt_key}: {correct}/{total} ({acc:.1f}%)")
                model_correct += correct
                model_total += total

        overall = model_correct / model_total * 100 if model_total > 0 else 0
        print(f"\n  Overall: {model_correct}/{model_total} ({overall:.1f}%)")


def save_summary(results: list[dict], output_path: str):
    """Save a summary JSON with per-task accuracy."""
    by_model = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        by_model[r["model"]][r["task"]][r["prompt_key"]].append(r["correct"])

    summary = {}
    for model, tasks in by_model.items():
        summary[model] = {}
        for task, prompts in tasks.items():
            summary[model][task] = {}
            for prompt_key, scores in prompts.items():
                correct = sum(scores)
                total = len(scores)
                summary[model][task][prompt_key] = {
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total if total > 0 else 0,
                }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_path}")
