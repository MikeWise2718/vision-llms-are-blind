"""Main evaluation runner — sends images to VLMs and records results."""

import json
import os
import time
from datetime import datetime, timezone

from .config import TASKS, RESULTS_DIR
from .ground_truth import TestImage, build_index
from .parsers import parse_response
from .scorer import score
from .backends.openrouter import OpenRouterClient


def run_task(
    client: OpenRouterClient,
    model: str,
    task_name: str,
    prompt_key: str,
    limit: int = 0,
    resume_from: str | None = None,
) -> list[dict]:
    """Evaluate a model on a specific task/prompt combination.

    Args:
        client: OpenRouter API client
        model: Model ID (e.g. "anthropic/claude-3.5-sonnet")
        task_name: Task name (e.g. "LineIntersection")
        prompt_key: Prompt variant (e.g. "Count-prompt")
        limit: Max images to evaluate (0 = all)
        resume_from: Path to existing results JSON to skip already-evaluated images

    Returns:
        List of result dicts
    """
    task_cfg = TASKS[task_name]
    prompt_text = task_cfg["prompts"][prompt_key]
    answer_type = task_cfg["answer_type"]

    images = build_index(task_name, prompt_key, limit=limit)
    if not images:
        print(f"  No images found for {task_name}/{prompt_key}")
        return []

    # Load existing results for resume
    done_paths = set()
    existing_results = []
    if resume_from and os.path.exists(resume_from):
        with open(resume_from) as f:
            existing_results = json.load(f)
        done_paths = {r["image_path"] for r in existing_results}
        print(f"  Resuming: {len(done_paths)} already evaluated")

    results = list(existing_results)
    remaining = [img for img in images if img.image_path not in done_paths]

    print(f"  {task_name}/{prompt_key}: {len(remaining)} images to evaluate "
          f"({len(images)} total, {len(done_paths)} done)")

    for i, img in enumerate(remaining):
        t0 = time.monotonic()
        response = client.query(model, prompt_text, img.image_path)
        elapsed = time.monotonic() - t0

        if response["error"]:
            print(f"    [{i+1}/{len(remaining)}] ERROR: {response['error']}")
            parsed = None
            is_correct = False
        else:
            parsed = parse_response(answer_type, response["response"])
            is_correct = score(answer_type, parsed, img.ground_truth)

        result = {
            "task": task_name,
            "prompt_key": prompt_key,
            "model": model,
            "image_path": img.image_path,
            "ground_truth": img.ground_truth,
            "raw_response": response["response"],
            "parsed_answer": parsed,
            "correct": is_correct,
            "error": response["error"],
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
            "tokens_used": response["tokens_used"],
            "latency_s": round(elapsed, 2),
            "metadata": img.metadata,
        }
        results.append(result)

        status = "OK" if is_correct else "WRONG"
        if response["error"]:
            status = "ERR"
        print(f"    [{i+1}/{len(remaining)}] {status} "
              f"gt={img.ground_truth} parsed={parsed} "
              f"resp={response['response'][:80]}")

    return results


def get_results_path(model: str, task_name: str, prompt_key: str) -> str:
    """Get the output path for a results JSON file."""
    model_slug = model.replace("/", "__")
    return os.path.join(RESULTS_DIR, f"{model_slug}__{task_name}__{prompt_key}.json")


def run_benchmark(
    model: str,
    tasks: list[str] | None = None,
    limit: int = 0,
    resume: bool = False,
    api_key: str = "",
):
    """Run the full benchmark for a model across all tasks.

    Args:
        model: Model ID
        tasks: List of task names to run (None = all)
        limit: Max images per task/prompt (0 = all)
        resume: Whether to resume from existing results
        api_key: OpenRouter API key (uses env var if empty)
    """
    client = OpenRouterClient(api_key=api_key)
    task_list = tasks or list(TASKS.keys())
    all_results = []

    os.makedirs(RESULTS_DIR, exist_ok=True)

    run_start = time.monotonic()
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"Running benchmark: model={model}, tasks={task_list}, limit={limit}")
    print(f"Results directory: {RESULTS_DIR}\n")

    for task_name in task_list:
        if task_name not in TASKS:
            print(f"Unknown task: {task_name}, skipping")
            continue

        task_cfg = TASKS[task_name]
        for prompt_key in task_cfg["prompts"]:
            results_path = get_results_path(model, task_name, prompt_key)
            resume_path = results_path if resume else None

            results = run_task(
                client, model, task_name, prompt_key,
                limit=limit, resume_from=resume_path,
            )

            # Save results incrementally
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved {len(results)} results to {results_path}\n")

            all_results.extend(results)

    wall_clock_s = time.monotonic() - run_start

    run_meta = {
        "timestamp": run_timestamp,
        "model": model,
        "tasks": task_list,
        "limit": limit,
        "total_requests": client.total_requests,
        "total_input_tokens": client.total_input_tokens,
        "total_output_tokens": client.total_output_tokens,
        "total_tokens": client.total_input_tokens + client.total_output_tokens,
        "wall_clock_s": round(wall_clock_s, 1),
        "errors": sum(1 for r in all_results if r["error"]),
    }

    print(f"\nDone. Requests: {run_meta['total_requests']}, "
          f"Tokens: {run_meta['total_input_tokens']} in / "
          f"{run_meta['total_output_tokens']} out, "
          f"Wall clock: {run_meta['wall_clock_s']:.0f}s")

    return all_results, run_meta
