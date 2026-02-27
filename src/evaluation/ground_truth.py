"""Extract ground truth answers from image filenames and build test indices."""

import os
import re
from dataclasses import dataclass

from .config import IMAGES_BASE, TASKS


@dataclass
class TestImage:
    task: str
    prompt_key: str
    image_path: str
    ground_truth: str
    metadata: dict


def _parse_line_intersection(filename: str) -> dict | None:
    """Parse: gpt-count_gt_{N}_image_{id}_thickness_{t}_resolution_{r}.png"""
    m = re.search(r"gt_(\d+)", filename)
    if not m:
        return None
    return {
        "answer": m.group(1),
        "metadata": {
            "gt": int(m.group(1)),
            "thickness": _extract_int(filename, r"thickness_(\d+)"),
            "resolution": _extract_int(filename, r"resolution_(\d+)"),
        },
    }


def _parse_touching_circle(filename: str, prompt_key: str) -> dict | None:
    """Parse: gpt_touch_pixels_{px}_rotation_{rot}_diameter_{d}_distance_{dist}.png

    Ground truth depends on prompt:
    - touching-prompt: distance <= 0 → Yes, distance > 0 → No
    - overlapping-prompt: distance < 0 → Yes, distance >= 0 → No
    """
    m = re.search(r"distance_([-\d.]+?)\.png", filename)
    if not m:
        return None
    distance = float(m.group(1))
    if prompt_key == "touching-prompt":
        answer = "Yes" if distance <= 0 else "No"
    else:  # overlapping-prompt
        answer = "Yes" if distance < 0 else "No"
    return {
        "answer": answer,
        "metadata": {
            "distance": distance,
            "diameter": _extract_float(filename, r"diameter_([\d.]+)"),
        },
    }


def _parse_nested_squares(filename: str) -> dict | None:
    """Parse: nested_squares_depth_{N}_image_{id}_thickness_{t}.png

    Answer is N (the depth = number of squares).
    """
    m = re.search(r"depth_(\d+)", filename)
    if not m:
        return None
    depth = int(m.group(1))
    return {
        "answer": str(depth),
        "metadata": {
            "depth": depth,
            "thickness": _extract_int(filename, r"thickness_(\d+)"),
        },
    }


def _parse_counting_rows_cols(filename: str) -> dict | None:
    """Parse: gpt_grid_{R}x{C}_{size}_{param}.png_blank.png
    or: gpt_text_grid_{R}x{C}_{size}_{param}.png_text.png
    """
    m = re.search(r"(\d+)x(\d+)", filename)
    if not m:
        return None
    rows, cols = m.group(1), m.group(2)
    return {
        "answer": f"{rows},{cols}",
        "metadata": {"rows": int(rows), "cols": int(cols)},
    }


def _extract_int(s: str, pattern: str) -> int | None:
    m = re.search(pattern, s)
    return int(m.group(1)) if m else None


def _extract_float(s: str, pattern: str) -> float | None:
    m = re.search(pattern, s)
    return float(m.group(1)) if m else None


PARSERS = {
    "LineIntersection": lambda f, pk: _parse_line_intersection(f),
    "TouchingCircle": _parse_touching_circle,
    "NestedSquares": lambda f, pk: _parse_nested_squares(f),
    "CountingRowsAndColumns": lambda f, pk: _parse_counting_rows_cols(f),
}


def build_index(task_name: str, prompt_key: str, limit: int = 0) -> list[TestImage]:
    """Build a list of TestImage entries for a given task and prompt variant.

    Collects images from the source model's correct/ and incorrect/ folders
    (these contain the full test set that was evaluated).

    Args:
        task_name: Name of the task (e.g. "LineIntersection")
        prompt_key: Prompt variant key (e.g. "Count-prompt")
        limit: Max images to return (0 = all)
    """
    task_cfg = TASKS[task_name]
    source_model = task_cfg["source_model"]
    parser = PARSERS[task_name]

    task_dir = os.path.join(IMAGES_BASE, task_name, "images", prompt_key, source_model)
    images = []

    for subfolder in ["correct", "incorrect"]:
        folder = os.path.join(task_dir, subfolder)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            result = parser(fname, prompt_key)
            if result is None:
                continue
            images.append(TestImage(
                task=task_name,
                prompt_key=prompt_key,
                image_path=os.path.join(folder, fname),
                ground_truth=result["answer"],
                metadata=result["metadata"],
            ))

    if limit > 0:
        images = images[:limit]

    return images


def build_full_index(limit_per_task: int = 0) -> list[TestImage]:
    """Build index across all configured tasks and prompt variants."""
    all_images = []
    for task_name, task_cfg in TASKS.items():
        for prompt_key in task_cfg["prompts"]:
            all_images.extend(build_index(task_name, prompt_key, limit=limit_per_task))
    return all_images
