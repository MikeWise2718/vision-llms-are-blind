"""Extract ground truth answers from image filenames and build test indices."""

import os
import re
from dataclasses import dataclass
from itertools import combinations

from .config import IMAGES_BASE, TASKS


EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

STATION_PAIRS = list(combinations("ABCD", 2))  # AB, AC, AD, BC, BD, CD


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


def _parse_counting_shapes(filename: str) -> dict | None:
    """Parse: gpt_count_pixels_{px}_linewidth_{lw}_diameter_{d}_numCircles_{N}_{color}_{id}.png

    Works for both circles and pentagons (both use numCircles_ key).
    """
    m = re.search(r"numCircles_(\d+)", filename)
    if not m:
        return None
    count = int(m.group(1))
    return {
        "answer": str(count),
        "metadata": {
            "count": count,
            "linewidth": _extract_float(filename, r"linewidth_([\d.]+)"),
        },
    }


def _parse_circled_word(filename: str) -> dict | None:
    """Parse: circled_{word}_idx{N}_char{C}_t{thickness}_p{padding}_f{font_id}.png

    Ground truth is the single character C.
    """
    m = re.search(r"circled_(\w+)_idx(\d+)_char([A-Za-z])_t(\d+)_p(\d+)_f(\d+)", filename)
    if not m:
        return None
    word, idx, char, thickness, padding, font_id = m.groups()
    return {
        "answer": char,
        "metadata": {
            "word": word,
            "circle_index": int(idx),
            "char": char,
            "thickness": int(thickness),
            "padding": int(padding),
            "font_id": int(font_id),
        },
    }


def _parse_subway_map(filename: str, station_pair: tuple[str, str]) -> dict | None:
    """Parse: subway_s{size}_lw{thickness}_{pair}{n}_..._seed{S}.png

    Extracts path count for a specific station pair.
    """
    pair_key = f"{''.join(station_pair)}"
    m = re.search(rf"{pair_key}_(\d+)", filename)
    if not m:
        return None
    count = int(m.group(1))
    return {
        "answer": str(count),
        "metadata": {
            "count": count,
            "station_pair": pair_key,
        },
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
    "CountingCircles": lambda f, pk: _parse_counting_shapes(f),
    "CountingPentagons": lambda f, pk: _parse_counting_shapes(f),
    # SubwayMap and CircledWord use special paths in build_index (no entry needed here)
}


def _build_index_standard(task_name: str, task_cfg: dict, prompt_key: str) -> list[TestImage]:
    """Build index for tasks with standard correct/incorrect folder layout."""
    source_model = task_cfg["source_model"]
    parser = PARSERS[task_name]

    # Support dir_name override (e.g. CountingPentagons -> CountingCircles dir)
    dir_name = task_cfg.get("dir_name", task_name)
    image_subdir = task_cfg.get("image_subdir")

    if image_subdir:
        # Layout: src/{dir_name}/images/{subdir}/{prompt_key}/{model}/...
        task_dir = os.path.join(IMAGES_BASE, dir_name, "images", image_subdir, prompt_key, source_model)
    else:
        # Layout: src/{dir_name}/images/{prompt_key}/{model}/...
        task_dir = os.path.join(IMAGES_BASE, dir_name, "images", prompt_key, source_model)

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
    return images


def _build_index_subway(task_cfg: dict, prompt_key: str) -> list[TestImage]:
    """Build index for SubwayMap — generated images, one TestImage per station pair per image."""
    image_dir = os.path.join(EVAL_DIR, task_cfg["image_dir"])
    prompt_template = task_cfg["prompts"][prompt_key]

    if not os.path.isdir(image_dir):
        return []

    images = []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        fpath = os.path.join(image_dir, fname)

        for pair in STATION_PAIRS:
            result = _parse_subway_map(fname, pair)
            if result is None:
                continue
            # Instantiate prompt template with station names
            prompt_text = prompt_template.format(station1=pair[0], station2=pair[1])
            images.append(TestImage(
                task="SubwayMap",
                prompt_key=prompt_key,
                image_path=fpath,
                ground_truth=result["answer"],
                metadata={
                    **result["metadata"],
                    "prompt_text": prompt_text,
                },
            ))
    return images


def _build_index_circledword(task_cfg: dict, prompt_key: str) -> list[TestImage]:
    """Build index for CircledWord — generated images, flat folder."""
    image_dir = os.path.join(EVAL_DIR, task_cfg["image_dir"])

    if not os.path.isdir(image_dir):
        return []

    images = []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        result = _parse_circled_word(fname)
        if result is None:
            continue
        images.append(TestImage(
            task="CircledWord",
            prompt_key=prompt_key,
            image_path=os.path.join(image_dir, fname),
            ground_truth=result["answer"],
            metadata=result["metadata"],
        ))
    return images


def build_index(task_name: str, prompt_key: str, limit: int = 0) -> list[TestImage]:
    """Build a list of TestImage entries for a given task and prompt variant.

    Collects images from the source model's correct/ and incorrect/ folders
    (these contain the full test set that was evaluated).
    For SubwayMap and CircledWord, uses generated images.

    Args:
        task_name: Name of the task (e.g. "LineIntersection")
        prompt_key: Prompt variant key (e.g. "Count-prompt")
        limit: Max images to return (0 = all)
    """
    task_cfg = TASKS[task_name]

    if task_name == "SubwayMap":
        images = _build_index_subway(task_cfg, prompt_key)
    elif task_name == "CircledWord":
        images = _build_index_circledword(task_cfg, prompt_key)
    else:
        images = _build_index_standard(task_name, task_cfg, prompt_key)

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
