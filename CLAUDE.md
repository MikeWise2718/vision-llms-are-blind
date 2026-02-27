# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for the paper **"Vision Language Models Are Blind"** (ACCV 2024 Oral). Contains the **BlindTest** benchmark: 7 tasks demonstrating that VLMs fail at simple low-level vision tasks (line intersection counting, circle overlap detection, letter identification, shape counting, grid counting, path following).

Paper: https://arxiv.org/abs/2407.06581 | Site: https://vlmsareblind.github.io/

## Repository Structure

- `src/` — All source code, organized by task
  - `src/{TaskName}/` — Each of the 7 benchmark tasks has its own directory with a Jupyter notebook for image generation, an `images/` folder, and a `commonly_incorrect/` folder
  - `src/prompts.md` — All evaluation prompts (2 variants per task)
  - `src/LinearProbe/` — Feature extraction and classifier training to analyze why VLMs fail
    - `extract_features/` — Scripts to extract vision encoder embeddings (CLIP, Phi-3.5, SigLIP, LLaVA)
    - `train_on_features/` — Train LogisticRegression/MLP classifiers on extracted features
    - `Phi3.5/` — Modified Phi-3.5-Vision model code for intermediate feature access
    - `LLaVA-NeXT/` — LLaVA-OneVision codebase (large submodule)
- `Figures/` — Benchmark result charts

## Running Code

**No requirements.txt or setup.py exists.** Key dependencies (inferred from imports):

- Image generation notebooks: `matplotlib`, `Pillow`, `numpy`, `freetype-py`
- Feature extraction: `torch` (CUDA), `transformers`, `Pillow`, `tqdm`
- Classifier training: `scikit-learn`, `pandas`, `numpy`, `torch`

**Image generation:** Open Jupyter notebooks in `src/{TaskName}/` directories. The only standalone Python module for generation is `src/CircledWord/text_image_generator.py`.

**Linear probe pipeline:**
1. Extract features: `python src/LinearProbe/extract_features/clip_and_phi_extract_vision_features.py` (or `siglip_and_llava_extract_vision_features.py`)
2. Train classifiers: `python src/LinearProbe/train_on_features/{line,circle}_images/{line,circle}_{llava,phi}_run.py`

## The 7 BlindTest Tasks

1. **LineIntersection** — Count intersections of colored lines
2. **TouchingCircle** — Detect if two circles touch/overlap (binary)
3. **CircledWord** — Identify which letter has a red oval around it
4. **CountingCircles** — Count overlapping circles/pentagons (Olympic-logo style)
5. **NestedSquares** — Count total squares in nested pattern
6. **CountingRowsAndColumns** — Count rows and columns in a grid
7. **SubwayMap** — Count single-colored paths between stations

## Key Models Referenced

- `microsoft/Phi-3.5-vision-instruct`
- `openai/clip-vit-base-patch32`
- `lmms-lab/llava-onevision-qwen2-0.5b-si`

## Development Philosophy

This is a fork of a published research paper's repository. The primary goal is reproducing and extending the original study's results. Avoid gratuitous changes to existing code, tooling, or project structure — don't adopt conventions (e.g., `uv`, `rich`, `rich-argparse`) that would diverge from the upstream repo. New code in `src/evaluation/` is ours; everything else should stay as-is unless there's a specific reason to change it.

## Notes

- Benchmark evaluation uses model APIs at default settings (temperature=1)
- The LinearProbe analysis tests whether task-relevant information exists in vision features before vs. after projection to language space
- `src/LinearProbe/Phi3.5/modeling_phi3_v.py` is a modified copy of the Phi-3.5-Vision model with hooks for extracting intermediate representations
