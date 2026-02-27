# BlindTest Replication via OpenRouter (+ Ollama)

## Goal

Replicate the BlindTest benchmark evaluation from the paper "Vision Language Models Are Blind" using OpenRouter API for VLM access, then extend to Ollama models.

## Status Tracker

| # | Task | Status |
|---|------|--------|
| 1 | Build ground truth index from existing images | Done |
| 2 | Create OpenRouter evaluation runner | Done |
| 3 | Create answer parser & scorer | Done |
| 4 | Create results aggregator & reporter | Done |
| 5 | Add Ollama backend | Pending |
| 6 | Test end-to-end on a small subset | Pending |
| 7 | Run full benchmark | Pending |

---

## Problem Analysis

### Ground Truth Situation

The repo has **no evaluation code** — the original authors manually sorted images into `correct/incorrect` folders per model. We need to reconstruct ground truth from image metadata:

| Task | Ground Truth Source | Example |
|------|-------------------|---------|
| **LineIntersection** | Filename: `gt_{N}` | `gpt-count_gt_0_image_100_thickness_2_resolution_384.png` → answer: `0` |
| **TouchingCircle** | Folder placement + distance param in filename | `distance_0.05` → touching if distance ≤ 0; need to verify threshold from notebook |
| **CircledWord** | **Not in filename** — must regenerate or extract from notebook | `text_image_{UUID}.png` — ground truth unknown without regeneration metadata |
| **CountingCircles** | **Not in filename** — UUID-based names | Need to count shapes in the generating notebook's output metadata |
| **NestedSquares** | Filename: `depth_{N}` | `nested_squares_depth_3_...` → answer: compute from depth (e.g., depth 3 = ? squares) |
| **CountingRowsAndColumns** | Filename: `{rows}x{cols}` | `blank_grid_3x3_2000_10.png` → rows=3, cols=3 |
| **SubwayMap** | Filename: `path_{N}` for path count; station pair in some names | `pixels_1024_linewidth_20_path_2_...` → answer: `2` |

**Key challenge:** CircledWord and CountingCircles have UUID-based filenames with no ground truth encoded. Options:
1. **Regenerate images** with ground truth in metadata/filename (preferred — notebooks are available)
2. **Use the existing correct/incorrect folders** to reverse-engineer ground truth (fragile)
3. **Start with the 5 tasks that have ground truth in filenames** and handle the other 2 separately

### Existing Image Counts (per model, per prompt variant)

The existing images inside `correct/incorrect` folders are **copies** of the same base images, just sorted by one model's results. The raw test images are few (3-8 per task at root level — these are just examples shown in README).

**The actual test sets were generated in bulk via notebooks but the raw generated sets are stored inside the model-result folders.** We need to either:
- Collect unique images from across model folders (they should be the same images evaluated by different models)
- Or regenerate fresh image sets from the notebooks

### Recommendation: Regenerate Fresh Test Sets

Since we want a clean replication:
1. Run each notebook to generate a fresh, controlled set of test images
2. Save ground truth metadata alongside each image (JSON sidecar or encoded in filename)
3. This gives us clean ground truth and avoids depending on the existing folder structure

**However**, if user prefers to use existing images, we can extract unique base images from the model folders and reconstruct ground truth where possible.

---

## Architecture

```
src/evaluation/
├── config.py              # Model list, API keys, task definitions
├── ground_truth.py        # Extract ground truth from filenames/metadata
├── generate_images.py     # (Optional) Wrapper to regenerate test images with metadata
├── runner.py              # Main evaluation loop: load image → send to model → record response
├── backends/
│   ├── openrouter.py      # OpenRouter API client (base64 image + prompt)
│   └── ollama.py          # Ollama API client (Phase 2)
├── parsers.py             # Parse model responses → extract answer per task type
├── scorer.py              # Compare parsed answer to ground truth
├── reporter.py            # Aggregate results, generate accuracy tables/charts
├── results/               # Output directory for results JSON/CSV
└── run_benchmark.py       # CLI entry point
```

### Phase 1: OpenRouter Evaluation

#### Step 1: Ground Truth Index (`ground_truth.py`)

Build a JSON index mapping image paths to ground truth answers for the 5 "easy" tasks:

```python
# Example index entry
{
    "task": "LineIntersection",
    "image_path": "src/LineIntersection/images/Count-prompt/gpt-4o/correct/gpt-count_gt_0_image_100_thickness_2_resolution_384.png",
    "ground_truth": "0",
    "metadata": {"thickness": 2, "resolution": 384}
}
```

For tasks where ground truth is in the filename:
- **LineIntersection**: parse `gt_{N}` → N
- **NestedSquares**: parse `depth_{N}` → compute total squares (depth N = sum of 1..N? or N²? — verify from notebook)
- **CountingRowsAndColumns**: parse `{rows}x{cols}` → (rows, cols)
- **SubwayMap**: parse `path_{N}` → N
- **TouchingCircle**: parse distance parameter → Yes/No based on threshold

For CircledWord and CountingCircles: defer or regenerate.

#### Step 2: OpenRouter Client (`backends/openrouter.py`)

```python
# OpenAI-compatible API via OpenRouter
# POST https://openrouter.ai/api/v1/chat/completions
# Headers: Authorization: Bearer $OPENROUTER_API_KEY
# Body: standard chat completion with image_url (base64)
```

Key considerations:
- **Rate limiting**: OpenRouter has per-model rate limits; add retry with exponential backoff
- **Cost tracking**: Log token usage per request for cost awareness
- **Base64 encoding**: Read PNG → base64 → data URI
- **Timeout handling**: Some models may be slow; configurable timeout

#### Step 3: Answer Parser (`parsers.py`)

Per-task parsing of free-text model responses:
- **TouchingCircle**: Extract Yes/No (case-insensitive, handle "yes, they are touching" etc.)
- **LineIntersection**: Extract number from `{N}` or plain text
- **NestedSquares**: Extract number from `{N}` or plain text
- **CountingRowsAndColumns**: Extract (rows, cols) pair from various formats
- **SubwayMap**: Extract number from `{N}` or plain text
- **CircledWord**: Extract single letter
- **CountingCircles**: Extract number

#### Step 4: Scorer (`scorer.py`)

Simple comparison: parsed_answer == ground_truth. Per-task, per-model, per-prompt-variant accuracy.

#### Step 5: Reporter (`reporter.py`)

- Per-model accuracy table (matching paper's Table format)
- Per-task breakdown
- Overall accuracy
- Save raw results as JSON for further analysis
- Optionally generate matplotlib charts similar to paper's figures

#### Step 6: CLI Entry Point (`run_benchmark.py`)

```bash
# Run all tasks on all configured models
python src/evaluation/run_benchmark.py

# Run specific task on specific model
python src/evaluation/run_benchmark.py --task LineIntersection --model google/gemini-2.0-flash-001

# Resume interrupted run (skip already-evaluated images)
python src/evaluation/run_benchmark.py --resume

# Generate report from existing results
python src/evaluation/run_benchmark.py --report-only
```

### Phase 1 Models (OpenRouter)

Suggested initial model set (replicating + extending the paper):
- `openai/gpt-4o` (original paper)
- `anthropic/claude-3.5-sonnet` (original paper)
- `google/gemini-2.0-flash-001` (updated from original paper's gemini-1.5)
- `anthropic/claude-sonnet-4-20250514` (new)
- `openai/gpt-4.1` (new)

User can configure via `config.py` or CLI args.

### Phase 2: Ollama Backend

Add `backends/ollama.py` using Ollama's OpenAI-compatible API:
- Endpoint: `http://localhost:11434/v1/chat/completions`
- Same message format (base64 images)
- No API key needed
- Models: `llava`, `llava-next`, `moondream`, `bakllava`, etc.

The runner should accept a `--backend openrouter|ollama` flag and model names appropriate to each.

---

## Decisions Needed from User

1. **Image source**: Regenerate fresh test images from notebooks, or extract/reuse existing images from the model result folders?
2. **Task scope for Phase 1**: Start with all 7 tasks, or begin with the 5 that have ground truth in filenames (skip CircledWord and CountingCircles initially)?
3. **Scale**: How many images per task? The original study used hundreds per condition. For initial testing, a smaller subset (e.g., 20-50 per task) would be faster and cheaper.
4. **Which OpenRouter models** to test first?
