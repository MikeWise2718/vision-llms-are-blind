"""Configuration for the BlindTest benchmark evaluation."""

import os

# OpenRouter API
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default model (best performer in original paper at 74.94%)
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"

# Base path to task image directories
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_BASE = SRC_DIR

# Results output
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Task definitions
# Each task maps to: image source info, prompts, and ground truth extraction method
TASKS = {
    "LineIntersection": {
        "prompts": {
            "Count-prompt": "Count the intersection points where the blue and red lines meet. Put your answer in curly brackets, e.g., {2}.",
            "HowMany-prompt": "How many times do the blue and red lines touch each other? Answer with a number in curly brackets, e.g., {5}.",
        },
        "answer_type": "integer",
        "source_model": "gpt-4o",  # Use this model's folder to get the test images
    },
    "TouchingCircle": {
        "prompts": {
            "touching-prompt": "Are the two circles touching each other? Answer with Yes/No.",
            "overlapping-prompt": "Are the two circles overlapping? Answer with Yes/No.",
        },
        "answer_type": "yes_no",
        "source_model": "gpt-4o",
    },
    "NestedSquares": {
        "prompts": {
            "count-prompt": "Count total number of squares in the image. Answer with only the number in numerical format in curly brackets e.g. {3}.",
            "how-many-prompt": "How many squares are in the image? Please answer with a number in curly brackets e.g., {10}.",
        },
        "answer_type": "integer",
        "source_model": "gpt-4o",
    },
    "CountingRowsAndColumns": {
        "prompts": {
            "CountRC-prompt": "Count the number of rows and columns and answer with numbers in curly brackets. For example, rows={5} columns={6}",
            "HowManyRC-prompt": "How many rows and columns are in the table? Answer with only the numbers in a pair (row, column), e.g., (5,6)",
        },
        "answer_type": "rows_cols",
        "source_model": "gpt-4o",
    },
}

# Models to evaluate (Phase 1: OpenRouter)
OPENROUTER_MODELS = [
    "anthropic/claude-3.5-sonnet",
]
