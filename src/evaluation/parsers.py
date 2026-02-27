"""Parse free-text model responses into structured answers per task type."""

import re


def parse_integer(response: str) -> str | None:
    """Extract an integer answer from response.

    Tries formats: {N}, plain number, or first number in text.
    """
    # Try curly bracket format first: {3}
    m = re.search(r"\{(\d+)\}", response)
    if m:
        return m.group(1)
    # Try standalone number
    m = re.search(r"\b(\d+)\b", response)
    if m:
        return m.group(1)
    return None


def parse_yes_no(response: str) -> str | None:
    """Extract Yes/No answer from response."""
    text = response.strip().lower()
    # Check for explicit yes/no at start or as standalone word
    if re.search(r"\byes\b", text):
        return "Yes"
    if re.search(r"\bno\b", text):
        return "No"
    return None


def parse_rows_cols(response: str) -> str | None:
    """Extract rows,cols pair from response.

    Tries formats:
    - rows={R} columns={C}
    - (R, C) or (R,C)
    - R rows and C columns
    - rows: R, columns: C
    """
    # rows={R} columns={C}
    m = re.search(r"rows?\s*=\s*\{?(\d+)\}?.*?columns?\s*=\s*\{?(\d+)\}?", response, re.IGNORECASE)
    if m:
        return f"{m.group(1)},{m.group(2)}"
    # (R, C) or (R,C)
    m = re.search(r"\((\d+)\s*,\s*(\d+)\)", response)
    if m:
        return f"{m.group(1)},{m.group(2)}"
    # R rows and C columns (or similar)
    m = re.search(r"(\d+)\s*rows?\b.*?(\d+)\s*columns?\b", response, re.IGNORECASE)
    if m:
        return f"{m.group(1)},{m.group(2)}"
    # columns first: C columns and R rows
    m = re.search(r"(\d+)\s*columns?\b.*?(\d+)\s*rows?\b", response, re.IGNORECASE)
    if m:
        return f"{m.group(2)},{m.group(1)}"  # swap to rows,cols
    return None


PARSERS = {
    "integer": parse_integer,
    "yes_no": parse_yes_no,
    "rows_cols": parse_rows_cols,
}


def parse_response(answer_type: str, response: str) -> str | None:
    """Parse a model response given the expected answer type."""
    parser = PARSERS.get(answer_type)
    if parser is None:
        raise ValueError(f"Unknown answer type: {answer_type}")
    return parser(response)
