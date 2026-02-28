"""Score parsed model answers against ground truth."""


def score(answer_type: str, parsed_answer: str | None, ground_truth: str) -> bool:
    """Return True if parsed_answer matches ground_truth."""
    if parsed_answer is None:
        return False

    if answer_type == "yes_no":
        return parsed_answer.lower() == ground_truth.lower()

    if answer_type == "rows_cols":
        # Both should be "R,C" format
        return parsed_answer.strip() == ground_truth.strip()

    if answer_type == "integer":
        try:
            return int(parsed_answer) == int(ground_truth)
        except ValueError:
            return False

    if answer_type == "letter":
        return parsed_answer.strip().lower() == ground_truth.strip().lower()

    return parsed_answer.strip() == ground_truth.strip()
