"""Column schemas for every pipeline table (shaped like future BigQuery tables)."""

LEVELS = ["product", "issue", "sub_issue", "path"]
SPLITS = ["train", "tuning", "test"]
VERDICTS = ["model right", "label right", "both OK", "both wrong", "ambiguous"]

COMPLAINTS_COLS = [
    "complaint_id",
    "date_received",
    "product",
    "issue",
    "sub_issue",
    "narrative",
    "company",
    "state",
]
TAXONOMY_COLS = ["product", "issue", "sub_issue", "path_id"]
SPLITS_COLS = ["complaint_id", "split", "hand_check", "self_consistency"]
PREDICTIONS_COLS = [
    "complaint_id",
    "classifier",
    "model_id",
    "prompt_version",
    "product",
    "issue",
    "sub_issue",
    "summary",
    "confidence",
    "taxonomy_hash",
    "run_ts",
]
VALIDATOR_RESULTS_COLS = ["complaint_id", "validator", "level", "verdict", "score"]
HUMAN_REVIEW_COLS = ["complaint_id", "level", "verdict", "summary_ok"]
METRICS_COLS = [
    "table",
    "classifier_or_validator",
    "level",
    "metric",
    "value",
    "ci_low",
    "ci_high",
]


def path_id(product: str, issue: str, sub_issue: str) -> str:
    return f"{product} || {issue} || {sub_issue}"
