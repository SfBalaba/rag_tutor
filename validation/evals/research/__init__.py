from pathlib import Path

import yaml

from .evals import (
    create_test_cases,
    evaluate_dataset,
    filter_best_examples,
    generate_report,
    save_results,
)


def load_research_config():
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


research_config = load_research_config()

__all__ = [
    "create_test_cases",
    "evaluate_dataset",
    "generate_report",
    "filter_best_examples",
    "save_results",
    "research_config",
]
