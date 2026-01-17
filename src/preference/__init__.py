"""Preference optimization module - DPO training for re-ranking."""

from .generate_pairs import (
    generate_preference_pairs,
    generate_hard_negative_pairs,
    load_msmarco_data
)

__all__ = [
    "generate_preference_pairs",
    "generate_hard_negative_pairs",
    "load_msmarco_data"
]
