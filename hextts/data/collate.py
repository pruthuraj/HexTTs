"""Collate compatibility layer."""

# Re-export keeps older imports stable while canonical implementation lives in raw_dataset.
from .raw_dataset import collate_fn_vits  # noqa: F401
