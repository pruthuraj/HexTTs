# Deprecated Files Index

This folder contains legacy files kept for reference and rollback safety.
They are not part of the active package-first workflow.

## Active Replacements

- Active base config: configs/base.yaml
- Active continuation auto config: configs/continue_auto.yaml
- Active data preprocessing entry: scripts/prepare_data.py (package-backed)
- Active cached data pipeline: hextts/data/cache_builder.py

## Deprecated Files

- vits_config.yaml
  - Replaced by configs/base.yaml

- vits_config.sanity.yaml
  - Replaced by configs/sanity.yaml

- vits_config.continue3.yaml
  - Replaced by configs/continue3.yaml

- vits_config.continue_auto.yaml
  - Replaced by configs/continue_auto.yaml

- vits_config.continue_auto.legacy_latest.yaml
  - Preserved alternate legacy snapshot of continuation config
  - Use only for historical comparison or rollback

- prepare_data.py
  - Historical copy kept in deprecated; active wrapper remains at project root and in scripts/

- vits_data_cached.py
  - Legacy copy retained; active API is hextts/data/dataloaders.py

- phoneme_duration_patch_record_2026-04-09.md
  - Historical patch notes

## Notes

- Prefer configs/_.yaml and scripts/_.py entrypoints for current workflows.
- If reactivating a deprecated file, verify compatibility with current hextts package contracts first.
