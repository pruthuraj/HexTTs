# Deprecated Files Index

This folder contains legacy files kept for reference and rollback safety.
They are not part of the active package-first workflow.

## Active Replacements

- Active base config: configs/base.yaml
- Active continuation auto config: configs/continue_auto.yaml
- Active data preprocessing entry: scripts/prepare_data.py (package-backed)
- Active cached data pipeline: hextts/data/cache_builder.py
- Active architecture doc: docs/architecture.md

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
  - Historical copy kept in deprecated; active entrypoint is scripts/prepare_data.py

- precompute_features.py
  - Active entrypoint is scripts/precompute_features.py

- vits_data_cached.py
  - Legacy copy retained; active API is hextts/data/dataloaders.py

- phoneme_duration_patch_record_2026-04-09.md
  - Historical patch notes

- doc_history/architecture.md
  - Archived architecture note previously stored under doc/

- doc_history/TTS_Phase_1_Foundation.docx
  - Archived phase document

- doc_history/TTS_Phase_2_Setup_Dataset_Preparation.docx
  - Archived phase document

- doc_history/TTS_Phase_3_Training_the_VITS_Model.docx
  - Archived phase document

- doc_history/TTS_Goal_Document.docx
  - Archived project goal document

## Notes

- Prefer configs/_.yaml and scripts/_.py entrypoints for current workflows.
- If reactivating a deprecated file, verify compatibility with current hextts package contracts first.
