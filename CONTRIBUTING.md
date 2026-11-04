# Contributing

Use Python 3.12 and a dedicated virtual environment. Keep the single entry point `run_pipeline.py`; add platform adapters behind its API. Verify the CLI and affected workflows before submitting changes.

Never modify source recordings. Preserve source SHA256, frame indices, timestamps, units, coordinate conventions, validity masks and subprocess provenance. Keep conflicts as NaN. Visual root assistance must not replace the local HumanV2 hand shape; flow is diagnostic. Never apply Tag→bottom→coil twice or promote `training_ready` from render success.

Do not commit personal references, device credentials, recordings, model binaries, generated runs or environment directories. Preserve upstream headers and licenses. Document behavior changes, reproduction commands and known limitations in the README and a sanitized provenance receipt. Contributions must be original or appropriately licensed; do not relicense third-party files under the root BSD license.
