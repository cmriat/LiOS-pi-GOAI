#!/usr/bin/env python3
"""Strip training-machine paths from a checkpoint norm_stats_manifest.json.

The submission checkpoint manifest carries training-side metadata
(dataset_uri, asset_id, source_path) that references machine-local paths
and is irrelevant for inference. Replace those values with placeholders
while keeping every inference contract field (config_name, use_task_embedding,
num_tasks, action_horizon, max_token_len, norm flags, sha256, ...) intact.

Usage:
    python scripts/inference/goai/sanitize_manifest.py /path/to/step30849/norm_stats_manifest.json

Writes the sanitized manifest next to the original as norm_stats_manifest.clean.json;
move it over the original before packaging the checkpoint for submission.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

STRIP_KEYS = {"dataset_uri", "asset_id", "source_path", "checkpoint_path", "repo_id"}


def sanitize(path: Path) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload.get("files", []):
        for key in STRIP_KEYS:
            if key in entry:
                entry[key] = "<redacted>"
    out = path.with_name(path.name.replace(".json", ".clean.json"))
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1])
    if not target.is_file():
        raise SystemExit(f"manifest not found: {target}")
    print(f"sanitized -> {sanitize(target)}")
