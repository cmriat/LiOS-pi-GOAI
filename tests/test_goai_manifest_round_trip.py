"""Round trip for the GOAI training manifest: the trainer writes it, inference reads it.

``archive_norm_stats`` / ``copy_norm_stats_archive`` (``scripts/train/utils.py``) are the
only producer of the ``norm_stats_manifest.json`` a checkpoint ships with; the frozen
inference chain (``load_checkpoint_manifest``, ``load_goai_checkpoint_training_contract``,
``validate_checkpoint``) is the only consumer. These tests drive the real producer over
real Lance datasets and read the result back through those consumers, so the two sides
stay pinned to each other field by field.
"""

from __future__ import annotations

import sys
import json
import hashlib
import pathlib
import dataclasses

import pytest

# The producer imports the training stack and reads Lance dataset metadata; the
# inference-only environment ships neither, so this is a training-environment test.
pytest.importorskip("wandb")
pytest.importorskip("lance")

from pi.training import instance_config  # noqa: E402
from pi.shared.embodiment import build_embodiment_contract  # noqa: E402
from pi.inference.goai_helpers import load_checkpoint_manifest  # noqa: E402
from pi.inference.goai_sim_policy import load_goai_checkpoint_training_contract  # noqa: E402
from pi.inference.goai_xpolicylab import validate_checkpoint  # noqa: E402
from pi.shared.goai_state_contract import validate_goai_policy_state_contract  # noqa: E402

# The trainer runs with scripts/train on sys.path ("from utils import ..."), so the
# tests import it the same way.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train"))
from utils import archive_norm_stats, copy_norm_stats_archive  # noqa: E402

GOAI_GEOMETRY = {
    "width": 640,
    "height": 640,
    "resize_mode": "pad",
    "source_width": 640,
    "source_height": 480,
}
GOAI_CONFIG_NAMES = ("pi05_goai_joint", "pi05_goai_joint_lance", "pi05_b1k_goai")
GOAI_CAMERA_KEYS = (
    "observation_images_cam_env",
    "observation_images_cam_left_wrist",
    "observation_images_cam_right_wrist",
)
DATASET_NAME = "goai_2026_joint"


def _write_lance(uri: pathlib.Path, metadata: dict[bytes, bytes]) -> str:
    import lance
    import pyarrow as pa

    schema = pa.schema([pa.field("state", pa.list_(pa.float32()))], metadata=metadata)
    lance.write_dataset(pa.table({"state": [[0.0] * 14]}, schema=schema), str(uri))
    return str(uri)


def _robot_frames_lance(tmp_path: pathlib.Path) -> str:
    """A Lance source storing 640x480 robot frames that are padded at read time."""
    return _write_lance(
        tmp_path / "goai_robot.lance",
        {
            b"video_lance:image_width": b"640",
            b"video_lance:image_height": b"480",
            b"video_lance:source": b"robot_to_lance",
        },
    )


def _padded_lance(tmp_path: pathlib.Path) -> str:
    """A Lance source that already stores square-padded frames."""
    source_sizes = {key: {"width": 640, "height": 480} for key in GOAI_CAMERA_KEYS}
    return _write_lance(
        tmp_path / "goai_padded.lance",
        {
            b"video_lance:image_width": b"640",
            b"video_lance:image_height": b"640",
            b"video_lance:resize_mode": b"pad",
            b"video_lance:source_camera_sizes_json": json.dumps(source_sizes).encode(),
        },
    )


def _goai_config(tmp_path: pathlib.Path, config_name: str, dataset_uri: str, **data_overrides):
    """Build a GOAI config laid out the way scripts/train/goai/train.sh launches it."""
    template = instance_config.get_config(config_name)
    data_fields = {
        "dataset_format": "video_lance",
        "dataset_uri": dataset_uri,
        "asset_id": DATASET_NAME,
        **data_overrides,
    }
    config = dataclasses.replace(
        template,
        data=dataclasses.replace(template.data, **data_fields),
        exp_name="round_trip",
        assets_base_dir=str(tmp_path / "assets"),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
    )
    stats_dir = config.assets_dirs / DATASET_NAME
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "norm_stats.json").write_bytes(json.dumps({"actions": {"mean": [0.0]}}).encode())
    return config


def _entry(manifest: dict, config) -> dict:
    entries = [entry for entry in manifest["files"] if entry["config_name"] == config.name]
    assert len(entries) == 1
    return entries[0]


@pytest.mark.parametrize("lance_fixture", [_robot_frames_lance, _padded_lance])
@pytest.mark.parametrize("config_name", GOAI_CONFIG_NAMES)
def test_produced_manifest_satisfies_the_inference_contract(tmp_path, config_name, lance_fixture) -> None:
    config = _goai_config(tmp_path, config_name, lance_fixture(tmp_path))
    checkpoint_dir = config.checkpoint_dir

    archive_norm_stats([config], checkpoint_dir, resuming=False)

    manifest = load_checkpoint_manifest(checkpoint_dir)
    contract = load_goai_checkpoint_training_contract(checkpoint_dir, config.name)
    entry = _entry(manifest, config)

    # Architecture and normalization markers, read by the model rebuild.
    assert manifest["version"] == 11
    assert contract["use_task_embedding"] is bool(config.model.use_task_embedding)
    assert contract["use_language_with_task_embedding"] is bool(config.model.use_language_with_task_embedding)
    assert contract["use_quantile_norm"] is config.effective_use_quantile_norm
    assert contract["use_per_timestamp_action_norm"] is bool(config.data.use_per_timestamp_action_norm)
    assert contract["apply_delta_transform"] is bool(config.data.apply_delta_transform)
    assert contract["task_embedding_target"] == config.model.task_embedding_target
    assert contract["state_conditioning_mode"] == config.model.state_conditioning_mode
    assert contract["num_tasks"] == config.model.num_tasks
    assert contract["action_horizon"] == config.model.action_horizon
    assert contract["max_token_len"] == config.model.max_token_len
    assert contract["use_embodiment_embedding"] is False
    assert (contract["num_embodiments"], contract["num_embodiment_tokens"]) == (2, 1)

    # Image geometry and the archived statistics the checkpoint is served with.
    assert contract["image_geometry"] == GOAI_GEOMETRY
    stats_name = "norm_stats_pt.json" if config.data.use_per_timestamp_action_norm else "norm_stats.json"
    assert entry["checkpoint_path"] == stats_name
    assert hashlib.sha256((checkpoint_dir / stats_name).read_bytes()).hexdigest() == entry["sha256"]

    # Policy-state contract: exactly what the shared GOAI contract declares.
    policy_state_contract = entry["policy_state_contract"]
    assert validate_goai_policy_state_contract(policy_state_contract) == policy_state_contract
    assert policy_state_contract["state_schema"] == "goai-policy-state-14d-v1"


def test_embodiment_run_is_archived_as_version_12(tmp_path) -> None:
    config = _goai_config(tmp_path, "pi05_goai_joint", _robot_frames_lance(tmp_path))
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            use_embodiment_embedding=True,
            num_embodiments=1,
            num_embodiment_tokens=2,
        ),
    )

    archive_norm_stats([config], config.checkpoint_dir, resuming=False)

    manifest = load_checkpoint_manifest(config.checkpoint_dir)
    contract = load_goai_checkpoint_training_contract(config.checkpoint_dir, config.name)

    assert manifest["version"] == 12
    assert contract["use_embodiment_embedding"] is True
    assert contract["num_embodiments"] == 1
    assert contract["num_embodiment_tokens"] == 2
    assert contract["embodiment_contract"] == build_embodiment_contract(num_embodiments=1, tokens_per_embodiment=2)


def test_served_checkpoint_copies_validate_through_the_frozen_checks(tmp_path) -> None:
    config = _goai_config(tmp_path, "pi05_goai_joint_lance", _robot_frames_lance(tmp_path))
    checkpoint_dir = config.checkpoint_dir
    archive_norm_stats([config], checkpoint_dir, resuming=False)

    # What fsdp_save_model_checkpoint does: the archive rides along with every step.
    step = checkpoint_dir / "step30849"
    copy_norm_stats_archive(checkpoint_dir, step)
    ema = step / "ema"
    ema.mkdir()
    (ema / ".metadata").touch()

    # The server is pointed at the EMA weights; the manifest comes from the step directory.
    contract = load_goai_checkpoint_training_contract(ema, config.name)
    assert contract["use_per_timestamp_action_norm"] is True
    assert contract["max_token_len"] == config.model.max_token_len

    checkpoint, stats = validate_checkpoint(ema)
    assert checkpoint == ema.resolve()
    assert stats == step / "norm_stats_pt.json"


def test_manifest_records_the_launcher_stats_source(tmp_path, monkeypatch) -> None:
    source = "/datasets/goai/goai_all_grip_ptnorm_h32/norm_stats_pt.json"
    monkeypatch.setenv("PI_NORM_STATS_SOURCE_PATH", source)
    config = _goai_config(tmp_path, "pi05_goai_joint_lance", _robot_frames_lance(tmp_path))

    archive_norm_stats([config], config.checkpoint_dir, resuming=False)

    entry = _entry(load_checkpoint_manifest(config.checkpoint_dir), config)
    assert entry["source_path"] == source
    # The staged copy the trainer actually read stays recorded next to it.
    assert entry["asset_id"] == DATASET_NAME


def test_resume_verifies_the_archived_manifest(tmp_path) -> None:
    config = _goai_config(tmp_path, "pi05_goai_joint_lance", _robot_frames_lance(tmp_path))
    checkpoint_dir = config.checkpoint_dir
    archive_norm_stats([config], checkpoint_dir, resuming=False)
    manifest = load_checkpoint_manifest(checkpoint_dir)

    archive_norm_stats([config], checkpoint_dir, resuming=True)

    assert load_checkpoint_manifest(checkpoint_dir) == manifest


def test_resume_rejects_a_changed_architecture(tmp_path) -> None:
    config = _goai_config(tmp_path, "pi05_goai_joint", _robot_frames_lance(tmp_path))
    checkpoint_dir = config.checkpoint_dir
    archive_norm_stats([config], checkpoint_dir, resuming=False)

    changed = dataclasses.replace(config, model=dataclasses.replace(config.model, num_tasks=6))
    with pytest.raises(ValueError, match="does not match archived manifest"):
        archive_norm_stats([changed], checkpoint_dir, resuming=True)


def test_resume_rejects_a_changed_training_recipe(tmp_path, monkeypatch) -> None:
    config = _goai_config(tmp_path, "pi05_goai_joint", _robot_frames_lance(tmp_path))
    checkpoint_dir = config.checkpoint_dir
    archive_norm_stats([config], checkpoint_dir, resuming=False)

    monkeypatch.setenv("PI_LR_DECAY_FLOOR_STEPS", "0")
    with pytest.raises(ValueError, match="Training recipe mismatch"):
        archive_norm_stats([config], checkpoint_dir, resuming=True)


def test_non_goai_config_archives_no_geometry_or_state_contract(tmp_path) -> None:
    template = instance_config.get_config("pi05_airbot")
    config = dataclasses.replace(
        template,
        data=dataclasses.replace(template.data, asset_id="airbot"),
        exp_name="round_trip",
        assets_base_dir=str(tmp_path / "assets"),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
    )
    stats_dir = config.assets_dirs / "airbot"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "norm_stats.json").write_bytes(b'{"actions": {"mean": [0.0]}}')

    archive_norm_stats([config], config.checkpoint_dir, resuming=False)

    manifest = load_checkpoint_manifest(config.checkpoint_dir)
    entry = _entry(manifest, config)
    assert entry["image_geometry"] is None
    assert entry["policy_state_contract"] is None
    assert manifest["version"] == 11


def test_goai_geometry_requires_a_lance_source(tmp_path) -> None:
    config = _goai_config(
        tmp_path,
        "pi05_goai_joint",
        _robot_frames_lance(tmp_path),
        dataset_format="lerobot",
    )

    with pytest.raises(ValueError, match="can only be archived from a VideoLance dataset"):
        archive_norm_stats([config], config.checkpoint_dir, resuming=False)
