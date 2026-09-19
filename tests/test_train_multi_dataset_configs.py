"""Multi-dataset training: one training config, several Lance sources.

A mixed run names its datasets through ``--data.dataset-uri``, and the launcher writes
the comma-separated form (``scripts/train/goai/train.sh`` joins the URIs it validated).
The run then has to reach ``MultiLeRobotLoader`` as one loader config per dataset, while
the run itself stays on a single ``TrainConfig``: the frozen inference chain resolves a
checkpoint through the single ``config_name`` its manifest records
(``load_checkpoint_manifest`` / ``resolve_checkpoint_config_name``), so a run that
expanded into several configs would produce a checkpoint the server cannot load.

These tests drive the real entry point (``train_pytorch_fsdp.main``) and the real config
builder (``data_loader._build_dataset_configs``); only the per-dataset loader is
stubbed, since a real one reads Lance video.
"""

from __future__ import annotations

import sys
import pathlib
import dataclasses

import numpy as np
import pytest

# The training stack reads Lance metadata (lance) through the training env (wandb,
# lerobot); the inference-only environment ships none of them.
pytest.importorskip("wandb")
pytest.importorskip("lerobot")
pytest.importorskip("lance")

import pi.data as pi_data  # noqa: E402
from pi.shared import normalize  # noqa: E402
from pi.training import instance_config  # noqa: E402
from pi.inference.goai_helpers import load_checkpoint_manifest, resolve_checkpoint_config_name  # noqa: E402
from pi.inference.goai_sim_policy import load_goai_checkpoint_training_contract  # noqa: E402

# The trainer runs with scripts/train on sys.path ("from utils import ...").
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train"))
import train_pytorch_fsdp  # noqa: E402
from utils import archive_norm_stats, build_configs_from_dataset_uris  # noqa: E402
from data_loader import _build_dataset_configs  # noqa: E402

# The submitted mixed-recipe config: one Lance source per real-robot dataset conversion.
CONFIG_NAME = "pi05_goai_joint_lance"
DATASET_DIR = "/datasets/goai"
GEOMETRY_METADATA = {
    b"video_lance:image_width": b"640",
    b"video_lance:image_height": b"480",
    b"video_lance:source": b"robot_to_lance",
}


def _lance_source(directory: pathlib.Path) -> str:
    """Write a minimal Lance source carrying the 640x480 robot-frames contract."""
    import lance
    import pyarrow as pa

    schema = pa.schema([pa.field("state", pa.list_(pa.float32()))], metadata=dict(GEOMETRY_METADATA))
    lance.write_dataset(pa.table({"state": [[0.0] * 14]}, schema=schema), str(directory))
    return str(directory)


def _stats(dim: int = 14) -> dict:
    zeros = np.zeros(dim, dtype=np.float32)
    ones = np.ones(dim, dtype=np.float32)
    return {
        "state": normalize.NormStats(mean=zeros, std=ones, q01=zeros, q99=ones),
        "actions": normalize.NormStats(mean=zeros, std=ones, q01=zeros, q99=ones),
    }


def _config(tmp_path: pathlib.Path, dataset_uri, **data_overrides) -> instance_config._config.TrainConfig:
    """Build a launched-looking config, with its joint norm stats staged on disk."""
    template = instance_config.get_config(CONFIG_NAME)
    config = dataclasses.replace(
        template,
        data=dataclasses.replace(template.data, dataset_uri=dataset_uri, **data_overrides),
        exp_name="multi_dataset",
        assets_base_dir=str(tmp_path / "assets"),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
    )
    stats_dir = config.assets_dirs / (config.data.asset_id or config.data.repo_id)
    stats_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(stats_dir, _stats())
    return config


def _run_main(monkeypatch, config) -> list:
    """Run the real entry point with the inspected config; return the configs it trained on."""
    captured: list = []
    monkeypatch.setattr(instance_config, "cli", lambda: config)
    monkeypatch.setattr(train_pytorch_fsdp, "train_loop", lambda cfgs: captured.append(cfgs))
    assert train_pytorch_fsdp.main() == 0
    (cfgs,) = captured
    return cfgs


# ------------------------- loader configs: one per dataset -------------------------


def test_multi_uri_config_expands_to_one_loader_config_per_dataset(tmp_path) -> None:
    uris = [f"{DATASET_DIR}/{name}.lance" for name in ("alpha", "beta", "gamma")]
    config = _config(tmp_path, ",".join(uris))

    ds_cfgs = _build_dataset_configs([config])

    # One loader config per dataset, in launch order, each naming exactly one source --
    # which is the single dataset a SimpleLeRobotLoader accepts.
    assert [ds_cfg.dataset_uri for ds_cfg in ds_cfgs] == uris
    assert [len(pi_data._dataset_uri_values(ds_cfg.dataset_uri, ds_cfg.repo_id)) for ds_cfg in ds_cfgs] == [1, 1, 1]
    # Each source keeps its own name; the run keeps the config's dataset identity.
    assert [ds_cfg.repo_id for ds_cfg in ds_cfgs] == ["alpha", "beta", "gamma"]
    assert {ds_cfg.dataset_format for ds_cfg in ds_cfgs} == {"video_lance"}
    assert {ds_cfg.asset_id for ds_cfg in ds_cfgs} == {config.data.asset_id}
    assert {ds_cfg.policy_name for ds_cfg in ds_cfgs} == {config.name}
    # Joint normalization: one stats file normalizes all three into one action space.
    assert ds_cfgs[0].norm_stats is not None
    assert all(ds_cfg.norm_stats is ds_cfgs[0].norm_stats for ds_cfg in ds_cfgs)
    assert {ds_cfg.use_quantile_norm for ds_cfg in ds_cfgs} == {config.effective_use_quantile_norm}
    # Per-dataset fields the loader forwards downstream are untouched by the split.
    assert {ds_cfg.use_per_timestamp_action_norm for ds_cfg in ds_cfgs} == {config.data.use_per_timestamp_action_norm}
    assert {ds_cfg.apply_delta_transform for ds_cfg in ds_cfgs} == {config.data.apply_delta_transform}
    assert {ds_cfg.policy_state_schema for ds_cfg in ds_cfgs} == {config.data.policy_state_schema}


def test_expanded_configs_build_one_multilerobotloader(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path, f"{DATASET_DIR}/alpha.lance,{DATASET_DIR}/beta.lance,{DATASET_DIR}/gamma.lance")
    ds_cfgs = _build_dataset_configs([config])

    built: list[dict] = []

    class _StubLoader:
        def __init__(self, _config, *_args, **kwargs) -> None:
            built.append(kwargs)
            self.repo_id = kwargs["repo_id"]
            self._length = len(built)  # 1, 2, 3 -> flattened length 6

        def __len__(self) -> int:
            return self._length

        def __getitem__(self, idx: int) -> dict:
            return {"repo_id": self.repo_id, "local_index": idx}

    monkeypatch.setattr(pi_data, "SimpleLeRobotLoader", _StubLoader)
    loader = pi_data.MultiLeRobotLoader(
        datasets=ds_cfgs,
        batch_size=4,
        action_horizon=int(config.model.action_horizon),
        action_dim=int(config.model.action_dim),
        max_token_len=int(config.model.max_token_len),
        use_per_timestamp_action_norm=config.data.use_per_timestamp_action_norm,
        mode="train",
    )

    assert [kwargs["dataset_uri"] for kwargs in built] == [
        f"{DATASET_DIR}/alpha.lance",
        f"{DATASET_DIR}/beta.lance",
        f"{DATASET_DIR}/gamma.lance",
    ]
    assert [kwargs["repo_id"] for kwargs in built] == ["alpha", "beta", "gamma"]
    assert all(kwargs["dataset_format"] == "video_lance" for kwargs in built)
    assert all(kwargs["norm_stats"] is ds_cfgs[0].norm_stats for kwargs in built)
    assert all(kwargs["use_per_timestamp_action_norm"] is True for kwargs in built)
    assert all(kwargs["mode"] == "train" for kwargs in built)

    # The datasets are read as one concatenated stream, in launch order.
    assert len(loader) == 6
    assert [loader[idx]["repo_id"] for idx in range(6)] == [
        "alpha",
        "beta",
        "beta",
        "gamma",
        "gamma",
        "gamma",
    ]


# ------------------------- single dataset: unchanged path -------------------------


@pytest.mark.parametrize(
    "dataset_uri",
    [None, "", f"{DATASET_DIR}/alpha.lance", [f"{DATASET_DIR}/alpha.lance"]],
)
def test_single_source_config_reaches_the_loader_untouched(tmp_path, dataset_uri) -> None:
    config = _config(tmp_path, dataset_uri)

    (ds_cfg,) = _build_dataset_configs([config])

    # Exactly the config the loader was given before multi-dataset support: the repo id
    # is the config's own (never re-derived from the URI) and the URI is passed through.
    assert ds_cfg == dataclasses.replace(
        config.data,
        norm_stats=ds_cfg.norm_stats,
        policy_name=config.name,
        use_quantile_norm=config.effective_use_quantile_norm,
    )
    assert ds_cfg.repo_id == config.data.repo_id
    assert ds_cfg.dataset_uri == config.data.dataset_uri


@pytest.mark.parametrize(
    "dataset_uri",
    [None, f"{DATASET_DIR}/alpha.lance", f"{DATASET_DIR}/alpha.lance,{DATASET_DIR}/beta.lance"],
)
def test_main_hands_both_uri_forms_to_the_loop_as_launched(tmp_path, monkeypatch, dataset_uri) -> None:
    config = _config(tmp_path, dataset_uri)

    captured = _run_main(monkeypatch, config)

    # A comma-separated URI stays on one config: the run keeps one config name, one
    # archived norm stats record and one manifest entry, so inference can resolve it.
    assert len(captured) == 1
    assert captured[0] is config


def test_non_lance_config_keeps_its_multi_uri_mismatch_for_the_loader(tmp_path) -> None:
    """Only Lance reads one source per dataset; other backends keep the old contract."""
    config = _config(tmp_path, f"{DATASET_DIR}/alpha.lance,{DATASET_DIR}/beta.lance", dataset_format="lerobot")

    (ds_cfg,) = _build_dataset_configs([config])

    # Left as launched, so SimpleLeRobotLoader keeps rejecting a config that names
    # several datasets instead of silently reading one of them.
    assert ds_cfg == dataclasses.replace(
        config.data,
        norm_stats=ds_cfg.norm_stats,
        policy_name=config.name,
        use_quantile_norm=config.effective_use_quantile_norm,
    )
    assert len(pi_data._dataset_uri_values(ds_cfg.dataset_uri, ds_cfg.repo_id)) == 2


# ------------------------- repeated flags: rejected loudly -------------------------


@pytest.mark.parametrize(
    "uri_args",
    [
        ["--data.dataset-uri", f"{DATASET_DIR}/alpha.lance"],
        ["--data.dataset-uri", f"{DATASET_DIR}/alpha.lance,{DATASET_DIR}/beta.lance"],
        ["--data.dataset-uri", f"{DATASET_DIR}/alpha.lance,{DATASET_DIR}/beta.lance,{DATASET_DIR}/gamma.lance"],
    ],
)
def test_cli_uri_forms_reach_the_loop_as_one_config(monkeypatch, uri_args) -> None:
    captured: list = []
    monkeypatch.setattr(sys, "argv", ["train_pytorch_fsdp.py", CONFIG_NAME, "--exp-name", "multi_dataset", *uri_args])
    monkeypatch.setattr(train_pytorch_fsdp, "train_loop", lambda cfgs: captured.append(cfgs))

    assert train_pytorch_fsdp.main() == 0

    (cfgs,) = captured
    # Both the launcher's comma-joined string and a lone URI arrive as one config; the
    # datasets it names are split per source further down, in _build_dataset_configs.
    assert len(cfgs) == 1
    assert cfgs[0].data.dataset_uri == uri_args[1]
    assert cfgs[0].data.dataset_format == "video_lance"


def test_cli_space_separated_uris_are_rejected_for_the_comma_form(monkeypatch) -> None:
    uris = [f"{DATASET_DIR}/alpha.lance", f"{DATASET_DIR}/beta.lance"]
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_pytorch_fsdp.py", CONFIG_NAME, "--exp-name", "multi_dataset", "--data.dataset-uri", *uris],
    )
    monkeypatch.setattr(train_pytorch_fsdp, "train_loop", lambda cfgs: pytest.fail(f"must not train on {cfgs}"))

    with pytest.raises(SystemExit, match="comma-separated"):
        train_pytorch_fsdp.main()


def test_repeated_dataset_uri_flags_build_one_config_per_uri(tmp_path) -> None:
    template = _config(tmp_path, None)

    cfgs = build_configs_from_dataset_uris([f"{DATASET_DIR}/alpha.lance", f"{DATASET_DIR}/beta.lance"], template)

    assert [cfg.data.dataset_uri for cfg in cfgs] == [
        f"{DATASET_DIR}/alpha.lance",
        f"{DATASET_DIR}/beta.lance",
    ]
    assert [cfg.data.repo_id for cfg in cfgs] == ["alpha", "beta"]
    assert {cfg.data.dataset_format for cfg in cfgs} == {"video_lance"}
    # Only the dataset fields are rewritten; stats identity and architecture are shared.
    assert {cfg.data.asset_id for cfg in cfgs} == {template.data.asset_id}
    assert {cfg.name for cfg in cfgs} == {template.name}
    assert all(cfg.model == template.model for cfg in cfgs)
    assert all(cfg is not template for cfg in cfgs)


def test_comma_separated_uri_string_is_read_as_several_datasets(tmp_path) -> None:
    template = _config(tmp_path, None)

    cfgs = build_configs_from_dataset_uris(
        f"{DATASET_DIR}/alpha.lance,{DATASET_DIR}/beta.lance,{DATASET_DIR}/gamma.lance", template
    )

    assert [cfg.data.dataset_uri for cfg in cfgs] == [
        f"{DATASET_DIR}/alpha.lance",
        f"{DATASET_DIR}/beta.lance",
        f"{DATASET_DIR}/gamma.lance",
    ]
    assert [cfg.data.repo_id for cfg in cfgs] == ["alpha", "beta", "gamma"]


def test_single_dataset_uri_keeps_the_template_repo_id(tmp_path) -> None:
    template = _config(tmp_path, None, repo_id="goai_2026_joint")

    (cfg,) = build_configs_from_dataset_uris(f"{DATASET_DIR}/alpha.lance", template)

    # A lone dataset is not renamed after its URI: the config's own identity wins.
    assert cfg.data.repo_id == "goai_2026_joint"
    assert cfg.data.dataset_uri == f"{DATASET_DIR}/alpha.lance"
    assert cfg.data.dataset_format == "video_lance"


def test_repeated_flags_rejected_under_parent_data_dir(tmp_path) -> None:
    template = _config(tmp_path, None)
    template = dataclasses.replace(template, parent_data_dir=str(tmp_path / "datasets"))

    with pytest.raises(ValueError, match="parent-data-dir"):
        build_configs_from_dataset_uris([f"{DATASET_DIR}/alpha.lance", f"{DATASET_DIR}/beta.lance"], template)


def test_no_dataset_uri_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="No dataset URIs"):
        build_configs_from_dataset_uris(" , ", _config(tmp_path, None))


def test_main_rejects_repeated_flags_that_share_one_config_name(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path, [f"{DATASET_DIR}/alpha.lance", f"{DATASET_DIR}/beta.lance"])

    with pytest.raises(SystemExit, match="comma-separated"):
        _run_main(monkeypatch, config)


# ------------------------- why a run stays on one config -------------------------


def test_archived_manifest_of_a_mixed_run_is_one_resolvable_record(tmp_path) -> None:
    uris = ",".join(_lance_source(tmp_path / f"{name}.lance") for name in ("alpha", "beta", "gamma"))
    config = _config(tmp_path, uris)
    checkpoint_dir = config.checkpoint_dir

    archive_norm_stats([config], checkpoint_dir, resuming=False)

    manifest = load_checkpoint_manifest(checkpoint_dir)
    assert len(manifest["files"]) == 1
    assert manifest["files"][0]["dataset_uri"] == uris
    assert manifest["files"][0]["asset_id"] == config.data.asset_id
    # The stats the three sources are normalized with ride along, and the checkpoint
    # still resolves to the one config it was trained as.
    assert (checkpoint_dir / manifest["files"][0]["checkpoint_path"]).is_file()
    assert resolve_checkpoint_config_name(checkpoint_dir) == config.name
    contract = load_goai_checkpoint_training_contract(checkpoint_dir, config.name)
    assert contract["use_per_timestamp_action_norm"] is True
    assert contract["image_geometry"]["width"] == 640


def test_one_config_name_cannot_describe_two_archived_configs(tmp_path) -> None:
    """The reason a mixed run is one config: the server reads one entry per name."""
    first = _config(tmp_path, _lance_source(tmp_path / "alpha.lance"))
    second = _config(tmp_path, _lance_source(tmp_path / "beta.lance"))
    checkpoint_dir = first.checkpoint_dir

    archive_norm_stats([first, second], checkpoint_dir, resuming=False)

    manifest = load_checkpoint_manifest(checkpoint_dir)
    assert {entry["config_name"] for entry in manifest["files"]} == {first.name}
    assert len(manifest["files"]) == 2
    with pytest.raises(ValueError, match="Expected one"):
        load_goai_checkpoint_training_contract(checkpoint_dir, first.name)
