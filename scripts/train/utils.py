"""Utility functions for FSDP training."""

from __future__ import annotations

import os
import glob
import json
import math
import shutil
import hashlib
import logging
import pathlib
import dataclasses
from typing import Tuple
from collections.abc import Sequence

import numpy as np
import torch
import wandb
import torch.distributed as dist
import torch.distributed.checkpoint
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor

import pi.models.model as _model  # noqa: E402
import pi.shared.download as _download  # noqa: E402
import pi.training.config as _config  # noqa: E402
from pi.shared.embodiment import build_embodiment_contract  # noqa: E402
from pi.shared.goai_state_contract import build_goai_policy_state_contract  # noqa: E402

logger = logging.getLogger()


def init_dist(*, backend: str | None = None) -> Tuple[int, int, int, torch.device]:
    """Initialize torch.distributed (env://) with optional backend override."""
    requested_backend = backend.lower() if backend else None
    is_initialized = dist.is_initialized()

    if not is_initialized:
        backend_in_use = requested_backend or "nccl"
        dist.init_process_group(backend=backend_in_use, init_method="env://")
        logging.info(f"Initialized torch.distributed with backend={backend_in_use}")
    else:
        backend_in_use = str(dist.get_backend()).lower()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if not is_initialized:
        if backend_in_use == "nccl":
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            else:
                raise RuntimeError("This script requires CUDA when using the NCCL backend.")
        else:
            logger.info("Skipping CUDA device setup because non-NCCL backend is in use.")

    if backend_in_use == "nccl":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    logger.info(f"{rank=},{device=}")
    return rank, world_size, local_rank, device


def validate_shared_fields(configs: list[_config.TrainConfig]) -> None:
    if not configs:
        raise ValueError("configs must be non-empty")
    fields = (
        ("action_horizon", lambda c: c.model.action_horizon),
        ("action_dim", lambda c: c.model.action_dim),
        ("max_token_len", lambda c: c.model.max_token_len),
        ("discrete_state_input", lambda c: getattr(c.model, "discrete_state_input", False)),
        ("batch_size", lambda c: c.batch_size),
        ("pytorch_training_precision", lambda c: c.pytorch_training_precision),
    )
    for name, fn in fields:
        v0 = fn(configs[0])
        for c in configs[1:]:
            if fn(c) != v0:
                raise ValueError(f"All configs must share {name}. Got {v0} vs {fn(c)}")


# ------------------------- norm stats archive and training manifest -------------------------
def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _dataset_uri_values(dataset_uri: str | Sequence[str] | None) -> list[str]:
    """Normalize a manifest dataset URI into the list of sources a run reads."""
    if dataset_uri is None:
        return []
    if isinstance(dataset_uri, str):
        # Comma-separated multi-URI support (e.g. "a.lance,b.lance").
        return [uri.strip() for uri in dataset_uri.split(",") if uri.strip()]
    return [str(uri) for uri in dataset_uri if str(uri)]


def _dataset_uri_for_manifest(dataset_uri: str | Sequence[str] | None) -> str | list[str] | None:
    if dataset_uri is None or isinstance(dataset_uri, str):
        return dataset_uri
    return [str(uri) for uri in dataset_uri]


def _image_geometry_for_manifest(config: _config.TrainConfig) -> dict[str, object] | None:
    """Read the image geometry contract that every GOAI Lance source must agree on."""
    if "goai" not in config.name.lower():
        return None
    if config.data.dataset_format != "video_lance":
        raise ValueError(f"{config.name} image geometry can only be archived from a VideoLance dataset")
    dataset_uris = _dataset_uri_values(config.data.dataset_uri)
    if not dataset_uris:
        raise ValueError(f"{config.name} training requires at least one dataset URI")

    import lance

    from pi import data as pi_data

    expected_geometry: dict[str, object] | None = None
    for dataset_uri in dataset_uris:
        dataset = lance.dataset(dataset_uri)
        geometry = pi_data._validate_goai_lance_image_metadata(dict(dataset.schema.metadata or {}), dataset_uri)
        if expected_geometry is None:
            expected_geometry = geometry
        elif geometry != expected_geometry:
            raise ValueError(
                f"{config.name} datasets must share one image geometry contract: "
                f"expected {expected_geometry}, got {geometry} for {dataset_uri!r}"
            )
    return expected_geometry


def _policy_state_contract_for_manifest(config: _config.TrainConfig) -> dict[str, object] | None:
    """Archive the state representation visible to the policy separately from source data."""
    if "goai" not in config.name.lower():
        return None
    return build_goai_policy_state_contract(config.data.policy_state_schema)


def _manifest_stable_value(entry: dict[str, object], key: str) -> object:
    """Apply explicit compatibility defaults for historical manifests."""
    value = entry.get(key)
    if key == "use_language_with_task_embedding" and value is None:
        return False
    if key == "use_embodiment_embedding" and value is None:
        return False
    if key == "task_embedding_target" and value is None:
        return "vlm"
    if key == "state_conditioning_mode" and value is None:
        return "discrete_vlm"
    if key == "policy_state_contract" and value is None and "goai" in str(entry.get("config_name", "")).lower():
        return build_goai_policy_state_contract("source")
    return value


def archive_norm_stats(configs: list[_config.TrainConfig], checkpoint_dir: pathlib.Path, *, resuming: bool) -> None:
    """Copy the active normalization stats into the experiment checkpoint directory.

    A resumed run must use the same archived stats and normalization mode. For old
    checkpoints without archived stats, the current stats are added on first resume.
    """
    if not configs:
        raise ValueError("configs must be non-empty")
    if resuming and not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist for resume")

    manifest_entries = []
    for index, config in enumerate(configs):
        asset_id = config.data.asset_id or config.data.repo_id
        if not asset_id:
            raise ValueError(f"Asset ID and repo ID are both missing for config '{config.name}'.")

        assets_location = str(config.assets_dirs / asset_id)
        source_dir = _download.maybe_download(assets_location)
        source = source_dir / "norm_stats.json"
        if not source.is_file():
            raise FileNotFoundError(f"Normalization stats file not found: {source}")

        stats_filename = "norm_stats_pt.json" if config.data.use_per_timestamp_action_norm else "norm_stats.json"
        if len(configs) == 1:
            relative_destination = pathlib.Path(stats_filename)
            legacy_relative_destination = pathlib.Path("norm_stats.json")
        else:
            relative_destination = pathlib.Path("norm_stats") / f"{index:02d}" / stats_filename
            legacy_relative_destination = pathlib.Path("norm_stats") / f"{index:02d}" / "norm_stats.json"
        if (
            resuming
            and config.data.use_per_timestamp_action_norm
            and not (checkpoint_dir / relative_destination).exists()
            and (checkpoint_dir / legacy_relative_destination).exists()
        ):
            relative_destination = legacy_relative_destination
        destination = checkpoint_dir / relative_destination
        source_sha256 = _file_sha256(source)

        if destination.exists():
            archived_sha256 = _file_sha256(destination)
            if archived_sha256 != source_sha256:
                raise ValueError(
                    f"Normalization stats mismatch for config '{config.name}': active file {source} "
                    f"has sha256={source_sha256}, but archived file {destination} "
                    f"has sha256={archived_sha256}. Refusing to resume with different stats."
                )
            logging.info(f"Verified archived norm stats for '{config.name}': {destination}")
        else:
            _atomic_copy(source, destination)
            if resuming:
                logging.warning(f"Added missing norm stats archive to legacy checkpoint: {destination}")
            else:
                logging.info(f"Archived norm stats for '{config.name}': {source} -> {destination}")

        source_path = str(source)
        if len(configs) == 1:
            # The launcher stages the active stats in a throwaway directory; the manifest
            # records where they came from instead.
            source_path = os.environ.get("PI_NORM_STATS_SOURCE_PATH") or source_path
        use_embodiment_embedding = bool(getattr(config.model, "use_embodiment_embedding", False))
        entry = {
            "config_name": config.name,
            "repo_id": config.data.repo_id,
            "dataset_uri": _dataset_uri_for_manifest(config.data.dataset_uri),
            "asset_id": config.data.asset_id,
            "source_path": source_path,
            "checkpoint_path": relative_destination.as_posix(),
            "sha256": source_sha256,
            "use_quantile_norm": bool(config.effective_use_quantile_norm),
            "apply_delta_transform": bool(config.data.apply_delta_transform),
            "use_per_timestamp_action_norm": bool(config.data.use_per_timestamp_action_norm),
            "use_task_embedding": bool(getattr(config.model, "use_task_embedding", False)),
            "use_language_with_task_embedding": bool(getattr(config.model, "use_language_with_task_embedding", False)),
            "task_embedding_target": str(getattr(config.model, "task_embedding_target", "vlm")),
            "state_conditioning_mode": str(getattr(config.model, "state_conditioning_mode", "discrete_vlm")),
            "num_tasks": int(getattr(config.model, "num_tasks", 0)),
            "max_token_len": int(config.model.max_token_len),
            "action_horizon": config.model.action_horizon,
            # Written for every entry so a mixed-architecture run still produces a
            # manifest whose every entry can be read back.
            "use_embodiment_embedding": use_embodiment_embedding,
            "image_geometry": _image_geometry_for_manifest(config),
            "policy_state_contract": _policy_state_contract_for_manifest(config),
        }
        if use_embodiment_embedding:
            num_embodiments = int(getattr(config.model, "num_embodiments", 2))
            num_embodiment_tokens = int(getattr(config.model, "num_embodiment_tokens", 1))
            entry.update(
                {
                    "num_embodiments": num_embodiments,
                    "num_embodiment_tokens": num_embodiment_tokens,
                    "embodiment_contract": build_embodiment_contract(
                        num_embodiments=num_embodiments,
                        tokens_per_embodiment=num_embodiment_tokens,
                    ),
                }
            )
        manifest_entries.append(entry)

    recipe = _resolve_training_recipe()
    manifest_version = 12 if any(entry["use_embodiment_embedding"] for entry in manifest_entries) else 11
    manifest = {"version": manifest_version, "files": manifest_entries, "training_recipe": recipe}
    manifest_path = checkpoint_dir / "norm_stats_manifest.json"
    if resuming and manifest_path.exists():
        try:
            archived_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read archived norm stats manifest: {manifest_path}") from error

        stable_keys = (
            "config_name",
            "repo_id",
            "dataset_uri",
            "checkpoint_path",
            "sha256",
            "use_quantile_norm",
            "use_per_timestamp_action_norm",
            "use_task_embedding",
            "use_language_with_task_embedding",
            "use_embodiment_embedding",
            "num_embodiments",
            "num_embodiment_tokens",
            "embodiment_contract",
            "task_embedding_target",
            "state_conditioning_mode",
            "num_tasks",
            "max_token_len",
            "action_horizon",
            "image_geometry",
            "policy_state_contract",
        )
        archived_signature = [
            {key: _manifest_stable_value(entry, key) for key in stable_keys}
            for entry in archived_manifest.get("files", [])
        ]
        current_signature = [
            {key: _manifest_stable_value(entry, key) for key in stable_keys} for entry in manifest_entries
        ]
        if archived_signature != current_signature:
            raise ValueError(
                f"Current normalization configuration does not match archived manifest {manifest_path}. "
                "Refusing to resume with different normalization settings."
            )
        _verify_training_recipe_on_resume(archived_manifest, recipe)
        logging.info(f"Verified archived norm stats manifest: {manifest_path}")
    else:
        _atomic_write_json(manifest_path, manifest)
        if resuming:
            logging.warning(f"Added missing norm stats manifest to legacy checkpoint: {manifest_path}")
        else:
            logging.info(f"Saved norm stats manifest: {manifest_path}")


def copy_norm_stats_archive(checkpoint_dir: pathlib.Path, destination: pathlib.Path) -> None:
    """Copy the experiment-level norm stats archive into a model checkpoint."""
    manifest = checkpoint_dir / "norm_stats_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Norm stats manifest not found: {manifest}")

    destination.mkdir(parents=True, exist_ok=True)
    _atomic_copy(manifest, destination / manifest.name)

    archived_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in archived_manifest.get("files", []):
        relative_path = pathlib.Path(entry["checkpoint_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Invalid norm stats path in {manifest}: {relative_path}")
        source = checkpoint_dir / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"Archived norm stats referenced by {manifest} are missing: {source}")
        _atomic_copy(source, destination / relative_path)
    logging.info(f"Copied norm stats archive into model checkpoint: {destination}")


def _resolve_training_recipe() -> dict[str, int]:
    """Parse the env-gated training knobs (defaults = historical behavior).

    Kept env-only to avoid growing TrainConfig; the values are archived in
    norm_stats_manifest.json and compared on resume, so a resumed run cannot
    silently switch the schedule it was launched with.

    The 20000 default is the floor runs archived before the knob existed, and stays the
    legacy value ``_verify_training_recipe_on_resume`` compares against, so it cannot
    move to the submission's 0 (docs-GOAI/technical_solution.md section 3) without
    refusing those resumes. The launcher sets the submission's value explicitly:
    ``scripts/train/goai/train.sh`` exports ``PI_LR_DECAY_FLOOR_STEPS=0``.
    """
    floor = int(os.environ.get("PI_LR_DECAY_FLOOR_STEPS", "20000"))
    if floor < 0:
        raise ValueError(f"PI_LR_DECAY_FLOOR_STEPS must be >= 0, got {floor}")
    return {"decay_floor_steps": floor}


def _verify_training_recipe_on_resume(archived_manifest: dict[str, object], recipe: dict[str, int]) -> None:
    """Reject a resume whose archived env-gated training knobs differ."""
    legacy_recipe = {"decay_floor_steps": 20000}
    archived_recipe = archived_manifest.get("training_recipe") or legacy_recipe
    if archived_recipe != recipe:
        raise ValueError(
            f"Training recipe mismatch: archived {archived_recipe} != current {recipe}. "
            "Refusing to resume with different env-gated training knobs."
        )


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


# ------------------------- logging utilities -------------------------
def init_logging():
    """Initialize logging with custom formatter."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def log_memory_usage(device, step, phase="unknown"):
    """Log GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return
    mem_alloc = torch.cuda.memory_allocated(device) / 1e9
    mem_resv = torch.cuda.memory_reserved(device) / 1e9
    mem_free = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1e9
    stats = torch.cuda.memory_stats(device)
    peak_alloc = stats.get("allocated_bytes.all.peak", 0) / 1e9
    peak_resv = stats.get("reserved_bytes.all.peak", 0) / 1e9
    ddp_info = f" | dist: rank={dist.get_rank()}, world={dist.get_world_size()}" if dist.is_initialized() else ""
    logging.info(
        f"Step {step} ({phase}): GPU mem alloc={mem_alloc:.2f}GB, resv={mem_resv:.2f}GB, free={mem_free:.2f}GB, "
        f"peak_alloc={peak_alloc:.2f}GB, peak_resv={peak_resv:.2f}GB{ddp_info}"
    )


def lr_schedule(step, total_steps, config):
    lr_config = config.lr_schedule
    # Parsed before the warmup early-return so a bad value fails at step 0.
    recipe = _resolve_training_recipe()
    if step == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
        logging.info("Effective training recipe: %s", recipe)
    if step < lr_config.warmup_steps:
        init_lr = lr_config.peak_lr / (lr_config.warmup_steps + 1)
        return init_lr + (lr_config.peak_lr - init_lr) * step / lr_config.warmup_steps
    decay_steps = max(lr_config.decay_steps, total_steps + recipe["decay_floor_steps"])
    progress = min(1.0, (step - lr_config.warmup_steps) / max(1, decay_steps - lr_config.warmup_steps))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_config.decay_lr + (lr_config.peak_lr - lr_config.decay_lr) * cos


def fsdp_wrap(model: torch.nn.Module, all_fp32=False) -> torch.nn.Module:
    def _select_mp_policy_bf16() -> MixedPrecisionPolicy | None:
        # Prefer bf16 on Ampere+; otherwise fp16. Inputs will be cast at forward.
        return MixedPrecisionPolicy(
            param_dtype=torch.bfloat16 if not all_fp32 else torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=None,
            cast_forward_inputs=True,
        )

    mp_policy = _select_mp_policy_bf16()

    with torch.no_grad():
        model.to(torch.float32)
    if dist.is_initialized():
        with torch.no_grad():  # critical path: avoid autograd tracking
            for t in model.state_dict().values():
                if torch.is_tensor(t) and t.numel() > 0:
                    dist.broadcast(t, src=0)
    # fully_shard mutates module in-place; create optimizer AFTER this.
    fully_shard(module=model, mp_policy=mp_policy, reshard_after_forward=False)
    return model


# ------------------------- gradient clipping -------------------------
@torch.no_grad()
def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh=None,
) -> torch.Tensor:
    """Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)

    # Group gradients and parameters by device mesh to handle mixed meshes (e.g., EP + non-EP layers)
    mesh_to_grads = {}
    mesh_to_params = {}
    for p in parameters:
        if p.grad is not None:
            if isinstance(p.grad, DTensor):
                mesh_key = str(p.grad.device_mesh)
            else:
                # Regular tensors
                mesh_key = "local"

            if mesh_key not in mesh_to_grads:
                mesh_to_grads[mesh_key] = []
                mesh_to_params[mesh_key] = []
            mesh_to_grads[mesh_key].append(p.grad)
            mesh_to_params[mesh_key].append(p)

    # Compute total norm for each mesh group separately, then combine
    group_norms = []
    for grad_group in mesh_to_grads.values():
        group_norm = torch.nn.utils.get_total_norm(grad_group, norm_type, error_if_nonfinite, foreach)
        if isinstance(group_norm, DTensor):
            group_norm = group_norm.full_tensor()
        group_norms.append(group_norm)

    # Combine norms from different meshes
    if math.isinf(norm_type):
        total_norm = torch.stack(group_norms).max()
    else:
        total_norm_p = sum(norm**norm_type for norm in group_norms)
        total_norm = total_norm_p ** (1.0 / norm_type)

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    # Apply gradient clipping to each mesh group separately using the global total_norm
    for params_in_group in mesh_to_params.values():
        torch.nn.utils.clip_grads_with_norm_(params_in_group, max_norm, total_norm, foreach)

    return total_norm


def build_configs_from_parent_dir(
    parent_dir: str | pathlib.Path, template: _config.TrainConfig
) -> list[_config.TrainConfig]:
    """Build a list of TrainConfig by globbing first-level subdirectories.

    Only repo_id differs across configs; all other fields follow the template.
    """
    # NOTE(critical path): use glob for first-level dirs and keep absolute paths for repo_id
    pattern = os.path.join(str(parent_dir), "*/")
    candidates = sorted(glob.glob(pattern))
    subdirs = [p for p in candidates if os.path.isdir(p)]
    if not subdirs:
        raise FileNotFoundError(f"No first-level subdirectories found under: {parent_dir}")

    cfgs: list[_config.TrainConfig] = []
    for d in subdirs:
        abs_d = os.path.abspath(d)
        # Replace only the nested DataConfigFactory.repo_id while keeping other fields intact
        new_data = dataclasses.replace(template.data, repo_id=abs_d)
        new_cfg = dataclasses.replace(template, data=new_data)
        cfgs.append(new_cfg)
        logging.info(f"Built config for repo_id: {abs_d}")
    return cfgs


def _repo_id_from_dataset_uri(uri: str) -> str:
    """Derive the per-source repo id a VideoLance URI is read under."""
    basename = uri.rstrip("/").split("?", 1)[0].split("#", 1)[0].rsplit("/", 1)[-1]
    return basename.removesuffix(".lance") or uri


def build_configs_from_dataset_uris(
    dataset_uri: str | Sequence[str] | None, template: _config.TrainConfig
) -> list[_config.TrainConfig]:
    """Build one VideoLance TrainConfig per dataset URI.

    Used for the repeated-flag form of ``--data.dataset-uri``. A run whose configs
    share a name must be launched with a comma-separated URI instead: a checkpoint
    manifest may record only one entry per config name (see ``main`` in
    ``train_pytorch_fsdp.py``), so several configs cannot describe one checkpoint.
    """
    uris = _dataset_uri_values(dataset_uri)
    if not uris:
        raise ValueError("No dataset URIs provided.")
    if template.parent_data_dir:
        raise ValueError("--parent-data-dir cannot be used together with multiple --data.dataset-uri values.")

    cfgs: list[_config.TrainConfig] = []
    for uri in uris:
        repo_id = template.data.repo_id if len(uris) == 1 and template.data.repo_id else _repo_id_from_dataset_uri(uri)
        new_data = dataclasses.replace(
            template.data,
            dataset_format="video_lance",
            repo_id=repo_id,
            dataset_uri=uri,
        )
        cfgs.append(dataclasses.replace(template, data=new_data))
        logging.info("Built VideoLance config for repo_id=%s uri=%s", repo_id, uri)
    return cfgs


def _tree_map_to_device(item, target_device):
    if isinstance(item, dict):
        return {k: _tree_map_to_device(v, target_device) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        converted = [_tree_map_to_device(v, target_device) for v in item]
        return type(item)(converted)
    if isinstance(item, np.ndarray):
        return torch.from_numpy(item).to(target_device)
    if hasattr(item, "to"):
        return item.to(target_device)
    return item


def run_test_evaluation(
    model: torch.nn.Module,
    test_dataloader,
    device: torch.device,
    rank: int,
    is_main: bool,
    epoch: int | None = None,
    global_step: int | None = None,
) -> float | None:
    if (epoch is None and global_step is None) or (epoch is not None and global_step is not None):
        raise ValueError("Exactly one of 'epoch' or 'global_step' must be provided, not both or neither")

    if is_main:
        if epoch is not None:
            logging.info(f"Starting test evaluation at epoch {epoch}")
        else:
            logging.info(f"Starting test evaluation at step {global_step}")
    model_was_training = model.training
    model.eval()
    test_losses: list[float] = []
    with torch.no_grad():
        for test_batch in test_dataloader:
            test_batch = _tree_map_to_device(test_batch, device)
            test_observation_dict = {k: v for k, v in test_batch.items() if k != "actions"}
            # Same bridge as the training loop: the model reads the observation's own
            # camera keys and converts the uint8 images, so it takes an Observation
            # rather than the raw collated dict.
            test_observation = _model.Observation.from_dict(test_observation_dict)
            test_actions = test_batch["actions"].to(torch.float32)
            test_loss_tensor = model(test_observation, test_actions)
            if isinstance(test_loss_tensor, (list, tuple)):
                test_loss_tensor = torch.stack(list(test_loss_tensor))
            elif not isinstance(test_loss_tensor, torch.Tensor):
                test_loss_tensor = torch.tensor(
                    test_loss_tensor,
                    dtype=torch.float32,
                    device=device,
                )
            test_losses.append(test_loss_tensor.mean().item())
    if test_losses:
        loss_sum = torch.tensor(
            [sum(test_losses), len(test_losses)],
            dtype=torch.float64,
            device=device,
        )
    else:
        loss_sum = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    total_loss, total_batches = loss_sum.tolist()
    test_loss_value = None
    if total_batches > 0:
        test_loss_value = total_loss / total_batches
        if rank == 0:
            if epoch is not None:
                logging.info(f"Finished test evaluation at epoch {epoch} with loss {test_loss_value:.6f}")
            else:
                logging.info(f"Finished test evaluation at step {global_step} with loss {test_loss_value:.6f}")
    if model_was_training:
        model.train()
    return test_loss_value


def fsdp_save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: _config.TrainConfig,
    is_main: bool,
    epoch: int = None,
    step: int = None,
    ema_model=None,
) -> None:
    """Save FSDP model checkpoint.

    Args:
        model: The FSDP-wrapped model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number (used as checkpoint directory name), if None, use step instead
        step: Current step number (used as checkpoint directory name), if None, use epoch instead
        config: Training configuration
        is_main: Whether this is the main process
        ema_model: Optional EMA model to save

    Note: This function no longer checks save_interval internally.
    The caller should decide when to save (e.g., based on epoch).
    """
    # Ensure exactly one of epoch or step is provided
    if (epoch is None and step is None) or (epoch is not None and step is not None):
        raise ValueError("Exactly one of 'epoch' or 'step' must be provided, not both or neither")

    # Use temporary directory for atomic save
    if epoch is not None:
        logging.info(f"Saving FSDP model checkpoint at epoch {epoch}...")
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_epoch{epoch}"
        final_ckpt_dir = config.checkpoint_dir / f"epoch{epoch}"
    else:
        logging.info(f"Saving FSDP model checkpoint at step {step}...")
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_step{step}"
        final_ckpt_dir = config.checkpoint_dir / f"step{step}"

    # Clean up any existing temp directory first
    if is_main:
        shutil.rmtree(tmp_ckpt_dir, ignore_errors=True)
    torch.distributed.barrier()

    # Save to temporary directory
    torch.distributed.checkpoint.save(model.state_dict(), checkpoint_id=tmp_ckpt_dir)

    # Save EMA parameters using distributed checkpoint (sharded across ranks)
    if ema_model is not None:
        ema_tmp_dir = tmp_ckpt_dir / "ema"
        torch.distributed.checkpoint.save(ema_model.shadow, checkpoint_id=ema_tmp_dir)

    # Save RNG states (only main process saves, all processes use same RNG states per rank)
    if is_main:
        rng_state = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()

        rng_path = tmp_ckpt_dir / "rng_state.pth"
        torch.save(rng_state, rng_path)
        logging.info(f"Saved RNG states to {rng_path}")
        copy_norm_stats_archive(config.checkpoint_dir, tmp_ckpt_dir)

    # Atomically rename temp directory to final (only main process does the rename)
    torch.distributed.barrier()
    if is_main:
        shutil.rmtree(final_ckpt_dir, ignore_errors=True)
        tmp_ckpt_dir.rename(final_ckpt_dir)
    torch.distributed.barrier()

    if epoch is not None:
        logging.info(f"Saved FSDP model checkpoint at epoch {epoch} -> {final_ckpt_dir}")
    else:
        logging.info(f"Saved FSDP model checkpoint at step {step} -> {final_ckpt_dir}")

    # Save optimizer checkpoint (also use atomic rename)
    tmp_optim_dir = config.checkpoint_dir / "tmp_last_optim"
    optim_ckpt_dir = config.checkpoint_dir / "last_optim"

    if is_main:
        shutil.rmtree(tmp_optim_dir, ignore_errors=True)
    torch.distributed.barrier()
    torch.distributed.checkpoint.save(optimizer.state_dict(), checkpoint_id=tmp_optim_dir)

    torch.distributed.barrier()
    if is_main:
        shutil.rmtree(optim_ckpt_dir, ignore_errors=True)
        tmp_optim_dir.rename(optim_ckpt_dir)
    torch.distributed.barrier()

    if epoch is not None:
        logging.info(f"Saved FSDP optimizer checkpoint at epoch {epoch} -> {optim_ckpt_dir}")
    else:
        logging.info(f"Saved FSDP optimizer checkpoint at step {step} -> {optim_ckpt_dir}")


def resume_from_fsdp_model_checkpoint(model, optimizer, checkpoint_dir: pathlib.Path, ema_model=None) -> int:
    """Load FSDP model and optimizer checkpoints from directory.

    Only loads from step-based checkpoints (e.g., step0, step10, step100).
    Epoch-based checkpoints are ignored for resume.

    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer
        checkpoint_dir: Directory containing checkpoints (must exist)
        ema_model: Optional EMA model

    Returns:
        int: The next global step to start training from (last_saved_step + 1).
             For example, if step10 was loaded, returns 11.

    Raises:
        FileNotFoundError: If no valid step-based checkpoint is found.
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

    last_global_step = -1
    last_ckpt_name = None

    for dir in checkpoint_dir.glob("*"):
        if not dir.is_dir():
            continue
        # Skip temporary directories, optimizer dir, and epoch-based checkpoints
        if dir.name.startswith("tmp_") or dir.name == "last_optim" or dir.name.startswith("epoch"):
            continue
        try:
            # Only accept step-based checkpoints (e.g., "step123")
            if dir.name.startswith("step"):
                step = int(dir.name[4:])
            else:
                # Legacy: pure number defaults to step
                step = int(dir.name)

            if step > last_global_step:
                last_global_step = step
                last_ckpt_name = dir.name
        except ValueError:
            continue

    # If no step checkpoint found, raise error
    if last_ckpt_name is None or last_global_step < 0:
        raise FileNotFoundError(
            f"No valid step-based checkpoint found in {checkpoint_dir}. "
            "If you want to start new training, use --overwrite instead of --resume."
        )

    last_model_ckpt_dir = checkpoint_dir / last_ckpt_name

    # Load model state dict
    torch.distributed.checkpoint.load(model.state_dict(), checkpoint_id=last_model_ckpt_dir)
    logging.info(f"Loaded FSDP model checkpoint from {last_model_ckpt_dir}")

    # Load EMA parameters if provided
    if ema_model is not None:
        ema_ckpt_dir = last_model_ckpt_dir / "ema"
        if ema_ckpt_dir.exists():
            torch.distributed.checkpoint.load(ema_model.shadow, checkpoint_id=ema_ckpt_dir)
            logging.info(f"Loaded FSDP EMA model checkpoint from {ema_ckpt_dir}")
        else:
            raise FileNotFoundError(
                f"EMA checkpoint directory not found: {ema_ckpt_dir}. "
                "The checkpoint was saved without EMA but you are trying to resume with EMA enabled. "
                "Either disable EMA (set ema_decay=None) or use a checkpoint that was saved with EMA."
            )

    # Load optimizer state dict
    optim_ckpt_dir = checkpoint_dir / "last_optim"
    if not optim_ckpt_dir.exists():
        raise FileNotFoundError(f"Optimizer checkpoint directory {optim_ckpt_dir} does not exist.")
    torch.distributed.checkpoint.load(optimizer.state_dict(), checkpoint_id=optim_ckpt_dir)
    logging.info(f"Loaded FSDP optimizer checkpoint from {optim_ckpt_dir}")

    # Load RNG states
    rng_path = last_model_ckpt_dir / "rng_state.pth"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
        torch.set_rng_state(rng_state["torch_rng_state"])
        np.random.set_state(rng_state["numpy_rng_state"])
        if torch.cuda.is_available() and "torch_cuda_rng_state" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda_rng_state"])
        logging.info(f"Loaded RNG states from {rng_path}")
    else:
        logging.warning(
            f"RNG state file not found: {rng_path}. RNG states will not be restored (checkpoint may be from older version)."
        )

    # Return the next step to train (the loaded step has already been trained)
    return last_global_step + 1
