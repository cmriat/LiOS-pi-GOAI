"""Distributed LeRobot dataloader with CUDA prefetching."""

import os
import glob
import logging
import pathlib
import dataclasses
from typing import Any, List

import numpy as np
import torch
from utils import init_dist as _init_dist, validate_shared_fields as _validate_shared_fields, _repo_id_from_dataset_uri

import pi.training.config as _config  # noqa: E402
from pi.data import MultiLeRobotLoader, _stack_tree, _dataset_uri_values  # noqa: E402
from pi.training.config import DatasetConfig  # noqa: E402


def worker_init_fn(worker_id: int):
    # Limit each worker to a single PyTorch thread — avoids oversubscription
    # when the DataLoader uses multiple workers per GPU.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _get_local_world_size(fallback: int) -> int:
    """Best-effort detection of processes per node."""
    env_value = os.environ.get("LOCAL_WORLD_SIZE")
    if env_value is not None:
        try:
            value = int(env_value)
            if value > 0:
                return value
        except ValueError:
            logging.warning("Invalid LOCAL_WORLD_SIZE=%s, ignoring.", env_value)
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return fallback


def _build_dataset_configs(configs: List[_config.TrainConfig]) -> List[DatasetConfig]:
    """Build the DatasetConfig each dataset is read through.

    Loads normalization stats and extracts dataset-specific parameters. A VideoLance
    config naming several datasets expands to one DatasetConfig per URI, so
    MultiLeRobotLoader reads each Lance source on its own while the run keeps a single
    training config -- and therefore a single archived manifest record.
    """
    ds_cfgs: List[DatasetConfig] = []
    for cfg in configs:
        # Load norm stats from assets directory
        assets_dir = cfg.assets_dirs
        norm_stats = cfg.data.load_norm_stats(assets_dir)

        if cfg.data.repo_id is None or cfg.data.repo_id == "":
            raise ValueError(f"Repo ID not set for config '{cfg.name}'.")
        if norm_stats is None:
            raise ValueError(f"Normalization stats missing for '{cfg.name}'. Run scripts/compute_norm_stats.py first.")

        # Determine if using quantile normalization (PI0.5 models use quantiles, PI0 uses z-score)
        use_quantile_norm = cfg.effective_use_quantile_norm

        uris = _dataset_uri_values(cfg.data.dataset_uri, cfg.data.repo_id)
        if len(uris) == 1 or cfg.data.dataset_format != "video_lance":
            ds_cfgs.append(
                dataclasses.replace(
                    cfg.data,
                    norm_stats=norm_stats,
                    policy_name=cfg.name,
                    use_quantile_norm=use_quantile_norm,
                )
            )
            continue

        # Mixed run: the URIs share one stats file, which is how the datasets are
        # normalized into a single action/state space, and each keeps its own name.
        for uri in uris:
            ds_cfgs.append(
                dataclasses.replace(
                    cfg.data,
                    repo_id=_repo_id_from_dataset_uri(uri),
                    dataset_uri=uri,
                    norm_stats=norm_stats,
                    policy_name=cfg.name,
                    use_quantile_norm=use_quantile_norm,
                )
            )
    return ds_cfgs


def build_configs_from_parent_dir(
    parent_dir: str | pathlib.Path, template: _config.TrainConfig
) -> List[_config.TrainConfig]:
    """Build a list of TrainConfig by globbing first-level subdirectories.

    Only repo_id differs across configs; all other fields follow the template.
    """
    # NOTE(critical path): use glob for first-level dirs and keep absolute paths for repo_id
    pattern = os.path.join(str(parent_dir), "*/")
    candidates = sorted(glob.glob(pattern))
    subdirs = [p for p in candidates if os.path.isdir(p)]
    if not subdirs:
        raise FileNotFoundError(f"No first-level subdirectories found under: {parent_dir}")

    cfgs: List[_config.TrainConfig] = []
    for d in subdirs:
        abs_d = os.path.abspath(d)
        # Replace only the nested DataConfigFactory.repo_id while keeping other fields intact
        new_data = dataclasses.replace(template.data, repo_id=abs_d)
        new_cfg = dataclasses.replace(template, data=new_data)
        cfgs.append(new_cfg)
        logging.info(f"Built config for repo_id: {abs_d}")
    return cfgs


def numpy_to_tensor(x):
    """Convert numpy or python scalar to tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (np.bool_, np.int64, np.float32, np.float64, int, float, bool)):
        return torch.tensor(x)
    return x


def collate_and_preprocess(batch_list: List[dict[str, Any]]) -> dict[str, Any]:
    """DataLoader collate_fn: batch the per-sample dicts and convert numpy to tensor.

    The samples keep the camera set and the dtypes the dataset produced (uint8 images
    in ``[H, W, C]``). The uint8 -> [-1, 1] conversion and the training augmentation
    both belong to the model, which applies each exactly once: ``Observation.from_dict``
    converts the dtypes, and ``PI0Pytorch.forward`` augments. Preprocessing here as well
    would augment twice.
    """
    return _stack_tree(batch_list)


def tree_map_tensor(fn, x):
    if torch.is_tensor(x):
        return fn(x)
    if isinstance(x, dict):
        return {k: tree_map_tensor(fn, v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(tree_map_tensor(fn, v) for v in x)
    return x


def tree_apply(fn, x):
    if torch.is_tensor(x):
        fn(x)
    elif isinstance(x, dict):
        for v in x.values():
            tree_apply(fn, v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            tree_apply(fn, v)


class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.iter = None
        self.next_batch = None
        logging.info(f"CUDAPrefetcher initialized on device {self.device}")

    def __iter__(self):
        self.iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            batch = tree_map_tensor(lambda t: t.to(self.device, non_blocking=True) if torch.is_tensor(t) else t, batch)

        self.next_batch = batch

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        batch = self.next_batch

        # Tell the caching allocator these tensors are used on the current stream
        # so it does not recycle them under a different stream.
        cur_stream = torch.cuda.current_stream(device=self.device)
        tree_apply(lambda t: t.record_stream(cur_stream) if torch.is_tensor(t) else None, batch)

        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)


def create_distributed_dataloader(
    configs: List[_config.TrainConfig],
    *,
    shuffle: bool = True,
    seed: int = 0,
    gpu_rank: int = 1,
) -> dict:
    # Create concatenated dataset and shard with DistributedSampler
    rank, world, _local_rank, _device = _init_dist()
    # Validate that all configs share required fields
    _validate_shared_fields(configs)
    base_model = configs[0].model
    global_batch = int(configs[0].batch_size)
    if gpu_rank <= 0:
        raise ValueError("gpu rank must be positive")
    if global_batch % gpu_rank != 0:
        raise ValueError(f"batch_size {global_batch} must be divisible by gpu rank {gpu_rank}")
    local_batch = global_batch // gpu_rank
    ds_cfgs = _build_dataset_configs(configs)

    logging.info(f"Action horizon: {base_model.action_horizon}, dim: {base_model.action_dim}")
    has_train_test_split = configs[0].data.test_ep_num > 0

    # --- Train dataset / sampler / loader ---
    g = torch.Generator()
    g.manual_seed(seed)
    train_multi_ds = MultiLeRobotLoader(
        datasets=ds_cfgs,
        batch_size=local_batch,
        action_horizon=int(base_model.action_horizon),
        action_dim=int(base_model.action_dim),
        max_token_len=int(base_model.max_token_len),
        discrete_state_input=bool(getattr(base_model, "discrete_state_input", True)),
        use_task_embedding=bool(getattr(base_model, "use_task_embedding", False)),
        use_language_with_task_embedding=bool(getattr(base_model, "use_language_with_task_embedding", False)),
        num_tasks=int(getattr(base_model, "num_tasks", 0)),
        use_embodiment_embedding=bool(getattr(base_model, "use_embodiment_embedding", False)),
        num_embodiments=int(getattr(base_model, "num_embodiments", 2)),
        apply_delta_transform=bool(getattr(configs[0].data, "apply_delta_transform", True)),
        use_per_timestamp_action_norm=bool(getattr(configs[0].data, "use_per_timestamp_action_norm", False)),
        state_history_frames=int(getattr(base_model, "state_history_frames", 1)),
        state_delay_frames=int(getattr(base_model, "state_delay_frames", 0)),
        mode="train",
    )
    total_train_samples = len(train_multi_ds)
    if rank == 0:
        logging.info(f"Total train samples across datasets: {total_train_samples}")

    local_world_size = _get_local_world_size(world)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_multi_ds, num_replicas=world, rank=rank, shuffle=shuffle, seed=seed, drop_last=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_multi_ds,
        batch_size=local_batch,
        sampler=train_sampler,
        shuffle=(train_sampler is None and shuffle),
        num_workers=configs[0].num_workers,
        worker_init_fn=worker_init_fn,
        prefetch_factor=None if configs[0].num_workers == 0 else 2,
        persistent_workers=True,
        generator=g,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_and_preprocess,
    )

    # --- Test dataset / sampler / loader ---
    test_loader = None
    if has_train_test_split:
        test_multi_ds = MultiLeRobotLoader(
            datasets=ds_cfgs,
            batch_size=local_batch,
            action_horizon=int(base_model.action_horizon),
            action_dim=int(base_model.action_dim),
            max_token_len=int(base_model.max_token_len),
            discrete_state_input=bool(getattr(base_model, "discrete_state_input", True)),
            use_task_embedding=bool(getattr(base_model, "use_task_embedding", False)),
            use_language_with_task_embedding=bool(getattr(base_model, "use_language_with_task_embedding", False)),
            num_tasks=int(getattr(base_model, "num_tasks", 0)),
            use_embodiment_embedding=bool(getattr(base_model, "use_embodiment_embedding", False)),
            num_embodiments=int(getattr(base_model, "num_embodiments", 2)),
            apply_delta_transform=bool(getattr(configs[0].data, "apply_delta_transform", True)),
            use_per_timestamp_action_norm=bool(getattr(configs[0].data, "use_per_timestamp_action_norm", False)),
            state_history_frames=int(getattr(base_model, "state_history_frames", 1)),
            state_delay_frames=int(getattr(base_model, "state_delay_frames", 0)),
            mode="test",
        )
        total_test_samples = len(test_multi_ds)
        if rank == 0:
            logging.info(f"Total test samples across datasets: {total_test_samples}")

        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_multi_ds, num_replicas=world, rank=rank, shuffle=False, seed=seed, drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_multi_ds,
            batch_size=local_batch,
            sampler=test_sampler,
            shuffle=False,  # Never shuffle test set for consistent evaluation
            num_workers=configs[0].num_workers,
            prefetch_factor=None if configs[0].num_workers == 0 else 2,  # Must be None when num_workers=0
            persistent_workers=False,
            drop_last=False,
            pin_memory=False,
            collate_fn=collate_and_preprocess,
        )

    cuda_prefetcher = CUDAPrefetcher(train_loader, device=torch.device(f"cuda:{_local_rank}"))
    data_loaders = {"train_loader": cuda_prefetcher}
    if test_loader is not None:
        data_loaders["test_loader"] = test_loader
    return data_loaders
