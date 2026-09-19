"""Sampling utilities for distributed robot-policy training."""

from __future__ import annotations

from collections.abc import Mapping, Iterator, Sequence

import numpy as np
import torch

TaskEpisodeRanges = Mapping[int, Sequence[tuple[int, int]]]
TaskEpisodeSkillRanges = Mapping[int, Sequence[Sequence[tuple[int, int]]]]


def task_episode_ranges_from_arrays(
    task_indices: np.ndarray,
    episode_indices: np.ndarray,
    *,
    offset: int = 0,
) -> dict[int, list[tuple[int, int]]]:
    """Build local half-open frame ranges grouped by task and episode."""
    tasks = np.asarray(task_indices).reshape(-1)
    episodes = np.asarray(episode_indices).reshape(-1)
    if tasks.shape != episodes.shape:
        raise ValueError(f"task/episode shapes differ: {tasks.shape} vs {episodes.shape}")
    if tasks.size == 0:
        raise ValueError("Cannot build balanced sampling groups from an empty dataset")

    changes = np.flatnonzero(episodes[1:] != episodes[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [episodes.size]))
    seen_episodes: set[int] = set()
    result: dict[int, list[tuple[int, int]]] = {}
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        episode_id = int(episodes[start])
        if episode_id in seen_episodes:
            raise ValueError(f"Episode {episode_id} is not stored as one contiguous frame range")
        seen_episodes.add(episode_id)

        task_id = int(tasks[start])
        if not np.all(tasks[start:end] == task_id):
            raise ValueError(f"Episode {episode_id} contains multiple task indices")
        result.setdefault(task_id, []).append((offset + start, offset + end))
    return result


class DistributedTaskEpisodeBalancedSampler(torch.utils.data.Sampler[int]):
    """Sample equal task counts, uniform episodes, then frames or skill segments."""

    def __init__(
        self,
        task_episode_ranges: TaskEpisodeRanges,
        *,
        num_samples: int,
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int,
        task_episode_skill_ranges: TaskEpisodeSkillRanges | None = None,
        skill_balance_ratio: float = 0.0,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank {rank} is outside [0, {num_replicas})")
        if batch_size <= 0 or num_samples <= 0:
            raise ValueError("batch_size and num_samples must be positive")
        if num_samples % batch_size != 0:
            raise ValueError("num_samples must contain a whole number of local batches")

        ranges = {
            int(task_id): tuple((int(start), int(end)) for start, end in episodes)
            for task_id, episodes in task_episode_ranges.items()
        }
        if not ranges:
            raise ValueError("task_episode_ranges must be non-empty")
        for task_id, episodes in ranges.items():
            if not episodes:
                raise ValueError(f"Task {task_id} has no episodes")
            if any(start < 0 or end <= start for start, end in episodes):
                raise ValueError(f"Task {task_id} has an invalid episode frame range")

        self.task_ids = tuple(sorted(ranges))
        self.approximate_task_balancing = batch_size % len(self.task_ids) != 0

        ratio = float(skill_balance_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"skill_balance_ratio must be in [0, 1], got {ratio}")
        if self.approximate_task_balancing and ratio > 0.0:
            raise ValueError(
                f"skill_balance_ratio > 0 requires exact task balancing "
                f"(local batch size {batch_size} must be divisible by {len(self.task_ids)} tasks)"
            )

        skill_ranges = None
        if ratio > 0.0:
            if task_episode_skill_ranges is None:
                raise ValueError("task_episode_skill_ranges is required when skill_balance_ratio > 0")
            skill_ranges = {
                int(task_id): tuple(
                    tuple((int(start), int(end)) for start, end in episode_skills) for episode_skills in episodes
                )
                for task_id, episodes in task_episode_skill_ranges.items()
            }
            if set(skill_ranges) != set(ranges):
                raise ValueError("Skill range task IDs must exactly match task_episode_ranges")
            for task_id, episodes in ranges.items():
                task_skill_ranges = skill_ranges[task_id]
                if len(task_skill_ranges) != len(episodes):
                    raise ValueError(
                        f"Task {task_id} has {len(episodes)} episodes but {len(task_skill_ranges)} skill-range groups"
                    )
                for episode_position, ((episode_start, episode_end), episode_skills) in enumerate(
                    zip(episodes, task_skill_ranges, strict=True)
                ):
                    if not episode_skills:
                        raise ValueError(f"Task {task_id} episode {episode_position} has no skill frame ranges")
                    if any(start < episode_start or end > episode_end or end <= start for start, end in episode_skills):
                        raise ValueError(
                            f"Task {task_id} episode {episode_position} has a skill range "
                            "outside its episode frame range"
                        )

        self.task_episode_ranges = ranges
        self.task_episode_skill_ranges = skill_ranges
        self.skill_balance_ratio = ratio
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, self.rank]))
        samples_per_task = self.batch_size // len(self.task_ids)
        num_batches = self.num_samples // self.batch_size

        if self.approximate_task_balancing:
            yield from self._iter_rotating_window(rng, num_batches)
            return

        for _ in range(num_batches):
            batch: list[int] = []
            for task_id in self.task_ids:
                episodes = self.task_episode_ranges[task_id]
                selected_episodes = rng.integers(0, len(episodes), size=samples_per_task)
                skill_samples = int(np.floor(samples_per_task * self.skill_balance_ratio))
                fractional_sample = samples_per_task * self.skill_balance_ratio - skill_samples
                if fractional_sample > 0.0 and rng.random() < fractional_sample:
                    skill_samples += 1
                use_skill = np.zeros(samples_per_task, dtype=bool)
                if skill_samples > 0:
                    use_skill[:skill_samples] = True
                    rng.shuffle(use_skill)

                for sample_position, episode_position in enumerate(selected_episodes):
                    start, end = episodes[int(episode_position)]
                    if use_skill[sample_position]:
                        assert self.task_episode_skill_ranges is not None
                        episode_skills = self.task_episode_skill_ranges[task_id][int(episode_position)]
                        skill_position = int(rng.integers(0, len(episode_skills)))
                        start, end = episode_skills[skill_position]
                    batch.append(int(rng.integers(start, end)))
            rng.shuffle(batch)
            yield from batch

    def _iter_rotating_window(self, rng: np.random.Generator, num_batches: int) -> Iterator[int]:
        """Approximate task balancing via a rotating batch-wide task window.

        Rank r at local batch b covers tasks [(b + r) * batch_size + offset) mod
        num_tasks for offset in [0, batch_size), one sample per covered task
        (episode and frame both drawn from rng, then the batch is shuffled like
        the exact path). With gcd(batch_size, num_tasks) = g the per-rank window
        start cycles through num_tasks / g distinct positions, so each task is
        sampled exactly batch_size / g times per sweep; across num_replicas
        ranks the per-global-step task count deviates by at most 1 sample.
        """
        num_tasks = len(self.task_ids)
        batch_size = self.batch_size
        for batch_idx in range(num_batches):
            start = (batch_idx * batch_size + self.rank * batch_size) % num_tasks
            batch: list[int] = []
            for offset in range(batch_size):
                task_id = self.task_ids[(start + offset) % num_tasks]
                episodes = self.task_episode_ranges[task_id]
                episode_position = int(rng.integers(0, len(episodes)))
                start_frame, end_frame = episodes[episode_position]
                batch.append(int(rng.integers(start_frame, end_frame)))
            rng.shuffle(batch)
            yield from batch
