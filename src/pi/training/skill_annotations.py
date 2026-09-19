"""Load episode skill annotations into flattened dataset frame ranges."""

from __future__ import annotations

import json
import pathlib

from pi.training.samplers import TaskEpisodeRanges


def load_task_episode_skill_ranges(
    task_episode_ranges: TaskEpisodeRanges,
    annotations_root: str | pathlib.Path,
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    """Map annotation frame spans onto flattened training indices.

    Each task directory holds one `episode_*.json` per episode of that task, in the
    same order as the episode ranges, so the two lists are zipped together.
    """
    root = pathlib.Path(annotations_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Skill annotations directory not found: {root}")

    result: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for task_id in sorted(task_episode_ranges):
        episodes = task_episode_ranges[task_id]
        task_dir = root / f"task-{int(task_id):04d}"
        annotation_files = sorted(task_dir.glob("episode_*.json"))
        if len(annotation_files) != len(episodes):
            raise ValueError(
                f"Task {task_id} has {len(episodes)} dataset episodes but "
                f"{len(annotation_files)} annotation files in {task_dir}"
            )

        task_result: list[tuple[tuple[int, int], ...]] = []
        for episode_position, (annotation_path, episode_range) in enumerate(
            zip(annotation_files, episodes, strict=True)
        ):
            try:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Could not read skill annotation: {annotation_path}") from error

            episode_start, episode_end = map(int, episode_range)
            episode_length = episode_end - episode_start
            metadata = annotation.get("meta_data", {})
            valid_duration = metadata.get("valid_duration")
            task_duration = metadata.get("task_duration")
            if (
                not isinstance(valid_duration, list)
                or len(valid_duration) != 2
                or task_duration is None
                or int(valid_duration[1]) - int(valid_duration[0]) != int(task_duration)
            ):
                raise ValueError(f"Invalid duration metadata in {annotation_path}")

            episode_skills: list[tuple[int, int]] = []
            for skill in annotation.get("skill_annotation", []):
                frame_duration = skill.get("frame_duration")
                if not isinstance(frame_duration, list) or len(frame_duration) != 2:
                    raise ValueError(f"Invalid skill frame_duration in {annotation_path}")
                local_start, local_end = map(int, frame_duration)
                if local_start < 0 or local_end <= local_start or local_end > episode_length:
                    raise ValueError(
                        f"Skill range [{local_start}, {local_end}) in {annotation_path} "
                        f"is outside dataset episode {episode_position} length {episode_length}"
                    )
                episode_skills.append((episode_start + local_start, episode_start + local_end))

            if not episode_skills:
                raise ValueError(f"No skill annotations found in {annotation_path}")
            task_result.append(tuple(episode_skills))
        result[int(task_id)] = task_result
    return result
