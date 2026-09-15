# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.

"""Pi model configuration."""

import dataclasses
from typing import Literal

from typing_extensions import override

import pi.models.gemma as _gemma
import pi.models.model as _model


@dataclasses.dataclass(frozen=True)
class PiConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - by default state is part of the discrete language tokens; dual mode also injects one continuous Expert token
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = False

    # Enable one learned embedding selected by task_index; language retention is controlled separately.
    use_task_embedding: bool = False
    # Keep the language prompt while also inserting the learned task embedding.
    use_language_with_task_embedding: bool = False
    num_tasks: int = 0
    # Keep the legacy VLM task token by default; new experiments may place it in the Action Expert.
    task_embedding_target: Literal["vlm", "expert"] = "vlm"
    # Pi05 normally exposes state only as discrete VLM tokens. Dual mode also adds one continuous Expert token.
    state_conditioning_mode: Literal["discrete_vlm", "dual"] = "discrete_vlm"

    # Optional learned VLM-prefix soft prompt selected by the robot embodiment.
    use_embodiment_embedding: bool = False
    num_embodiments: int = 2
    num_embodiment_tokens: int = 1

    # Number of historical state frames to use as input (default=1 for backward compatibility)
    state_history_frames: int = 1  # 目前是如果=1，只使用当前状态，如果>1，使用历史状态

    # Maximum random delay frames for state input (0 means no delay, 5 means random delay 0-5 frames)
    # Used to simulate real-world state acquisition delays during training
    state_delay_frames: int = 0  # 如果=0，不使用延迟状态，如果>0，使用延迟状态

    def __post_init__(self):
        """实例化时会被调用."""
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)

        if self.pi05 is False:  # 确保在pi0模式下，state_input为连续输入
            object.__setattr__(self, "discrete_state_input", False)

        if self.task_embedding_target not in ("vlm", "expert"):
            raise ValueError(f"Unsupported task_embedding_target={self.task_embedding_target!r}")
        if self.state_conditioning_mode not in ("discrete_vlm", "dual"):
            raise ValueError(f"Unsupported state_conditioning_mode={self.state_conditioning_mode!r}")

        if self.use_task_embedding:
            if not self.pi05 or not self.discrete_state_input:
                raise ValueError("Task embedding requires Pi05 with discrete_state_input=True")
            if self.num_tasks <= 0:
                raise ValueError("num_tasks must be positive when use_task_embedding=True")
        elif self.use_language_with_task_embedding:
            raise ValueError("use_language_with_task_embedding requires use_task_embedding=True")
        elif self.task_embedding_target != "vlm":
            raise ValueError("task_embedding_target='expert' requires use_task_embedding=True")

        if self.state_conditioning_mode == "dual":
            if not self.pi05 or not self.discrete_state_input:
                raise ValueError("Dual state conditioning requires Pi05 with discrete_state_input=True")
            if self.state_history_frames != 1:
                raise ValueError("Dual state conditioning currently requires state_history_frames=1")

        if self.use_embodiment_embedding:
            if not self.pi05 or not self.discrete_state_input:
                raise ValueError("Embodiment embedding requires Pi05 with discrete_state_input=True")
            if self.num_embodiments <= 0:
                raise ValueError("num_embodiments must be positive when use_embodiment_embedding=True")
            if self.num_embodiment_tokens <= 0:
                raise ValueError("num_embodiment_tokens must be positive when use_embodiment_embedding=True")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0
