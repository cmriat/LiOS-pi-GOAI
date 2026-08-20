# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.

"""PyTorch implementation of Pi0 model."""

import math
import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

import pi.models.gemma as _gemma
import pi.models_pytorch.preprocessing_pytorch as _preprocessing
from pi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from pi.models_pytorch.attention_pooling import PerceiverResampler


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.use_task_embedding = bool(getattr(config, "use_task_embedding", False))

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        if self.use_task_embedding:
            self.task_embedding = nn.Embedding(config.num_tasks, paligemma_config.width)
            nn.init.normal_(self.task_embedding.weight, std=paligemma_config.width**-0.5)
            token_embedding = self.paligemma_with_expert.paligemma.language_model.embed_tokens
            self.task_embedding.to(dtype=token_embedding.weight.dtype)

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
            # When pi05 + state_history_frames > 1, we need state_proj for historical states
            if config.state_history_frames > 1:
                self.state_proj = nn.Linear(32, action_expert_config.width)
                # Add Perceiver Resampler for compressing historical states
                # Compress T=state_history_frames -> M=64 summary tokens
                self.perceiver_resampler = PerceiverResampler(
                    d_model=action_expert_config.width,
                    num_latents=32,  # M=64 summary tokens
                    num_heads=8,
                    num_layers=2,  # Number of cross-attention layers # 最好最大是2，设置多了增益不大
                    use_self_attn=True,  # Use self-attention between cross-attention layers
                    ffn_ratio=4,
                    dropout=0.0,
                )
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        # try:
        #     from transformers.models.siglip import check

        #     if not check.check_whether_transformers_replace_is_installed_correctly():
        #         raise ValueError(msg)
        # except ImportError:
        #     raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        # print(f"observation.state.shape: {observation.state.shape}")
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.task_index,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        task_indices=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare.

        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        if self.use_task_embedding:
            if task_indices is None:
                raise ValueError("task_index is required when use_task_embedding=True")
            if task_indices.ndim == 2 and task_indices.shape[1] == 1:
                task_indices = task_indices[:, 0]
            if task_indices.ndim != 1:
                raise ValueError(f"Expected task_index shape [batch], got {tuple(task_indices.shape)}")
            if task_indices.shape[0] != bsize:
                raise ValueError(
                    f"task_index batch {task_indices.shape[0]} does not match image batch {bsize}"
                )
            task_emb = self.task_embedding(task_indices.to(dtype=torch.long))
            task_emb = task_emb[:, None, :] * math.sqrt(task_emb.shape[-1])
            task_emb = task_emb.to(dtype=embs[0].dtype)
            embs.append(task_emb)
            pad_masks.append(torch.ones((bsize, 1), dtype=torch.bool, device=task_emb.device))
            att_masks.append(0)

        # Tokens contain normalized state and may also contain the language prompt.
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process state if:
        # 1. pi0 mode (not self.pi05), OR
        # 2. pi05 mode with state_history_frames > 1 (historical states enter action expert)
        use_state_proj = (not self.pi05) or (self.pi05 and self.config.state_history_frames > 1)

        if use_state_proj:
            # Embed state - now supports historical states
            # state shape: (batch, history, state_dim) or (batch, state_dim)
            # Ensure state has history dimension
            if state.dim() == 2:
                # (batch, state_dim) -> (batch, 1, state_dim) for backward compatibility
                state = state.unsqueeze(1)

            bsize, history_frames = state.shape[0], state.shape[1]
            device = state.device

            # Project each historical frame independently using the same weight
            # state_proj: Linear(state_dim, hidden) will be applied to last dim
            def state_proj_func(state):
                # Convert state to match state_proj dtype for mixed precision training
                target_dtype = self.state_proj.weight.dtype
                state = state.to(dtype=target_dtype)
                return self.state_proj(state)  # (batch, history, state_dim) -> (batch, history, hidden)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            # Add temporal positional encoding for historical states only when history_frames > 1
            # When history_frames == 1, we only have the current state, no need for temporal encoding
            if history_frames > 1:
                # state[:, 0, :] is the most recent (current) state
                # state[:, history_frames-1, :] is the oldest state
                # So we use reversed time indices: (history_frames-1, history_frames-2, ..., 1, 0)
                # to reflect that earlier positions in the sequence correspond to more recent states
                time_indices = torch.arange(history_frames - 1, -1, -1, dtype=torch.float32, device=device)

                # Generate sinusoidal positional embeddings for historical states
                # Use similar parameters as action timestep but adjusted for historical indexing
                temporal_emb = create_sinusoidal_pos_embedding(
                    time_indices,
                    state_emb.shape[-1],  # hidden dimension
                    min_period=1.0,  # Adjusted for frame indices
                    max_period=float(history_frames),  # Max period based on history length
                    device=device
                )
                # temporal_emb shape: (history_frames, hidden)

                # Expand temporal embedding to match batch dimension and add to state embeddings
                temporal_emb = temporal_emb.unsqueeze(0).expand(bsize, -1, -1)  # (1, history, hidden) -> (batch, history, hidden)
                state_emb = state_emb + temporal_emb.to(state_emb.dtype)

            # Apply Perceiver Resampler to compress historical states
            # from [batch, T=history_frames, hidden] to [batch, M=num_latents, hidden]
            if self.pi05 and hasattr(self, 'perceiver_resampler') and history_frames > self.perceiver_resampler.num_latents:
                # Only apply compression if history_frames > num_latents
                state_emb = self.perceiver_resampler(state_emb)
                compressed_frames = self.perceiver_resampler.num_latents
            else:
                # No compression needed or not available
                compressed_frames = history_frames

            # state_emb shape after compression: (batch, M, hidden) where M=num_latents or original history_frames
            embs.append(state_emb)

            # Create masks for compressed state tokens
            state_mask = torch.ones(bsize, compressed_frames, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            # All state tokens can attend to each other (same cumsum value)
            # First state starts a new attention block, rest share the same block
            att_masks += [1] + ([0] * (compressed_frames - 1)) if compressed_frames > 1 else [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions.to(dtype=self.action_in_proj.weight.dtype))

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb.to(dtype=self.time_mlp_in.weight.dtype))
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb
        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks)) # [batch_size, seq_len]

        return embs, pad_masks, att_masks, adarms_cond


    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)."""
        # 1. 预处理观测值
        images, img_masks, lang_tokens, lang_masks, task_indices, state = self._preprocess_observation(
            observation, train=True
        )
        # print(f"state dtype: {state.dtype}, actions dtype: {actions.dtype}, images[0] dtype: {images[0].dtype}")

        # 2. 扩散过程：采样噪声和时间步
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]

        # 3. 加噪：x_t = t * noise + (1-t) * actions
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # 4. 编码前缀（图像 + 语言）
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, task_indices
        )

        # 5. 编码后缀（状态 + 噪声动作 + 时间步）
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        # if (
        #     self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        #     == torch.bfloat16
        # ):
        #     suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
        #     prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # print("prefix_embs dtype:", prefix_embs.dtype)
        # print("suffix_embs dtype:", suffix_embs.dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            dtype = suffix_out.dtype
            return F.linear(
                suffix_out, self.action_out_proj.weight.to(dtype=dtype), self.action_out_proj.bias.to(dtype=dtype)
            )
        # 预测速度场
        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # 计算 MSE 损失: loss = ||u_t - v_t||²
        # 其中 u_t = noise - actions（真实速度场）
        return F.mse_loss(u_t, v_t, reduction="none")


    # @torch.compile
    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)."""
        bsize = observation.state.shape[0]
        # print(f"observation.state.shape: {observation.state.shape}")
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, task_indices, state = self._preprocess_observation(
            observation, train=False
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, task_indices
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks) # [seq_len, seq_len] # [4, 968, 968]
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1 # 计算每个有效 token 的 位置 ID # [4, 968]

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks) # [batch, 1, seq_len, seq_len] # [4, 1, 968, 968] # 这个 1 是 广播维度（broadcast dimension），会自动扩展到所有注意力头。 所有 8 个头使用相同的 mask
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )


        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        # print(f"prefix_pad_2d_masks.shape: {prefix_pad_2d_masks.shape}")

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        # print(f"suffix_att_2d_masks.shape: {suffix_att_2d_masks.shape}")

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2) #[4, 10, 978] 10是action_horizon，978是prefix_len + suffix_len
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
