# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.

"""Tokenizer utilities for Pi models."""

import os
import logging
from pathlib import Path

import numpy as np
import sentencepiece

import pi.shared.download as download

LOGGER = logging.getLogger(__name__)


def _tokenizer_path() -> Path:
    """Resolve the paligemma tokenizer model.

    Priority: GOAI_TOKENIZER_PATH env → repo-bundled assets/ copy (offline
    self-contained submissions) → user cache → GCS download fallback.
    """
    env_path = os.environ.get("GOAI_TOKENIZER_PATH")
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GOAI_TOKENIZER_PATH does not exist: {path}")
        return path

    bundled = Path(__file__).resolve().parents[3] / "assets" / "paligemma_tokenizer.model"
    if bundled.is_file():
        return bundled

    cached = Path.home() / ".cache" / "openpi" / "big_vision" / "paligemma_tokenizer.model"
    if cached.is_file():
        LOGGER.info("Using cached tokenizer: %s", cached)
        return cached

    return download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = _tokenizer_path()
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            state_str = self._discretize_state(state)
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        return self._pad_tokens(tokens)

    def tokenize_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize only Pi05's discrete state, without a language task prompt."""
        state_str = self._discretize_state(state)
        tokens = self._tokenizer.encode(f"State: {state_str};\nAction: ", add_bos=True)
        return self._pad_tokens(tokens)

    @staticmethod
    def _discretize_state(state: np.ndarray) -> str:
        state = np.asarray(state)
        if state.ndim != 1:
            raise ValueError(f"Expected a 1D state vector, got shape {state.shape}")
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return " ".join(map(str, discretized_state))

    def _pad_tokens(self, tokens: list[int]) -> tuple[np.ndarray, np.ndarray]:
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)
