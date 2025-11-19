"""
Multimodal RL trainer for vision-language models (VLMs).

This trainer extends the verifiers RL framework to support true multimodal
(text + image) training with GRPO-style optimization.
"""

from .config import MultimodalRLConfig  # noqa: F401
from .trainer import (  # noqa: F401
    MultimodalAdapter,
    MultimodalBatch,
    MultimodalGRPOTrainer,
)

__all__ = [
    "MultimodalRLConfig",
    "MultimodalAdapter",
    "MultimodalBatch",
    "MultimodalGRPOTrainer",
]

