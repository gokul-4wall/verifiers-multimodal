"""
Simple multimodal RL trainer that integrates Verifiers environments, a vLLM
server running a VLM, and a local Hugging Face VLM for GRPO-style training.

This lives outside the core `verifiers` RL trainer to avoid changing existing
APIs and to keep multimodal support experimental and opt-in.
"""

from .config import MultimodalRLConfig  # noqa: F401
from .trainer import MultimodalAdapter, MultimodalGRPOTrainer  # noqa: F401


