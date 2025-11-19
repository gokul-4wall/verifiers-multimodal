from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from transformers import TrainingArguments


@dataclass
class MultimodalRLConfig(TrainingArguments):
    """
    Configuration for the multimodal GRPO trainer.

    This extends `TrainingArguments` with the minimal set of RL-specific fields
    required for online training with a Verifiers environment and a vLLM VLM.

    Key differences from `verifiers.rl.trainer.RLConfig`:
    - We keep the config surface smaller and focused on single-node runs.
    - We don't hard-wire any text-only assumptions (e.g., no max_prompt_len).
    """

    # ---- RL hyperparameters ----
    max_steps: int = 500
    batch_size: int = 512
    micro_batch_size: int = 4

    # GRPO-style clipping for importance weights
    mask_ratio_low: float = 0.1
    mask_ratio_high: float = 10.0

    # temperature used when computing trainer logprobs
    temperature: float = 1.0

    # ---- Environment + rollout config ----
    env_id: str = ""
    env_args: Dict[str, Any] = field(default_factory=dict)

    # Number of rollouts to collect per training step
    rollouts_per_step: int = 512

    # vLLM server configuration (behavior policy)
    vllm_server_host: str = "0.0.0.0"
    vllm_server_port: int = 8000
    vllm_server_group_port: int = 51216
    vllm_server_timeout: float = 600.0

    # How long we wait for environment generation per batch
    generation_timeout: float = 600.0
    max_concurrent: int = 128

    # How often (in steps) to sync weights from the HF VLM to vLLM.
    # Set to 0 to disable weight syncing (offline RL).
    sync_to_vllm_every: int = 1

    # ---- Logging ----
    log_every_steps: int = 10

    # Optional: name of the reward key to optimize (if env exposes multiple).
    primary_reward_key: Optional[str] = None


