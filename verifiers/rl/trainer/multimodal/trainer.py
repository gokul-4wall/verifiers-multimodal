from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import asyncio
import torch
from openai import AsyncOpenAI
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import verifiers as vf
from verifiers.rl.inference.client import VLLMClient
from verifiers.types import GenerateOutputs

from .config import MultimodalRLConfig


@dataclass
class MultimodalBatch:
    """
    Minimal batch representation for multimodal GRPO training.

    All 2D tensors are shape (B, L) except rewards, which is (B,).
    `pixel_values` (if provided) is typically (B, C, H, W) or model-specific.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    behavior_logprobs: torch.Tensor
    rewards: torch.Tensor
    pixel_values: torch.Tensor | None = None
    extra_model_kwargs: Dict[str, torch.Tensor] = field(default_factory=dict)


class MultimodalAdapter(ABC):
    """
    Adapter that knows how to turn Verifiers rollouts + vLLM responses
    into token-level sequences suitable for training a Hugging Face VLM.

    This is intentionally model-family specific. Users should subclass this
    for their particular VLM (e.g., Qwen-VL, LLaVA, etc.) and implement:

    - how to map `GenerateOutputs` (prompts/completions/states/rewards) to a
      sequence of token ids and masks that align with vLLM's tokenization.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    @abstractmethod
    def build_batch(self, outputs: GenerateOutputs) -> MultimodalBatch:
        """
        Convert Verifiers `GenerateOutputs` into a `MultimodalBatch`.

        Responsibilities:
        - Align `input_ids` with the tokens used by vLLM (ideally by using the
          same tokenizer and chat template that vLLM uses).
        - Construct `loss_mask` so that only completion tokens (or a subset)
          contribute to the RL loss.
        - Extract behavior logprobs from vLLM responses and align them with
          `input_ids` (padding with zeros for prompt tokens).
        - Aggregate rewards per sequence (one scalar per row in the batch).
        """
        raise NotImplementedError


class StepsDataset(Dataset):
    """
    Simple dataset that produces a fixed number of "steps" to drive the
    training loop. Each item just returns its index; the trainer is
    responsible for doing environment rollouts per step.
    """

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, int]:
        return {"step": idx}


class MultimodalGRPOTrainer:
    """
    Minimal GRPO-style trainer for multimodal RL with Verifiers.

    Design:
    - Uses a Verifiers Environment + vLLM for rollouts.
    - Uses a local Hugging Face VLM (PreTrainedModel) for training.
    - Uses a model-family-specific MultimodalAdapter to bridge from
      Verifiers rollouts to token-level tensors.
    - Optionally syncs weights back to vLLM for online RL.
    """

    def __init__(
        self,
        model: PreTrainedModel | str,
        env: vf.Environment | None = None,  # Match text-only: accepts env
        tokenizer: PreTrainedTokenizerBase | None = None,
        adapter: MultimodalAdapter | None = None,
        config: MultimodalRLConfig | None = None,
        args: MultimodalRLConfig | None = None,
        **kwargs,
    ):
        # Handle args vs config (match text-only trainer pattern)
        if args is not None and config is None:
            config = args
        if config is None:
            raise ValueError("Must provide either 'config' or 'args'")
        
        # Auto-load VLM model if string provided
        # For VLMs, we need special loading (can't use get_model_and_tokenizer for vision models)
        if isinstance(model, str):
            from transformers import AutoModelForImageTextToText
            import torch
            print(f"Loading VLM: {model}")
            # Match text-only pattern: no device_map, let manual .to(device) handle it
            model = AutoModelForImageTextToText.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                # No device_map - we'll move to device manually below
            )
        
        if tokenizer is None:
            raise ValueError("Must provide tokenizer")
        
        if adapter is None:
            raise ValueError(
                "Must provide adapter. For Qwen3-VL, use: "
                "from adapters import Qwen3VLAdapter; "
                "adapter = Qwen3VLAdapter(tokenizer, processor)"
            )
        
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.config = config

        # Store env (match text-only pattern)
        self.env: vf.Environment | None = env

        # vLLM client for behavior policy + weight sync
        self.vllm_client = VLLMClient(
            host=self.config.vllm_server_host,
            port=self.config.vllm_server_port,
            group_port=self.config.vllm_server_group_port,
            connection_timeout=self.config.vllm_server_timeout,
        )
        # initialize NCCL communicator for weight updates (lazy init on first sync)
        # Only init if we're actually going to sync weights
        if self.config.sync_to_vllm_every > 0:
            try:
                self.vllm_client.init_communicator()
            except Exception as e:
                print(f"Warning: Failed to initialize NCCL communicator: {e}")
                print("Weight syncing will be disabled. Training will continue without syncing to vLLM.")
                self.config.sync_to_vllm_every = 0

        # standard optimizer; users can swap this out if needed
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # simple LR schedule: cosine decay over max_steps
        self._global_step = 0

    def _get_env(self) -> vf.Environment:
        # Match text-only: env can be passed in __init__ or lazy-loaded from config
        if self.env is None:
            if self.config.env_id:
                self.env = vf.load_environment(
                    env_id=self.config.env_id, **self.config.env_args
                )
            else:
                raise ValueError("Environment not provided. Pass env to __init__ or set env_id in config.")
        return self.env

    async def _collect_rollouts(self) -> GenerateOutputs:
        """
        Collect a batch of rollouts from the environment using vLLM.

        We use the same async generate API that `vf-eval` uses, but with
        a custom AsyncOpenAI client pointing at our vLLM server.
        """
        env = self._get_env()
        dataset = env.get_dataset()
        num_rows = len(dataset)
        if num_rows == 0:
            raise ValueError("Environment dataset is empty")

        # cap by available rows
        num_examples = min(self.config.rollouts_per_step, num_rows)
        ds_slice = dataset.select(range(num_examples))

        client = AsyncOpenAI(
            base_url=f"http://{self.config.vllm_server_host}:{self.config.vllm_server_port}/v1",
            api_key="EMPTY",
        )

        try:
            outputs = await env.a_generate(
                ds_slice,
                client=client,
                model=self.model.config._name_or_path,
                sampling_args={},
                num_examples=num_examples,
                rollouts_per_example=1,
                score_rollouts=True,
                max_concurrent=self.config.max_concurrent,
                max_concurrent_generation=None,
                max_concurrent_scoring=None,
            )
        finally:
            await client.close()
        return outputs

    def _compute_logprobs_and_entropy(
        self, logits: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token logprobs and entropies for a causal LM.

        We follow the pattern from the built-in RLTrainer: drop the last logit,
        shift targets by one, and compute logprobs for the observed tokens.
        """
        # logits: (B, L, V)
        # we want logprobs for tokens 1..L-1
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]

        logits = logits / self.config.temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        # gather logprobs of the actual tokens
        token_logprobs = log_probs.gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)  # (B, L-1)

        # entropy per token
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, L-1)

        return token_logprobs, entropy

    def _compute_loss(
        self,
        batch: MultimodalBatch,
        trainer_logprobs: torch.Tensor,
        entropies: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO-style loss using behavior and trainer logprobs.
        """
        # Align shapes: both are (B, L-1); batch.loss_mask is (B, L)
        loss_mask = batch.loss_mask[:, 1:].bool()
        behavior_lp = batch.behavior_logprobs[:, 1:]
        trainer_lp = trainer_logprobs

        # importance weights
        log_ratio = trainer_lp - behavior_lp
        ratio = torch.exp(log_ratio)

        # clip high/low ratios
        is_low = ratio < self.config.mask_ratio_low
        is_high = ratio > self.config.mask_ratio_high
        is_masked = is_low | is_high
        keep_mask = (~is_masked) & loss_mask

        # advantages: centered rewards per sequence
        rewards = batch.rewards
        advantages = rewards - rewards.mean()
        # broadcast advantages to token dimension
        advantages_tokens = advantages.unsqueeze(-1).expand_as(trainer_lp)

        # core loss
        loss = (-ratio * advantages_tokens)[keep_mask].sum()

        # tracking statistics
        mismatch_kl = torch.exp(log_ratio) - log_ratio - 1
        with torch.no_grad():
            ir_vals = ratio[loss_mask]
            ent_vals = entropies[loss_mask]
            kl_vals = mismatch_kl[loss_mask]

            ir_mean = float(ir_vals.mean().item()) if ir_vals.numel() > 0 else 0.0
            ent_mean = float(ent_vals.mean().item()) if ent_vals.numel() > 0 else 0.0
            kl_mean = float(kl_vals.mean().item()) if kl_vals.numel() > 0 else 0.0

        summaries = {
            "importance_ratio": ir_mean,
            "entropy": ent_mean,
            "mismatch_kl": kl_mean,
        }
        return loss, summaries

    def _sync_weights_to_vllm(self) -> None:
        """
        Push updated model weights to vLLM for online RL.
        """
        for name, param in self.model.named_parameters():
            # send parameter shape + dtype, then broadcast actual data
            self.vllm_client.update_named_param(name, param.data)

        # reset prefix cache and wait for any background tasks
        self.vllm_client.reset_prefix_cache()
        while self.vllm_client.get_num_background_tasks() > 0:
            pass

    def train(self) -> None:
        """
        Run the full training loop.

        This is deliberately simple: single-process, single-node, synchronous
        rollouts + updates. It can be extended to use Accelerate/Deepspeed
        if needed, but we keep the initial implementation minimal.
        """
        self.model.train()

        dataset = StepsDataset(self.config.max_steps)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        progress = tqdm(dataloader, desc="Training", total=len(dataloader))

        for batch_idx, _ in enumerate(progress):
            self._global_step = batch_idx

            # 1. Collect rollouts from vLLM + environment (async)
            outputs = asyncio.run(self._collect_rollouts())

            # 2. Adapter builds a multimodal batch
            mm_batch = self.adapter.build_batch(outputs)

            # move tensors to device
            input_ids = mm_batch.input_ids.to(self.device)
            attention_mask = mm_batch.attention_mask.to(self.device)
            loss_mask = mm_batch.loss_mask.to(self.device)
            behavior_logprobs = mm_batch.behavior_logprobs.to(self.device)
            rewards = mm_batch.rewards.to(self.device)
            pixel_values = (
                mm_batch.pixel_values.to(self.device)
                if mm_batch.pixel_values is not None
                else None
            )
            extra_kwargs = {
                k: v.to(self.device) for k, v in mm_batch.extra_model_kwargs.items()
            }

            # 3. Forward pass through HF VLM
            model_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if pixel_values is not None:
                model_kwargs["pixel_values"] = pixel_values
            model_kwargs.update(extra_kwargs)

            outputs_model = self.model(**model_kwargs)
            logits = outputs_model.logits
            trainer_lp, entropies = self._compute_logprobs_and_entropy(
                logits, input_ids
            )

            # 4. Compute loss
            mm_batch_device = MultimodalBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                behavior_logprobs=behavior_logprobs,
                rewards=rewards,
            )
            loss, summaries = self._compute_loss(
                mm_batch_device, trainer_lp, entropies
            )

            # 5. Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 6. Optionally sync weights to vLLM
            if (
                self.config.sync_to_vllm_every > 0
                and (self._global_step + 1) % self.config.sync_to_vllm_every == 0
            ):
                self._sync_weights_to_vllm()

            # 7. Logging
            if (self._global_step + 1) % self.config.log_every_steps == 0:
                progress.set_postfix(
                    loss=float(loss.item()),
                    ir=summaries["importance_ratio"],
                    ent=summaries["entropy"],
                    kl=summaries["mismatch_kl"],
                )

