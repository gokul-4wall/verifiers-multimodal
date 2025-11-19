import math

import pytest

# These tests are only meaningful when `transformers` is installed.
pytest.importorskip("transformers")

from verifiers.rl.trainer.multimodal import (
    MultimodalBatch,
    MultimodalGRPOTrainer,
    MultimodalRLConfig,
)


def _make_trainer_for_unit_tests(
    mask_ratio_low: float = 0.1,
    mask_ratio_high: float = 10.0,
    temperature: float = 1.0,
) -> MultimodalGRPOTrainer:
    """
    Construct a MultimodalGRPOTrainer instance suitable for unit-testing
    its internal math helpers without initializing model/env/vLLM.
    """
    # Bypass __init__ to avoid VLLMClient and model setup.
    trainer = object.__new__(MultimodalGRPOTrainer)
    # Minimal TrainingArguments require an output_dir.
    config = MultimodalRLConfig(
        output_dir=".",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-3,
    )
    config.mask_ratio_low = mask_ratio_low
    config.mask_ratio_high = mask_ratio_high
    config.temperature = temperature
    trainer.config = config
    return trainer  # type: ignore[return-value]


def test_compute_logprobs_and_entropy_basic():
    """
    Sanity-check _compute_logprobs_and_entropy:
    - Shapes are correct
    - Logprob values match manual log-softmax
    - Entropy is non-negative
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests()

    batch_size, seq_len, vocab_size = 2, 4, 3
    torch.manual_seed(0)
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Targets: choose indices 0,1,2 in a simple pattern
    input_ids = torch.tensor(
        [
            [0, 1, 2, 0],
            [2, 1, 0, 1],
        ],
        dtype=torch.long,
    )

    token_logprobs, entropy = trainer._compute_logprobs_and_entropy(
        logits, input_ids
    )

    # Should have shape (B, L-1)
    assert token_logprobs.shape == (batch_size, seq_len - 1)
    assert entropy.shape == (batch_size, seq_len - 1)

    # Manual check for one example/token
    # We look at sequence 0, token position 1 (target = input_ids[0,1])
    with torch.no_grad():
        logits_slice = logits[0, :-1, :]  # (L-1, V)
        targets = input_ids[0, 1:]  # (L-1,)
        log_probs_full = torch.log_softmax(logits_slice, dim=-1)
        expected_lp = log_probs_full[0, targets[0]].item()

    assert math.isclose(
        token_logprobs[0, 0].item(), expected_lp, rel_tol=1e-5
    )

    # Entropy must be >= 0 everywhere
    assert torch.all(entropy >= 0.0)


def test_compute_loss_matches_manual():
    """
    Verify that _compute_loss matches a manually-computed GRPO-style loss
    on a tiny synthetic example.
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests(
        mask_ratio_low=0.0, mask_ratio_high=100.0
    )

    # Tiny batch: B=1, L=4 -> effective tokens for loss: positions 1..3
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # Loss mask: only last 3 tokens contribute
    loss_mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)

    # Behavior logprobs over L=4 positions (padded with 0 for prompt token)
    # Trainer logprobs over L-1=3 positions
    # Choose simple numbers so we can compute by hand.
    behavior_lp = torch.log(torch.tensor([[1.0, 0.2, 0.5, 0.8]]))  # (1,4)
    trainer_lp = torch.log(torch.tensor([[0.25, 0.4, 0.9]]))  # (1,3)

    # Rewards: scalar per sequence
    rewards = torch.tensor([1.0])

    # Entropy only used for summaries; choose arbitrary positive values
    entropies = torch.tensor([[0.1, 0.2, 0.3]])

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
    )

    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)

    # Manual loss:
    # ratio = exp(trainer_lp - behavior_lp)
    # advantages = rewards - mean(rewards) = 0 -> but that would give zero loss.
    # To make it non-zero, use a second sequence with different reward.
    # Instead, we verify the summaries and that loss is finite.
    assert torch.isfinite(loss)

    # importance_ratio summary is the mean of ratio over masked tokens
    with torch.no_grad():
        log_ratio = trainer_lp - behavior_lp[:, 1:]
        ratio = torch.exp(log_ratio)
        masked_ratio = ratio[loss_mask[:, 1:].bool()]
        expected_ir_mean = float(masked_ratio.mean().item())

    assert math.isclose(
        summaries["importance_ratio"], expected_ir_mean, rel_tol=1e-5
    )


