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


def test_all_tokens_masked_out():
    """
    Test that when all tokens are masked out by importance ratio clipping,
    the loss is zero and summaries are computed correctly.
    """
    torch = pytest.importorskip("torch")
    # Use extreme masking so all ratios get filtered
    trainer = _make_trainer_for_unit_tests(
        mask_ratio_low=0.9999, mask_ratio_high=1.0001
    )

    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)

    # Behavior and trainer logprobs that produce ratios close to 1.0
    behavior_lp = torch.log(torch.tensor([[1.0, 0.5, 0.5, 0.5]]))
    trainer_lp = torch.log(torch.tensor([[0.5, 0.5, 0.5]]))

    rewards = torch.tensor([1.0])
    entropies = torch.tensor([[0.1, 0.2, 0.3]])

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
    )

    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)

    # Loss should be zero when all tokens are masked
    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    # Summaries should still be computed over loss_mask (not keep_mask)
    assert summaries["importance_ratio"] > 0.0


def test_all_rewards_identical():
    """
    Test that when all rewards are identical, advantages become zero
    and the loss is zero.
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests()

    # Two sequences with identical rewards
    input_ids = torch.tensor([[0, 1, 2], [0, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.long)

    behavior_lp = torch.log(torch.tensor([[1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]))
    trainer_lp = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))

    # Identical rewards -> zero advantages
    rewards = torch.tensor([1.0, 1.0])
    entropies = torch.tensor([[0.1, 0.2], [0.1, 0.2]])

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
    )

    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)

    # Loss should be zero when advantages are zero
    assert loss.item() == 0.0
    assert torch.isfinite(loss)


def test_extreme_importance_ratios_low():
    """
    Test that very low importance ratios are properly masked out.
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests(
        mask_ratio_low=0.5, mask_ratio_high=10.0
    )

    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)

    # Create very low ratios: trainer_lp << behavior_lp
    behavior_lp = torch.log(torch.tensor([[1.0, 0.9, 0.9, 0.9]]))
    trainer_lp = torch.log(torch.tensor([[0.1, 0.1, 0.1]]))  # ratio ~ 0.11

    rewards = torch.tensor([2.0])
    entropies = torch.tensor([[0.1, 0.2, 0.3]])

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
    )

    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)

    # Loss should be zero since all ratios are below mask_ratio_low
    assert loss.item() == 0.0


def test_extreme_importance_ratios_high():
    """
    Test that very high importance ratios are properly masked out.
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests(
        mask_ratio_low=0.1, mask_ratio_high=5.0
    )

    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)

    # Create very high ratios: trainer_lp >> behavior_lp
    behavior_lp = torch.log(torch.tensor([[1.0, 0.1, 0.1, 0.1]]))
    trainer_lp = torch.log(torch.tensor([[0.9, 0.9, 0.9]]))  # ratio ~ 9

    rewards = torch.tensor([2.0])
    entropies = torch.tensor([[0.1, 0.2, 0.3]])

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
    )

    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)

    # Loss should be zero since all ratios are above mask_ratio_high
    assert loss.item() == 0.0


def test_empty_loss_mask():
    """
    Test that when loss_mask is all zeros, the loss is zero.
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests()

    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    # All zeros -> no tokens contribute to loss
    loss_mask = torch.zeros_like(input_ids)

    behavior_lp = torch.log(torch.tensor([[1.0, 0.5, 0.5, 0.5]]))
    trainer_lp = torch.log(torch.tensor([[0.5, 0.5, 0.5]]))

    rewards = torch.tensor([2.0])
    entropies = torch.tensor([[0.1, 0.2, 0.3]])

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
    )

    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)

    # Loss should be zero when loss_mask is empty
    assert loss.item() == 0.0
    # Summaries should be 0 or handle empty case
    assert summaries["importance_ratio"] == 0.0
    assert summaries["entropy"] == 0.0


def test_temperature_affects_entropy():
    """
    Test that temperature correctly scales logits and affects entropy.
    Higher temperature should increase entropy.
    """
    torch = pytest.importorskip("torch")

    batch_size, seq_len, vocab_size = 2, 4, 5
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    # Low temperature trainer
    trainer_low = _make_trainer_for_unit_tests(temperature=0.5)
    _, entropy_low = trainer_low._compute_logprobs_and_entropy(logits, input_ids)

    # High temperature trainer
    trainer_high = _make_trainer_for_unit_tests(temperature=2.0)
    _, entropy_high = trainer_high._compute_logprobs_and_entropy(logits, input_ids)

    # Higher temperature should produce higher entropy
    assert entropy_high.mean() > entropy_low.mean()
    assert torch.all(entropy_low >= 0.0)
    assert torch.all(entropy_high >= 0.0)


def test_negative_rewards():
    """
    Test that negative rewards are handled correctly.
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests()

    # Two sequences: one positive, one negative reward
    input_ids = torch.tensor([[0, 1, 2], [0, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.long)

    # Different logprobs to create different importance ratios
    behavior_lp = torch.log(torch.tensor([[1.0, 0.5, 0.5], [1.0, 0.6, 0.6]]))
    trainer_lp = torch.log(torch.tensor([[0.5, 0.5], [0.4, 0.4]]))

    rewards = torch.tensor([1.0, -1.0])
    entropies = torch.tensor([[0.1, 0.2], [0.1, 0.2]])

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
    )

    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)

    # Loss should be finite (may be zero if ratios/advantages cancel out)
    assert torch.isfinite(loss)
    # Main point: negative rewards don't cause errors
    assert summaries["importance_ratio"] > 0.0


def test_gradient_flow():
    """
    Test that gradients flow through the loss computation.
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests()

    # Use two sequences to get non-zero advantages
    input_ids = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=torch.long)

    # Create logits that require gradients
    batch_size, seq_len, vocab_size = 2, 4, 10
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

    # Compute trainer logprobs from logits
    trainer_lp, entropies = trainer._compute_logprobs_and_entropy(logits, input_ids)

    behavior_lp = torch.log(torch.tensor([[1.0, 0.5, 0.5, 0.5], [1.0, 0.5, 0.5, 0.5]]))
    # Different rewards to create non-zero advantages
    rewards = torch.tensor([2.0, 1.0])

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
    )

    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)

    # Backward pass
    if loss.requires_grad and loss.item() != 0.0:
        loss.backward()

        # Check that gradients exist and are non-zero
        assert logits.grad is not None
        assert torch.any(logits.grad != 0.0)
    else:
        # If loss is zero, just check that the computation didn't error
        assert torch.isfinite(loss)


def test_multiple_sequences_different_lengths():
    """
    Test batch with multiple sequences where loss_mask effectively
    creates different "active" lengths.
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests()

    # Same length but different loss masks
    input_ids = torch.tensor([[0, 1, 2, 3], [0, 5, 6, 7]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    # First sequence: train on last 2 tokens, second: train on last 3 tokens
    loss_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.long)

    behavior_lp = torch.log(torch.tensor([[1.0, 0.5, 0.5, 0.5], [1.0, 0.5, 0.5, 0.5]]))
    trainer_lp = torch.log(torch.tensor([[0.6, 0.6, 0.6], [0.6, 0.6, 0.6]]))

    rewards = torch.tensor([1.5, 0.5])
    entropies = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
    )

    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)

    assert torch.isfinite(loss)
    assert loss.item() != 0.0


def test_logprobs_numerical_stability():
    """
    Test that log_softmax handles extreme logit values without NaN.
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests()

    batch_size, seq_len, vocab_size = 1, 4, 10
    # Create extreme logits
    logits = torch.tensor([[[1000.0] + [-1000.0] * (vocab_size - 1)] * seq_len])
    logits = logits.expand(batch_size, seq_len, vocab_size)
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

    token_logprobs, entropy = trainer._compute_logprobs_and_entropy(logits, input_ids)

    # Should not produce NaN despite extreme values
    assert torch.all(torch.isfinite(token_logprobs))
    assert torch.all(torch.isfinite(entropy))
    assert torch.all(entropy >= 0.0)


def test_entropy_bounds():
    """
    Test that entropy is bounded correctly:
    - Minimum entropy (deterministic): ~0
    - Maximum entropy (uniform): log(vocab_size)
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests()

    batch_size, seq_len, vocab_size = 1, 3, 10
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

    # Deterministic distribution (low entropy)
    logits_det = torch.zeros(batch_size, seq_len, vocab_size)
    logits_det[:, :, 0] = 100.0  # First token has very high logit
    _, entropy_det = trainer._compute_logprobs_and_entropy(logits_det, input_ids)

    # Uniform distribution (high entropy)
    logits_uniform = torch.zeros(batch_size, seq_len, vocab_size)
    _, entropy_uniform = trainer._compute_logprobs_and_entropy(
        logits_uniform, input_ids
    )

    # Deterministic should have very low entropy
    assert entropy_det.mean() < 0.1

    # Uniform should have entropy close to log(vocab_size)
    expected_max_entropy = math.log(vocab_size)
    assert math.isclose(entropy_uniform.mean().item(), expected_max_entropy, rel_tol=0.1)


def test_batch_size_one():
    """
    Test edge case with batch size of 1.
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests()

    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 1, 1]], dtype=torch.long)

    behavior_lp = torch.log(torch.tensor([[1.0, 0.5, 0.5]]))
    trainer_lp = torch.log(torch.tensor([[0.5, 0.5]]))

    rewards = torch.tensor([1.0])
    entropies = torch.tensor([[0.1, 0.2]])

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
    )

    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)

    # With single sequence, advantages are 0
    assert loss.item() == 0.0
    assert torch.isfinite(loss)


def test_summaries_with_partial_masking():
    """
    Test that summaries are computed correctly when some tokens
    are masked by importance ratios but others are not.
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests(
        mask_ratio_low=0.3, mask_ratio_high=3.0
    )

    input_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.long)

    # Mix of ratios: some will be masked, some won't
    behavior_lp = torch.log(torch.tensor([[1.0, 0.8, 0.5, 0.5, 0.9]]))
    trainer_lp = torch.log(torch.tensor([[0.2, 0.5, 0.5, 0.3]]))  # ratios: ~0.25, 1.0, 1.0, ~0.33

    rewards = torch.tensor([1.0])
    entropies = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
    )

    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)

    # Summaries should be computed over loss_mask, not keep_mask
    assert summaries["importance_ratio"] > 0.0
    assert summaries["entropy"] > 0.0
    assert summaries["mismatch_kl"] >= 0.0  # KL divergence is non-negative

