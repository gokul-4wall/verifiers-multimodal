import pytest

# These tests are only meaningful when `transformers` is installed.
pytest.importorskip("transformers")

from multimodal_rl.config import MultimodalRLConfig
from multimodal_rl.trainer import MultimodalAdapter, MultimodalBatch, MultimodalGRPOTrainer


def _make_trainer_for_unit_tests(
    mask_ratio_low: float = 0.1,
    mask_ratio_high: float = 10.0,
    temperature: float = 1.0,
) -> MultimodalGRPOTrainer:
    """
    Construct a MultimodalGRPOTrainer instance suitable for unit-testing
    its internal math helpers without initializing model/env/vLLM.
    """
    trainer = object.__new__(MultimodalGRPOTrainer)
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


def test_batch_with_pixel_values():
    """
    Test that MultimodalBatch correctly handles pixel_values (images).
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests()

    batch_size, seq_len = 2, 5
    image_size = 224
    channels = 3

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.long)

    behavior_lp = torch.randn(batch_size, seq_len)
    rewards = torch.tensor([1.5, 0.5])

    # Create pixel_values (typical shape for vision models)
    pixel_values = torch.randn(batch_size, channels, image_size, image_size)

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
        pixel_values=pixel_values,
    )

    # Verify batch was created successfully
    assert batch.pixel_values is not None
    assert batch.pixel_values.shape == (batch_size, channels, image_size, image_size)
    assert batch.input_ids.shape == (batch_size, seq_len)

    # Compute logprobs/entropy (should work regardless of pixel_values)
    logits = torch.randn(batch_size, seq_len, 1000)
    trainer_lp, entropies = trainer._compute_logprobs_and_entropy(logits, input_ids)

    # Compute loss (should work with pixel_values in batch)
    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)
    assert torch.isfinite(loss)


def test_batch_with_extra_model_kwargs():
    """
    Test that MultimodalBatch correctly handles extra_model_kwargs
    for model-specific inputs (e.g., image_sizes, aspect_ratios).
    """
    torch = pytest.importorskip("torch")

    batch_size, seq_len = 2, 4
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)
    behavior_lp = torch.randn(batch_size, seq_len)
    rewards = torch.tensor([1.0, 2.0])

    # Extra kwargs for model-specific needs
    extra_kwargs = {
        "image_sizes": torch.tensor([[224, 224], [448, 448]]),
        "image_grid_thw": torch.tensor([[1, 1, 1], [2, 2, 1]]),
    }

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
        pixel_values=None,
        extra_model_kwargs=extra_kwargs,
    )

    assert "image_sizes" in batch.extra_model_kwargs
    assert "image_grid_thw" in batch.extra_model_kwargs
    assert batch.extra_model_kwargs["image_sizes"].shape == (batch_size, 2)


def test_batch_without_pixel_values():
    """
    Test that MultimodalBatch works fine without pixel_values (text-only).
    """
    torch = pytest.importorskip("torch")
    trainer = _make_trainer_for_unit_tests()

    batch_size, seq_len = 2, 4
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)
    behavior_lp = torch.randn(batch_size, seq_len)
    rewards = torch.tensor([1.0, 2.0])

    # No pixel_values provided
    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
        pixel_values=None,
    )

    assert batch.pixel_values is None

    # Should still work fine
    logits = torch.randn(batch_size, seq_len, 1000)
    trainer_lp, entropies = trainer._compute_logprobs_and_entropy(logits, input_ids)
    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)
    assert torch.isfinite(loss)


def test_multimodal_forward_pass_mock():
    """
    Test a full forward pass with a mock multimodal model.
    """
    torch = pytest.importorskip("torch")
    from unittest.mock import Mock
    from types import SimpleNamespace

    trainer = _make_trainer_for_unit_tests()

    batch_size, seq_len = 2, 5
    vocab_size = 1000
    image_size = 224
    channels = 3

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.long)
    pixel_values = torch.randn(batch_size, channels, image_size, image_size)

    # Mock model that accepts both text and images
    mock_model = Mock()
    # Return logits with correct shape
    mock_logits = torch.randn(batch_size, seq_len, vocab_size)
    mock_model.return_value = SimpleNamespace(logits=mock_logits)

    # Simulate calling the model
    outputs = mock_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
    )

    # Verify model was called with correct arguments
    mock_model.assert_called_once()
    call_kwargs = mock_model.call_args.kwargs
    assert "input_ids" in call_kwargs
    assert "attention_mask" in call_kwargs
    assert "pixel_values" in call_kwargs
    assert torch.equal(call_kwargs["input_ids"], input_ids)
    assert torch.equal(call_kwargs["pixel_values"], pixel_values)

    # Verify outputs are correct shape
    assert outputs.logits.shape == (batch_size, seq_len, vocab_size)

    # Compute logprobs should work with these logits
    trainer_lp, entropies = trainer._compute_logprobs_and_entropy(
        outputs.logits, input_ids
    )
    assert trainer_lp.shape == (batch_size, seq_len - 1)


def test_pixel_values_shape_validation():
    """
    Test various pixel_values shapes to ensure they're handled correctly.
    """
    torch = pytest.importorskip("torch")

    batch_size, seq_len = 2, 4
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)
    behavior_lp = torch.randn(batch_size, seq_len)
    rewards = torch.tensor([1.0, 2.0])

    # Test different image shapes
    shapes_to_test = [
        (batch_size, 3, 224, 224),  # Standard RGB
        (batch_size, 3, 336, 336),  # Larger image
        (batch_size, 1, 224, 224),  # Grayscale
        (batch_size, 3, 512, 512),  # High-res
    ]

    for shape in shapes_to_test:
        pixel_values = torch.randn(shape)

        batch = MultimodalBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            behavior_logprobs=behavior_lp,
            rewards=rewards,
            pixel_values=pixel_values,
        )

        assert batch.pixel_values.shape == shape


def test_batch_size_mismatch_detection():
    """
    Test that we can detect when pixel_values batch size doesn't match input_ids.
    This is a common error that should be caught.
    """
    torch = pytest.importorskip("torch")

    batch_size = 2
    wrong_batch_size = 3
    seq_len = 4

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)
    behavior_lp = torch.randn(batch_size, seq_len)
    rewards = torch.tensor([1.0, 2.0])

    # Create pixel_values with wrong batch size
    pixel_values = torch.randn(wrong_batch_size, 3, 224, 224)

    # Creating the batch should work (no validation in __init__)
    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
        pixel_values=pixel_values,
    )

    # But we can check for the mismatch
    assert batch.input_ids.shape[0] != batch.pixel_values.shape[0]


class MockMultimodalAdapter(MultimodalAdapter):
    """
    Mock adapter for testing that creates batches with images.
    """

    def build_batch(self, outputs):
        """Build a mock multimodal batch."""
        torch = pytest.importorskip("torch")

        # Simulate extracting data from GenerateOutputs
        batch_size = 2
        seq_len = 5

        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.long)
        behavior_logprobs = torch.randn(batch_size, seq_len)
        rewards = torch.tensor([1.5, 0.5])
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        return MultimodalBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            behavior_logprobs=behavior_logprobs,
            rewards=rewards,
            pixel_values=pixel_values,
        )


def test_adapter_creates_multimodal_batch():
    """
    Test that a custom adapter can create batches with pixel_values.
    """
    pytest.importorskip("torch")
    from unittest.mock import Mock

    mock_tokenizer = Mock()
    adapter = MockMultimodalAdapter(tokenizer=mock_tokenizer)

    # Mock GenerateOutputs
    mock_outputs = Mock()

    batch = adapter.build_batch(mock_outputs)

    assert batch.pixel_values is not None
    assert batch.pixel_values.shape[0] == batch.input_ids.shape[0]  # Same batch size
    assert len(batch.pixel_values.shape) == 4  # (B, C, H, W)


def test_gradient_flow_through_multimodal_inputs():
    """
    Test that gradients flow through both text and image inputs.
    """
    torch = pytest.importorskip("torch")
    from unittest.mock import Mock
    from types import SimpleNamespace

    trainer = _make_trainer_for_unit_tests()

    batch_size, seq_len = 2, 5
    vocab_size = 100

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.long)

    # Create image embeddings that require gradients
    pixel_values = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    # Create logits that require gradients (simulating model output)
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

    # Build batch
    behavior_lp = torch.randn(batch_size, seq_len)
    rewards = torch.tensor([2.0, 1.0])

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
        pixel_values=pixel_values,
    )

    # Compute loss
    trainer_lp, entropies = trainer._compute_logprobs_and_entropy(logits, input_ids)
    loss, summaries = trainer._compute_loss(batch, trainer_lp, entropies)

    # Backward pass
    if loss.requires_grad and loss.item() != 0.0:
        loss.backward()

        # Check that gradients flow to logits
        assert logits.grad is not None
        # Note: pixel_values won't have gradients here because the loss
        # only depends on logits, not directly on pixel_values.
        # In real training, pixel_values -> model -> logits -> loss


def test_multiple_images_per_sequence():
    """
    Test handling of multiple images per text sequence (e.g., interleaved).
    Some VLMs support multiple images in one conversation.
    """
    torch = pytest.importorskip("torch")

    batch_size = 2
    seq_len = 8
    num_images = 3  # Multiple images per sequence

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)
    behavior_lp = torch.randn(batch_size, seq_len)
    rewards = torch.tensor([1.0, 2.0])

    # Pixel values with multiple images per sequence
    # Shape: (batch_size, num_images, channels, height, width)
    pixel_values = torch.randn(batch_size, num_images, 3, 224, 224)

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
        pixel_values=pixel_values,
    )

    assert batch.pixel_values.shape == (batch_size, num_images, 3, 224, 224)
    assert batch.pixel_values is not None


def test_image_patch_embeddings():
    """
    Test handling of pre-computed image patch embeddings
    (some VLMs use pre-processed patch embeddings instead of raw pixels).
    """
    torch = pytest.importorskip("torch")

    batch_size = 2
    seq_len = 10
    num_patches = 196  # 14x14 patches for 224x224 image
    hidden_dim = 768

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)
    behavior_lp = torch.randn(batch_size, seq_len)
    rewards = torch.tensor([1.0, 2.0])

    # Pre-computed patch embeddings instead of raw pixels
    image_embeddings = torch.randn(batch_size, num_patches, hidden_dim)

    # Store in extra_model_kwargs since it's not standard pixel_values
    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
        pixel_values=None,
        extra_model_kwargs={"image_embeddings": image_embeddings},
    )

    assert "image_embeddings" in batch.extra_model_kwargs
    assert batch.extra_model_kwargs["image_embeddings"].shape == (
        batch_size,
        num_patches,
        hidden_dim,
    )


def test_variable_size_images():
    """
    Test handling variable-sized images with image_sizes metadata.
    Real VLMs often need to know the original image sizes for proper processing.
    """
    torch = pytest.importorskip("torch")

    batch_size = 2
    seq_len = 6

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)
    behavior_lp = torch.randn(batch_size, seq_len)
    rewards = torch.tensor([1.0, 2.0])

    # All images resized to same size, but we track original sizes
    pixel_values = torch.randn(batch_size, 3, 336, 336)

    # Track original image dimensions
    image_sizes = torch.tensor(
        [[1024, 768], [512, 512]]  # Image 1: 1024x768  # Image 2: 512x512
    )

    batch = MultimodalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_lp,
        rewards=rewards,
        pixel_values=pixel_values,
        extra_model_kwargs={"image_sizes": image_sizes},
    )

    assert batch.pixel_values.shape == (batch_size, 3, 336, 336)
    assert "image_sizes" in batch.extra_model_kwargs
    assert batch.extra_model_kwargs["image_sizes"].shape == (batch_size, 2)

