# Qwen3-VL-8B-Instruct Adapter

This directory contains the adapter for training **Qwen3-VL-8B-Instruct** with the multimodal RL trainer.

## Quick Start

### 1. Start vLLM Server

```bash
vf-vllm \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    --limit-mm-per-prompt image=4
```

### 2. Run Training

```bash
python scripts/vf-train-multimodal.py @ configs/vf-rl/qwen3-vl-8b.toml
```

## What the Adapter Does

The `Qwen3VLAdapter` handles:

1. **Image Loading** - From URLs, file paths, or base64 strings
2. **Conversation Formatting** - Qwen3-VL's message format with image tokens
3. **Tokenization** - Using Qwen2VLProcessor for text + images
4. **Loss Masking** - Only train on assistant responses
5. **Logprob Extraction** - Get behavior logprobs from vLLM
6. **Batch Construction** - Create MultimodalBatch with all necessary tensors

## Customization

### For Your Environment

Update `_format_conversation()` if your environment uses a different message structure:

```python
def _format_conversation(self, prompt, completion, image):
    # Your custom logic here
    # e.g., extract specific fields from prompt/state
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt['question']}
            ]
        },
        {
            'role': 'assistant', 
            'content': completion
        }
    ]
    return messages
```

### Image Sources

The adapter expects images in `state['image_url']` or `state['image']`. Update `build_batch()` if your environment stores images differently:

```python
# In build_batch():
for state in outputs.state:
    # Your custom image extraction
    image_data = state['my_custom_image_field']
    images.append(self._load_image(image_data))
```

### vLLM Logprobs

Currently uses dummy logprobs. To extract real vLLM logprobs, update `_extract_behavior_logprobs()`:

```python
def _extract_behavior_logprobs(self, outputs, input_ids):
    # Extract from vLLM response
    # vLLM returns logprobs in the completion metadata
    for i, metadata in enumerate(outputs.metadata):
        if 'logprobs' in metadata:
            # Align logprobs with input_ids
            vllm_logprobs = metadata['logprobs']
            # ... alignment logic ...
```

## Testing

Test your adapter with a small batch:

```python
from qwen3_vl_adapter import Qwen3VLAdapter
from transformers import Qwen2VLProcessor
from verifiers.types import GenerateOutputs

processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
adapter = Qwen3VLAdapter(processor.tokenizer, processor)

# Mock outputs
outputs = GenerateOutputs(
    prompt=[{"role": "user", "content": "What's in this image?"}],
    completion=["A cat sitting on a couch"],
    state=[{"image_url": "https://example.com/cat.jpg"}],
    reward=[{"reward": 1.0}],
    metadata=[{}]
)

batch = adapter.build_batch(outputs)
print(f"Batch shapes:")
print(f"  input_ids: {batch.input_ids.shape}")
print(f"  pixel_values: {batch.pixel_values.shape}")
print(f"  loss_mask: {batch.loss_mask.shape}")
```

## Qwen3-VL Specifics

### Image Tokens

Qwen3-VL uses:
- `<|vision_start|>` and `<|vision_end|>` to mark image regions
- `<|image_pad|>` tokens for each image patch

The processor handles this automatically.

### Dynamic Resolution

Qwen3-VL supports dynamic resolution via `image_grid_thw`:
- `t`: temporal dimension (1 for static images)
- `h`, `w`: grid height and width

This is automatically computed by the processor and passed in `extra_model_kwargs`.

### Chat Template

Uses Qwen's chat template:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|>...<|vision_end|>
What's in this image?<|im_end|>
<|im_start|>assistant
A cat<|im_end|>
```

## Environment Requirements

Your verifiers environment should return states with images:

```python
class MyMultimodalEnv(vf.Environment):
    def get_dataset(self):
        return Dataset.from_dict({
            "prompt": [...],
            "image_url": [...]  # or "image_path", "image_data"
        })
```

## Common Issues

**Images not loading:**
- Check `state['image_url']` exists
- Verify URLs are accessible or files exist
- Check image format (should be RGB)

**Token alignment errors:**
- Ensure completion tokens match vLLM output
- Check chat template is applied consistently

**Shape mismatches:**
- `behavior_logprobs` must be `(B, L)` not `(B, L-1)`
- Pad with zeros for prompt tokens

**CUDA OOM:**
- Reduce `micro_batch_size` in config
- Use smaller images or max_model_len
- Lower gpu_memory_utilization in vLLM

