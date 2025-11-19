# Multimodal RL Trainer

A specialized GRPO-style trainer for vision-language models (VLMs) that extends the `verifiers` RL framework to support true multimodal (text + image) training.

## Why a Separate Multimodal Trainer?

The existing `RLTrainer` in `verifiers/rl/trainer/` is designed exclusively for **text-only** language models and cannot support multimodal inputs without significant architectural changes. This multimodal trainer was built to address those limitations while maintaining compatibility with the verifiers environment system.

---

## Key Differences: Multimodal vs. Text-Only Trainer

### 1. **Batch Representation**

**Text-Only Trainer (`RLTrainer`)**
- Uses simple dictionaries with text tensors only
- Batch inputs: `input_ids`, `loss_mask`, `inference_logprobs`, `advantages`
- Directly calls `model(input_ids=..., attention_mask=...)`
- No extensibility for additional modalities

**Multimodal Trainer (`MultimodalGRPOTrainer`)**
- Uses structured `MultimodalBatch` dataclass
- Supports additional fields:
  - `pixel_values`: Image tensors (B, C, H, W) or (B, N, C, H, W)
  - `extra_model_kwargs`: Model-specific inputs (image_sizes, grid_thw, embeddings, etc.)
- Flexible batch construction for different VLM architectures

```python
@dataclass
class MultimodalBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    behavior_logprobs: torch.Tensor
    rewards: torch.Tensor
    pixel_values: torch.Tensor | None = None           # NEW
    extra_model_kwargs: Dict[str, torch.Tensor] = ...  # NEW
```

---

### 2. **Model Forward Pass**

**Text-Only Trainer**
```python
# Fixed: only text inputs
logits = model(
    input_ids=input_ids_batch,
    attention_mask=attention_mask_batch,
).logits
```

**Multimodal Trainer**
```python
# Flexible: supports text + images + model-specific kwargs
model_kwargs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
}
if pixel_values is not None:
    model_kwargs["pixel_values"] = pixel_values
model_kwargs.update(extra_kwargs)  # image_sizes, grid_thw, etc.

outputs = model(**model_kwargs)
```

This allows VLMs to receive:
- **Images** as `pixel_values`
- **Image metadata** (original sizes, aspect ratios)
- **Pre-computed embeddings** (patch features)
- **Multi-image inputs** (for interleaved conversations)

---

### 3. **Adapter Pattern**

**Text-Only Trainer**
- Uses `env.process_env_results_vllm()` method
- Environment-specific processing is hardcoded
- Assumes text tokenization aligns perfectly
- No customization for different model families

**Multimodal Trainer**
- Introduces `MultimodalAdapter` abstract class
- Separates rollout processing from training logic
- Users implement `build_batch()` for their VLM family

```python
class MultimodalAdapter(ABC):
    @abstractmethod
    def build_batch(self, outputs: GenerateOutputs) -> MultimodalBatch:
        """Convert rollouts to model-specific batch format."""
        raise NotImplementedError
```

**Why This Matters:**
- **Qwen-VL** tokenizes images differently than **LLaVA**
- **Pixtral** uses different image preprocessing than **InternVL**
- Each VLM family needs custom logic for:
  - Image token insertion/alignment
  - Loss mask construction (which tokens to train on)
  - Handling variable-sized/multiple images

---

### 4. **Integration with Accelerate/DeepSpeed**

**Text-Only Trainer**
- Inherits from `transformers.Trainer`
- Uses `Accelerator` for distributed training
- Built-in DeepSpeed integration
- Complex setup but production-ready

**Multimodal Trainer**
- Standalone trainer (no `Trainer` inheritance)
- Simpler, more transparent implementation
- Single-process, single-node design
- Easy to understand and modify for research

**Trade-offs:**
- Text-only trainer: Better for large-scale production
- Multimodal trainer: Better for research iteration and debugging

---

### 5. **Data Flow Architecture**

**Text-Only Trainer**
```
Environment → vLLM → Orchestrator → process_env_results_vllm()
                                    ↓
                            Text tokens + logprobs
                                    ↓
                            Model forward pass
```

**Multimodal Trainer**
```
Environment → vLLM → GenerateOutputs
                           ↓
                    MultimodalAdapter.build_batch()
                           ↓
                    Text + Images + Metadata
                           ↓
                    VLM forward pass
```

The adapter layer provides flexibility to:
- Load images from disk/URLs
- Process image features
- Align text/image tokens
- Handle model-specific preprocessing

---

### 6. **Why Text-Only Trainer Can't Support Multimodal**

#### Problem 1: Fixed Input Signature
The `get_logprobs()` method hardcodes text-only inputs:
```python
def get_logprobs(self, model, input_ids, attention_mask, batch_size=None):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
```
There's no way to pass `pixel_values` or other modality inputs.

#### Problem 2: Orchestrator Assumptions
The `Orchestrator` class assumes `process_env_results_vllm()` returns only text:
```python
processed_results = self.env.process_env_results_vllm(
    prompts=..., completions=..., states=..., rewards=...,
    processing_class=self.processing_class,
    ...
)
```
No mechanism to attach or process images.

#### Problem 3: No Image Tensor Management
- No code to load/process images
- No device movement for pixel_values
- No handling of variable-sized images
- No support for model-specific image kwargs

#### Problem 4: Tokenizer-Centric Design
The text-only trainer assumes tokenization is all that's needed:
```python
input_ids = pad([torch.tensor(x) for x in microbatch.input_ids], ...)
```
VLMs need:
- Image token placeholders (`<image>`)
- Alignment between text and vision tokens
- Special handling for interleaved text/images

---

## Multimodal-Specific Features

### 1. Flexible Image Shapes
Supports various image tensor formats:
- Standard: `(B, 3, 224, 224)` - RGB images
- High-res: `(B, 3, 336, 336)` or `(B, 3, 512, 512)`
- Multi-image: `(B, N, 3, 224, 224)` - multiple images per sequence
- Grayscale: `(B, 1, H, W)`

### 2. Model-Specific Metadata
Supports VLM-specific inputs via `extra_model_kwargs`:
- `image_sizes`: Original image dimensions before resize
- `image_grid_thw`: Grid layout for high-res patches
- `aspect_ratios`: For dynamic resolution handling
- `image_embeddings`: Pre-computed vision features

### 3. Adapter Pattern Examples

**Qwen-VL Adapter**
```python
class QwenVLAdapter(MultimodalAdapter):
    def build_batch(self, outputs: GenerateOutputs) -> MultimodalBatch:
        # Load images from URLs in states
        images = [load_image(s['image_url']) for s in outputs.state]
        pixel_values = self.processor(images=images)['pixel_values']
        
        # Tokenize with <|image_pad|> tokens
        input_ids = self.tokenizer.apply_chat_template(...)
        
        # Build loss mask (only train on assistant responses)
        loss_mask = self.create_loss_mask(input_ids, ...)
        
        return MultimodalBatch(
            input_ids=input_ids,
            pixel_values=pixel_values,
            extra_model_kwargs={'image_grid_thw': grid_thw},
            ...
        )
```

**LLaVA Adapter**
```python
class LLaVAAdapter(MultimodalAdapter):
    def build_batch(self, outputs: GenerateOutputs) -> MultimodalBatch:
        # LLaVA uses simple <IMAGE> tokens
        # Different preprocessing pipeline
        ...
```

---

## Usage Example

```python
from multimodal_rl import MultimodalGRPOTrainer, MultimodalRLConfig
from my_adapters import QwenVLAdapter

# Load VLM
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Create adapter
adapter = QwenVLAdapter(tokenizer=tokenizer)

# Configure training
config = MultimodalRLConfig(
    output_dir="./outputs",
    learning_rate=1e-5,
    max_steps=1000,
    rollouts_per_step=16,
    vllm_server_host="localhost",
    vllm_server_port=8000,
)

# Train!
trainer = MultimodalGRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    adapter=adapter,
    config=config,
)
trainer.set_environment("my-vlm-task")
trainer.train()
```

---

## When to Use Each Trainer

### Use Text-Only Trainer (`RLTrainer`) when:
- ✅ Training text-only language models
- ✅ Need production-grade distributed training
- ✅ Want DeepSpeed/Accelerate integration
- ✅ Standard text tokenization is sufficient

### Use Multimodal Trainer (`MultimodalGRPOTrainer`) when:
- ✅ Training vision-language models (VLMs)
- ✅ Need image + text inputs
- ✅ Want to experiment with different VLM architectures
- ✅ Need fine-grained control over batch construction
- ✅ Handling complex multimodal scenarios (multiple images, video, etc.)

---

## Architecture Summary

| Feature | Text-Only Trainer | Multimodal Trainer |
|---------|------------------|-------------------|
| **Input Modalities** | Text only | Text + Images + Metadata |
| **Batch Type** | Dict | `MultimodalBatch` dataclass |
| **Model Forward** | Fixed signature | Flexible kwargs |
| **Extensibility** | Limited | Adapter pattern |
| **VLM Support** | ❌ Not possible | ✅ Native support |
| **Distributed** | DeepSpeed/Accelerate | Single-node |
| **Complexity** | High (inherits Trainer) | Low (standalone) |
| **Use Case** | Production LLM training | Research VLM training |

---

## Testing

See comprehensive test suite:
- `tests/test_multimodal_trainer_math.py` - Core math/loss computation
- `tests/test_multimodal_trainer_edge_cases.py` - Edge cases and failure modes
- `tests/test_multimodal_inputs.py` - **Multimodal-specific tests**
  - Image tensor handling
  - Adapter pattern
  - Variable-sized images
  - Multiple images per sequence
  - Model-specific kwargs
  - Gradient flow through images

Run tests:
```bash
pytest tests/test_multimodal*.py -v
```

---

## Future Extensions

The adapter pattern makes it easy to extend to:
- **Video inputs**: Add temporal dimension to pixel_values
- **Audio inputs**: Add waveforms or spectrograms
- **3D data**: Point clouds, meshes via extra_model_kwargs
- **Multi-modal retrieval**: Embeddings for retrieved context

The core trainer logic remains unchanged; only the adapter needs modification.

---

## Limitations

1. **Single-node only**: No distributed training support (yet)
2. **Synchronous rollouts**: No async orchestrator like text-only trainer
3. **Manual adapter creation**: Users must implement adapters for each VLM family
4. **No LoRA support**: Full parameter training only (for now)

These are deliberate simplifications to keep the multimodal implementation transparent and hackable for research use cases.

---

## Summary

The multimodal trainer was necessary because the existing text-only trainer's architecture fundamentally assumes text-only inputs. Key innovations:

1. **`MultimodalBatch`** - Structured container for text + images + metadata
2. **`MultimodalAdapter`** - Extensible pattern for VLM-specific processing
3. **Flexible forward pass** - Supports arbitrary model kwargs
4. **Transparent implementation** - Simple, hackable codebase for research

This enables true multimodal RL training with vision-language models while maintaining compatibility with the verifiers environment ecosystem.

