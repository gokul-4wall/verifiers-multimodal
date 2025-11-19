# Multimodal Trainer Quickstart

This guide shows you how to run the multimodal trainer, following the same pattern as the text-only `RLTrainer`.

## Setup

### 1. Install Dependencies

```bash
# Install verifiers with RL extras
uv sync --extra rl

# Or if using pip:
pip install torch transformers accelerate peft wandb trl vllm liger-kernel deepspeed
```

### 2. Directory Structure

```
your-project/
├── configs/
│   └── vf-rl/
│       └── your-vlm-config.toml  # Your config
├── scripts/
│   └── vf-train-multimodal.py    # Training script
└── adapters/
    └── your_vlm_adapter.py       # Your custom adapter
```

## Running the Trainer

### Method 1: Using tmux (Recommended - Follows text-only pattern)

Like the text-only trainer, you'll run two processes:
1. **vLLM server** (behavior policy for rollouts)
2. **Training script** (updates the policy)

**Step 1: Start vLLM server**

In one terminal:
```bash
vf-vllm \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    --limit-mm-per-prompt image=4
```

**Step 2: Start training**

In another terminal:
```bash
python scripts/vf-train-multimodal.py @ configs/vf-rl/your-vlm-config.toml
```

**Or use tmux** (like `vf-rl`):
```bash
# Create tmux session with both processes
tmux new-session -s vlm-rl -d "vf-vllm --model Qwen/Qwen2-VL-7B-Instruct --port 8000"
tmux split-window -v "python scripts/vf-train-multimodal.py @ configs/vf-rl/your-vlm-config.toml"
tmux attach-session -t vlm-rl
```

### Method 2: Python Script (For simple testing)

```python
from transformers import AutoModelForVision2Seq, AutoTokenizer
from verifiers.rl.trainer.multimodal import (
    MultimodalGRPOTrainer,
    MultimodalRLConfig,
)
from your_adapters import YourVLMAdapter

# Load model
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Create adapter (you implement this)
adapter = YourVLMAdapter(tokenizer=tokenizer)

# Configure
config = MultimodalRLConfig(
    output_dir="./outputs",
    env_id="your-multimodal-env",
    learning_rate=1e-5,
    max_steps=100,
    rollouts_per_step=16,
    vllm_server_host="localhost",
    vllm_server_port=8000,
)

# Train
trainer = MultimodalGRPOTrainer(model, tokenizer, adapter, config)
trainer.train()
```

## Configuration

### TOML Config File (Like text-only trainer)

Create `configs/vf-rl/my-vlm.toml`:

```toml
model = "Qwen/Qwen2-VL-7B-Instruct"

[env]
id = "your-multimodal-env"

[env.args]
# Environment-specific args

[inference]
gpus = 1

[inference.args]
enforce_eager = true
max_model_len = 8192
limit_mm_per_prompt = {"image": 4}

[trainer]
gpus = 1

[trainer.args]
run_name = "my-vlm-training"
output_dir = "./outputs"
learning_rate = 1e-5
max_steps = 100
batch_size = 512
micro_batch_size = 4
rollouts_per_step = 16
mask_ratio_low = 0.1
mask_ratio_high = 10.0
temperature = 1.0
vllm_server_host = "0.0.0.0"
vllm_server_port = 8000
max_concurrent = 128
sync_to_vllm_every = 1
log_every_steps = 10
```

## Implementing Your Adapter

The key difference from text-only training is you need a **custom adapter** for your VLM family:

```python
from verifiers.rl.trainer.multimodal import MultimodalAdapter, MultimodalBatch
from verifiers.types import GenerateOutputs
import torch

class QwenVLAdapter(MultimodalAdapter):
    """Adapter for Qwen2-VL models."""
    
    def build_batch(self, outputs: GenerateOutputs) -> MultimodalBatch:
        """
        Convert verifiers GenerateOutputs into MultimodalBatch.
        
        Args:
            outputs: Contains prompts, completions, states, rewards from rollouts
        
        Returns:
            MultimodalBatch with all tensors ready for training
        """
        # 1. Load images from states
        images = [self._load_image(state['image_url']) 
                  for state in outputs.state]
        
        # 2. Process images (use your VLM's image processor)
        pixel_values = self.image_processor(images)['pixel_values']
        
        # 3. Tokenize text with image tokens
        texts = [self._format_with_images(p, c) 
                 for p, c in zip(outputs.prompt, outputs.completion)]
        tokenized = self.tokenizer(texts, padding=True, return_tensors='pt')
        
        # 4. Create loss mask (train only on completion tokens)
        loss_mask = self._create_loss_mask(
            tokenized['input_ids'],
            outputs.prompt,
            outputs.completion
        )
        
        # 5. Extract behavior logprobs from vLLM
        behavior_logprobs = self._extract_vllm_logprobs(outputs)
        
        # 6. Aggregate rewards
        rewards = torch.tensor([r['reward'] for r in outputs.reward])
        
        return MultimodalBatch(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
            loss_mask=loss_mask,
            behavior_logprobs=behavior_logprobs,
            rewards=rewards,
            pixel_values=pixel_values,
            extra_model_kwargs={'image_grid_thw': grid_thw},  # VLM-specific
        )
```

See `README.md` in this directory for detailed adapter examples for different VLM families.

## Comparison with Text-Only Trainer

| Component | Text-Only Trainer | Multimodal Trainer |
|-----------|-------------------|-------------------|
| **Config File** | TOML | TOML (same format) |
| **Start vLLM** | `vf-vllm --model <text-model>` | `vf-vllm --model <vlm-model>` |
| **Run Training** | `vf-train @ config.toml` | `vf-train-multimodal @ config.toml` |
| **Main Class** | `RLTrainer` | `MultimodalGRPOTrainer` |
| **Config Class** | `RLConfig` | `MultimodalRLConfig` |
| **Adapter** | Built-in | **Custom required** |
| **Inputs** | Text only | Text + Images |
| **Dependencies** | peft, deepspeed, etc. | Same |

## Workflow

1. **Write config** → `configs/vf-rl/my-vlm.toml`
2. **Implement adapter** → `adapters/my_vlm_adapter.py`
3. **Start vLLM** → Terminal 1: `vf-vllm --model <vlm>`
4. **Start training** → Terminal 2: `python scripts/vf-train-multimodal.py @ config.toml`
5. **Monitor** → Check logs, tmux panes, or WandB

## Example: Full Training Run

```bash
# Terminal 1: Start vLLM
vf-vllm \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.8

# Terminal 2: Start training
python scripts/vf-train-multimodal.py @ configs/vf-rl/qwen-vl.toml
```

Training output:
```
Loading VLM: Qwen/Qwen2-VL-7B-Instruct
Loading environment: your-multimodal-env
Initializing MultimodalGRPOTrainer
Starting training...
Training:   1%|▏         | 1/100 [00:12<20:05, 12.18s/it, loss=0.45, ir=1.02, ent=2.34, kl=0.01]
```

## Troubleshooting

**vLLM not connecting:**
- Check vLLM is running: `curl http://localhost:8000/health`
- Verify port in config matches vLLM port

**Adapter errors:**
- Make sure your adapter returns properly shaped tensors
- Check `behavior_logprobs` is shape `(B, L)` not `(B, L-1)`
- Verify pixel_values match your VLM's expected input

**CUDA OOM:**
- Reduce `micro_batch_size` in config
- Lower `gpu_memory_utilization` in vLLM
- Use smaller `max_model_len`

## Next Steps

- Implement your adapter for your specific VLM family
- Create a multimodal environment (images + text)
- Test with small batch sizes first
- Scale up once working

See the full `README.md` for architecture details and VLM-specific adapter examples.

