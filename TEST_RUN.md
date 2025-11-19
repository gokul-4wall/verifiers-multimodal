# Multimodal Trainer Test Run

## ✅ Progress So Far

1. ✅ Model loads successfully with `AutoModelForImageTextToText`
2. ✅ Processor and adapter initialize correctly
3. ✅ Trainer initializes
4. ⚠️ Needs vLLM server running

## 🚀 How to Run Complete Test

### Step 1: Start vLLM Server

In **terminal 1**, start the vLLM server for Qwen3-VL:

```bash
cd /home/antim/gokul/verifiers-multimodal
source .venv/bin/activate

# Start vLLM for Qwen3-VL (this will keep running)
NCCL_P2P_DISABLE=1 uv run vf-vllm \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --port 8000 \
  --max-model-len 8192 \
  --enforce-eager
```

**Note**: 
- `NCCL_P2P_DISABLE=1` fixes NCCL peer-to-peer GPU communication issues
- This will take ~1-2 minutes to load the model and start the server
- Wait for "Uvicorn running" message before proceeding to Step 2

### Step 2: Run Trainer

In **terminal 2** (once vLLM server says "Uvicorn running"):

```bash
cd /home/antim/gokul/verifiers-multimodal
source .venv/bin/activate

# Run the multimodal trainer (must use same NCCL settings as vLLM!)
NCCL_P2P_DISABLE=1 python scripts/vf-train-multimodal.py @ configs/vf-rl/test-multimodal.toml
```

**Important**: Use `NCCL_P2P_DISABLE=1` for **both** vLLM and trainer to ensure they can communicate.

## 📊 Expected Output

### vLLM Server (Terminal 1)
```
INFO: Started server process [...]
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Trainer (Terminal 2)
```
Loading environment: test_env_multimodal
Loading VLM: Qwen/Qwen3-VL-8B-Instruct
Loading checkpoint shards: 100%|██████████| 4/4
Server is up!
vLLM world size: 1
Starting training...
Step 1/100: loss=X.XXX
...
```

## 🐛 Common Errors

### Error 1: NCCL Error - No vLLM Server
```
RuntimeError: NCCL error: invalid usage
```
**Cause**: No vLLM server running.  
**Fix**: Start vLLM server first (Step 1 above).

### Error 2: NCCL Error - Communication Mismatch
```
RuntimeError: NCCL error: remote process exited or there was a network error
```
**Cause**: NCCL settings mismatch between vLLM and trainer.  
**Fix**: Use `NCCL_P2P_DISABLE=1` for **BOTH** processes:
```bash
# Terminal 1 - vLLM
NCCL_P2P_DISABLE=1 uv run vf-vllm --model ... --port 8000

# Terminal 2 - Trainer (must match!)
NCCL_P2P_DISABLE=1 python scripts/vf-train-multimodal.py @ config.toml
```

### Error 3: vLLM Not Ready
If trainer starts before vLLM finishes loading, wait for vLLM to show:
```
INFO: Uvicorn running on http://0.0.0.0:8000
```

## 🔧 Quick Test (Without vLLM)

If you want to test just the model loading without starting vLLM:

```python
from transformers import AutoModelForImageTextToText, Qwen2VLProcessor
import torch

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

print("✅ Model and processor loaded successfully!")
```

## 📝 Test Environment Details

- **Model**: Qwen/Qwen3-VL-8B-Instruct (8B parameters, vision-language)
- **Environment**: `test_env_multimodal` (5 colored image examples)
- **Task**: Model describes colored images, rewarded for correct color
- **Training**: 5 rollouts, 100 steps max

## 🎯 Success Criteria

1. ✅ vLLM server starts without errors
2. ✅ Trainer connects to vLLM
3. ✅ Generates completions for colored images
4. ✅ Computes loss and updates model
5. ✅ No crashes during training loop

## 📚 Files Involved

- `scripts/vf-train-multimodal.py` - Training script
- `configs/vf-rl/test-multimodal.toml` - Config
- `test_env_multimodal.py` - Dummy environment
- `adapters/qwen3_vl_adapter.py` - Qwen3-VL adapter
- `verifiers/rl/trainer/multimodal/trainer.py` - Trainer core

## 🔄 Comparison with Text-Only

| Aspect | Text-Only | Multimodal |
|--------|-----------|-----------|
| **Model Loading** | `get_model_and_tokenizer()` | `AutoModelForImageTextToText.from_pretrained()` |
| **Extra Components** | None | Processor + Adapter |
| **vLLM Args** | Standard | + `--max-model-len`, supports images |
| **Env State** | Just `prompt` | `prompt` + `image` |
| **Everything Else** | ✅ Same | ✅ Same |

---

**Next**: Start vLLM server to complete the test run!
