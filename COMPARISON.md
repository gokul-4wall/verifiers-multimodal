# Text-Only vs Multimodal Trainer Comparison

## Side-by-Side Structure

### Text-Only (`vf-train`)
```python
model = config["model"]
env_id = config["env"]["id"]
env_args = config["env"].get("args", {})
env = vf.load_environment(env_id=env_id, **env_args)
rl_config = vf.RLConfig(**config["trainer"].get("args", {}))
trainer = vf.RLTrainer(model=model, env=env, args=rl_config)
trainer.train()
```

### Multimodal (`vf-train-multimodal`)
```python
model = config["model"]
env_id = config["env"]["id"]
env_args = config["env"].get("args", {})
env = vf.load_environment(env_id=env_id, **env_args)
processor = Qwen2VLProcessor.from_pretrained(model)  # ← Only difference
adapter = Qwen3VLAdapter(processor.tokenizer, processor)  # ← Only difference
rl_config = vf.MultimodalRLConfig(**config["trainer"].get("args", {}))
trainer = vf.MultimodalGRPOTrainer(model=model, env=env, adapter=adapter, args=rl_config)
trainer.train()
```

## What's the Same ✅

1. **Config loading** - Identical TOML structure
2. **Environment loading** - `vf.load_environment()` 
3. **Trainer pattern** - `Trainer(model, env, args)`
4. **Training call** - `trainer.train()`

## What's Different 🔄

1. **Adapter** - VLMs need `MultimodalAdapter` for image+text processing
2. **Processor** - Load `Qwen2VLProcessor` instead of just tokenizer
3. **Config class** - `MultimodalRLConfig` vs `RLConfig` (minor)

## Line Count

- **Text-only script**: 40 lines
- **Multimodal script**: 130 lines (extra: unused adapter skeleton, could be removed)
- **Actual difference**: ~5 lines of code

## API Comparison

| Component | Text-Only | Multimodal |
|-----------|-----------|-----------|
| **Script** | `vf-train` | `vf-train-multimodal` |
| **Trainer** | `vf.RLTrainer` | `vf.MultimodalGRPOTrainer` |
| **Config** | `vf.RLConfig` | `vf.MultimodalRLConfig` |
| **Init Args** | `(model, env, args)` | `(model, env, adapter, args)` |
| **Model Loading** | Auto (`get_model_and_tokenizer`) | Auto (`Qwen2VLForConditionalGeneration`) |
| **Env Loading** | `vf.load_environment()` | `vf.load_environment()` ✅ Same |
| **Env Base** | `SingleTurnEnv` / `MultiTurnEnv` | `SingleTurnEnv` / `MultiTurnEnv` ✅ Same |
| **Rubric** | `Rubric` with reward funcs | `Rubric` with reward funcs ✅ Same |

## Merger Path

To merge into one repo later:

1. ✅ Both use same environment system
2. ✅ Both use same config pattern
3. ✅ Both use same trainer interface pattern  
4. ⚠️ Only difference: VLMs need adapter parameter
5. 💡 Could make adapter optional in future: `adapter=None` for text-only

**Minimal change to support both:**
```python
# Unified trainer
if is_multimodal_model(model):
    processor = load_processor(model)
    adapter = create_adapter(processor)
    trainer = MultimodalGRPOTrainer(model, env, adapter, args)
else:
    trainer = RLTrainer(model, env, args)
```

