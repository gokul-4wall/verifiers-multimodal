#!/usr/bin/env python3
"""
Multimodal trainer script - matches vf-train structure exactly.

Usage:
    python scripts/vf-train-multimodal.py @ configs/vf-rl/qwen3-vl-8b.toml
"""

import argparse
from pathlib import Path

try:
    import tomllib  # type: ignore[unresolved-import]
except ImportError:
    import tomli as tomllib  # type: ignore[unresolved-import]

import verifiers as vf
from transformers import Qwen2VLProcessor
from verifiers.rl.trainer.multimodal import MultimodalRLConfig, MultimodalGRPOTrainer

# Import adapters
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "adapters"))
from qwen3_vl_adapter import Qwen3VLAdapter


class YourVLMAdapter_UNUSED:
    """
    Custom adapter for your VLM family.
    
    You need to implement build_batch() to convert GenerateOutputs
    into a MultimodalBatch suitable for your specific VLM.
    
    Different VLM families tokenize images differently:
    - Qwen-VL: Uses <|image_pad|> tokens and image_grid_thw
    - LLaVA: Uses <IMAGE> tokens
    - Pixtral: Uses <IMG> and </IMG> tags
    
    See the multimodal trainer README for detailed examples.
    """
    
    def build_batch(self, outputs):
        """
        Convert verifiers GenerateOutputs into MultimodalBatch.
        
        Steps:
        1. Load images from outputs.state (e.g., from URLs or base64)
        2. Process images with your VLM's image processor
        3. Tokenize text with image tokens properly inserted
        4. Create loss_mask (which tokens to train on)
        5. Extract behavior logprobs from vLLM outputs
        6. Aggregate rewards per sequence
        7. Return MultimodalBatch
        """
        import torch
        
        # TODO: Implement your adapter logic here
        # This is a skeleton - you need to customize for your VLM
        
        batch_size = len(outputs.prompt)
        seq_len = 128  # Example
        
        # Dummy data - replace with actual implementation
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.ones_like(input_ids)
        behavior_logprobs = torch.randn(batch_size, seq_len)
        rewards = torch.tensor([r["reward"] for r in outputs.reward])
        
        # Load and process images
        # pixel_values = your_image_processor(images)
        pixel_values = torch.randn(batch_size, 3, 224, 224)  # Dummy
        
        return MultimodalBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            behavior_logprobs=behavior_logprobs,
            rewards=rewards,
            pixel_values=pixel_values,
            extra_model_kwargs={},  # Add model-specific kwargs if needed
        )


def main():
    # Match vf-train structure EXACTLY
    parser = argparse.ArgumentParser()
    parser.add_argument("at", type=str)
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    
    if args.at != "@":
        raise SystemExit("Usage: vf-train-multimodal @ path/to/file.toml")
    
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise SystemExit(f"TOML config not found: {config_path}")
    
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    
    model = config["model"]
    env_id = config["env"]["id"]
    env_args = config["env"].get("args", {})
    
    # Load environment (same as text-only)
    env = vf.load_environment(env_id=env_id, **env_args)
    
    # Load processor for VLM (only difference from text-only)
    processor = Qwen2VLProcessor.from_pretrained(model)
    
    # Create adapter (VLM-specific)
    adapter = Qwen3VLAdapter(
        tokenizer=processor.tokenizer,
        processor=processor
    )
    
    # Create trainer (same pattern as text-only: vf.RLTrainer(model, env, args))
    rl_config = MultimodalRLConfig(**config["trainer"].get("args", {}))
    trainer = MultimodalGRPOTrainer(
        model=model,  # Pass model string like text-only
        env=env,      # Pass env like text-only
        tokenizer=processor.tokenizer,  # VLM-specific: need tokenizer from processor
        adapter=adapter,  # VLM-specific: handles image+text inputs
        args=rl_config,
    )
    trainer.train()


if __name__ == "__main__":
    main()

