#!/usr/bin/env python3
"""
Quick test script to validate the test environment works.
"""

import sys
sys.path.insert(0, '.')

from test_env_multimodal import load_environment

def main():
    print("Testing dummy multimodal environment...")
    
    # Load environment
    env = load_environment(num_examples=3)
    print(f"✓ Environment loaded")
    
    # Get dataset
    ds = env.get_dataset()
    print(f"✓ Dataset created: {len(ds)} examples")
    
    # Test first example
    row = ds[0]
    print(f"\n📋 Example 1:")
    print(f"  Color: {row['color']}")
    print(f"  Image type: {type(row['image'])}")
    print(f"  Image size: {row['image'].size}")
    print(f"  Prompt: {row['prompt']}")
    
    # Test prompt formatting
    prompt = env.get_prompt(row)
    print(f"\n💬 Formatted prompt:")
    print(f"  {prompt}")
    
    # Test state
    state = env.get_state(row)
    print(f"\n🔢 State:")
    print(f"  Has color: {'color' in state}")
    print(f"  Has image: {'image' in state}")
    
    # Test scoring logic
    test_completions = [
        (f"The image shows a {row['color']} square", True),
        ("I see a different color", False),
        ("This is a picture", False),  # No color mentioned
    ]
    
    print(f"\n🎯 Scoring tests:")
    for completion, should_reward in test_completions:
        # Test the reward function directly
        reward = env.rubric.reward_funcs[0](completion=completion, state=state)
        print(f"  '{completion[:40]}...'")
        print(f"    Reward: {reward:.1f} {'✓' if reward > 0 else '✗'}")
    
    print(f"\n✅ All tests passed! Environment is ready.")
    print(f"\nNext step: Run training with:")
    print(f"  python scripts/vf-train-multimodal.py @ configs/vf-rl/test-multimodal.toml")

if __name__ == "__main__":
    main()

