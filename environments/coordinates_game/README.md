# Coordinates Game Environment

A multimodal RL environment where models must identify the (x, y) coordinates of colored dots on a 10x10 grid.

## Overview

This environment tests spatial reasoning and coordinate understanding by presenting models with grid images containing one blue, one green, and one red dot. The model must output all three coordinates in the format `\boxed{blue:x,y; green:x,y; red:x,y}`.

## Features

- **Procedurally Generated**: Infinite examples generated on-demand
- **Clear Visual Indicators**:
  - (0,0) marked at bottom-left corner
  - x-axis arrow (→) showing rightward increase
  - y-axis arrow (↑) showing upward increase
  - No numeric tick labels (to reduce clutter)
- **Dense Reward**: Linear decay based on Manhattan distance from target
- **0-Indexed Coordinates**: Range from (0,0) to (9,9)

## Installation

```bash
cd environments/coordinates_game
uv pip install -e .
```

## Usage

### Basic Evaluation

```bash
vf-eval coordinates-game
```

### Python API

```python
from verifiers.utils.env_utils import load_environment

# Load environment
env = load_environment("coordinates-game", num_examples=100)

# Generate examples
results = await env.generate(model="gpt-4o", num_examples=10)
```

### RL Training

Update your config file:

```toml
[[env]]
id = "coordinates-game"
args = { num_examples = 1000 }
```

## Coordinate System

```
     0  1  2  3  4  5  6  7  8  9
   ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
 0 │  │  │  │  │  │  │  │  │  │  │
   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤  y
 1 │  │  │  │● │  │  │  │  │  │  │  ↓
   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
 ...
 9 │  │  │  │  │  │  │  │  │  │  │
   └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
              x →
```

- **Origin**: (0,0) at bottom-left corner
- **X-axis**: Increases rightward (0 → 9)
- **Y-axis**: Increases upward (0 → 9)

## Reward Function

Uses **linear Manhattan distance** per dot, averaged across the 3 dots:

```python
d = abs(pred_x - true_x) + abs(pred_y - true_y)  # Manhattan distance
reward = max(0.0, 1.0 - d / k)  # k = 5
```

The environment then scales the final per-sample reward by `reward_scale` (default `10.0`).

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_examples` | int | 1000 | Number of examples to generate |
| `grid_size` | int | 10 | Grid dimensions (N×N) |
| `seed` | int | 42 | Random seed for reproducibility |
| `k` | int | 5 | Manhattan distance scale for reward |
| `reward_scale` | float | 10.0 | Multiplier applied to the final averaged reward |

## Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Linear Manhattan reward scaled by `reward_scale` (default 0-10) |
| `correct_answer` | Same as reward (single reward function) |

## Example Output Format

The model should respond with reasoning followed by the boxed answer:

```
Let me analyze the grid carefully. Looking at the blue dot, I can see it's positioned:
- In the 4th column from the left (x = 3, since 0-indexed)
- In the 2nd row from the top (y = 1, since 0-indexed)

Therefore, the coordinate is \boxed{3,1}
```

## Notes

- Images are 500×500 pixels with 60px margins for labels
- Grid cells are 40×40 pixels
- Blue dot has a 15px radius for clear visibility
- All examples are deterministic based on seed + index


