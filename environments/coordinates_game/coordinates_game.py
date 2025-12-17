import base64
import random
from io import BytesIO
from typing import Iterator

from PIL import Image, ImageDraw, ImageFont
from datasets import Dataset

import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer


def generate_grid_image(
    dots: dict[str, tuple[int, int]],
    grid_size: int = 10,
    cell_size: int = 40,
) -> Image.Image:
    """
    Generate a grid image with colored dots at specified positions.
    Uses standard mathematical coordinate system: (0,0) at bottom-left,
    x increases rightward, y increases upward.
    
    Args:
        dots: Mapping from color name to (x, y) coordinate.
            Supported keys: "blue", "green", "red".
            Coordinates are 0-indexed with (0,0) at bottom-left.
        grid_size: Number of cells per side (default 10 for 10x10 grid)
        cell_size: Size of each cell in pixels (default 40)
    
    Returns:
        PIL Image with grid, axis labels, and colored dots
    """
    # Calculate dimensions with margins for labels
    margin = 60  # Space for axis labels and arrows
    grid_pixel_size = grid_size * cell_size
    canvas_size = grid_pixel_size + 2 * margin
    
    # Create white canvas
    img = Image.new('RGB', (canvas_size, canvas_size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    
    # Grid offset (top-left corner of the grid on canvas)
    grid_offset_x = margin
    grid_offset_y = margin
    
    # Draw grid lines
    for i in range(grid_size + 1):
        # Vertical lines (thicker at x=0, which is the left edge)
        x_pos = grid_offset_x + i * cell_size
        draw.line([(x_pos, grid_offset_y), (x_pos, grid_offset_y + grid_pixel_size)], 
                  fill='black', width=2 if i == 0 else 1)
        
        # Horizontal lines (thicker at y=0, which is the bottom edge)
        y_pos = grid_offset_y + i * cell_size
        is_bottom = (i == grid_size)  # Bottom edge in pixel space
        draw.line([(grid_offset_x, y_pos), (grid_offset_x + grid_pixel_size, y_pos)], 
                  fill='black', width=2 if is_bottom else 1)
    
    # Mark (0,0) at bottom-left corner
    bottom_left_x = grid_offset_x - 25
    bottom_left_y = grid_offset_y + grid_pixel_size + 5
    draw.text((bottom_left_x, bottom_left_y), "(0,0)", fill='red', font=font)
    
    # Draw x-axis label and arrow at the bottom
    x_label_y = grid_offset_y + grid_pixel_size + 35
    x_label_center = grid_offset_x + grid_pixel_size // 2
    draw.text((x_label_center - 20, x_label_y), "x →", fill='blue', font=font)
    
    # Draw arrow for x-axis (pointing right)
    arrow_y = x_label_y + 5
    draw.line([(grid_offset_x + 20, arrow_y), (grid_offset_x + grid_pixel_size - 20, arrow_y)], 
              fill='blue', width=2)
    # Arrowhead
    draw.polygon([(grid_offset_x + grid_pixel_size - 20, arrow_y - 5),
                  (grid_offset_x + grid_pixel_size - 20, arrow_y + 5),
                  (grid_offset_x + grid_pixel_size - 10, arrow_y)], 
                 fill='blue')
    
    # Draw y-axis label and arrow on the LEFT side
    y_label_x = grid_offset_x - 45
    y_label_center = grid_offset_y + grid_pixel_size // 2
    draw.text((y_label_x, y_label_center - 10), "y ↑", fill='green', font=font)
    
    # Draw arrow for y-axis (pointing upward)
    arrow_x = y_label_x + 15
    draw.line([(arrow_x, grid_offset_y + 20), (arrow_x, grid_offset_y + grid_pixel_size - 20)], 
              fill='green', width=2)
    # Arrowhead (pointing up)
    draw.polygon([(arrow_x - 5, grid_offset_y + 20),
                  (arrow_x + 5, grid_offset_y + 20),
                  (arrow_x, grid_offset_y + 10)], 
                 fill='green')
    
    # Note: we intentionally do NOT draw numeric tick labels (0..9) on the axes.
    # The origin marker (0,0) plus axis arrows/labels provide directionality.
    
    # Draw colored dots
    palette = {
        "blue": ("blue", "darkblue"),
        "green": ("limegreen", "darkgreen"),
        "red": ("red", "darkred"),
    }
    dot_radius = 14
    for color, (x, y) in dots.items():
        if color not in palette:
            continue
        fill, outline = palette[color]

        # x increases left to right (standard)
        dot_center_x = grid_offset_x + x * cell_size + cell_size // 2
        # y increases bottom to top, so flip the y coordinate
        dot_center_y = grid_offset_y + (grid_size - 1 - y) * cell_size + cell_size // 2

        draw.ellipse(
            [
                dot_center_x - dot_radius,
                dot_center_y - dot_radius,
                dot_center_x + dot_radius,
                dot_center_y + dot_radius,
            ],
            fill=fill,
            outline=outline,
            width=2,
        )
    
    return img


class ProceduralDataset:
    """
    Procedurally generated dataset for coordinate detection.
    Generates examples on-demand without storing data.
    """
    
    def __init__(self, num_examples: int = 1000, grid_size: int = 10, seed: int = 42):
        self.num_examples = num_examples
        self.grid_size = grid_size
        self.seed = seed
        self._rng = random.Random(seed)
    
    def __len__(self) -> int:
        return self.num_examples
    
    def __getitem__(self, idx: int) -> dict:
        """Generate a single example."""
        # Use deterministic random based on index for reproducibility
        local_rng = random.Random(self.seed + idx)
        # Sample 3 unique coordinates (no overlap)
        coords: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()
        while len(coords) < 3:
            x = local_rng.randint(0, self.grid_size - 1)
            y = local_rng.randint(0, self.grid_size - 1)
            if (x, y) in used:
                continue
            used.add((x, y))
            coords.append((x, y))

        dots = {
            "blue": coords[0],
            "green": coords[1],
            "red": coords[2],
        }

        # Generate the grid image
        image = generate_grid_image(dots, grid_size=self.grid_size)

        (bx, by) = dots["blue"]
        (gx, gy) = dots["green"]
        (rx, ry) = dots["red"]
        return {
            "blue_x": bx,
            "blue_y": by,
            "green_x": gx,
            "green_y": gy,
            "red_x": rx,
            "red_y": ry,
            "image": image,
            "answer": f"blue:{bx},{by}; green:{gx},{gy}; red:{rx},{ry}",
        }
    
    def __iter__(self) -> Iterator[dict]:
        """Iterate over the dataset."""
        for i in range(self.num_examples):
            yield self[i]


def format_prompt(example: dict) -> list[dict]:
    """
    Format a single example into a multimodal prompt.
    
    Args:
        example: Dict with 'image' (PIL Image) and coordinate data
    
    Returns:
        List of message dicts in OpenAI format
    """
    pil_image = example["image"]
    
    # Encode image as base64
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    b64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Create instruction text
    instruction = (
        "You must output the coordinates of the blue, green, and red dots based on the image given. "
        "The grid uses standard mathematical coordinates where (0,0) is at the bottom-left corner. "
        "The x-axis increases to the right (→) and the y-axis increases upward (↑). "
        "All coordinates are positive integers from 0 to 9. "
        "Think step-by-step and give your final answer in exactly this format: "
        "\\boxed{blue:x,y; green:x,y; red:x,y}.\n\n"
        "Example (how to reason):\n"
        "- Blue dot is at x=3 (3 squares right), y=7 (7 squares up).\n"
        "- Green dot is at x=0 (left edge), y=0 (origin).\n"
        "- Red dot is at x=9 (right edge), y=2 (2 squares up).\n"
        "Final: \\boxed{blue:3,7; green:0,0; red:9,2}"
    )
    
    # Format as multimodal message
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                },
            ],
        }
    ]
    
    return prompt


def load_environment(
    num_examples: int = 10_000,
    grid_size: int = 10,
    seed: int = 42,
    k: int = 5,
    reward_scale: float = 10.0,
    **kwargs
) -> vf.Environment:
    """
    Load the coordinates-game environment.
    
    Args:
        num_examples: Number of examples to generate (default 10,000)
        grid_size: Size of the grid (default 10 for 10x10)
        seed: Random seed for reproducibility (default 42)
        **kwargs: Additional arguments passed to SingleTurnEnv
    
    Returns:
        Configured verifiers Environment
    """
    # Create procedural dataset
    procedural_ds = ProceduralDataset(
        num_examples=num_examples,
        grid_size=grid_size,
        seed=seed
    )
    
    # Convert to HuggingFace Dataset format
    dataset = Dataset.from_generator(
        lambda: procedural_ds,
        features=None,
    ).map(lambda x: {"prompt": format_prompt(x), "answer": x["answer"]})
    
    # Remove intermediate columns, keep only prompt and answer
    cols_to_remove = [col for col in dataset.column_names if col not in ["prompt", "answer"]]
    dataset = dataset.remove_columns(cols_to_remove)
    
    # Create parser
    parser = vf.Parser(extract_fn=extract_boxed_answer)
    
    def correct_answer(parser, completion, answer) -> float:
        """
        Reward function using linear Manhattan distance per dot, averaged.

        Returns reward in [0, 1]. Perfect match for all 3 dots = 1.0.
        """
        import re

        parsed_answer = (parser.parse_answer(completion) or "").strip()
        answer = (answer or "").strip()

        def parse_by_color(text: str) -> dict[str, tuple[int, int] | None]:
            t = text.lower()
            out: dict[str, tuple[int, int] | None] = {"blue": None, "green": None, "red": None}
            for color in out.keys():
                m = re.search(rf"{color}\s*[:=]\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", t)
                if m:
                    out[color] = (int(m.group(1)), int(m.group(2)))
            # Fallback: if no labels, take first three coordinate pairs in order blue, green, red
            if all(v is None for v in out.values()):
                pairs = re.findall(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", t)
                if len(pairs) >= 3:
                    out["blue"] = (int(pairs[0][0]), int(pairs[0][1]))
                    out["green"] = (int(pairs[1][0]), int(pairs[1][1]))
                    out["red"] = (int(pairs[2][0]), int(pairs[2][1]))
            return out

        pred = parse_by_color(parsed_answer)
        gold = parse_by_color(answer)

        # Linear reward on Manhattan distance:
        # reward = max(0, 1 - d / k), where d = |dx| + |dy|.
        def manhattan_reward(p: tuple[int, int] | None, g: tuple[int, int] | None) -> float:
            if p is None or g is None:
                return 0.0
            (px, py) = p
            (gx, gy) = g
            d = abs(px - gx) + abs(py - gy)
            # Defensive: avoid division-by-zero if misconfigured.
            denom = float(k) if k != 0 else 1.0
            return max(0.0, 1.0 - (float(d) / denom))

        rewards = [
            manhattan_reward(pred["blue"], gold["blue"]),
            manhattan_reward(pred["green"], gold["green"]),
            manhattan_reward(pred["red"], gold["red"]),
        ]
        return (sum(rewards) / 3.0) * float(reward_scale)
    
    # Create rubric with reward function
    rubric = vf.Rubric(funcs=[correct_answer], parser=parser)
    
    # System prompt to reinforce coordinate system understanding
    system_prompt = (
        "You are a precise coordinate detector. "
        "Use standard mathematical coordinates: (0,0) is at bottom-left, "
        "x increases rightward, y increases upward. "
        "All coordinates are positive integers. "
        "Output must be \\boxed{blue:x,y; green:x,y; red:x,y}."
    )
    
    # Create and return environment
    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs
    )
    
    return vf_env

