"""
Dummy multimodal environment for testing the multimodal trainer.

Uses SingleTurnEnv base class (same as text-only environments).
Simple image description task: describe colored images, get rewarded for mentioning the color.
"""

import random
from typing import Any

from datasets import Dataset
from PIL import Image, ImageDraw

from verifiers import Rubric, SingleTurnEnv


def _create_colored_image(color_name: str, size: int = 224) -> Image.Image:
    """Create a simple colored square image."""
    colors = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
    }
    
    rgb = colors.get(color_name, (128, 128, 128))
    image = Image.new('RGB', (size, size), color=rgb)
    
    # Add a simple shape
    draw = ImageDraw.Draw(image)
    draw.rectangle([size//4, size//4, 3*size//4, 3*size//4], 
                   outline=(255, 255, 255), width=3)
    
    return image


class DummyMultimodalEnv(SingleTurnEnv):
    """
    Simple test environment: describe the color of an image.
    
    Inherits from SingleTurnEnv (same base as text-only environments).
    Uses Rubric for reward functions (standard verifiers pattern).
    """
    
    def __init__(self, num_examples: int = 10, **kwargs):
        self.num_examples = num_examples
        self.colors = ["red", "blue", "green", "yellow"]
        
        # Create dataset
        dataset = self._create_dataset()
        
        # Create rubric with reward function (standard pattern)
        rubric = Rubric()
        
        def color_reward_func(completion, state, **kwargs) -> float:
            """Reward if correct color is mentioned."""
            correct_color = state["color"].lower()
            completion_lower = completion.lower()
            return 1.0 if correct_color in completion_lower else 0.0
        
        rubric.add_reward_func(color_reward_func)
        
        # Initialize parent (same as text-only envs)
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            message_type="chat",
            **kwargs
        )
    
    def _create_dataset(self) -> Dataset:
        """Create dataset with colored images."""
        data = {
            "color": [],
            "image": [],
            "question": [],
        }
        
        for i in range(self.num_examples):
            color = random.choice(self.colors)
            image = _create_colored_image(color)
            
            data["color"].append(color)
            data["image"].append(image)
            data["question"].append("What color is in this image? Describe it.")
        
        return Dataset.from_dict(data)
    
    def get_prompt(self, row: dict[str, Any]) -> list[dict[str, str]]:
        """Format prompt as messages (standard verifiers pattern)."""
        return [
            {
                "role": "user",
                "content": row["question"]
            }
        ]
    
    def get_state(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return state with image (standard verifiers pattern)."""
        return {
            "color": row["color"],
            "image": row["image"],  # PIL Image
        }


def load_environment(**kwargs):
    """Load the dummy multimodal environment."""
    return DummyMultimodalEnv(**kwargs)

