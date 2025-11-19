"""
Multi-turn multimodal environment for testing the multimodal trainer.

Simple color guessing game: model gets multiple turns to identify the color in an image.
After each guess, it receives feedback and can try again.
"""

import base64
import logging
import random
from io import BytesIO
from typing import Any

from datasets import Dataset
from PIL import Image, ImageDraw

from verifiers import MultiTurnEnv, Rubric

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _create_colored_image(color_name: str, size: int = 224) -> Image.Image:
    """Create a simple colored square image (solid color, no distractions)."""
    colors = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
        "orange": (255, 165, 0),
        "crimson": (220, 20, 60),
        "navy": (0, 0, 128),
        "teal": (0, 128, 128),
        "mustard": (255, 219, 88),
        "coral": (255, 127, 80),
        "violet": (238, 130, 238),
    }
    
    rgb = colors.get(color_name.lower(), (128, 128, 128))
    # Just solid color - no borders or shapes that distract the model
    image = Image.new('RGB', (size, size), color=rgb)
    
    return image


class DummyMultimodalMultiTurnEnv(MultiTurnEnv):
    """
    Multi-turn color guessing game.
    
    The model sees a colored image and has up to 3 turns to guess the color.
    After each guess, it receives feedback and can try again.
    """
    
    def __init__(self, num_examples: int = 10, max_turns: int = 3, **kwargs):
        self.num_examples = num_examples
        # Use simple colors that change each turn
        self.colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        
        # Create dataset
        dataset = self._create_dataset()
        
        # Create rubric with reward functions
        rubric = Rubric()
        
        def correct_guess_reward(state: dict, **kwargs) -> float:
            """Reward for guessing correctly."""
            reward = 1.0 if state.get("guessed_correctly", False) else 0.0
            return reward
        
        def efficiency_reward(state: dict, **kwargs) -> float:
            """Bonus for guessing quickly (fewer turns)."""
            if not state.get("guessed_correctly", False):
                return 0.0
            # Bonus decreases with more turns: 1.0, 0.5, 0.0 for turns 1, 2, 3
            turn = state.get("turn", 1)
            bonus = max(0.0, 1.0 - (turn - 1) * 0.5)
            return bonus
        
        rubric.add_reward_func(correct_guess_reward)
        rubric.add_reward_func(efficiency_reward)
        
        # Initialize parent (pass max_turns to parent)
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            message_type="chat",
            question_key="question",
            answer_key="answer",
            max_turns=max_turns,
            **kwargs
        )
    
    def format_dataset(
        self,
        dataset: Dataset,
        system_prompt: str | None = None,
        few_shot: list[dict] | None = None,
        question_key: str = "question",
        answer_key: str = "answer",
        map_kwargs: dict = {},
    ) -> Dataset:
        """Override to use get_prompt() for multimodal prompts."""
        # Add example_id if not present
        if "example_id" not in dataset.column_names:
            dataset = dataset.add_column("example_id", range(len(dataset)))
        
        # Add prompt column using get_prompt() to create multimodal messages
        if "prompt" not in dataset.column_names:
            dataset = dataset.map(
                lambda row: {"prompt": self.get_prompt(row)},
                **map_kwargs
            )
        
        return dataset
    
    def _create_dataset(self) -> Dataset:
        """Create dataset with colored images."""
        data = {
            "color": [],
            "image": [],
            "question": [],
            "answer": [],
            "info": [],  # Store color/image in info dict (preserved through rollout)
        }
        
        for i in range(self.num_examples):
            color = random.choice(self.colors)
            image = _create_colored_image(color)
            
            data["color"].append(color)
            data["image"].append(image)
            data["question"].append("What color is in this image?")
            data["answer"].append(color)
            # Store color/image in info dict so it's available during rollout
            data["info"].append({
                "color": color,
                "image": image,
            })
        
        return Dataset.from_dict(data)
    
    def get_prompt(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        """Initial prompt for the task with image."""
        # vLLM expects nested dict format for image_url
        image = row["image"]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "You will see a sequence of SOLID colored squares (like video game frames). Just remember what colors you see. First color:"
                    }
                ]
            }
        ]
        
        # LOG EXACT PAYLOAD BEING SENT TO VLLM
        print("\n" + "="*80)
        print("📤 SENDING TO VLLM:")
        print("="*80)
        print(f"Message structure: {type(prompt)}")
        print(f"Number of messages: {len(prompt)}")
        print(f"First message role: {prompt[0]['role']}")
        print(f"Content type: {type(prompt[0]['content'])}")
        print(f"Content parts: {len(prompt[0]['content'])}")
        for i, part in enumerate(prompt[0]['content']):
            print(f"  Part {i}: type={part.get('type')}")
            if part['type'] == 'image_url':
                print(f"    - image_url is dict: {isinstance(part['image_url'], dict)}")
                print(f"    - has 'url' key: {'url' in part['image_url']}")
                print(f"    - url starts with: {part['image_url']['url'][:50]}...")
                print(f"    - base64 length: {len(img_base64)}")
            elif part['type'] == 'text':
                print(f"    - text: {part['text'][:80]}")
        print("="*80 + "\n")
        
        return prompt
    
    def get_state(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return initial state with image and correct color."""
        return {
            "color": row["color"],
            "image": row["image"],
            "guessed_correctly": False,
            "guesses": [],
        }
    
    async def setup_state(self, state: dict, **kwargs) -> dict:
        """Setup state - extract color/image from info dict."""
        # Extract color and image from info dict (populated from dataset)
        info = state.get("info", {})
        state["initial_color"] = info.get("color", "")
        state["initial_image"] = info.get("image", None)
        
        # Initialize tracking fields
        if "guessed_correctly" not in state:
            state["guessed_correctly"] = False
        if "guesses" not in state:
            state["guesses"] = []
        if "color_sequence" not in state:
            # Start with the initial color
            state["color_sequence"] = [info.get("color", "")]
        
        return state
    
    async def is_completed(self, messages: list[dict], state: dict, **kwargs) -> bool:
        """Task completes when correct guess or max turns reached."""
        if state.get("guessed_correctly", False):
            # Log completion
            print("\n" + "="*80)
            print("MULTI-TURN ROLLOUT COMPLETE")
            print("="*80)
            print(f"Ground truth color sequence: {state.get('color_sequence', [])}")
            print(f"Total turns: {state.get('turn', 0)}")
            print(f"Final guess: {state.get('guesses', [])[-1] if state.get('guesses') else 'None'}")
            print(f"Success: ✓")
            print("="*80 + "\n")
            return True
        # Check if max turns reached using parent method
        max_turns_done = await self.max_turns_reached(state)
        if max_turns_done:
            # Log failure
            print("\n" + "="*80)
            print("MULTI-TURN ROLLOUT COMPLETE")
            print("="*80)
            print(f"Ground truth color sequence: {state.get('color_sequence', [])}")
            print(f"Total turns: {state.get('turn', 0)}")
            print(f"Final guess: {state.get('guesses', [])[-1] if state.get('guesses') else 'None'}")
            print(f"Success: ✗ (ran out of turns)")
            print("="*80 + "\n")
        return max_turns_done
    
    async def env_response(self, messages: list[dict], state: dict, **kwargs) -> tuple[list[dict], dict]:
        """Provide feedback after each guess AND send a new colored image (like a game frame)."""
        turn = state.get("turn", 1)
        
        # Generate a NEW color for this turn (simulate changing game state)
        import random
        new_color = random.choice(self.colors)
        new_image = _create_colored_image(new_color)
        
        # Add to color sequence
        if "color_sequence" not in state:
            state["color_sequence"] = [state.get("initial_color", "")]
        state["color_sequence"].append(new_color)
        
        # Get the last message (model's response)
        if messages and len(messages) > 0:
            last_message = messages[-1]
            if isinstance(last_message, dict):
                guess_text = last_message.get("content", "")
            else:
                guess_text = str(last_message)
            
            # Store the guess
            if "guesses" not in state:
                state["guesses"] = []
            state["guesses"].append(guess_text.lower().strip())
            
            # Log the turn
            print(f"\n--- Turn {turn} ---")
            print(f"Model response: {guess_text[:200]}...")
            print(f"Colors shown so far: {state['color_sequence']}")
        
        # Check if this is the last turn
        remaining = self.max_turns - turn
        
        # Encode new image as base64
        buffered = BytesIO()
        new_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        if remaining > 0:
            # Not the last turn - show next frame and ask to continue
            feedback_text = f"Turn {turn} complete. Here's the next frame (turn {turn + 1}). Continue tracking all colors."
            print(f"Sending new frame: {new_color}")
        else:
            # Last turn - ask for final answer
            feedback_text = f"Final frame! Now tell me ALL the colors you saw in order from the beginning."
            print(f"Final frame: {new_color}. Asking for complete sequence.")
            # Check if the guess contains all colors
            if messages and len(messages) > 0:
                guess_lower = guess_text.lower()
                all_correct = all(color.lower() in guess_lower for color in state["color_sequence"])
                if all_correct:
                    state["guessed_correctly"] = True
                    print(f"Result: ✓ Model correctly identified all colors!")
        
        # Return multimodal message with NEW IMAGE + text feedback
        env_message = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": feedback_text
                }
            ]
        }]
        
        return env_message, state


def load_environment(**kwargs):
    """Load the multi-turn multimodal environment."""
    return DummyMultimodalMultiTurnEnv(**kwargs)

