"""
Adapter for Qwen3-VL-8B-Instruct model.

This adapter handles the conversion from verifiers GenerateOutputs to
MultimodalBatch suitable for Qwen3-VL training.
"""

from typing import List
import torch
from PIL import Image
import requests
from io import BytesIO
import base64

from transformers import PreTrainedTokenizerBase, Qwen2VLProcessor
from verifiers.rl.trainer.multimodal import MultimodalAdapter, MultimodalBatch
from verifiers.types import GenerateOutputs


class Qwen3VLAdapter(MultimodalAdapter):
    """
    Adapter for Qwen3-VL models.
    
    Qwen3-VL uses:
    - <|vision_start|> and <|vision_end|> tokens for image regions
    - <|image_pad|> tokens for image patches
    - image_grid_thw metadata for dynamic resolution
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase, processor: Qwen2VLProcessor, image_scale: float = 1.0):
        super().__init__(tokenizer)
        self.processor = processor
        self.image_scale = image_scale  # Scale factor for resizing images (0.5 = half size)
        
    def _load_image(self, image_data: str | Image.Image | dict | None) -> Image.Image:
        """
        Load image from URL, file path, base64 string, PIL Image, or dict.
        
        Args:
            image_data: URL, file path, base64 encoded image, PIL Image, or dict (HF datasets format)
            
        Returns:
            PIL Image
        """
        # If already a PIL Image, just ensure it's RGB
        if isinstance(image_data, Image.Image):
            image = image_data
        elif image_data is None or (isinstance(image_data, dict) and not image_data):
            # Handle None or empty dict - return a dummy image
            return Image.new('RGB', (224, 224), color='white')
        elif isinstance(image_data, dict):
            # Handle dict (HuggingFace datasets sometimes serialize PIL Images to dicts)
            # This happens when PIL Images are stored in the info column
            # Try to extract the image if it's in a known format
            if 'bytes' in image_data:
                # HF format with bytes
                image = Image.open(BytesIO(image_data['bytes']))
            elif 'path' in image_data:
                # HF format with path
                image = Image.open(image_data['path'])
            else:
                # Unknown dict format - log and return dummy
                print(f"Warning: Unknown image dict format with keys: {list(image_data.keys())}")
                return Image.new('RGB', (224, 224), color='white')
        elif isinstance(image_data, str):
            if image_data.startswith('http://') or image_data.startswith('https://'):
                # Load from URL
                response = requests.get(image_data)
                image = Image.open(BytesIO(response.content))
            elif image_data.startswith('data:image'):
                # Base64 encoded image
                image_data_split = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data_split)
                image = Image.open(BytesIO(image_bytes))
            else:
                # File path
                image = Image.open(image_data)
        else:
            # Unknown type, return dummy
            return Image.new('RGB', (224, 224), color='white')
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if scale factor is set
        if self.image_scale != 1.0:
            w, h = image.size
            new_w = max(1, int(w * self.image_scale))
            new_h = max(1, int(h * self.image_scale))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return image
    
    def _format_conversation(
        self, 
        prompt: List[dict], 
        completion: str | List[dict],
        image: Image.Image
    ) -> List[dict]:
        """
        Format conversation in Qwen3-VL's expected format.
        
        Args:
            prompt: List of messages (verifiers format)
            completion: Model completion
            image: PIL Image
            
        Returns:
            Formatted conversation with image
        """
        # Qwen3-VL expects messages with content that can be:
        # - Text: {"type": "text", "text": "..."}
        # - Image: {"type": "image", "image": <PIL.Image>}
        
        messages = []
        
        # Add system message if present
        for msg in prompt:
            if msg['role'] == 'system':
                messages.append({
                    'role': 'system',
                    'content': msg['content']
                })
        
        def _content_to_text(content: object) -> str:
            # OpenAI-style multimodal: content is a list of parts
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text")
                        if isinstance(t, str) and t:
                            parts.append(t)
                    elif isinstance(part, str) and part:
                        parts.append(part)
                return "\n".join(parts).strip()
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()

        def _completion_to_text(comp: str | List[dict]) -> str:
            if isinstance(comp, str):
                return comp
            # If completion is message list, take assistant content(s)
            texts: list[str] = []
            for msg in comp:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    texts.append(_content_to_text(msg.get("content")))
            return "\n".join([t for t in texts if t]).strip()

        # Add user message with image
        user_content: list[dict] = [
            {"type": "image", "image": image},
        ]

        # Add text from user messages (handle multimodal content parts)
        user_texts: list[str] = []
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_texts.append(_content_to_text(msg.get("content")))
        user_text = "\n".join([t for t in user_texts if t]).strip()
        if user_text:
            user_content.append({"type": "text", "text": user_text})
        
        messages.append({
            'role': 'user',
            'content': user_content
        })
        
        # Add assistant completion
        messages.append({
            'role': 'assistant',
            'content': _completion_to_text(completion)
        })
        
        return messages
    
    def _create_loss_mask(
        self,
        input_ids: torch.Tensor,
        assistant_start_positions: List[int]
    ) -> torch.Tensor:
        """
        Create loss mask that only trains on assistant responses.
        
        Args:
            input_ids: Token IDs (B, L)
            assistant_start_positions: Starting position of assistant tokens for each sequence
            
        Returns:
            Loss mask (B, L) with 1s for assistant tokens, 0s elsewhere
        """
        batch_size, seq_len = input_ids.shape
        loss_mask = torch.zeros_like(input_ids)
        
        for i, start_pos in enumerate(assistant_start_positions):
            if start_pos < seq_len:
                loss_mask[i, start_pos:] = 1
        
        return loss_mask
    
    def _extract_behavior_logprobs(
        self,
        outputs: GenerateOutputs,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract behavior (vLLM) logprobs and align with input_ids.
        
        Args:
            outputs: GenerateOutputs from vLLM
            input_ids: Token IDs (B, L)
            
        Returns:
            Behavior logprobs (B, L) padded with zeros for prompt tokens
        """
        batch_size, seq_len = input_ids.shape
        behavior_logprobs = torch.zeros(batch_size, seq_len)
        
        # Extract logprobs from vLLM outputs
        for i, completion_data in enumerate(outputs.completion):
            # vLLM returns logprobs per token in the completion
            # We need to align these with input_ids
            
            # For now, use dummy logprobs
            # In production, you'd extract from vLLM's logprobs field
            # and align with the completion tokens
            
            # TODO: Extract actual logprobs from vLLM response
            # This depends on how your vLLM is configured to return logprobs
            if completion_data and isinstance(completion_data, str):
                completion_tokens = len(self.tokenizer.encode(completion_data))
                if completion_tokens > 0:
                    # Fill in the completion portion with dummy values
                    # Replace with actual vLLM logprobs extraction
                    behavior_logprobs[i, -completion_tokens:] = torch.randn(completion_tokens)
        
        return behavior_logprobs
    
    def _extract_image_from_prompt(self, prompt: List[dict]) -> str | None:
        """
        Extract image URL/data from prompt content (for MMMU-style prompts).
        
        Args:
            prompt: List of message dicts with potential image content
            
        Returns:
            Image URL/data string or None
        """
        for msg in prompt:
            if isinstance(msg, dict) and 'content' in msg:
                content = msg['content']
                # Handle list of content parts (MMMU format)
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'image_url':
                            # Extract from {"type": "image_url", "image_url": {"url": "..."}}
                            image_url_obj = part.get('image_url', {})
                            if isinstance(image_url_obj, dict):
                                return image_url_obj.get('url')
                            return image_url_obj
        return None
    
    def build_batch(self, outputs: GenerateOutputs) -> MultimodalBatch:
        """
        Convert verifiers GenerateOutputs into MultimodalBatch for Qwen3-VL.
        
        Supports multiple image sources:
        1. state['image_url'] or state['image'] (direct state)
        2. Embedded in prompt content (MMMU-style)
        
        Args:
            outputs: GenerateOutputs containing prompts, completions, states, rewards
            
        Returns:
            MultimodalBatch ready for training
        """
        batch_size = len(outputs.prompt)
        
        # 1. Load images from states OR prompts
        images = []
        for i, state in enumerate(outputs.state):
            image_data = None
            
            # Try to get image from state first
            image_data = state.get('image_url') or state.get('image')
            
            # If not in state, try extracting from prompt content (MMMU format)
            if not image_data:
                prompt = outputs.prompt[i]
                if isinstance(prompt, list):
                    image_data = self._extract_image_from_prompt(prompt)
            
            if image_data:
                images.append(self._load_image(image_data))
            else:
                # Create dummy image if none provided
                images.append(Image.new('RGB', (224, 224), color='white'))
        
        # 2. Format conversations
        conversations = []
        for i in range(batch_size):
            conv = self._format_conversation(
                prompt=outputs.prompt[i] if isinstance(outputs.prompt[i], list) else [{'role': 'user', 'content': outputs.prompt[i]}],
                completion=outputs.completion[i],
                image=images[i]
            )
            conversations.append(conv)
        
        # 3. Process with Qwen3-VL processor
        # This handles tokenization + image processing in one go
        processed = self.processor(
            text=[self.processor.apply_chat_template(conv, tokenize=False) 
                  for conv in conversations],
            images=images,
            padding=True,
            return_tensors='pt'
        )
        
        input_ids = processed['input_ids']
        attention_mask = processed['attention_mask']
        pixel_values = processed['pixel_values']
        image_grid_thw = processed.get('image_grid_thw')
        
        # 4. Create loss mask (train only on assistant responses)
        # Find where assistant tokens start
        assistant_token_id = self.tokenizer.encode('assistant', add_special_tokens=False)[0]
        assistant_start_positions = []
        
        for i in range(batch_size):
            # Find first occurrence of assistant role token
            assistant_positions = (input_ids[i] == assistant_token_id).nonzero(as_tuple=True)[0]
            if len(assistant_positions) > 0:
                # Start training after the assistant marker
                assistant_start_positions.append(assistant_positions[-1].item() + 1)
            else:
                # If not found, train on last half (fallback)
                assistant_start_positions.append(input_ids.shape[1] // 2)
        
        loss_mask = self._create_loss_mask(input_ids, assistant_start_positions)
        
        # 5. Extract behavior logprobs from vLLM
        behavior_logprobs = self._extract_behavior_logprobs(outputs, input_ids)
        
        # 6. Aggregate rewards
        rewards = torch.tensor([
            r['reward'] if isinstance(r, dict) else r 
            for r in outputs.reward
        ], dtype=torch.float32)
        
        # 7. Build extra model kwargs for Qwen3-VL
        extra_model_kwargs = {}
        if image_grid_thw is not None:
            extra_model_kwargs['image_grid_thw'] = image_grid_thw
        
        return MultimodalBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            behavior_logprobs=behavior_logprobs,
            rewards=rewards,
            pixel_values=pixel_values,
            extra_model_kwargs=extra_model_kwargs,
        )

