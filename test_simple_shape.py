#!/usr/bin/env python3
"""Test with a simple recognizable shape + text."""

import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import requests

# Create an image with clear text
image = Image.new('RGB', (400, 200), color='white')
draw = ImageDraw.Draw(image)

# Draw big text
draw.text((50, 70), "HELLO", fill='black')

# Encode as base64
buffered = BytesIO()
image.save(buffered, format="PNG")
img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

print("Testing if vLLM can read text from an image...")
print("Sending image with 'HELLO' written on it...\n")

# Send request
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "messages": [{
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
                    "text": "What text do you see in this image? Reply with only the text you see."
                }
            ]
        }],
        "max_tokens": 20,
        "temperature": 0.0
    },
    timeout=30
)

if response.status_code == 200:
    result = response.json()
    answer = result['choices'][0]['message']['content'].strip()
    print("="*80)
    print(f"Model's response: '{answer}'")
    print("="*80)
    
    if "HELLO" in answer.upper():
        print("\n✅ SUCCESS! The model can read text from images!")
        print("   This confirms images ARE being processed correctly.")
    else:
        print("\n⚠️  Model didn't see 'HELLO'. Response doesn't match expected text.")
else:
    print(f"ERROR: {response.status_code} - {response.text}")

