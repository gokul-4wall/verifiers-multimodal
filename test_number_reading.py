#!/usr/bin/env python3
"""Test if vLLM can read numbers from an image."""

import base64
from io import BytesIO
from PIL import Image, ImageDraw
import requests

# Create an image with a clear number
image = Image.new('RGB', (400, 200), color='white')
draw = ImageDraw.Draw(image)

# Draw a big number
draw.text((150, 70), "42069", fill='black')

# Encode as base64
buffered = BytesIO()
image.save(buffered, format="PNG")
img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

print("Testing if vLLM can read numbers from an image...")
print("Sending image with '42069' written on it...\n")

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
                    "text": "What number do you see in this image? Reply with only the number."
                }
            ]
        }],
        "max_tokens": 10,
        "temperature": 0.0
    },
    timeout=30
)

if response.status_code == 200:
    result = response.json()
    answer = result['choices'][0]['message']['content'].strip()
    print("="*80)
    print(f"Expected: 42069")
    print(f"Model's response: '{answer}'")
    print("="*80)
    
    if "42069" in answer:
        print("\n✅ PERFECT! Model read the exact number correctly!")
    else:
        print(f"\n⚠️  Model gave: '{answer}' instead of '42069'")
else:
    print(f"ERROR: {response.status_code} - {response.text}")

