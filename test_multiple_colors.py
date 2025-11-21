#!/usr/bin/env python3
"""Test vLLM with multiple clearly different colored images."""

import base64
from io import BytesIO
from PIL import Image
import requests

def test_color(color_name, rgb_tuple):
    """Test vLLM with a specific color."""
    # Create solid color image
    image = Image.new('RGB', (224, 224), color=rgb_tuple)
    
    # Encode as base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
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
                        "text": "What is the main color of this solid colored square? Answer with ONE word only - the color name."
                    }
                ]
            }],
            "max_tokens": 5,
            "temperature": 0.0
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        answer = result['choices'][0]['message']['content'].strip()
        return answer
    else:
        return f"ERROR: {response.status_code}"

# Test multiple colors
colors_to_test = [
    ("RED", (255, 0, 0)),
    ("GREEN", (0, 255, 0)),
    ("BLUE", (0, 0, 255)),
    ("WHITE", (255, 255, 255)),
    ("BLACK", (0, 0, 0)),
]

print("Testing vLLM with different colored images...")
print("="*80)

results = []
for color_name, rgb in colors_to_test:
    print(f"Testing {color_name} {rgb}...", end=" ")
    answer = test_color(color_name, rgb)
    results.append((color_name, answer))
    print(f"Model says: '{answer}'")

print("="*80)
print("\nRESULTS:")
print("="*80)

correct = 0
for expected, actual in results:
    match = "✅" if expected.lower() in actual.lower() else "❌"
    print(f"{match} Expected: {expected:10s} | Got: {actual}")
    if expected.lower() in actual.lower():
        correct += 1

print("="*80)
print(f"\nAccuracy: {correct}/{len(results)} ({100*correct//len(results)}%)")

if correct == 0:
    print("\n🚨 CRITICAL: Model got 0 correct. The image is likely NOT being processed!")
    print("   The model may be ignoring the image and just guessing.")
elif correct < len(results):
    print(f"\n⚠️  WARNING: Only {correct}/{len(results)} correct. Image processing may be unreliable.")
else:
    print("\n🎉 SUCCESS: All colors identified correctly!")

