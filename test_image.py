#!/usr/bin/env python3
"""Test script to verify vLLM server can process images correctly."""

import base64
from io import BytesIO
from PIL import Image
import requests

def test_vllm_image():
    # Create a bright red square image
    print("Creating a red test image...")
    image = Image.new('RGB', (224, 224), color='red')
    
    # Encode as base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    print(f"Image encoded to base64 (length: {len(img_base64)} chars)")
    print("\nSending request to vLLM server...")
    
    # Send request to vLLM in the exact format used by your environments
    try:
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
                            "text": "What color is this image? Reply with just one word."
                        }
                    ]
                }],
                "max_tokens": 10,
                "temperature": 0.0
            },
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            print("\n" + "="*80)
            print("✅ SUCCESS! Model's response:")
            print("="*80)
            print(f"  {answer}")
            print("="*80)
            
            if 'red' in answer.lower():
                print("\n🎉 PERFECT! The model correctly identified the red color!")
            else:
                print("\n⚠️  WARNING: Expected 'red' but got different answer.")
                print("   This might indicate the image isn't being processed correctly.")
        else:
            print("\n❌ ERROR:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to vLLM server at http://localhost:8000")
        print("   Make sure the vLLM server is running first!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    test_vllm_image()

