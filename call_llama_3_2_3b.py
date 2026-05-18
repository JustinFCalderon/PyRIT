"""
Simple script to directly call HuggingFace Llama-3.2-3B model.

Usage:
    python call_llama_3_2_3b.py "Your prompt here"

Or run interactively:
    python call_llama_3_2_3b.py
"""

import asyncio
import os
import sys
import httpx
from dotenv import load_dotenv

load_dotenv()


async def call_llama_3_2_3b(prompt: str):
    """
    Directly call HuggingFace Llama-3.2-3B model via Router API.
    
    Args:
        prompt: The prompt to send to the model
        
    Returns:
        The model's response text
    """
    # Configuration
    endpoint = "https://router.huggingface.co/v1/chat/completions"
    model_id = "meta-llama/Llama-3.2-3B-Instruct:novita"
    
    # Get API key from environment
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not api_key:
        raise ValueError(
            "HF_TOKEN or HUGGINGFACE_TOKEN environment variable not set. "
            "Please set it in your .env file or environment."
        )
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Prepare request body (OpenAI-compatible format)
    request_body = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }
    
    # Set longer timeout for HuggingFace models (60 minutes)
    timeout = httpx.Timeout(connect=300.0, read=3600.0, write=30.0, pool=300.0)
    
    print(f"[*] Calling Llama-3.2-3B model...")
    print(f"[*] Prompt: {prompt}")
    print(f"[*] This may take a while (timeout set to 60 minutes)...\n")
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                headers=headers,
                json=request_body,
            )
            
            # Check for errors
            if response.is_error:
                print(f"[ERROR] HTTP {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Extract the message content
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0]["message"]["content"]
                return message
            else:
                print(f"[WARNING] Unexpected response format: {result}")
                return str(result)
                
    except httpx.TimeoutException:
        print("[ERROR] Request timed out. The model may be taking longer than expected.")
        raise
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] HTTP Error: {e}")
        print(f"Response: {e.response.text}")
        raise
    except Exception as e:
        print(f"[ERROR] Error calling model: {e}")
        raise


async def main():
    """Main function to handle command-line arguments or interactive mode."""
    if len(sys.argv) > 1:
        # Use command-line argument
        prompt = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        print("=" * 60)
        print("HuggingFace Llama-3.2-3B Direct Call")
        print("=" * 60)
        print("\nEnter your prompt (or 'quit' to exit):")
        prompt = input("> ").strip()
        
        if not prompt or prompt.lower() == "quit":
            print("Goodbye!")
            return
    
    try:
        response = await call_llama_3_2_3b(prompt)
        print("\n" + "=" * 60)
        print("[SUCCESS] Response from Llama-3.2-3B:")
        print("=" * 60)
        print(response)
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Failed to get response: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

