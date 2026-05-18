"""
Call HuggingFace Llama-3.2-3B using PyRIT's OpenAIChatTarget.

This version uses PyRIT's infrastructure for better integration.

Usage:
    python call_llama_3_2_3b_pyrit.py "Your prompt here"
"""

import asyncio
import os
import sys
import httpx
from dotenv import load_dotenv
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.models import PromptRequestPiece, PromptRequestResponse

load_dotenv()


async def call_llama_3_2_3b_pyrit(prompt: str):
    """
    Call HuggingFace Llama-3.2-3B using PyRIT's OpenAIChatTarget.
    
    Args:
        prompt: The prompt to send to the model
        
    Returns:
        The model's response text
    """
    # Initialize PyRIT
    initialize_pyrit(memory_db_type=IN_MEMORY)
    
    # Get API key
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not api_key:
        raise ValueError(
            "HF_TOKEN or HUGGINGFACE_TOKEN environment variable not set. "
            "Please set it in your .env file or environment."
        )
    
    # Create the target with extended timeout
    target = OpenAIChatTarget(
        endpoint="https://router.huggingface.co/v1/chat/completions",
        api_key=api_key,
        model_name="meta-llama/Llama-3.2-3B-Instruct:novita",
        api_version=None,  # HF Router doesn't use Azure's api-version parameter
        httpx_client_kwargs={
            "timeout": httpx.Timeout(connect=300.0, read=3600.0, write=30.0, pool=300.0)
        },
    )
    
    print(f"🤖 Calling Llama-3.2-3B model via PyRIT...")
    print(f"📝 Prompt: {prompt}")
    print(f"⏳ This may take a while (timeout set to 60 minutes)...\n")
    
    try:
        # Create a prompt request
        prompt_request = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    original_value=prompt,
                    converted_value=prompt,
                )
            ]
        )
        
        # Send the prompt
        response = await target.send_prompt_async(prompt_request=prompt_request)
        
        # Extract the response text
        if response.response_pieces and len(response.response_pieces) > 0:
            return response.response_pieces[0].converted_value
        else:
            return "No response received"
            
    except Exception as e:
        print(f"❌ Error calling model: {e}")
        raise


async def main():
    """Main function to handle command-line arguments or interactive mode."""
    if len(sys.argv) > 1:
        # Use command-line argument
        prompt = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        print("=" * 60)
        print("HuggingFace Llama-3.2-3B via PyRIT")
        print("=" * 60)
        print("\nEnter your prompt (or 'quit' to exit):")
        prompt = input("> ").strip()
        
        if not prompt or prompt.lower() == "quit":
            print("Goodbye!")
            return
    
    try:
        response = await call_llama_3_2_3b_pyrit(prompt)
        print("\n" + "=" * 60)
        print("✅ Response from Llama-3.2-3B:")
        print("=" * 60)
        print(response)
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Failed to get response: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())



