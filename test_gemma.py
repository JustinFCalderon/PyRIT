# test_gemma.py
# Tests Gemma via OpenAIChatTarget exactly as 3_crescendo_attack.py uses it
import asyncio
import os
import httpx
from dotenv import load_dotenv
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.models import PromptRequestPiece, PromptRequestResponse

load_dotenv()

async def main():
    initialize_pyrit(memory_db_type=IN_MEMORY)

    print("Creating OpenAIChatTarget for Gemma...")
    target = OpenAIChatTarget(
        endpoint="https://router.huggingface.co/v1/chat/completions",
        api_key=os.environ.get("HF_TOKEN"),
        model_name="google/gemma-3-27b-it",
        api_version=None,
        httpx_client_kwargs={
            "timeout": httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
        }
    )
    print("Target created. Sending test prompt...")

    request_piece = PromptRequestPiece(
        role="user",
        original_value="Reply with one word: ready",
    )
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])

    print("Waiting for response (timeout: 120s)...")
    response = await target.send_prompt_async(prompt_request=prompt_request)
    print(f"✅ Response: {response.request_pieces[0].converted_value}")

if __name__ == "__main__":
    asyncio.run(main())