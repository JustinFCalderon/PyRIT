import os
import sys
import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi, InferenceClient

from pathlib import Path
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)
print(f"Loaded .env from: {env_path} (exists={env_path.exists()})")

# Ensure terminal handles special characters
sys.stdout.reconfigure(encoding='utf-8')
# load_dotenv()

def check_huggingface_connection(api):
    print("\n--- CHECKING HUGGING FACE CONNECTION ---")
    try:
        user_info = api.whoami()
        print(f"✅ [SUCCESS] Authenticated as user: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ [FAILURE] Authentication failed. Check your HF_TOKEN in .env.")
        print(f"   Error: {e}")
        return False

def check_token_permissions(api):
    try:
        info = api.whoami()
        # Retrieve all authorized scopes from the token
        scopes = info.get("auth", {}).get("accessToken", {}).get("scopes", [])
        
        # Check for any variation of the required permissions
        has_read = any("read" in s.lower() for s in scopes)
        has_inference = any("inference" in s.lower() for s in scopes)
        
        if has_read and has_inference:
            print("✅ [SUCCESS] Token has sufficient scopes.")
        else:
            missing = []
            if not has_read: missing.append("Read (Gated Repos)")
            if not has_inference: missing.append("Inference Providers")
            print("ℹ️  Token scopes not introspectable (fine-grained token).")

    except Exception as e:
        print(f"⚠️ [WARNING] Could not read token scopes via whoami(): {e}")
        print("    This is common. Rely on the router chat test + model_info checks above.")


def check_model_access(api, model_id):
    # Strip provider suffix (e.g., :nebius) for repo validation
    clean_id = model_id.split(":")[0] 
    print(f"  > Checking access for: {clean_id}...", end="", flush=True)
    try:
        api.model_info(repo_id=clean_id)
        print(" [SUCCESS] Access verified.")
        return True
    except Exception as e:
        print(f" [FAILURE] Error: {e}")
        return False

def check_inference_status(token, model_id):
    clean_id = model_id.split(":")[0]
    print(f"  > Testing inference for: {clean_id}...", end="", flush=True)
    try:
        # Use provider="auto" to bypass specific 404s
        client = InferenceClient(api_key=token, provider="auto")
        
        # FIX: Call chat.completions (Conversational task) instead of text_generation
        client.chat.completions.create(
            model=clean_id,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1
        )
        print(" [SUCCESS] Inference live.")
    except Exception as e:
        print(f" [FAILURE] Error: {e}")

def derive_victim_nickname(victim_id: str) -> str:
    """
    Convert VICTIM_MODEL (HF repo id or ollama tag) -> nickname used by ollama_tag_map.
    Examples:
      "meta-llama/Llama-3.2-3B-Instruct" -> "llama-3.2-3b"
      "meta-llama/Llama-3.2-1B-Instruct:novita" -> "llama-3.2-1b"
      "llama3.2:3b" -> "llama-3.2-3b"
    """
    s = (victim_id or "").strip().lower()

    # Remove provider suffix like ":novita"
    base = s.split(":", 1)[0]

    # Handle ollama tags
    if base.startswith("llama3.2:"):
        size = base.split(":", 1)[1]
        if size == "1b":
            return "llama-3.2-1b"
        if size == "3b":
            return "llama-3.2-3b"

    # Handle HF repo ids
    if "llama-3.2-1b" in base or "llama-3.2-1b-instruct" in base:
        return "llama-3.2-1b"
    if "llama-3.2-3b" in base or "llama-3.2-3b-instruct" in base:
        return "llama-3.2-3b"
    if "llama-3.1-8b" in base or "llama-3.1-8b-instruct" in base:
        return "llama-3.1-8b"

    # Safe default
    return "llama-3.2-1b"

def check_ollama_local():
    print("\n--- CHECKING OLLAMA (Local) ---")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ [SUCCESS] Ollama server is running and reachable.")
            return True
        else:
            print(f"⚠️ [WARNING] Ollama server responded with status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ [FAILURE] Could not connect to Ollama. Run 'ollama serve'.")
    return False

def check_router_chat(token, model_id):
    url = "https://router.huggingface.co/v1/chat/completions"
    clean_id = model_id.split(":")[0]
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": clean_id,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        },
        timeout=20,
    )
    print(f"  > Router chat test: HTTP {r.status_code}")
    if r.status_code != 200:
        print("  > Response:", r.text[:400])


def check_ollama_model(model_tag):
    # Verify the specific tag pulled in run_batch.py is ready
    print(f"\n--- CHECKING OLLAMA MODEL: {model_tag} ---")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            if model_tag in models:
                print(f"✅ [SUCCESS] {model_tag} is downloaded and ready.")
            else:
                print(f"❌ [FAILURE] {model_tag} not found. Pull it locally first.")
    except Exception as e:
        print(f"❌ [FAILURE] Connection error: {e}")

def main():
    # ------------------------------------------------------------------
    # 1) Load token
    # ------------------------------------------------------------------
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("❌ [FAILURE] Neither HF_TOKEN nor HUGGINGFACE_TOKEN found in .env.")
        return

    api = HfApi(token=token)

    # ------------------------------------------------------------------
    # 2) Hugging Face authentication (ONCE)
    # ------------------------------------------------------------------
    print("\n--- CHECKING HUGGING FACE CONNECTION ---")
    if check_huggingface_connection(api):
        check_token_permissions(api)

    # ------------------------------------------------------------------
    # 3) Load model config from .env
    # ------------------------------------------------------------------
    attacker = os.getenv("ATTACKER_MODEL", "google/gemma-3-27b-it")
    victim_id = os.getenv("VICTIM_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    judge = os.getenv("JUDGE_MODEL", "google/gemma-3-27b-it")

    victim_nickname = derive_victim_nickname(victim_id)

    ollama_tag_map = {
        "llama-3.2-1b": "llama3.2:1b",
        "llama-3.2-3b": "llama3.2:3b",
        "llama-3.1-8b": "llama3.1:8b",
    }
    victim_tag = ollama_tag_map.get(victim_nickname, "llama3.2:1b")

    # ------------------------------------------------------------------
    # 4) Ollama checks (ONCE)
    # ------------------------------------------------------------------
    print("\n--- LOCAL (OLLAMA) CHECKS ---")
    print(f"Victim model (from .env): {victim_id}")
    print(f"Derived victim nickname:  {victim_nickname}")
    print(f"Expected Ollama tag:      {victim_tag}")

    if check_ollama_local():
        check_ollama_model(victim_tag)

    # ------------------------------------------------------------------
    # 5) Cloud validation (ONCE per model)
    # ------------------------------------------------------------------
    print("\n--- VALIDATING EXPERIMENT MODELS (CLOUD) ---")

    models = {
        "Attacker": attacker,
        "Victim": victim_id,
        "Judge": judge,
    }

    for role, model in models.items():
        print(f"\n[{role}] {model}")
        if check_model_access(api, model):
            check_router_chat(token, model)
            check_inference_status(token, model)


if __name__ == "__main__":
    main()