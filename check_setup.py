import os
import sys
import json
import requests
import importlib.util
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, InferenceClient

# ── Load .env ─────────────────────────────────────────────────────────────
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)
print(f"Loaded .env from: {env_path} (exists={env_path.exists()})")

sys.stdout.reconfigure(encoding='utf-8')

# ── Load config from run_batch.py ─────────────────────────────────────────
# Single source of truth — no hardcoded models here
def load_run_batch_config():
    run_batch_path = Path(__file__).resolve().parent / "run_batch.py"
    try:
        spec = importlib.util.spec_from_file_location("run_batch", run_batch_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {
            'VICTIM_MODEL':  getattr(mod, 'VICTIM_MODEL',  None),
            'ATTACK_MODEL':  getattr(mod, 'ATTACK_MODEL',  None),
            'MODE':          getattr(mod, 'MODE',          None),
            'active_runs':   getattr(mod, 'active_runs',   None),
            'active_objectives': getattr(mod, 'active_objectives', []),
        }
    except Exception as e:
        print(f"⚠️  Could not import run_batch.py: {e}")
        print("    Falling back to .env for model config")
        return {
            'VICTIM_MODEL':  os.getenv("VICTIM_MODEL", "llama-3.1-8b-local"),
            'ATTACK_MODEL':  os.getenv("ATTACKER_MODEL", "gemma-3-27b"),
            'MODE':          None,
            'active_runs':   None,
            'active_objectives': [],
        }

# ── Load MODEL_CONFIGS from 3_crescendo_attack.py ────────────────────────
def load_model_configs():
    script_path = Path(__file__).resolve().parent / \
        "doc/code/executor/attack/3_crescendo_attack.py"
    try:
        spec = importlib.util.spec_from_file_location("crescendo", script_path)
        mod  = importlib.util.module_from_spec(spec)
        # Prevent the script from running its __main__ block
        mod.__spec__.name = "crescendo"
        spec.loader.exec_module(mod)
        return getattr(mod, 'MODEL_CONFIGS', {})
    except Exception as e:
        print(f"⚠️  Could not load MODEL_CONFIGS from 3_crescendo_attack.py: {e}")
        return {}

# ── Check functions ───────────────────────────────────────────────────────

def check_huggingface_connection(api):
    print("\n--- CHECKING HUGGING FACE CONNECTION ---")
    try:
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Authentication failed. Check HF_TOKEN in .env")
        print(f"   Error: {e}")
        return False

def check_token_permissions(api):
    try:
        info   = api.whoami()
        scopes = info.get("auth", {}).get("accessToken", {}).get("scopes", [])
        has_read      = any("read"      in s.lower() for s in scopes)
        has_inference = any("inference" in s.lower() for s in scopes)
        if has_read and has_inference:
            print("✅ Token has sufficient scopes")
        else:
            print("ℹ️  Token scopes not introspectable (fine-grained token) — OK")
    except Exception as e:
        print(f"⚠️  Could not read token scopes: {e}")
        print("    Rely on router chat test below")

def check_model_access(api, model_id):
    clean_id = model_id.split(":")[0]
    print(f"  > Checking repo access: {clean_id}...", end="", flush=True)
    try:
        api.model_info(repo_id=clean_id)
        print(" ✅ Access verified")
        return True
    except Exception as e:
        print(f" ❌ {e}")
        return False

def check_router_chat(token, model_id):
    clean_id = model_id.split(":")[0]
    print(f"  > Router chat test: {clean_id}...", end="", flush=True)
    try:
        r = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": model_id,   # keep provider suffix (e.g. :novita)
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=20,
        )
        if r.status_code == 200:
            print(f" ✅ HTTP {r.status_code}")
        else:
            print(f" ❌ HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f" ❌ {e}")

def check_inference_status(token, model_id):
    clean_id = model_id.split(":")[0]
    print(f"  > Inference client test: {clean_id}...", end="", flush=True)
    try:
        client = InferenceClient(api_key=token, provider="auto")
        client.chat.completions.create(
            model=clean_id,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1,
        )
        print(" ✅ Inference live")
    except Exception as e:
        print(f" ❌ {e}")

def check_ollama_server():
    print("\n--- CHECKING OLLAMA SERVER ---")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            print(f"✅ Ollama running  |  models: {', '.join(models) or 'none'}")
            return True, models
        else:
            print(f"⚠️  Ollama responded with status {r.status_code}")
            return False, []
    except requests.exceptions.ConnectionError:
        print("❌ Ollama not reachable — run: ollama serve")
        return False, []

def check_ollama_model(ollama_model_id, available_models):
    print(f"  > Model '{ollama_model_id}'...", end="", flush=True)
    if any(ollama_model_id in m for m in available_models):
        print(" ✅ Available")
        return True
    else:
        print(f" ❌ Not found — run: ollama pull {ollama_model_id}")
        return False

def check_ollama_inference(ollama_model_id):
    print(f"  > Quick inference test...", end="", flush=True)
    try:
        r = requests.post(
            "http://localhost:11434/v1/chat/completions",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer none"},
            json={
                "model": ollama_model_id,
                "messages": [{"role": "user", "content": "Reply with one word: ready"}],
                "max_tokens": 5,
            },
            timeout=30,
        )
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"].strip()
            print(f" ✅ Response: '{text}'")
        else:
            print(f" ❌ HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f" ❌ {e}")

def check_scripts_exist():
    print("\n--- CHECKING SCRIPTS ---")
    scripts = [
        ("3_crescendo_attack.py", "doc/code/executor/attack/3_crescendo_attack.py"),
        ("last_turn_replay.py",   "last_turn_replay.py"),
        ("attack_utils.py",       "attack_utils.py"),
    ]
    for label, path in scripts:
        exists = Path(path).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {label}: {Path(path).resolve()}")

def check_model_registered(model_configs, model_key, label):
    registered = model_key in model_configs
    status = "✅" if registered else "❌"
    detail = "" if registered else f"Add '{model_key}' to MODEL_CONFIGS in 3_crescendo_attack.py"
    print(f"  {status} {label} '{model_key}' registered in MODEL_CONFIGS"
          + (f"\n     {detail}" if detail else ""))
    return registered

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    # Load configs
    cfg          = load_run_batch_config()
    model_configs = load_model_configs()

    VICTIM_MODEL  = cfg['VICTIM_MODEL']
    ATTACK_MODEL  = cfg['ATTACK_MODEL']
    MODE          = cfg['MODE']
    active_runs   = cfg['active_runs']
    objectives    = cfg['active_objectives']

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    # Resolve victim config
    victim_cfg      = model_configs.get(VICTIM_MODEL, {})
    victim_model_id = victim_cfg.get("model_id", VICTIM_MODEL)
    victim_endpoint = victim_cfg.get("endpoint", "")
    victim_is_local = "localhost" in victim_endpoint

    attacker_cfg      = model_configs.get(ATTACK_MODEL, {})
    attacker_model_id = attacker_cfg.get("model_id", ATTACK_MODEL)

    print("\n" + "=" * 60)
    print("Setup Check")
    print(f"  Victim:   {VICTIM_MODEL} → {victim_model_id}")
    print(f"  Attacker: {ATTACK_MODEL} → {attacker_model_id}")
    print(f"  Mode:     {MODE}")
    if objectives and active_runs:
        print(f"  Batch:    {len(objectives)} objectives × {active_runs} runs "
              f"= {len(objectives)*active_runs} total")
    print("=" * 60)

    # 1. Scripts
    check_scripts_exist()

    # 2. Model registration
    print("\n--- MODEL REGISTRATION ---")
    check_model_registered(model_configs, VICTIM_MODEL,  "Victim")
    check_model_registered(model_configs, ATTACK_MODEL, "Attacker")

    # 3. HuggingFace auth (always needed for attacker)
    if not token:
        print("\n❌ HF_TOKEN not set in .env — needed for attacker model")
        return
    api = HfApi(token=token)
    if check_huggingface_connection(api):
        check_token_permissions(api)

    # 4. Victim checks — local or cloud
    if victim_is_local:
        ollama_ok, available = check_ollama_server()
        if ollama_ok:
            if check_ollama_model(victim_model_id, available):
                check_ollama_inference(victim_model_id)
    else:
        print("\n--- CHECKING VICTIM MODEL (CLOUD) ---")
        print(f"  Model: {victim_model_id}")
        if check_model_access(api, victim_model_id):
            check_router_chat(token, victim_model_id)
            check_inference_status(token, victim_model_id)

    # 5. Attacker checks (always cloud)
    print("\n--- CHECKING ATTACKER MODEL (CLOUD) ---")
    print(f"  Model: {attacker_model_id}")
    if check_model_access(api, attacker_model_id):
        check_router_chat(token, attacker_model_id)
        check_inference_status(token, attacker_model_id)

    print("\n" + "=" * 60)
    print("✅ Setup check complete — review any ❌ above before running")
    print("=" * 60)

if __name__ == "__main__":
    main()