import json
from pathlib import Path


def load_jsonl_rows(path):
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_last_row(path):
    rows = load_jsonl_rows(path)
    if not rows:
        raise ValueError(f"No rows found in: {path}")
    return rows[-1]


def extract_last_attacker_turn(row):
    history = row.get("conversation_history", [])
    user_turns = [m for m in history if m.get("role") == "user"]
    if not user_turns:
        raise ValueError("No user turns found in conversation_history")
    return user_turns[-1]["content"]


def save_one_row_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def list_jsonl_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(folder.glob("*.jsonl"))


def get_selected_jsonl_files(folder, include_files=None, exclude_files=None):
    """
    include_files: list of exact filenames to include, or None for all
    exclude_files: list/set of exact filenames to exclude
    """
    files = list_jsonl_files(folder)

    if include_files:
        include_set = set(include_files)
        files = [f for f in files if f.name in include_set]

    if exclude_files:
        exclude_set = set(exclude_files)
        files = [f for f in files if f.name not in exclude_set]

    return files


def build_isolated_output_row(full_row, last_attacker_turn, isolated_row, source_file):
    isolated_response = isolated_row.get("response", "")

    return {
        "scenario":             isolated_row.get("scenario", "unknown"),
        "jailbroken":           isolated_row.get("jailbroken", False),
        "turn":                 1,
        "objective":            full_row.get("objective"),
        "victim_model":         isolated_row.get("victim_model", ""),
        "source_full_jsonl":    str(source_file),
        "source_full_scenario": full_row.get("scenario"),
        "source_full_turn":     full_row.get("turn"),
        "conversation_history": [
            {"role": "user",      "content": last_attacker_turn},
            {"role": "assistant", "content": isolated_response},
        ],
        "response": isolated_response,
    }