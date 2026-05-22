#!/usr/bin/env python3
"""PostToolUse hook: auto-append entry to WORKLOG.md when Python/config files are edited."""
import json
import sys
import os
import datetime

WORKLOG = os.path.join(os.path.dirname(__file__), "..", "WORKLOG.md")
WORKLOG = os.path.abspath(WORKLOG)

TRACKED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".sh", ".toml", ".md"}

def get_short_path(full_path):
    base = "/Users/andrewchen/vscode/rl_project/"
    if full_path.startswith(base):
        return full_path[len(base):]
    return os.path.basename(full_path)

def ensure_date_header(today_str, lines):
    header = f"## {today_str}"
    for line in lines:
        if line.strip().startswith(header):
            return lines
    # Add blank line + header
    sep = ["", f"## {today_str}", ""]
    return lines + sep

def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if not file_path:
        sys.exit(0)

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in TRACKED_EXTENSIONS:
        sys.exit(0)

    # Skip WORKLOG itself to avoid recursion
    if os.path.abspath(file_path) == WORKLOG:
        sys.exit(0)

    now = datetime.datetime.now()
    today_str = now.strftime("%Y-%m-%d (%A)")
    time_str = now.strftime("%H:%M")
    short_path = get_short_path(file_path)

    # Read existing log
    if os.path.exists(WORKLOG):
        with open(WORKLOG, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    else:
        lines = ["# RL Project Work Log", ""]

    lines = ensure_date_header(today_str, lines)

    entry = f"- `{time_str}` **{tool_name}** → `{short_path}`"
    lines.append(entry)

    with open(WORKLOG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
