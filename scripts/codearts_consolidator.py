"""
CodeArts session consolidator — merge session files into a single trajectory JSON.

Reads a CodeArts agent session folder containing:
  - chat_baseInfo.json  (metadata: title, chatId, timestamp, agent info)
  - messages_0.json     (raw message list — preserved as-is)

and produces a single consolidated JSON file with metadata wrapper.
The original messages are kept verbatim in the output.

Usage:
    python scripts/codearts_consolidator.py samples/<session-id>
    python scripts/codearts_consolidator.py samples/<session-id> --output out.json
    python scripts/codearts_consolidator.py samples/ --batch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_base_info(session_dir: str) -> dict:
    """Read chat_baseInfo.json and return metadata dict."""
    path = os.path.join(session_dir, "chat_baseInfo.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_messages(session_dir: str) -> list[dict]:
    """Read messages_0.json and return raw message list."""
    path = os.path.join(session_dir, "messages_0.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def consolidate_session(session_dir: str) -> dict:
    """Merge chat_baseInfo.json + messages_0.json into a single trajectory dict.

    Messages are preserved verbatim — no normalization or transformation.
    """
    base_info = load_base_info(session_dir)
    messages = load_messages(session_dir)

    session_id = base_info.get("chatId", os.path.basename(session_dir))
    title = base_info.get("title", "")
    start_ts = base_info.get("timestamp", "")
    agent = base_info.get("selectedGpt", {})

    # Timing from first/last message timestamps
    first_ts = messages[0].get("timestamp", "") if messages else ""
    last_ts = messages[-1].get("timestamp", "") if messages else ""

    # Count by sender
    user_count = sum(1 for m in messages if m.get("sender") == "User")
    asst_count = len(messages) - user_count

    return {
        "format": "codearts",
        "metadata": {
            "session_id": session_id,
            "title": title,
            "agent": agent.get("en_name", agent.get("agent_id", "CodeArts")),
            "agent_id": agent.get("real_agent_id", ""),
            "model": "",  # extracted per-message by loader
            "timestamp_utc": start_ts,
            "context_tokens": base_info.get("contextToken", 0),
            "generator_name": "codearts",
        },
        "timing": {
            "started_at": first_ts,
            "finished_at": last_ts,
        },
        "messages": messages,  # raw messages preserved as-is
        "chat_base_info": base_info,  # full original metadata
        "statistics": {
            "total_messages": len(messages),
            "user_messages": user_count,
            "assistant_messages": asst_count,
        },
    }


def write_output(data: dict, output_path: str) -> None:
    """Write consolidated trajectory JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _is_session_dir(path: str) -> bool:
    """Check if a directory looks like a CodeArts session folder."""
    return (
        os.path.isfile(os.path.join(path, "chat_baseInfo.json"))
        and os.path.isfile(os.path.join(path, "messages_0.json"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate CodeArts session files into a single trajectory JSON."
    )
    parser.add_argument("input", help="Path to session folder or parent directory (with --batch)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path (default: <session-dir>/<session-id>_trajectory.json)")
    parser.add_argument("--batch", action="store_true",
                        help="Process all session subfolders in the given directory")
    args = parser.parse_args()

    if args.batch:
        parent = args.input
        if not os.path.isdir(parent):
            print(f"Error: {parent} is not a directory", file=sys.stderr)
            sys.exit(1)

        processed = 0
        skipped = 0
        for entry in sorted(os.listdir(parent)):
            subdir = os.path.join(parent, entry)
            if not os.path.isdir(subdir) or not _is_session_dir(subdir):
                skipped += 1
                continue

            try:
                data = consolidate_session(subdir)
                session_id = data["metadata"]["session_id"]
                out_path = os.path.join(subdir, f"{session_id}_trajectory.json")
                write_output(data, out_path)
                n_msgs = data["statistics"]["total_messages"]
                print(f"  {session_id}: {n_msgs} messages → {out_path}")
                processed += 1
            except Exception as exc:
                print(f"  Error in {entry}: {exc}", file=sys.stderr)
                skipped += 1

        print(f"\nDone: {processed} sessions consolidated, {skipped} skipped.")
        return

    # Single session mode
    session_dir = args.input
    if not os.path.isdir(session_dir):
        print(f"Error: {session_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    if not _is_session_dir(session_dir):
        print(f"Error: {session_dir} does not contain chat_baseInfo.json and messages_0.json",
              file=sys.stderr)
        sys.exit(1)

    data = consolidate_session(session_dir)
    session_id = data["metadata"]["session_id"]

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(session_dir, f"{session_id}_trajectory.json")

    write_output(data, output_path)

    stats = data["statistics"]
    print(f"Consolidated: {session_id}")
    print(f"  Messages: {stats['total_messages']} total ({stats['user_messages']} user, {stats['assistant_messages']} assistant)")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
