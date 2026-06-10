'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import argparse
import json
import os
import shlex
import sys
from datetime import datetime


def _quote_command(argv):
    if os.name == "nt":
        return " ".join(f'"{arg}"' if any(char.isspace() for char in arg) else arg for arg in argv)
    return " ".join(shlex.quote(arg) for arg in argv)


def record_training_command(logdir, argv=None, metadata=None, text_filename="training_command.txt",
                            jsonl_filename="training_command.jsonl"):
    """Record a training command as both human-readable text and JSONL."""
    argv = list(sys.argv if argv is None else argv)
    metadata = dict(metadata or {})
    os.makedirs(logdir, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "command": _quote_command(argv),
        "argv": argv,
        "metadata": metadata,
    }

    text_path = os.path.join(logdir, text_filename)
    with open(text_path, "a", encoding="utf-8") as file:
        file.write(f"[{record['timestamp']}]\n")
        file.write(f"cwd: {record['cwd']}\n")
        file.write(f"command: {record['command']}\n")
        if metadata:
            file.write(f"metadata: {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}\n")
        file.write("\n")

    jsonl_path = os.path.join(logdir, jsonl_filename)
    with open(jsonl_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    return text_path, jsonl_path


def parse_args():
    parser = argparse.ArgumentParser(description="Record a training command to a log directory.")
    parser.add_argument("--logdir", required=True, help="Directory where command logs will be written.")
    parser.add_argument("--metadata", default=None, help="Optional JSON metadata to include in the record.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to record. Prefix with -- before the command.")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = json.loads(args.metadata) if args.metadata else None
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    text_path, jsonl_path = record_training_command(args.logdir, argv=command or sys.argv, metadata=metadata)
    print(f"Wrote command log: {text_path}")
    print(f"Wrote command log: {jsonl_path}")


if __name__ == "__main__":
    main()
