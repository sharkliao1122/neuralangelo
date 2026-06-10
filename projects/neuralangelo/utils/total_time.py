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

import os
import time
from datetime import datetime

from imaginaire.utils.distributed import is_master, master_only_print as print


def format_total_time(seconds):
    """Format elapsed wall-clock seconds into a readable string."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_total_time(output_path, status, start_time, end_time):
    """Write total wall-clock training time to a text file."""
    elapsed = end_time - start_time
    formatted = format_total_time(elapsed)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"status: {status}\n")
        file.write(f"start_time: {datetime.fromtimestamp(start_time).isoformat(timespec='seconds')}\n")
        file.write(f"end_time: {datetime.fromtimestamp(end_time).isoformat(timespec='seconds')}\n")
        file.write(f"total_time: {formatted}\n")
        file.write(f"total_seconds: {elapsed:.2f}\n")


def run_with_total_time(enabled, train_fn, *args, output_path=None, **kwargs):
    """Run the training function and optionally print total wall-clock time."""
    if not enabled:
        return train_fn(*args, **kwargs)

    start_time = time.time()
    status = "completed"
    try:
        return train_fn(*args, **kwargs)
    except Exception:
        status = "failed"
        raise
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        formatted = format_total_time(elapsed)
        print(f"Training {status}. Total wall-clock time: {formatted} ({elapsed:.2f} seconds).")
        if output_path and is_master():
            write_total_time(output_path, status, start_time, end_time)
