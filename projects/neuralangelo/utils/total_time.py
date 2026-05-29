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

import time

from imaginaire.utils.distributed import master_only_print as print


def format_total_time(seconds):
    """Format elapsed wall-clock seconds into a readable string."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def run_with_total_time(enabled, train_fn, *args, **kwargs):
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
        elapsed = time.time() - start_time
        formatted = format_total_time(elapsed)
        print(f"Training {status}. Total wall-clock time: {formatted} ({elapsed:.2f} seconds).")
