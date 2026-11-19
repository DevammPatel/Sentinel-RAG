"""Facts about the machine we are running on."""
from __future__ import annotations

import math
import os
from pathlib import Path


def available_cpus() -> int:
    """CPUs this process may actually use.

    `os.cpu_count()` reports the host's cores even inside a container limited to
    a fraction of them. Sizing a thread pool from it there is a real performance
    bug: ONNX Runtime starts one thread per reported core, they all fight over
    the container's CPU quota, and inference slows down by an order of
    magnitude. So we also honour the cgroup quota and the affinity mask.
    """
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)) or count)

    quota = _cgroup_cpu_quota()
    if quota:
        count = min(count, max(1, math.ceil(quota)))
    return max(1, count)


def _cgroup_cpu_quota() -> float | None:
    """CPU quota in cores from cgroup v2 or v1, or None when unlimited."""
    try:
        v2 = Path("/sys/fs/cgroup/cpu.max")
        if v2.exists():
            quota, period = v2.read_text().split()
            return None if quota == "max" else int(quota) / int(period)
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        return quota / period if quota > 0 and period > 0 else None
    except (OSError, ValueError):
        return None
