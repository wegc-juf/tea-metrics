#!/usr/bin/env python3
"""Run calc_TEA with optional profiling and a bounded Dask CPU count."""

import argparse
import cProfile
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--metrics-file', type=Path, required=True)
    parser.add_argument('--profile-file', type=Path)
    parser.add_argument('--dask-workers', type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dask_workers is not None:
        from teametrics.common import dask_config

        dask_config.os.cpu_count = lambda: args.dask_workers

    from teametrics import calc_TEA

    sys.argv = [
        'calc_TEA', '--config-file', args.config_file, '--loglevel', 'INFO',
    ]
    profiler = cProfile.Profile() if args.profile_file else None
    started = time.perf_counter()
    if profiler:
        profiler.enable()
    try:
        calc_TEA.run()
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(args.profile_file)
        elapsed = time.perf_counter() - started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        args.metrics_file.write_text(json.dumps({
            'elapsed_seconds': elapsed,
            'max_rss_kb': usage.ru_maxrss,
            'user_cpu_seconds': usage.ru_utime,
            'system_cpu_seconds': usage.ru_stime,
            'dask_workers': args.dask_workers,
            'pid': os.getpid(),
        }, indent=2) + '\n')


if __name__ == '__main__':
    main()
