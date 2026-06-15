#!/usr/bin/env python3
"""Run the full forecasting pipeline in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PIPELINE_DIR = Path("pipeline")

PIPELINE_STEPS = [
    PIPELINE_DIR / "1-classify-content.py",
    PIPELINE_DIR / "2-find-topics.py",
    PIPELINE_DIR / "3-forecast.py",
    PIPELINE_DIR / "5-yoy-trends.py",
    PIPELINE_DIR / "6-report.py",
    PIPELINE_DIR / "7-metrics.py",
    PIPELINE_DIR / "9-contextual-analysis-v3.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full event forecasting pipeline."
    )
    parser.add_argument(
        "--from-step",
        help="Resume from a script, e.g. 5-yoy-trends.py or pipeline/5-yoy-trends.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the scripts that would run without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    steps = PIPELINE_STEPS
    if args.from_step:
        matches = [
            index
            for index, step in enumerate(PIPELINE_STEPS)
            if args.from_step in {str(step), step.name}
        ]
        if not matches:
            valid_steps = ", ".join(step.name for step in PIPELINE_STEPS)
            print(
                f"Unknown --from-step value: {args.from_step}\n"
                f"Valid steps: {valid_steps}",
                file=sys.stderr,
            )
            return 2
        steps = steps[matches[0] :]

    print("Forecasting pipeline")
    print(f"Project: {project_root}")
    print(f"Python : {sys.executable}")
    print()

    if args.dry_run:
        for step in steps:
            print(f"Would run: {step}")
        return 0

    total_start = time.monotonic()
    for index, step in enumerate(steps, start=1):
        step_path = project_root / step
        if not step_path.exists():
            print(f"Missing script: {step}", file=sys.stderr)
            return 1

        print("=" * 72)
        print(f"[{index}/{len(steps)}] Running {step}")
        print("=" * 72)
        start = time.monotonic()

        result = subprocess.run(
            [sys.executable, str(step_path)],
            cwd=project_root,
            check=False,
        )

        elapsed = time.monotonic() - start
        if result.returncode != 0:
            print()
            print(
                f"Pipeline stopped: {step} failed with exit code "
                f"{result.returncode} after {elapsed:.1f}s.",
                file=sys.stderr,
            )
            return result.returncode

        print()
        print(f"Finished {step} in {elapsed:.1f}s")
        print()

    total_elapsed = time.monotonic() - total_start
    print("=" * 72)
    print(f"Pipeline complete in {total_elapsed:.1f}s")
    print(f"Reports: {project_root / 'reports'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
