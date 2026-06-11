#!/usr/bin/env python
# coding=utf-8
"""Summarize xBench evaluation results from one or more result jsonl files.

Workaround for eval_utils.py report bug: generate_unified_report counts
records by the string field `judgement` (GAIA schema), but the xBench
runner writes a numeric `score` field, so its report always shows 0.
This script computes accuracy and resource stats from `score` directly.

Usage:
    python summarize_results.py xbench_output/nomem_20.jsonl [more.jsonl ...]
"""
import json
import sys
import os


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # keep the LAST record per task_id (reruns append to the same file)
    dedup = {}
    for r in rows:
        dedup[r["task_id"]] = r
    return list(dedup.values())


def summarize(path):
    rows = load(path)
    n = len(rows)
    correct = sum(1 for r in rows if r.get("score") == 1)
    m = [r.get("metrics", {}) for r in rows]
    avg = lambda key: sum(x.get(key, 0) for x in m) / n if n else 0
    steps = [len(r.get("agent_trajectory", [])) for r in rows]
    return {
        "name": os.path.basename(path).replace(".jsonl", ""),
        "n": n,
        "correct": correct,
        "acc": correct / n * 100 if n else 0,
        "avg_time": avg("elapsed_time"),
        "avg_tokens": avg("total_tokens"),
        "avg_calls": avg("api_calls"),
        "avg_steps": sum(steps) / n if n else 0,
        "per_task": [(r["task_id"], r.get("score", 0)) for r in rows],
    }


def main(paths):
    stats = [summarize(p) for p in paths]
    header = f"{'setting':<22}{'acc':>10}{'avg_time(s)':>13}{'avg_tokens':>12}{'avg_calls':>11}{'avg_steps':>11}"
    print(header)
    print("-" * len(header))
    for s in stats:
        print(f"{s['name']:<22}{s['correct']}/{s['n']} ({s['acc']:.0f}%)"
              f"{s['avg_time']:>10.1f}{s['avg_tokens']:>12.0f}{s['avg_calls']:>11.1f}{s['avg_steps']:>11.1f}")
    # per-task correctness grid for case analysis
    print("\nper-task score (row = setting, column = task index):")
    for s in stats:
        marks = "".join("o" if sc == 1 else "x" for _, sc in s["per_task"])
        print(f"{s['name']:<22}{marks}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
