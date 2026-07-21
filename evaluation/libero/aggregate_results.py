"""Aggregate per-task LIBERO eval results produced by eval_policy_client.py across all GPUs.

Each client writes ``<out_dir>/<benchmark>_<task_id>.json``. With parallel
evaluation we have one ``<out_dir>`` per GPU (``outputs/libero/gpu{i}/``).

New rich schema (written by eval_policy_client.py):
    {benchmark, task_id, task_name, num_episodes, max_steps, success_count,
     success_rate, episode_manifest:[{episode_index, episode_tag, episode_name,
     seed, success, steps, client_video, server_pred_video}],
     done_episode_ids, obs_schema, host, port}

Old minimal schema (backward compat):
    {succ_num, total_num, succ_rate, done_episode_ids}

This script:
  * walks all GPU subdirs and MERGES per-task partial results across GPUs
  * de-duplicates by episode_index (Variant A) or done_episode_ids (Variant B)
  * works even while eval is still running (no need to wait for completion)
  * optionally cross-checks with ``logs/plan/plan_summary.json`` for progress
  * writes a top-level ``summary.json`` with overall_success_rate (aligned
    with robocasa's aggregate_results.py)

Usage:
    python evaluation/libero/aggregate_results.py --run-root outputs/libero_my_ckpt_step900/libero_10
    python evaluation/libero/aggregate_results.py --run-root outputs/libero --benchmark libero_10 --watch 30
    python evaluation/libero/aggregate_results.py --run-root outputs/libero --summary-json outputs/libero/summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

def _load_plan(root: Path) -> dict | None:
    plan_path = root / "logs" / "plan" / "plan_summary.json"
    if not plan_path.is_file():
        return None
    try:
        return json.loads(plan_path.read_text())
    except Exception:  # noqa: BLE001
        return None

def _gather(root: Path, benchmark: str) -> tuple[dict, list[str]]:
    """Return merged per-task stats and the list of parse warnings.

    Merge strategy (3 variants for backward compat):
      Variant A (new rich schema, has episode_manifest): exact per-episode
          merge — union by episode_index, count actual successes.
      Variant B (old schema, has done_episode_ids): approximate merge —
          union done_episode_ids, use file-level success ratio as proxy.
      Variant C (legacy, only succ_num/total_num): keep largest total_num.
    """
    json_files = sorted(root.rglob(f"{benchmark}_*.json"))
    per_task: dict[int, dict] = defaultdict(
        lambda: {"succ_num": 0.0, "total_num": 0.0, "episode_results": {}, "manifest_entries": {}, "task_name": "", "sources": []}
    )
    parse_warnings: list[str] = []

    for p in json_files:
        stem = p.stem  # e.g. libero_10_3
        suffix = stem[len(benchmark) + 1:]
        try:
            task_id = int(suffix)
        except ValueError:
            parse_warnings.append(f"skip unrecognized file: {p}")
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            parse_warnings.append(f"failed to parse {p}: {e}")
            continue

        bucket = per_task[task_id]
        bucket["sources"].append(str(p))
        if data.get("task_name") and not bucket["task_name"]:
            bucket["task_name"] = data["task_name"]

        # Variant A: episode_manifest (new rich schema) — exact per-episode merge
        manifest = data.get("episode_manifest")
        if isinstance(manifest, list) and manifest:
            for entry in manifest:
                eid = int(entry.get("episode_index", -1))
                if eid < 0 or eid in bucket["episode_results"]:
                    continue
                success = bool(entry.get("success", False))
                bucket["episode_results"][eid] = success
                bucket["manifest_entries"][eid] = entry
                bucket["total_num"] += 1
                if success:
                    bucket["succ_num"] += 1
            continue

        # Variant B/C: old schema (succ_num/total_num/done_episode_ids)
        s = float(data.get("success_count", data.get("succ_num", 0.0)))
        n = float(data.get("num_episodes", data.get("total_num", 0.0)))
        done_ids = data.get("done_episode_ids", []) or []

        if done_ids:
            # We don't know per-episode success from old schema alone, so we
            # use the per-file success ratio as a proxy. Exact when no GPU
            # overlap exists (the normal case).
            new_ids = [e for e in done_ids if e not in bucket["episode_results"]]
            num_new = len(new_ids)
            if num_new > 0 and n > 0:
                file_rate = s / n
                approx_new_succ = int(round(file_rate * num_new))
                approx_new_succ = max(0, min(num_new, approx_new_succ))
                for eid in new_ids:
                    bucket["episode_results"][eid] = None  # presence-only
                bucket["succ_num"] += approx_new_succ
                bucket["total_num"] += num_new
        else:
            # No per-episode ids: keep the json with the largest total.
            if n > bucket["total_num"]:
                bucket["succ_num"] = s
                bucket["total_num"] = n

    return dict(per_task), parse_warnings

def _print_report(root: Path, benchmark: str, per_task: dict, plan: dict | None, warnings: list[str]) -> None:
    for w in warnings:
        print(f"[aggregate][warn] {w}")

    # Planned totals per task (if plan available)
    planned_per_task: dict[int, int] = {}
    planned_total = 0
    if plan is not None:
        plan_dir = root / "logs" / "plan"
        for gpu_plan in sorted(plan_dir.glob("gpu*.json")):
            try:
                gp = json.loads(gpu_plan.read_text())
            except Exception:  # noqa: BLE001
                continue
            for task in gp.get("tasks", []):
                tid = int(task.get("task_id", -1))
                eps = task.get("episode_ids", []) or []
                planned_per_task[tid] = planned_per_task.get(tid, 0) + len(eps)
                planned_total += len(eps)

    print(f"\n[aggregate] benchmark={benchmark}   root={root}")
    print(f"[aggregate] tasks with results: {len(per_task)}"
          + (f"  /  planned tasks: {len(planned_per_task)}" if planned_per_task else ""))
    if plan is not None:
        print(f"[aggregate] eval plan: num_gpus={plan.get('num_gpus')}  "
              f"test_num_per_task={plan.get('test_num_per_task')}  "
              f"total_pending_episodes={plan.get('total_pending_episodes')}")

    print("-" * 90)
    header_plan = " planned" if planned_per_task else ""
    print(f"{'task_id':>8} | {'succ':>5} / {'done':>5}{header_plan:>9} | {'rate':>6} | sources")
    print("-" * 90)

    all_task_ids = sorted(set(per_task.keys()) | set(planned_per_task.keys()))
    total_succ = 0.0
    total_done = 0.0
    rates = []
    for task_id in all_task_ids:
        d = per_task.get(task_id, {"succ_num": 0.0, "total_num": 0.0, "sources": []})
        s = float(d.get("succ_num", 0.0))
        n = float(d.get("total_num", 0.0))
        r = (s / n) if n > 0 else 0.0
        planned = planned_per_task.get(task_id, 0)
        total_succ += s
        total_done += n
        if n > 0:
            rates.append(r)
        plan_col = f"/{planned:>4d}" if planned_per_task else ""
        nsrc = len(d.get("sources", []))
        print(f"{task_id:>8d} | {s:>5.0f} / {n:>5.0f}{plan_col:>9} | {r:>6.3f} | {nsrc} gpu file(s)")

    print("-" * 90)
    micro = (total_succ / total_done) if total_done > 0 else 0.0
    macro = (sum(rates) / len(rates)) if rates else 0.0
    print(f"micro success_rate (sum_succ / sum_done)  = {total_succ:.0f} / {total_done:.0f} = {micro:.4f}")
    print(f"macro success_rate (mean over {len(rates):>2d} done tasks) = {macro:.4f}")
    if planned_total:
        progress = total_done / planned_total if planned_total > 0 else 0.0
        print(f"overall progress: done {total_done:.0f} / planned {planned_total} = {progress*100:.1f}%")
    print()

def _build_summary(run_root: Path, per_task: dict) -> dict:
    """Build top-level summary dict (aligned with robocasa's summary.json schema)."""
    per_task_list = []
    for task_id in sorted(per_task.keys()):
        d = per_task[task_id]
        s = float(d.get("succ_num", 0.0))
        n = float(d.get("total_num", 0.0))
        r = (s / n) if n > 0 else 0.0
        per_task_list.append({
            "task_id": int(task_id),
            "task_name": d.get("task_name", ""),
            "num_episodes": int(n),
            "success_count": int(s),
            "success_rate": r,
        })
    total_succ = sum(t["success_count"] for t in per_task_list)
    total_eps = sum(t["num_episodes"] for t in per_task_list)
    return {
        "run_root": str(run_root),
        "num_tasks_aggregated": len(per_task_list),
        "per_task": per_task_list,
        "overall_success_rate": total_succ / max(1, total_eps),
    }

def _detect_benchmarks(root: Path) -> list[str]:
    """Auto-detect benchmark names by scanning for subdirs that contain gpu*/<benchmark>_*.json files.

    Returns a sorted list of unique benchmark prefixes found under root.
    Falls back to scanning root directly if no benchmark subdirs exist.
    """
    benchmarks: set[str] = set()
    # Check for benchmark subdirs (e.g. root/libero_spatial/gpu0/libero_spatial_0.json)
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        # Look for json files matching <subdir_name>_*.json pattern inside this subdir
        json_files = list(subdir.rglob(f"{subdir.name}_*.json"))
        if json_files:
            benchmarks.add(subdir.name)
    # If no benchmark subdirs found, scan root directly for any libero_*_*.json
    if not benchmarks:
        for json_file in root.rglob("libero_*_*.json"):
            stem = json_file.stem  # e.g. libero_10_3
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                benchmarks.add(parts[0])
    return sorted(benchmarks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-task LIBERO eval results across GPUs."
    )
    parser.add_argument(
        "--run-root",
        type=str,
        required=True,
        help="Root directory containing per-GPU subdirs (e.g. gpu0/, gpu1/, ...) "
             "or a single client's out_dir.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional global summary path. Writes summary.json with overall_success_rate.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="auto",
        help="Benchmark name prefix used in the per-task json files. "
             "Use 'auto' (default) to detect all benchmarks under --run-root. "
             "Specify a concrete name (e.g. libero_10, libero_spatial) to restrict to one.",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="If > 0, re-aggregate every N seconds (live progress mode).",
    )
    args = parser.parse_args()

    root = Path(args.run_root)
    if not root.exists():
        raise SystemExit(f"[aggregate] run-root not found: {root}")

    # Resolve benchmark(s) to process
    if args.benchmark == "auto":
        benchmarks = _detect_benchmarks(root)
        if not benchmarks:
            raise SystemExit(f"[aggregate] auto-detect found no benchmarks under {root}")
    else:
        benchmarks = [args.benchmark]

    def _once() -> dict:
        all_summaries = []
        for benchmark in benchmarks:
            benchmark_root = root / benchmark if (root / benchmark).is_dir() else root
            per_task, warnings = _gather(benchmark_root, benchmark)
            plan = _load_plan(benchmark_root)
            if not per_task and not plan:
                print(f"[aggregate] no result files found under {benchmark_root} matching {benchmark}_*.json yet")
                continue
            _print_report(benchmark_root, benchmark, per_task, plan, warnings)
            all_summaries.append((benchmark, per_task))
        return all_summaries

    if args.watch <= 0:
        all_summaries = _once()
        if args.summary_json:
            _write_summary_json(args.summary_json, root, all_summaries)
        return

    try:
        while True:
            print(f"\n========== [{time.strftime('%F %T')}] aggregate tick ==========")
            all_summaries = _once()
            if args.summary_json:
                _write_summary_json(args.summary_json, root, all_summaries)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\n[aggregate] interrupted, bye.")


def _write_summary_json(summary_json_path: str, root: Path, all_summaries: list[tuple[str, dict]]) -> None:
    """Write top-level summary.json (aligned with robocasa's summary.json schema)."""
    per_task_all = []
    for benchmark, per_task in all_summaries:
        for task_id, d in per_task.items():
            s = float(d.get("succ_num", 0.0))
            n = float(d.get("total_num", 0.0))
            r = (s / n) if n > 0 else 0.0
            per_task_all.append({
                "benchmark": benchmark,
                "task_id": int(task_id),
                "task_name": d.get("task_name", ""),
                "num_episodes": int(n),
                "success_count": int(s),
                "success_rate": r,
            })
    total_succ = sum(t["success_count"] for t in per_task_all)
    total_eps = sum(t["num_episodes"] for t in per_task_all)
    global_summary = {
        "run_root": str(root),
        "num_tasks_aggregated": len(per_task_all),
        "per_task": per_task_all,
        "overall_success_rate": total_succ / max(1, total_eps),
    }
    sp = Path(summary_json_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(global_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[aggregate] global summary -> {sp}")


if __name__ == "__main__":
    main()
