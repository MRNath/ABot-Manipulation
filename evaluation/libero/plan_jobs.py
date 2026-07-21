"""Episode-level job planner for parallel LIBERO evaluation.

Given:
  - benchmark name (libero_10 / libero_goal / libero_spatial / libero_object)
  - per-task episode count (test_num)
  - number of GPUs / clients
  - out_root where per-task results are written (and may already exist)

Produces NUM_GPUS plan JSON files (one per GPU), each containing the exact
(task_id, episode_ids) pairs that GPU should run.

Skipping logic:
  - For each task we scan any existing ``out_root/**/<benchmark>_<task>.json``
    files; the episode ids already in ``done_episode_ids`` (or, if missing,
    inferred from ``total_num``) are removed from the pending set.

Balancing:
  - All pending (task, episode) pairs are flattened, then distributed to GPUs
    in a round-robin fashion. This keeps the per-GPU episode count
    differences at most 1 even when task counts don't divide evenly by NUM_GPUS.

Output:
  - Per-GPU plan: ``<plan_dir>/gpu{i}.json``
    {
      "gpu_id": i,
      "benchmark": "libero_10",
      "out_dir": "<per-gpu out dir>",
      "items": [{"task_id": int, "episode_ids": [int, ...]}, ...]
    }
  - Summary: ``<plan_dir>/plan_summary.json``
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_done_per_task(out_root: Path, benchmark: str, num_tasks: int) -> dict[int, set[int]]:
    """Scan out_root recursively for <benchmark>_<task>.json files and collect
    already-completed episode ids per task."""
    done: dict[int, set[int]] = {t: set() for t in range(num_tasks)}
    files = list(out_root.rglob(f"{benchmark}_*.json"))
    for p in files:
        try:
            stem = p.stem
            suffix = stem[len(benchmark) + 1:]
            tid = int(suffix)
        except Exception:
            continue
        if tid < 0 or tid >= num_tasks:
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        # New schema: episode_manifest (rich schema from eval_policy_client.py)
        manifest = data.get("episode_manifest")
        if isinstance(manifest, list) and manifest:
            for entry in manifest:
                if isinstance(entry, dict) and "episode_index" in entry:
                    done[tid].add(int(entry["episode_index"]))
            continue
        ep_ids = data.get("done_episode_ids")
        if isinstance(ep_ids, list) and len(ep_ids) > 0:
            done[tid].update(int(e) for e in ep_ids)
        else:
            # Older format: only total_num. Assume eps 0..total_num-1 are done.
            total = int(data.get("total_num", 0))
            done[tid].update(range(total))
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--test-num", type=int, required=True, help="episodes per task")
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--out-root", required=True, help="root where per-gpu subdirs live (e.g. outputs/libero/libero_10)")
    parser.add_argument("--plan-dir", required=True, help="where to write gpu{i}.json plan files")
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    plan_dir = Path(args.plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)

    # 1) Figure out which (task,episode) pairs are pending.
    if args.skip_completed:
        done_per_task = _load_done_per_task(out_root, args.benchmark, args.num_tasks)
    else:
        done_per_task = {t: set() for t in range(args.num_tasks)}

    pending: list[tuple[int, int]] = []
    per_task_pending: dict[int, list[int]] = defaultdict(list)
    for tid in range(args.num_tasks):
        for eid in range(args.test_num):
            if eid in done_per_task[tid]:
                continue
            pending.append((tid, eid))
            per_task_pending[tid].append(eid)

    # 2) Round-robin distribute to GPUs.
    buckets: dict[int, dict[int, list[int]]] = {g: defaultdict(list) for g in range(args.num_gpus)}
    for idx, (tid, eid) in enumerate(pending):
        gpu = idx % args.num_gpus
        buckets[gpu][tid].append(eid)

    # 3) Write per-GPU plan files.
    plan_summary = {
        "benchmark": args.benchmark,
        "num_tasks": args.num_tasks,
        "test_num_per_task": args.test_num,
        "num_gpus": args.num_gpus,
        "total_pending_episodes": len(pending),
        "skipped_episodes_by_task": {
            int(t): sorted(int(e) for e in done_per_task[t]) for t in range(args.num_tasks)
        },
        "per_gpu": [],
    }
    for gpu in range(args.num_gpus):
        items = []
        ep_count = 0
        for tid, eps in sorted(buckets[gpu].items()):
            items.append({"task_id": int(tid), "episode_ids": sorted(int(e) for e in eps)})
            ep_count += len(eps)
        plan = {
            "gpu_id": gpu,
            "benchmark": args.benchmark,
            "out_dir": str(out_root / f"gpu{gpu}"),
            "items": items,
            "total_episodes": ep_count,
        }
        plan_file = plan_dir / f"gpu{gpu}.json"
        plan_file.write_text(json.dumps(plan, indent=2))
        plan_summary["per_gpu"].append({
            "gpu_id": gpu,
            "total_episodes": ep_count,
            "task_ids": sorted(int(t) for t in buckets[gpu].keys()),
            "plan_file": str(plan_file),
        })

    summary_file = plan_dir / "plan_summary.json"
    summary_file.write_text(json.dumps(plan_summary, indent=2))

    # Human-readable summary on stdout for the launcher to log.
    print(f"[plan] benchmark={args.benchmark} pending_episodes={len(pending)}")
    for g in range(args.num_gpus):
        print(f"[plan]   gpu{g}: episodes={plan_summary['per_gpu'][g]['total_episodes']}  tasks={plan_summary['per_gpu'][g]['task_ids']}")
    print(f"[plan] summary written to {summary_file}")


if __name__ == "__main__":
    main()
