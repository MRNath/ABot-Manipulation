"""Aggregate dynamic-wrapper per-batch JSONs into per-env JSONs.

The dynamic wrapper writes one ``robocasa_eval.json`` per executed
batch under::

    <run_root>/envs/<env_name>/batches/ep<NNN>_n<COUNT>_seed<SEED>/robocasa_eval.json

This tool merges all batches of one env into a single
``<run_root>/envs/<env_name>/robocasa_eval.json`` whose schema mirrors the
output of the original env-sweep wrapper,
so any downstream tooling that consumed the static wrapper's results keeps
working unchanged.

Should be invoked once after all workers finish. It is idempotent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _merge_one_env(env_dir: Path) -> dict | None:
    batches_dir = env_dir / "batches"
    if not batches_dir.is_dir():
        return None
    batch_dirs = sorted(p for p in batches_dir.iterdir() if p.is_dir())
    if not batch_dirs:
        return None

    episodes: list[dict] = []
    success_count = 0
    split_seen: set[str] = set()
    max_steps_seen: set[int] = set()
    obs_schema: list = []
    host = ""
    port = 0
    frame_chunk_size = 0
    env_name = env_dir.name
    sources: list[str] = []

    # Each batch's ep_start tells us how to remap that batch's
    # episode_index (which is local to the batch, 0..N-1) back to the
    # global ep_idx the dispatcher assigned.
    for batch_dir in batch_dirs:
        # Parse "epNNN_nM_seedS" out of the directory name.
        name = batch_dir.name
        try:
            ep_part, n_part, seed_part = name.split("_")
            assert ep_part.startswith("ep") and n_part.startswith("n") and seed_part.startswith("seed")
            ep_start = int(ep_part[2:])
        except Exception:  # pragma: no cover - defensive
            ep_start = -1

        json_path = batch_dir / "robocasa_eval.json"
        if not json_path.is_file():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[aggregate] WARNING: failed to read {json_path}: {exc!r}",
                  file=sys.stderr)
            continue
        sources.append(str(json_path))

        env_name_in_json = str(data.get("env_name", env_name)).strip() or env_name
        if env_name_in_json:
            env_name = env_name_in_json

        for entry in data.get("episode_manifest") or []:
            entry = dict(entry)  # shallow copy
            local_ep = int(entry.get("episode_index", 0))
            if ep_start >= 0:
                entry["episode_index"] = ep_start + local_ep
                entry["batch_local_episode_index"] = local_ep
            entry["batch_dir"] = str(batch_dir)
            episodes.append(entry)
            if bool(entry.get("success", False)):
                success_count += 1

        if data.get("split"):
            split_seen.add(str(data["split"]))
        if data.get("max_steps") is not None:
            try:
                max_steps_seen.add(int(data["max_steps"]))
            except Exception:
                pass
        if not obs_schema and data.get("obs_schema"):
            obs_schema = data["obs_schema"]
        if not host and data.get("host"):
            host = str(data["host"])
        if not port and data.get("port"):
            try:
                port = int(data["port"])
            except Exception:
                pass
        if not frame_chunk_size and data.get("frame_chunk_size"):
            try:
                frame_chunk_size = int(data["frame_chunk_size"])
            except Exception:
                pass

    # Sort episodes by global episode_index for stable output.
    episodes.sort(key=lambda e: int(e.get("episode_index", 0)))
    num_episodes = len(episodes)
    return {
        "env_name": env_name,
        "split": next(iter(split_seen), ""),
        "num_episodes": num_episodes,
        "max_steps": max(max_steps_seen) if max_steps_seen else 0,
        "success_count": success_count,
        "success_rate": success_count / max(1, num_episodes),
        "episode_manifest": episodes,
        "obs_schema": obs_schema,
        "host": host,
        "port": port,
        "frame_chunk_size": frame_chunk_size,
        "_aggregated_from": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate dynamic-wrapper per-batch JSONs."
    )
    parser.add_argument("--run-root", type=str, required=True,
                        help="Same value as the wrapper's --run-root.")
    parser.add_argument("--summary-json", type=str, default="",
                        help="Optional global summary path.")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    envs_root = run_root / "envs"
    if not envs_root.is_dir():
        print(f"[aggregate] ERROR: {envs_root} does not exist", file=sys.stderr)
        return 2

    summary: list[dict] = []
    aggregated = 0
    skipped: list[str] = []

    for env_dir in sorted(p for p in envs_root.iterdir() if p.is_dir()):
        result = _merge_one_env(env_dir)
        if result is None:
            skipped.append(env_dir.name)
            continue
        out_path = env_dir / "robocasa_eval.json"
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary.append({
            "env_name": env_dir.name,
            "num_episodes": result["num_episodes"],
            "success_count": result["success_count"],
            "success_rate": result["success_rate"],
            "aggregated_json": str(out_path),
        })
        aggregated += 1
        print(
            f"[aggregate] {env_dir.name}: episodes={result['num_episodes']} "
            f"success={result['success_count']} "
            f"rate={result['success_rate']:.4f} -> {out_path}"
        )

    if skipped:
        print(f"[aggregate] skipped envs (no batches): {skipped}", file=sys.stderr)

    if args.summary_json:
        sp = Path(args.summary_json)
        sp.parent.mkdir(parents=True, exist_ok=True)
        global_obj = {
            "run_root": str(run_root),
            "num_envs_aggregated": aggregated,
            "skipped_envs": skipped,
            "per_env": summary,
            "overall_success_rate": (
                sum(s["success_count"] for s in summary)
                / max(1, sum(s["num_episodes"] for s in summary))
            ),
        }
        sp.write_text(
            json.dumps(global_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[aggregate] global summary -> {sp}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
