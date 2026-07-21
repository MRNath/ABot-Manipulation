#!/usr/bin/env python3
"""RoboCasa ENV sweep wrapper (dynamic episode-level work stealing).

Each worker becomes a consumer of a centralized shared task pool whose atomic unit
is ``(env_name, ep_idx)`` instead of ``env_name``.

Differences vs. the static wrapper:

  - The static wrapper performs an LPT (longest-processing-time) bin-packing
    of envs to workers up-front, then each worker serially evaluates its
    fixed env list. Slow workers tail-latency the whole job.
  - This dynamic wrapper writes one shared TSV task pool on NAS at startup
    (rank-0 only). Every worker repeatedly:
        1. acquires an exclusive ``flock`` on the shared lock file,
        2. pops the head task,
        3. *opportunistically extends* the pop into a contiguous batch of
           same-env, increasing-``ep_idx`` tasks (up to ``--batch-max``),
        4. invokes ``evaluation/robocasa/launch_server_env_sweep.sh`` with
           ``NUM_EPISODES=batch_size`` and ``SEED=base_seed+first_ep_idx``,
           which is bit-exact with what the static wrapper would have run.
    The batching step is critical: ``launch_server_env_sweep.sh`` starts and
    stops the policy server on every invocation, so we want to amortize that
    cost across as many same-env episodes as possible while still allowing
    work to migrate across workers when the queue is short.

"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import typer

try:
    from robocasa.utils.dataset_registry_utils import get_task_horizon
except Exception:  # pragma: no cover - robocasa optional at parse time.
    get_task_horizon = None


# ---------------------------------------------------------------------------
# Rank discovery (mirrors the original wrapper's behavior)
# ---------------------------------------------------------------------------


def get_worker_rank() -> int:
    for key in ["RANK", "WORKER_RANK", "LOCAL_RANK"]:
        value = os.environ.get(key)
        if value is not None:
            return int(value)
    lock_dir = Path("/tmp/robocasa_eval_dynamic_ranks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(60):
        for rank in range(1000):
            lock_file = lock_dir / f"rank_{rank}.lock"
            try:
                fp = lock_file.open("w")
                fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fp.write(str(os.getpid()))
                fp.flush()
                # Intentionally leak the fp so the lock is held for the
                # lifetime of this worker process.
                typer.echo(f"Acquired rank via file lock: {rank}")
                return rank
            except OSError:
                continue
        time.sleep(1)
    raise RuntimeError("Failed to acquire worker rank; check environment variables or filesystem state.")


def resolve_effective_gpu_id(gpu_id: str, rank: int) -> str:
    """Pick ``CUDA_VISIBLE_DEVICES`` for ``launch_server_env_sweep.sh``.

    - **One GPU visible** (typical 1 worker / 1 node): use ``0`` (or
      ``--gpu-id``), regardless of global worker ``rank``.
    - **Several GPUs on this node**: prefer platform ``LOCAL_RANK`` /
      ``LOCAL_PROCESS_RANK``; otherwise ``rank % torch.cuda.device_count()``.
    """
    fallback = str(gpu_id).strip() if str(gpu_id).strip() else "0"
    for key in ("LOCAL_PROCESS_RANK", "LOCAL_RANK"):
        raw = os.environ.get(key)
        if raw is None or not str(raw).strip().lstrip("-").isdigit():
            continue
        local = int(raw)
        if local >= 0:
            return str(local)
    try:
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n = 0
    if n > 1:
        return str(rank % n)
    return fallback


# ---------------------------------------------------------------------------
# One-shot ckpt prefetch from NAS to container-local /tmp
# ---------------------------------------------------------------------------


def prefetch_ckpt_to_local(
    ckpt_path: str,
    local_root: str = "/tmp/eval_ckpt_cache",
    rank: int = 0,
) -> str:
    """Copy a NAS ckpt directory to container-local disk one time per node.

    Why: ``launch_server_env_sweep.sh`` is invoked per batch, and each invocation
    re-runs ``from_pretrained(CKPT_PATH)`` which streams the full safetensors
    shards from disk. With CKPT_PATH on NAS this means re-reading multi-GB of
    weights for every batch — under NAS contention this single read has been
    observed to exceed 10 minutes, which is why ``WAIT_TIMEOUT`` was raised to
    3600s. Copying once to ``/tmp`` cuts every subsequent batch start to
    container-local I/O.

    Concurrency: multiple workers may share the same physical node (and the
    same ``/tmp``). We coordinate via ``fcntl.LOCK_EX`` on a per-source-path
    lock file plus a ``.copy_done`` sentinel inside the destination, so that
    only the first worker on the node actually performs the copy. The copy is
    staged to ``<dst>.inprogress`` and renamed atomically once finished, so a
    half-finished copy from a previous crash is never mistaken for a complete
    one.

    Returns the local destination path that the caller should pass as
    ``CKPT_PATH`` to ``launch_server_env_sweep.sh``.
    """
    src = Path(ckpt_path.rstrip("/"))
    if not ckpt_path or not src.exists():
        typer.echo(
            f"[rank {rank}][prefetch] ckpt_path missing or empty, skip prefetch: {ckpt_path!r}"
        )
        return ckpt_path

    digest = hashlib.md5(str(src.resolve()).encode()).hexdigest()[:10]
    cache_root = Path(local_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    dst = cache_root / f"{src.name}_{digest}"
    sentinel = dst / ".copy_done"
    lock_path = cache_root / f"{src.name}_{digest}.lock"

    if sentinel.exists():
        typer.echo(
            f"[rank {rank}][prefetch] cache hit, skip copy: {dst} (src={src})"
        )
        return str(dst)

    typer.echo(
        f"[rank {rank}][prefetch] waiting for node-level flock: {lock_path} (src={src})"
    )
    t_wait_start = time.time()
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            t_wait_end = time.time()
            typer.echo(
                f"[rank {rank}][prefetch] acquired flock, waited {t_wait_end - t_wait_start:.1f}s"
            )
            if sentinel.exists():
                typer.echo(
                    f"[rank {rank}][prefetch] another worker on this node finished copying: {dst}"
                )
                return str(dst)

            tmp_dst = cache_root / f"{src.name}_{digest}.inprogress"
            if tmp_dst.exists():
                typer.echo(
                    f"[rank {rank}][prefetch] cleaning up previous incomplete in-progress directory: {tmp_dst}"
                )
                shutil.rmtree(tmp_dst, ignore_errors=True)
            if dst.exists():
                typer.echo(
                    f"[rank {rank}][prefetch] cleaning up previous leftover without sentinel: {dst}"
                )
                shutil.rmtree(dst, ignore_errors=True)

            try:
                src_size = subprocess.check_output(
                    ["du", "-sb", str(src)], text=True
                ).split()[0]
                typer.echo(
                    f"[rank {rank}][prefetch] source size: {int(src_size) / 1e9:.2f} GB"
                )
            except Exception as e:
                typer.echo(f"[rank {rank}][prefetch] failed to measure source size (ignored): {e}")

            typer.echo(
            f"[rank {rank}][prefetch] start copy: {src} -> {tmp_dst}"
            )
            t_copy_start = time.time()
            ret = subprocess.run(
                ["cp", "-a", f"{src}/.", str(tmp_dst)],
                check=False,
            )
            if ret.returncode != 0:
                shutil.rmtree(tmp_dst, ignore_errors=True)
                raise RuntimeError(
                    f"cp -a {src} -> {tmp_dst} failed: returncode={ret.returncode}"
                )
            tmp_dst.rename(dst)
            sentinel.touch()
            t_copy_end = time.time()
            typer.echo(
                f"[rank {rank}][prefetch] copy complete: {dst} took {t_copy_end - t_copy_start:.1f}s"
            )
            return str(dst)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Env list / horizon parsing (compatible with the static wrapper)
# ---------------------------------------------------------------------------


def parse_env_names(env_names_raw: str) -> list[str]:
    raw = env_names_raw.strip()
    if not raw:
        raise ValueError("env_names must not be empty")
    if "," in raw:
        env_names = [v.strip() for v in raw.split(",") if v.strip()]
    else:
        env_names = [v.strip() for v in raw.split() if v.strip()]
    if not env_names:
        raise ValueError("parsed env list is empty")
    return list(dict.fromkeys(env_names))


def load_env_costs_from_horizon_json(horizon_json: str) -> tuple[list[str], dict[str, int]]:
    path = Path(horizon_json)
    if not path.is_file():
        raise FileNotFoundError(f"horizon_json does not exist: {horizon_json}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    env_costs: dict[str, int] = {}
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = str(item.get("task_name", "")).strip()
            if not name:
                continue
            env_costs[name] = max(1, int(item.get("task_horizon", 0)))
    elif isinstance(payload, dict):
        if all(isinstance(v, (int, float)) for v in payload.values()):
            for k, v in payload.items():
                name = str(k).strip()
                if name:
                    env_costs[name] = max(1, int(v))
        elif isinstance(payload.get("tasks"), list):
            for item in payload.get("tasks", []):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("task_name", "")).strip()
                if not name:
                    continue
                env_costs[name] = max(1, int(item.get("task_horizon", 0)))
    if not env_costs:
        raise ValueError(f"horizon_json has no valid task_name/task_horizon: {horizon_json}")
    return list(env_costs.keys()), env_costs


def estimate_env_cost(env_name: str, default_horizon: int) -> int:
    if get_task_horizon is None:
        return max(1, int(default_horizon))
    clean = env_name.replace("robocasa/", "")
    try:
        return max(1, int(get_task_horizon(clean)))
    except Exception:
        return max(1, int(default_horizon))


# ---------------------------------------------------------------------------
# Mesa install (same behavior as static wrapper)
# ---------------------------------------------------------------------------


def ensure_mesa_osmesa_devel_installed() -> None:
    if os.environ.get("SKIP_MESA_INSTALL", "0").lower() in {"1", "true", "yes"}:
        typer.echo("[debug] SKIP_MESA_INSTALL=1, skip mesa-libOSMesa-devel auto-install")
        return
    local_process_rank = os.environ.get(
        "LOCAL_PROCESS_RANK", os.environ.get("LOCAL_RANK", "-1")
    )
    typer.echo(
        f"[debug] LOCAL_PROCESS_RANK={os.environ.get('LOCAL_PROCESS_RANK')}, "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
        f"effective_local_rank={local_process_rank}"
    )
    if local_process_rank == "0":
        typer.echo("[debug] rank0: running mesa-libOSMesa-devel install command")
        install_ret = os.system("sudo -n yum install -y mesa-libOSMesa-devel")
        if install_ret != 0:
            raise RuntimeError(
                "Auto-install of mesa-libOSMesa-devel failed. Ensure sudo is available, or install manually."
            )
        os.system("touch /tmp/flag_install_mesa_libosmesa_devel && sync")
        typer.echo("[debug] rank0 install complete, wrote /tmp/flag_install_mesa_libosmesa_devel")
    if "LOCAL_PROCESS_RANK" in os.environ:
        typer.echo("[debug] Detected LOCAL_PROCESS_RANK env var, waiting for rank0 install flag")
        while not os.path.exists("/tmp/flag_install_mesa_libosmesa_devel"):
            time.sleep(1)
        typer.echo("[debug] Install flag detected, continuing")


# ---------------------------------------------------------------------------
# Resume scanning: detect already-completed (env_name, ep_idx) from disk
# ---------------------------------------------------------------------------

def scan_completed_episodes(run_roots: list[str]) -> set[tuple[str, int]]:
    """Scan one or more run_root dirs for already-completed (env_name, ep_idx).

    A batch is considered "completed" if its ``robocasa_eval.json`` exists and
    is parseable. We treat every episode listed in ``episode_manifest`` as
    done regardless of ``success``, because re-running an already-evaluated
    episode would only waste compute (the metric for that ep is already
    determined). If ``episode_manifest`` is missing/empty, we fall back to
    inferring ep indices from the batch directory name pattern
    ``ep<ep_start:03d>_n<ep_count>_seed<seed>``.

    Layout assumed:
        <run_root>/envs/<env_name>/batches/<batch_dir>/robocasa_eval.json

    Returns a set of (env_name, ep_idx) tuples already covered.
    """
    completed: set[tuple[str, int]] = set()
    for run_root in run_roots:
        root = Path(run_root)
        envs_dir = root / "envs"
        if not envs_dir.is_dir():
            continue
        for env_dir in sorted(envs_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            env_name = env_dir.name
            batches_dir = env_dir / "batches"
            if not batches_dir.is_dir():
                continue
            for batch_dir in sorted(batches_dir.iterdir()):
                if not batch_dir.is_dir():
                    continue
                eval_json = batch_dir / "robocasa_eval.json"
                if not eval_json.is_file():
                    continue

                # Parse "ep<NNN>_n<M>_seed<S>" from the dir name to obtain the
                # GLOBAL ep_start and ep_count. This is the *source of truth*
                # for which ep_idx range this batch covers, because the
                # episode_manifest inside robocasa_eval.json stores
                # *batch-local* episode_index (0..M-1), not the global one.
                ep_start: int | None = None
                n_eps: int | None = None
                try:
                    name = batch_dir.name
                    ep_part, _, rest = name.partition("_n")
                    n_part, _, _ = rest.partition("_seed")
                    if ep_part.startswith("ep") and n_part:
                        ep_start = int(ep_part[2:])
                        n_eps = int(n_part)
                except Exception:
                    ep_start = None
                    n_eps = None

                covered: set[int] = set()

                # Path 1 (preferred): trust the dir name. As soon as a batch
                # has a robocasa_eval.json on disk, the underlying client has
                # reset env.reset(seed=base_seed+ep) for every ep in
                # [ep_start, ep_start+n_eps), even if some eps failed -- the
                # JSON's episode_manifest is the proof. So mark all of them.
                if ep_start is not None and n_eps is not None and n_eps > 0:
                    for i in range(n_eps):
                        covered.add(ep_start + i)

                # Path 2 (cross-check / fallback): also consult the JSON's
                # episode_manifest. Convert each entry's local
                # episode_index -> global by adding ep_start. This catches
                # the unlikely case where the dir name is malformed but the
                # JSON is valid, and also gives us robustness against partial
                # batches (e.g. a future client that only writes the eps it
                # actually finished).
                try:
                    payload = json.loads(eval_json.read_text(encoding="utf-8"))
                    manifest = payload.get("episode_manifest") or []
                    base = ep_start if ep_start is not None else 0
                    for item in manifest:
                        if not isinstance(item, dict):
                            continue
                        if "episode_index" in item:
                            try:
                                covered.add(base + int(item["episode_index"]))
                            except (TypeError, ValueError):
                                pass
                except Exception as e:  # noqa: BLE001 - best-effort parse
                    typer.echo(
                        f"[resume] WARN failed to parse {eval_json}: {e}; "
                        "relying on dir-name fallback only"
                    )

                if not covered:
                    typer.echo(
                        f"[resume] WARN could not infer ep range for "
                        f"{batch_dir}, skipping"
                    )
                    continue
                for ep_idx in covered:
                    completed.add((env_name, ep_idx))
    return completed

# ---------------------------------------------------------------------------
# Shared NAS task pool
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """One atomic unit of work: a single (env_name, ep_idx).

    ``retries_left`` is a *task-bound* retry budget, persisted into the shared
    task_queue.tsv. Putting the budget on the task (not in a per-worker dict)
    is what prevents the tail-end "same task forever" loop: when a failed
    task is re-queued, whichever worker eventually picks it up sees the
    decremented budget and will not re-queue again once it hits zero.
    """

    neg_priority: int  # smaller pops first; we use -horizon so long tasks lead.
    env_name: str
    ep_idx: int
    seed: int
    insertion_order: int
    retries_left: int = 0

    def to_line(self) -> str:
        return (
            f"{self.neg_priority}\t{self.env_name}\t{self.ep_idx}\t"
            f"{self.seed}\t{self.insertion_order}\t{self.retries_left}\n"
        )

    @classmethod
    def from_line(cls, line: str) -> "Task":
        parts = line.rstrip("\n").split("\t")
        # Backward-compatible: queues written by older versions only have 5
        # columns; treat such tasks as having 0 retries left.
        if len(parts) == 5:
            return cls(
                neg_priority=int(parts[0]),
                env_name=parts[1],
                ep_idx=int(parts[2]),
                seed=int(parts[3]),
                insertion_order=int(parts[4]),
                retries_left=0,
            )
        if len(parts) == 6:
            return cls(
                neg_priority=int(parts[0]),
                env_name=parts[1],
                ep_idx=int(parts[2]),
                seed=int(parts[3]),
                insertion_order=int(parts[4]),
                retries_left=int(parts[5]),
            )
        raise ValueError(f"Malformed task line: {line!r}")


class SharedPool:
    """File-locked TSV queue stored on NAS. Safe across nodes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.queue_path = root / "task_queue.tsv"
        self.lock_path = root / "task_queue.lock"
        self.init_flag = root / "INITIALIZED"
        self.progress_path = root / "progress.tsv"
        self.progress_lock = root / "progress.lock"
        # Rebuild barrier: rank-0 creates this *before* it touches the queue
        # and removes it only after the new INITIALIZED is on disk. Non-rank-0
        # workers refuse to proceed while this barrier exists. This closes
        # a TOCTOU window where, in resume mode, the old INITIALIZED + old
        # task_queue.tsv are still present when non-rank-0 workers wake up,
        # causing them to consume stale tasks before rank-0 had a chance to
        # rebuild the pool.
        self.rebuild_barrier = root / "REBUILDING"

    # -- eager barrier plant (rank-0 only) ----------------------------------

    def plant_rebuild_barrier_eagerly(self, rank: int, reason: str = "") -> None:
        """Plant the REBUILDING barrier as early as possible from rank-0.

        Why this exists: ``initialize_if_rank0`` plants the barrier *inside*
        its own body, but the caller typically does slow work *before*
        invoking it (e.g. scanning thousands of robocasa_eval.json on NAS via
        :func:`scan_completed_episodes`). During that pre-init window, the
        old INITIALIZED + task_queue.tsv from a previous interrupted run
        are still on disk, and non-rank-0 workers will happily start
        consuming stale tasks.

        The fix is to call this method *immediately* on rank-0 startup,
        before any scanning, so non-rank-0 workers (whose wait loop checks
        for the barrier's presence) are blocked from the very first second.
        ``initialize_if_rank0`` will then re-plant the barrier idempotently
        (overwrite is fine) and remove it once the queue has been rebuilt.

        No-op for non-rank-0 workers, so callers can invoke this
        unconditionally.
        """
        if rank != 0:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.rebuild_barrier.write_text(
                json.dumps({
                    "pid": os.getpid(),
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": reason or "rank0_eager_plant_pre_scan",
                    "stage": "pre_scan",
                }),
                encoding="utf-8",
            )
            typer.echo(
                f"[rank 0] planted REBUILDING barrier eagerly "
                f"({reason or 'pre_scan'}) at {self.rebuild_barrier}"
            )
        except Exception as e:  # noqa: BLE001 - best-effort, never fatal.
            typer.echo(
                f"[rank 0] WARN failed to plant rebuild barrier eagerly: {e}"
            )

    # -- initialization (rank-0 only) ---------------------------------------

    def initialize_if_rank0(
        self,
        rank: int,
        tasks: list[Task],
        wait_seconds: int,
        completed_tasks: set[tuple[str, int]] | None = None,
        force_rebuild: bool = False,
    ) -> None:
        """Initialize the shared task queue.

        - completed_tasks: any (env_name, ep_idx) here is filtered out from
          the queue so the new run only evaluates remaining episodes.
        - force_rebuild: when True, an existing INITIALIZED flag plus the
          previous task_queue.tsv are wiped before re-initialization. This
          is what makes "re-export RUN_TAG and resubmit" actually re-plan
          the task pool against the new completed_tasks snapshot, instead
          of silently reusing a stale (possibly empty) queue from the
          prior run. progress.tsv is preserved (and appended to) so
          historical worker activity stays auditable.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)
        self.progress_lock.touch(exist_ok=True)
        completed_tasks = completed_tasks or set()
        if rank == 0:
            # Plant the rebuild barrier *before* taking the lock so any
            # non-rank-0 worker that wakes up early (and only sees the old
            # INITIALIZED + queue) is forced to wait. We always plant it on
            # every rank-0 init, even on a fresh run, to keep the wait
            # protocol uniform; we always remove it at the end.
            try:
                self.rebuild_barrier.write_text(
                    json.dumps({
                        "pid": os.getpid(),
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "force_rebuild": bool(force_rebuild),
                    }),
                    encoding="utf-8",
                )
            except Exception as e:  # noqa: BLE001
                typer.echo(f"[rank 0] WARN failed to plant rebuild barrier: {e}")
            with self.lock_path.open("a+") as lock_fp:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
                try:
                    if self.init_flag.exists():
                        if force_rebuild:
                            typer.echo(
                                f"[rank 0][resume] force_rebuild=True; wiping "
                                f"stale INITIALIZED + task_queue.tsv at {self.root}"
                            )
                            try:
                                self.init_flag.unlink()
                            except FileNotFoundError:
                                pass
                            try:
                                self.queue_path.unlink()
                            except FileNotFoundError:
                                pass
                        else:
                            typer.echo(
                                f"[rank 0] task pool already initialized at {self.queue_path}"
                            )
                            try:
                                self.rebuild_barrier.unlink()
                            except FileNotFoundError:
                                pass
                            return
                    original_total = len(tasks)
                    if completed_tasks:
                        filtered = [
                            t for t in tasks
                            if (t.env_name, t.ep_idx) not in completed_tasks
                        ]
                        skipped = original_total - len(filtered)
                        typer.echo(
                            f"[rank 0][resume] skipping {skipped} already-completed "
                            f"episodes; {len(filtered)}/{original_total} remain"
                        )
                        tasks = filtered
                    # Sort so that:
                    #   1. Lower neg_priority (= longer horizon) comes first.
                    #   2. Within the same neg_priority, tasks with the same
                    #      env_name are grouped together (alphabetical).
                    #   3. Within the same (neg_priority, env_name), ep_idx
                    #      is ascending so pop_batch can greedily extend
                    #      contiguous same-env batches as large as possible.
                    tasks_sorted = sorted(
                        tasks,
                        key=lambda t: (t.neg_priority, t.env_name, t.ep_idx),
                    )
                    with self.queue_path.open("w", encoding="utf-8") as f:
                        for t in tasks_sorted:
                            f.write(t.to_line())
                    # Append-mode for progress.tsv on resume so old history
                    # is preserved; write the header only when the file is
                    # brand-new / empty.
                    need_header = (
                        not self.progress_path.exists()
                        or self.progress_path.stat().st_size == 0
                    )
                    with self.progress_path.open("a", encoding="utf-8") as f:
                        if need_header:
                            f.write(
                                "timestamp\trank\thostname\tenv_name\tep_start\tep_count\t"
                                "status\twall_seconds\tlog_file\n"
                            )
                    self.init_flag.write_text(
                        json.dumps(
                            {
                                "total_tasks": len(tasks_sorted),
                                "original_total": original_total,
                                "skipped_resume": original_total - len(tasks_sorted),
                                "force_rebuild": bool(force_rebuild),
                                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        ),
                        encoding="utf-8",
                    )
                    typer.echo(
                        f"[rank 0] initialized task pool with {len(tasks_sorted)} tasks "
                        f"-> {self.queue_path}"
                    )
                finally:
                    # Drop the rebuild barrier *before* releasing the lock so
                    # any non-rank-0 waiter that immediately grabs the lock
                    # next sees a fully consistent state (no barrier + new
                    # INITIALIZED + new queue).
                    try:
                        self.rebuild_barrier.unlink()
                    except FileNotFoundError:
                        pass
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
        else:
            # Non-rank-0 wait protocol:
            #   1. Spin until the rebuild barrier is gone AND the new
            #      INITIALIZED + queue files exist.
            #   2. Even after that, briefly grab a SHARED flock to make sure
            #      rank-0 has fully released the EXCLUSIVE lock (i.e. the
            #      rebuild critical section has finished). Only then is it
            #      safe to start consuming.
            deadline = time.monotonic() + wait_seconds
            while time.monotonic() < deadline:
                if (
                    not self.rebuild_barrier.exists()
                    and self.init_flag.exists()
                    and self.queue_path.exists()
                ):
                    # Confirm rank-0 has released the EX lock by acquiring
                    # a non-blocking SH lock. If it's still held EX, retry.
                    try:
                        with self.lock_path.open("a+") as lock_fp:
                            fcntl.flock(
                                lock_fp.fileno(),
                                fcntl.LOCK_SH | fcntl.LOCK_NB,
                            )
                            try:
                                # Re-verify under the lock (defense in
                                # depth: rank-0 might have re-planted the
                                # barrier for a follow-up rebuild on a
                                # very fast resume cycle).
                                ok = (
                                    not self.rebuild_barrier.exists()
                                    and self.init_flag.exists()
                                    and self.queue_path.exists()
                                )
                            finally:
                                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
                        if ok:
                            typer.echo(
                                f"[rank {rank}] detected initialized pool"
                            )
                            return
                    except OSError:
                        # SH lock not available -> rank-0 still EX-locked.
                        pass
                time.sleep(1)
            raise RuntimeError(
                f"[rank {rank}] timed out waiting for rank-0 to initialize task pool"
            )

    # -- pop a contiguous same-env batch ------------------------------------

    def pop_batch(self, batch_max: int) -> list[Task]:
        """Pop the head task; greedily extend with same-env contiguous ep_idx tasks.

        We require the batch to have:
          - identical ``env_name``,
          - strictly contiguous, ascending ``ep_idx`` (head, head+1, ...),
          - identical ``seed - ep_idx`` base offset,
        so that invoking the underlying client with ``NUM_EPISODES=batch_size``
        and ``SEED=base_seed`` reproduces the exact same trajectories the static
        wrapper would produce (which relies on ``env.reset(seed=seed+ep)``).
        """
        if batch_max < 1:
            batch_max = 1
        with self.lock_path.open("a+") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                if not self.queue_path.exists():
                    return []
                with self.queue_path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                if not lines:
                    return []

                head = Task.from_line(lines[0])
                taken = [head]
                consumed = 1
                base_seed_offset = head.seed - head.ep_idx
                expected_ep = head.ep_idx + 1
                for line in lines[1:]:
                    if len(taken) >= batch_max:
                        break
                    cand = Task.from_line(line)
                    if cand.env_name != head.env_name:
                        break
                    if cand.ep_idx != expected_ep:
                        break
                    if (cand.seed - cand.ep_idx) != base_seed_offset:
                        break
                    taken.append(cand)
                    expected_ep += 1
                    consumed += 1

                rest = lines[consumed:]
                tmp_path = self.queue_path.with_suffix(self.queue_path.suffix + ".tmp")
                # Defensive: NAS may race rank-0's mkdir; ensure self.root exists
                # before any write on a .tmp sibling (FileNotFoundError otherwise).
                self.root.mkdir(parents=True, exist_ok=True)
                with tmp_path.open("w", encoding="utf-8") as f:
                    f.writelines(rest)
                os.replace(tmp_path, self.queue_path)
                return taken
            finally:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    def push_back(self, tasks: list[Task]) -> None:
        if not tasks:
            return
        # Defensive: ensure self.root exists before opening the lock file
        # (lock_path.open("a+") would FileNotFoundError otherwise on NAS).
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                with self.queue_path.open("a", encoding="utf-8") as f:
                    for t in tasks:
                        f.write(t.to_line())
            finally:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    def push_back_unique(self, tasks: list[Task]) -> int:
        """Append tasks to the queue tail, but skip any (env_name, ep_idx)
        that is already present in the queue.

        This is the per-pool deduplication guard against the tail-end
        "multi-worker re-queue storm": several workers can pop the same
        (env, ep) on overlapping retry cycles (after a transient infra
        hiccup), each fail, and each call ``push_back``. Without dedup,
        the queue grows unboundedly with copies of the same task and the
        worker swarm thrashes on it. With dedup, only one copy survives
        and the budget on that single copy decides when to give up.

        Returns the number of tasks actually appended.
        """
        if not tasks:
            return 0
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                existing: set[tuple[str, int]] = set()
                if self.queue_path.exists():
                    with self.queue_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                t_existing = Task.from_line(line)
                            except ValueError:
                                continue
                            existing.add(
                                (t_existing.env_name, t_existing.ep_idx)
                            )
                appended = 0
                with self.queue_path.open("a", encoding="utf-8") as f:
                    for t in tasks:
                        if (t.env_name, t.ep_idx) in existing:
                            continue
                        f.write(t.to_line())
                        existing.add((t.env_name, t.ep_idx))
                        appended += 1
                return appended
            finally:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    def remaining_count(self) -> int:
        try:
            with self.lock_path.open("a+") as lock_fp:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_SH)
                try:
                    if not self.queue_path.exists():
                        return 0
                    with self.queue_path.open("r", encoding="utf-8") as f:
                        return sum(1 for _ in f)
                finally:
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
        except FileNotFoundError:
            return 0

    # -- progress logging ---------------------------------------------------

    def log_progress(
        self,
        rank: int,
        env_name: str,
        ep_start: int,
        ep_count: int,
        status: str,
        wall_seconds: float,
        log_file: str,
    ) -> None:
        line = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{rank}\t"
            f"{socket.gethostname()}\t{env_name}\t{ep_start}\t{ep_count}\t"
            f"{status}\t{wall_seconds:.2f}\t{log_file}\n"
        )
        with self.progress_lock.open("a+") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                with self.progress_path.open("a", encoding="utf-8") as f:
                    f.write(line)
            finally:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------


def build_tasks(
    env_names: list[str],
    num_episodes: int,
    base_seed: int,
    horizons: dict[str, int],
    default_horizon: int,
    max_retries: int = 0,
) -> list[Task]:
    tasks: list[Task] = []
    counter = 0
    initial_budget = max(0, int(max_retries))
    for env_name in env_names:
        horizon = int(horizons.get(env_name, default_horizon))
        if horizon < 1:
            horizon = 1
        for ep in range(num_episodes):
            tasks.append(
                Task(
                    neg_priority=-horizon,
                    env_name=env_name,
                    ep_idx=ep,
                    seed=base_seed + ep,
                    insertion_order=counter,
                    retries_left=initial_budget,
                )
            )
            counter += 1

    # Within the same priority bucket, interleave envs so that no single env's
    # entire episode list sits at the head of the queue. This gives the dynamic
    # scheduler more opportunities to load-balance. We preserve ascending
    # ep_idx within each env so contiguous batching at pop time still works.
    by_priority: dict[int, list[Task]] = defaultdict(list)
    for t in tasks:
        by_priority[t.neg_priority].append(t)

    flat: list[Task] = []
    for neg_p in sorted(by_priority.keys()):
        bucket = by_priority[neg_p]
        # Group by env, each group already in ep_idx ascending order.
        groups: dict[str, list[Task]] = defaultdict(list)
        for t in bucket:
            groups[t.env_name].append(t)
        # Interleave one ep_idx at a time across envs (round-robin).
        env_iters = {name: iter(group) for name, group in groups.items()}
        active = list(groups.keys())
        # Stable env order for reproducibility: by env_name asc.
        active.sort()
        while active:
            next_active: list[str] = []
            for name in active:
                try:
                    flat.append(next(env_iters[name]))
                    next_active.append(name)
                except StopIteration:
                    pass
            active = next_active

    for i, t in enumerate(flat):
        t.insertion_order = i
    return flat

# ---------------------------------------------------------------------------
# Worker execution: invoke launch_server_env_sweep.sh per batch
# ---------------------------------------------------------------------------

def _episode_already_completed(
    run_root: str, env_name: str, ep_idx: int
) -> bool:
    """Cheap on-disk check: has any batch in this env already produced a
    robocasa_eval.json that covers ``ep_idx``?

    We answer by inspecting batch directory names under
    ``<run_root>/envs/<env_name>/batches/``: a directory named
    ``ep<NNN>_n<M>_seed<S>`` covers the global ep range
    ``[NNN, NNN+M)`` if its ``robocasa_eval.json`` exists. This mirrors the
    "Path 1" logic in :func:`scan_completed_episodes` and is the source of
    truth at runtime: as soon as the underlying client finishes a batch,
    that JSON appears, so any ep in that range should not be rerun.

    Performance: this walks one env's batches dir per call, which is small
    (tens to low hundreds of entries). Returning early on first match.
    Failures (NAS unavailable, parse error) are conservatively treated as
    "not completed" so we err on running rather than silently skipping.
    """
    env_dir = Path(run_root) / "envs" / env_name / "batches"
    if not env_dir.is_dir():
        return False
    try:
        for batch_dir in env_dir.iterdir():
            if not batch_dir.is_dir():
                continue
            name = batch_dir.name
            if not name.startswith("ep"):
                continue
            try:
                ep_part, _, rest = name.partition("_n")
                n_part, _, _ = rest.partition("_seed")
                bd_ep_start = int(ep_part[2:])
                bd_n = int(n_part)
            except (ValueError, IndexError):
                continue
            if not (bd_ep_start <= ep_idx < bd_ep_start + bd_n):
                continue
            if (batch_dir / "robocasa_eval.json").is_file():
                return True
    except OSError:
        return False
    return False

def run_one_batch(
    env_name: str,
    base_seed_for_first_ep: int,
    ep_start: int,
    ep_count: int,
    rank: int,
    cmd_env: dict[str, str],
    run_root: str,
    wrapper_log_dir: Path,
) -> dict:
    """Run a contiguous batch of episodes for one env via launch_server_env_sweep.sh.

    The underlying client iterates ``ep in range(ep_count)`` and resets with
    ``seed = SEED_arg + ep``. To reproduce the original ``ep_idx`` trajectory,
    we set ``SEED_arg = base_seed_for_first_ep``; the client will then run
    ``ep'=0..ep_count-1`` resetting with ``base_seed_for_first_ep + ep'``,
    which equals the original ``BASE + ep_start + ep' = BASE + ep_idx``.
    """
    env_root = Path(run_root) / "envs" / env_name
    batch_dir = env_root / "batches" / f"ep{ep_start:03d}_n{ep_count}_seed{base_seed_for_first_ep}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    save_root = batch_dir / "save_root"
    save_root.mkdir(parents=True, exist_ok=True)
    save_video_dir = batch_dir / "videos"
    save_video_dir.mkdir(parents=True, exist_ok=True)
    save_json = batch_dir / "robocasa_eval.json"
    server_log = batch_dir / "server.log"
    log_file = wrapper_log_dir / f"{env_name}__ep{ep_start:03d}_n{ep_count}.log"

    env = os.environ.copy()
    env.update(cmd_env)
    env["ENV_NAME"] = env_name
    env["SEED"] = str(base_seed_for_first_ep)
    env["NUM_EPISODES"] = str(ep_count)
    env["RESULT_DIR"] = str(batch_dir)
    env["SAVE_ROOT"] = str(save_root)
    env["SAVE_VIDEO_DIR"] = str(save_video_dir)
    env["SAVE_JSON"] = str(save_json)
    env["SERVER_LOG"] = str(server_log)

    cmd = ["bash", "evaluation/robocasa/launch_server_env_sweep.sh"]
    start = time.monotonic()
    with log_file.open("w", encoding="utf-8") as f:
        f.write(
            f"# rank={rank} env={env_name} ep_start={ep_start} ep_count={ep_count} "
            f"seed={base_seed_for_first_ep}\n"
        )
        f.flush()
        ret = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    wall = time.monotonic() - start
    return {
        "env_name": env_name,
        "ep_start": ep_start,
        "ep_count": ep_count,
        "seed": base_seed_for_first_ep,
        "returncode": int(ret.returncode),
        "wall_seconds": wall,
        "log_file": str(log_file),
        "save_json": str(save_json),
        "save_video_dir": str(save_video_dir),
        "server_log": str(server_log),
        "batch_dir": str(batch_dir),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    worker_count: int = 16,
    env_names: str = (
        "CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug "
        "NavigateKitchen OpenCabinet OpenDrawer OpenStandMixerHead "
        "PickPlaceCounterToCabinet PickPlaceCounterToStove PickPlaceDrawerToCounter "
        "PickPlaceSinkToCounter PickPlaceToasterToCounter SlideDishwasherRack "
        "TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet"
    ),
    seed: int = 0,
    ckpt_path: str = "",
    start_port: int = 29056,
    master_port_base: int = 29100,
    config_name: str = "robocasa_train",
    gpu_id: str = "0",
    wait_timeout: int = 1200,
    host: str = "127.0.0.1",
    server_python: str = "",
    client_python: str = "python",
    split: str = "target",
    num_episodes: int = 50,
    max_steps: int = 500,
    debug_actions: str = "0",
    horizon_json: str = "",
    run_root: str = "outputs/robocasa_eval_env_sweep_dynamic/default_run",
    shared_pool_root: str = "",
    batch_max: int = 5,
    init_wait_seconds: int = 600,
    max_retries: int = 0,
    resume_run_root: str = "",
    resume_from_self: int = 1,
    horizon_multiplier: float = 1.0,
):
    """Dynamic episode-level distributed RoboCasa eval wrapper.

    --shared-pool-root must point at an NAS-shared directory so that all
    Workers see the same task queue and lock files.
    --batch-max bounds how many same-env contiguous episodes a worker may
    pull in one shot; larger values amortize the server-restart cost but
    reduce dynamic load balancing granularity. Default 5.

    Resume / break-point continuation:
      * --resume-from-self 1 (default): rank-0 scans the *current* ``run_root``
        for any pre-existing ``robocasa_eval.json`` files and skips the
        matching (env_name, ep_idx) pairs. Set 0 to disable.
      * --resume-run-root <path>: optionally also scan an *additional* (older)
        run_root for completed episodes; their (env_name, ep_idx) pairs are
        also removed from the new task pool. Use this when you start a fresh
        run_tag but want to inherit progress from a previous interrupted run.
    """
    ensure_mesa_osmesa_devel_installed()
    rank = get_worker_rank()

    # 1) Resolve env list and per-env horizon (used as priority weight).
    if horizon_json.strip():
        parsed_env_names, horizons = load_env_costs_from_horizon_json(horizon_json)
    else:
        parsed_env_names = parse_env_names(env_names)
        horizons = {
            name: estimate_env_cost(name, default_horizon=max_steps)
            for name in parsed_env_names
        }

    # 2) Resolve the shared NAS pool location (default: under run_root).
    if not shared_pool_root.strip():
        shared_pool_root = str(Path(run_root) / "shared_pool")
    pool = SharedPool(Path(shared_pool_root))

    # CRITICAL: rank-0 must plant the REBUILDING barrier *before* doing any
    # slow work (especially scan_completed_episodes, which walks NAS and can
    # take 30+ seconds at scale). If we wait until initialize_if_rank0 to
    # plant the barrier, non-rank-0 workers spinning in the wait loop will
    # see "no barrier + stale INITIALIZED + stale queue" from the previous
    # interrupted run and start consuming stale tasks immediately, causing
    # the same (env, ep_idx) to be evaluated by both an old in-flight worker
    # and a new worker -- with the second producer overwriting the first
    # one's robocasa_eval.json. Eager plant closes that window.
    pool.plant_rebuild_barrier_eagerly(rank, reason="rank0_main_pre_scan")

    tasks = build_tasks(
        env_names=parsed_env_names,
        num_episodes=num_episodes,
        base_seed=int(seed),
        horizons=horizons,
        default_horizon=max_steps,
        max_retries=int(max_retries),
    )

    # Resume scan (rank-0 only): figure out which (env, ep_idx) pairs are
    # already covered by an existing robocasa_eval.json on disk, so we do not
    # re-evaluate them. Non-rank-0 workers skip scanning to avoid touching
    # NAS unnecessarily; they pick up whatever pool rank-0 writes.
    completed_tasks: set[tuple[str, int]] = set()
    if rank == 0:
        scan_roots: list[str] = []
        if resume_from_self:
            scan_roots.append(run_root)
        if resume_run_root.strip():
            scan_roots.append(resume_run_root.strip())
        # Deduplicate while preserving order.
        scan_roots = list(dict.fromkeys(scan_roots))
        if scan_roots:
            typer.echo(
                f"[rank 0][resume] scanning for completed episodes in: {scan_roots}"
            )
            completed_tasks = scan_completed_episodes(scan_roots)
            typer.echo(
                f"[rank 0][resume] found {len(completed_tasks)} already-completed "
                f"(env_name, ep_idx) pairs"
            )

    # In resume mode, ask the pool to wipe any stale INITIALIZED flag /
    # task_queue.tsv from a previous interrupted run so the new run actually
    # re-plans against the latest completed_tasks snapshot.
    is_resume_mode = bool(resume_from_self) or bool(resume_run_root.strip())
    pool.initialize_if_rank0(
        rank=rank,
        tasks=tasks,
        wait_seconds=init_wait_seconds,
        completed_tasks=completed_tasks,
        force_rebuild=is_resume_mode,
    )

    # 3) Per-worker output area.
    worker_output = Path(run_root) / "workers" / f"worker_{rank}"
    worker_output.mkdir(parents=True, exist_ok=True)
    wrapper_log_dir = worker_output / "logs"
    wrapper_log_dir.mkdir(parents=True, exist_ok=True)

    effective_gpu = resolve_effective_gpu_id(gpu_id, rank)

    # Skip prefetch to /tmp for local eval — read ckpt directly from original path.
    effective_ckpt_path = ckpt_path

    cmd_env = {
        "CKPT_PATH": effective_ckpt_path,
        "START_PORT": str(start_port + rank),
        "MASTER_PORT": str(master_port_base + rank),
        "CONFIG_NAME": config_name,
        "GPU_ID": effective_gpu,
        "WAIT_TIMEOUT": str(wait_timeout),
        "HOST": host,
        "SERVER_PYTHON": server_python,
        "CLIENT_PYTHON": client_python,
        "SPLIT": split,
        "MAX_STEPS": str(max_steps),
        "DEBUG_ACTIONS": str(debug_actions),
        "HORIZON_MULTIPLIER": str(horizon_multiplier),
    }

    typer.echo("=" * 60)
    typer.echo(f"RoboCasa ENV Sweep DYNAMIC Worker {rank}/{worker_count}")
    typer.echo(f"shared_pool_root: {shared_pool_root}")
    typer.echo(f"run_root        : {run_root}")
    typer.echo(f"total_envs      : {len(parsed_env_names)}")
    typer.echo(f"total_tasks     : {len(tasks)}")
    typer.echo(f"batch_max       : {batch_max}")
    typer.echo(f"gpu_id (cli)    : {gpu_id}")
    typer.echo(f"gpu_id (effective): {effective_gpu}")
    typer.echo(f"start_port(this): {start_port + rank}")
    typer.echo(f"master_port(this): {master_port_base + rank}")
    typer.echo(f"ckpt_path (src) : {ckpt_path}")
    typer.echo(f"ckpt_path (used): {effective_ckpt_path}")
    typer.echo("=" * 60)

    # 4) Consume tasks until the pool is empty.
    completed_batches: list[dict] = []
    failed_batches: list[dict] = []
    total_episodes_done = 0

    # Episodes already produced on disk by *this* worker (or visible via NAS
    # sync from peer workers). Used at pop time to short-circuit batches whose
    # robocasa_eval.json already exists -- this happens at tail-end when
    # multiple workers may have re-queued the same failed task before any
    # of them noticed a peer succeeded on a fresh attempt. Without this
    # check, the same (env, ep) can ping-pong through the queue forever.
    locally_seen_completed: set[tuple[str, int]] = set()

    while True:
        batch = pool.pop_batch(batch_max=batch_max)
        if not batch:
            typer.echo(f"[rank {rank}] no more tasks, exiting consumer loop")
            break

        env_name = batch[0].env_name
        ep_start = batch[0].ep_idx
        ep_count = len(batch)
        base_seed_for_first_ep = batch[0].seed
        log_path = wrapper_log_dir / f"{env_name}__ep{ep_start:03d}_n{ep_count}.log"

        # Tail-end safety: if every (env, ep) in this batch already has a
        # robocasa_eval.json on NAS, skip running the server entirely. This
        # is the second line of defense against the "same task forever"
        # loop; the primary defense is the per-task retries_left budget.
        all_already_done = True
        for t in batch:
            key = (t.env_name, t.ep_idx)
            if key in locally_seen_completed:
                continue
            if _episode_already_completed(run_root, t.env_name, t.ep_idx):
                locally_seen_completed.add(key)
                continue
            all_already_done = False
            break
        if all_already_done:
            typer.echo(
                f"[rank {rank}] SKIP env={env_name} ep_start={ep_start} "
                f"ep_count={ep_count} (already completed on disk)"
            )
            pool.log_progress(
                rank, env_name, ep_start, ep_count,
                "skip_already_done", 0.0, str(log_path),
            )
            continue

        typer.echo(
            f"[rank {rank}] start env={env_name} ep_start={ep_start} "
            f"ep_count={ep_count} seed={base_seed_for_first_ep} "
            f"remaining_in_pool={pool.remaining_count()}"
        )
        pool.log_progress(
            rank, env_name, ep_start, ep_count, "start", 0.0, str(log_path)
        )
        item = run_one_batch(
            env_name=env_name,
            base_seed_for_first_ep=base_seed_for_first_ep,
            ep_start=ep_start,
            ep_count=ep_count,
            rank=rank,
            cmd_env=cmd_env,
            run_root=run_root,
            wrapper_log_dir=wrapper_log_dir,
        )
        status = "done" if item["returncode"] == 0 else "failed"
        pool.log_progress(
            rank, env_name, ep_start, ep_count, status,
            float(item["wall_seconds"]), str(log_path),
        )
        if item["returncode"] == 0:
            completed_batches.append(item)
            total_episodes_done += ep_count
            for t in batch:
                locally_seen_completed.add((t.env_name, t.ep_idx))
            typer.echo(
                f"[rank {rank}] DONE env={env_name} ep_start={ep_start} "
                f"ep_count={ep_count} wall={item['wall_seconds']:.1f}s"
            )
        else:
            failed_batches.append(item)
            typer.echo(
                f"[rank {rank}] FAIL env={env_name} ep_start={ep_start} "
                f"ep_count={ep_count} rc={item['returncode']}"
            )
            # Retry: split the failed batch back into per-episode tasks so
            # any worker can pick them up next time. The retry budget lives
            # on the Task itself (persisted in task_queue.tsv), so we cannot
            # re-queue forever even if peer workers also touch this (env, ep).
            requeue: list[Task] = []
            for t in batch:
                # Defensive: if disk shows this ep is already done (peer
                # worker raced ahead and finished it on another attempt),
                # don't re-queue.
                if _episode_already_completed(run_root, t.env_name, t.ep_idx):
                    locally_seen_completed.add((t.env_name, t.ep_idx))
                    continue
                if t.retries_left > 0:
                    requeue.append(
                        Task(
                            neg_priority=t.neg_priority,
                            env_name=t.env_name,
                            ep_idx=t.ep_idx,
                            seed=t.seed,
                            insertion_order=t.insertion_order,
                            retries_left=t.retries_left - 1,
                        )
                    )
                else:
                    typer.echo(
                        f"[rank {rank}] GIVEUP env={t.env_name} "
                        f"ep_idx={t.ep_idx} (retries exhausted)"
                    )
            if requeue:
                pool.push_back_unique(requeue)

    # 5) Worker summary on NAS for downstream aggregation.
    summary = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "worker_count": worker_count,
        "run_root": run_root,
        "shared_pool_root": shared_pool_root,
        "total_episodes_done": total_episodes_done,
        "completed_batches": completed_batches,
        "failed_batches": failed_batches,
        "env_names": parsed_env_names,
        "horizons": horizons,
        "batch_max": batch_max,
    }
    summary_path = worker_output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(f"[rank {rank}] worker summary: {summary_path}")

    if failed_batches:
        # Keep non-zero exit semantics for failed workers.
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
