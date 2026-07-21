import sys as _sys
import os as _os
print(f"[eval_policy_client.py] === STARTUP === pid={_os.getpid()} python={_sys.executable}", flush=True)
print(f"[eval_policy_client.py] argv: {_sys.argv}", flush=True)
print(f"[eval_policy_client.py] PYTHONPATH={_os.environ.get('PYTHONPATH', '<unset>')}", flush=True)
print(f"[eval_policy_client.py] LIBERO_CONFIG_PATH={_os.environ.get('LIBERO_CONFIG_PATH', '<unset>')}", flush=True)

if not _os.environ.get("LIBERO_CONFIG_PATH"):
    _default_cfg = _os.path.expanduser("~/.libero")
    _os.makedirs(_default_cfg, exist_ok=True)
    _os.environ["LIBERO_CONFIG_PATH"] = _default_cfg
    print(f"[eval_policy_client.py] LIBERO_CONFIG_PATH not set, defaulting to {_default_cfg}", flush=True)

# stdin fallback
import io as _io
_sys.stdin = _io.StringIO("N\n" * 10)


def set_seed(seed: int):
    """Fix all random seeds for reproducibility.

    Sets seeds for Python's random, NumPy, PyTorch (CPU + CUDA), and
    configures CuDNN / MIopen deterministic mode where available.
    Should be called **before** any model loading or env creation.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms (may reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Set via env var so that sub-processes / lazy imports also pick it up
    _os.environ["PYTHONHASHSEED"] = str(seed)
    # Also export for LIBERO/mujoco env reset (read by init_single_env)
    _os.environ["_LIBERO_EVAL_SEED"] = str(seed)
    print(f"[eval_policy_client.py] ✅ Global seed set to {seed}", flush=True)


print("[eval_policy_client.py] importing numpy...", flush=True)
import numpy as np
import typer
import json


def _disable_robosuite_numba_cache():
    """Patch robosuite macros before env imports to avoid numba cache locator failures."""
    import importlib.util

    spec = importlib.util.find_spec("robosuite")
    if spec is None or spec.origin is None:
        return
    pkg_dir = _os.path.dirname(_os.path.abspath(spec.origin))
    macros_path = _os.path.join(pkg_dir, "macros.py")
    private_path = _os.path.join(pkg_dir, "macros_private.py")
    source_path = private_path if _os.path.isfile(private_path) else macros_path
    if not _os.path.isfile(source_path):
        return

    for module_name in ("robosuite.macros", "robosuite.macros_private"):
        module_spec = importlib.util.spec_from_file_location(module_name, source_path)
        if module_spec is None or module_spec.loader is None:
            continue
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.CACHE_NUMBA = False
        _sys.modules[module_name] = module
    print("[eval_policy_client.py] robosuite numba cache disabled", flush=True)


_disable_robosuite_numba_cache()


print("[eval_policy_client.py] importing libero package...", flush=True)
import libero as _libero_pkg
print(f"[eval_policy_client.py] libero package path: {getattr(_libero_pkg, '__file__', None) or _libero_pkg.__path__}", flush=True)
print("[eval_policy_client.py] importing libero.benchmark...", flush=True)
from libero.libero import benchmark
print(f"[eval_policy_client.py] libero.libero.benchmark path: {benchmark.__file__}", flush=True)
import time
print("[eval_policy_client.py] importing OffScreenRenderEnv (mujoco, slow)...", flush=True)
from libero.libero.envs import OffScreenRenderEnv
print("[eval_policy_client.py] importing lerobot/imageio/cv2...", flush=True)
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.utils import write_json
import os
import imageio
import cv2
print("[eval_policy_client.py] all imports done", flush=True)

try:
    from wam_client import WebsocketClientPolicy
except ModuleNotFoundError:  # allow running without PYTHONPATH set
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from wam_client import WebsocketClientPolicy

# ============================================================
# Camera / observation key configuration
# ------------------------------------------------------------
# `LIBERO_OBS_KEY_MAP` maps the LeRobot-style camera key (used by the
# wam server / save_video) to the raw libero env obs key. Centralising
# this mapping keeps `_extract_obs`, `compose_libero_multiview_frame` and
# the CLI in sync.
# ============================================================
LIBERO_OBS_KEY_MAP = {
    "observation.images.agentview_rgb": "agentview_image",
    "observation.images.eye_in_hand_rgb": "robot0_eye_in_hand_image",
}
DEFAULT_CAMERA_KEYS = list(LIBERO_OBS_KEY_MAP.keys())

# ============================================================
# xlab data compatibility flag
# When True: (1) do NOT flip obs images, (2) remap gripper {0,1}->{-1,+1}
# Controlled via CLI --xlab-data or env var XLAB_DATA=1.
# ============================================================
_XLAB_DATA = False


def compose_libero_multiview_frame(obs, cam_keys=None):
    """Concatenate per-camera frames horizontally into a single multiview frame.

    Each camera is resized to the height/width of the first camera before
    concatenation. Returns a uint8 [H, W*N, 3] numpy array.
    """
    cam_keys = list(cam_keys) if cam_keys is not None else list(DEFAULT_CAMERA_KEYS)
    if not cam_keys:
        raise ValueError("compose_libero_multiview_frame requires at least one camera key")
    base_h, base_w = obs[cam_keys[0]].shape[:2]
    target_size = (base_w, base_h)  # cv2.resize takes (w, h)
    return np.hstack([cv2.resize(obs[k], target_size) for k in cam_keys]).astype(np.uint8)


def save_video(real_obs_list, save_path, fps=15, video_names=None):
    if video_names is None:
        video_names = list(DEFAULT_CAMERA_KEYS)
    if not real_obs_list:
        print("❌ No real observation frames")
        return

    print(f"Saving video: {len(real_obs_list)} frames...")
    final_frames = [compose_libero_multiview_frame(obs, video_names) for obs in real_obs_list]

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def construct_single_env(env_args):
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    """Extract camera images from raw env obs dict, keyed by LeRobot-style names.

    Two transforms are applied:
    1. Vertical flip (always): robosuite OpenGL convention -> OpenCV convention.
    2. Left-right flip (xlab only): xlab training data is horizontally mirrored
       vs official data, but actions are NOT mirrored. We flip eval images to match.
    """
    out = {}
    for lerobot_key, raw_key in LIBERO_OBS_KEY_MAP.items():
        img = obs[raw_key]
        # Vertical flip (always): OpenGL -> OpenCV convention
        img = img[::-1]
        if _XLAB_DATA:
            # Left-right flip: xlab images are horizontally mirrored vs official
            img = img[:, ::-1, :]
        out[lerobot_key] = np.ascontiguousarray(img)
    return out


def init_single_env(env_in, init_state):
    # Re-seed numpy before env reset to ensure mujoco physics randomness
    # is deterministic when a global seed has been set via set_seed().
    # LIBERO/robosuite uses np.random internally during reset/step.
    _seed_env = os.environ.get("_LIBERO_EVAL_SEED")
    if _seed_env is not None:
        np.random.seed(int(_seed_env))
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.] * 7)
    return _extract_obs(obs)


def _remap_xlab_action(action):
    """Remap action from xlab-trained model to robosuite execution space.

    The xlab training data stores images left-right mirrored vs official, but
    actions are identical (NOT mirrored). The model learns: mirrored image -> correct
    action. At eval time we mirror the env image to match training, so the model
    outputs correct actions directly. Only gripper encoding needs remapping:
    xlab uses {0, 1} while robosuite expects {-1, +1}.

    Action layout: [dx, dy, dz, droll, dpitch, dyaw, gripper]
                     0    1    2    3      4      5      6
    """
    action = np.array(action, dtype=np.float64)
    # Gripper: {0,1} -> {-1,+1}
    action[6] = action[6] * 2.0 - 1.0
    return action

def env_one_step(env_in, action):
    if _XLAB_DATA:
        action = _remap_xlab_action(action)
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done

def _run_single_episode(model, libero_benchmark, task_idx, out_dir, episode_idx, cam_keys=None, debug=False, seed=None, server_save_root=None):
    """Run one episode. Returns a manifest entry dict (aligned with robocasa's episode_manifest)."""
    cam_keys = list(cam_keys) if cam_keys is not None else list(DEFAULT_CAMERA_KEYS)
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    prompt = benchmark_instance.get_task(task_idx).language
    env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    first_obs = init_single_env(cur_env, init_states[episode_idx % init_states.shape[0]])

    # Use short episode_name to avoid filesystem path-length limits in server visualization.
    # The full prompt is saved separately as a .txt file alongside the video.
    short_ep_name = f"task{task_idx}_ep{episode_idx}"
    episode_tag = f"ep{episode_idx:03d}"
    ret = model.infer(dict(reset=True, prompt=prompt, episode_name=short_ep_name))

    full_obs_list = [first_obs]
    done = False
    first = True
    step_count = 0
    final_path = None

    try:
        while cur_env.env.timestep < 800:
            ret = model.infer(dict(obs=first_obs, prompt=prompt))
            action = ret['action']

            key_frame_list = []
            assert action.shape[2] % 4 == 0
            action_per_frame = action.shape[2] // 4
            start_idx = 1 if first else 0
            for i in range(start_idx, action.shape[1]):
                for j in range(action.shape[2]):
                    ee_action = action[:, i, j]
                    observes, done = env_one_step(cur_env, ee_action)
                    step_count += 1
                    if done:
                        break
                    if (j + 1) % action_per_frame == 0:
                        full_obs_list.append(observes)
                        key_frame_list.append(observes)

                if done:
                    break

            first = False

            if done:
                break
            else:
                model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))
    finally:
        # Always try to dump whatever frames we have, even on exception, so
        # mid-episode crashes still leave a debuggable video on disk.
        final_path = (
            Path(out_dir) / libero_benchmark
            / f"{task_idx}" / f"{episode_idx}_{done}.mp4"
        )
        final_path.parent.mkdir(exist_ok=True, parents=True)
        try:
            save_video(
                real_obs_list=full_obs_list,
                save_path=final_path,
                fps=60,
                video_names=cam_keys,
            )
        except Exception as e:
            print(f"[warn] save_video failed for task {task_idx} ep {episode_idx}: {e}")

        # Save prompt text for human inspection (matches eval_policy_client.py)
        prompt_path = final_path.parent / f"{episode_idx}_{done}_prompt.txt"
        try:
            prompt_path.write_text(prompt, encoding="utf-8")
        except Exception as e:
            print(f"[warn] save_prompt failed for task {task_idx} ep {episode_idx}: {e}")

        try:
            cur_env.close()
        except Exception:
            pass

    # Construct server_pred_video path (aligned with robocasa's server_save_root layout)
    server_pred_video = None
    if server_save_root:
        server_pred_video = str(Path(server_save_root) / "real" / f"{episode_tag}_{short_ep_name}" / "pred_video.mp4")

    return {
        "episode_index": int(episode_idx),
        "episode_tag": episode_tag,
        "episode_name": short_ep_name,
        "seed": int(seed) + int(episode_idx) if seed is not None else None,
        "success": bool(done),
        "steps": int(step_count),
        "client_video": str(final_path) if final_path else None,
        "server_pred_video": server_pred_video,
    }


def _load_existing_stat(out_dir, libero_benchmark, task_idx):
    """Load existing per-task json if present.

    Returns (succ_num, total_num, done_episode_ids_set, episode_manifest).
    Handles both new rich schema (episode_manifest) and old minimal schema
    (succ_num/total_num/done_episode_ids).
    """
    out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
    if not out_file.exists():
        return 0.0, 0.0, set(), []
    try:
        data = json.loads(out_file.read_text())
    except Exception:
        return 0.0, 0.0, set(), []
    succ_num = float(data.get("success_count", data.get("succ_num", 0.0)))
    total_num = float(data.get("num_episodes", data.get("total_num", 0.0)))
    done_set = set(int(e) for e in data.get("done_episode_ids", []))
    manifest = data.get("episode_manifest", []) or []
    return succ_num, total_num, done_set, manifest


def _run_episodes_for_task(model, libero_benchmark, task_idx, out_dir, episode_ids, init_succ_num=0.0, init_total_num=0.0, init_done_set=None, init_manifest=None, cam_keys=None, debug=False, task_name="", seed=None, server_save_root=None, host="127.0.0.1", port=23908):
    """Run a given list of episode_ids for a single task, updating per-task json with rich schema."""
    if init_done_set is None:
        init_done_set = set()
    if init_manifest is None:
        init_manifest = []
    succ_num = init_succ_num
    total_num = init_total_num
    done_set = set(init_done_set)
    manifest = list(init_manifest)

    if len(episode_ids) == 0:
        print(f"[task {task_idx}] no episodes to run, skip.")
        return succ_num, total_num

    out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
    out_file.parent.mkdir(exist_ok=True, parents=True)

    pbar = tqdm(episode_ids, total=len(episode_ids), desc=f"task {task_idx}")
    for episode_idx in pbar:
        if episode_idx in done_set:
            print(f"[task {task_idx}] episode {episode_idx} already done, skip.")
            continue
        entry = _run_single_episode(model, libero_benchmark, task_idx, out_dir, episode_idx, cam_keys=cam_keys, debug=debug, seed=seed, server_save_root=server_save_root)
        manifest.append(entry)
        succ_num += float(bool(entry["success"]))
        total_num += 1.0
        done_set.add(int(episode_idx))
        succ_rate = succ_num / total_num if total_num > 0 else 0.0
        print(f"[task {task_idx} ep {episode_idx}] success={entry['success']} steps={entry['steps']}  rate={succ_rate:.3f}  ({int(succ_num)}/{int(total_num)})")
        write_json({
            "benchmark": libero_benchmark,
            "task_id": int(task_idx),
            "task_name": task_name,
            "num_episodes": int(total_num),
            "max_steps": 800,
            "success_count": int(succ_num),
            "success_rate": succ_rate,
            "episode_manifest": manifest,
            "done_episode_ids": sorted(int(e) for e in done_set),
            "obs_schema": [],
            "host": host,
            "port": int(port),
        }, out_file)
    return succ_num, total_num


def _plan_from_args(libero_benchmark, test_num, task_range, task_ids, episode_ids, plan_file, out_dir, skip_completed):
    """Resolve which (task -> episode_ids) to run from CLI args / plan file.

    Returns: list of (task_id, episode_ids_list_for_this_task)
    Precedence: plan_file > (task_ids + episode_ids) > task_range

    All string params use truthy checks so both None and "" skip cleanly.
    task_range, when provided, is a list[int] of length 2.
    """
    if plan_file:
        plan = json.loads(Path(plan_file).read_text())
        items = plan.get("items") or plan.get("jobs") or []
        out = []
        for it in items:
            tid = int(it["task_id"])
            eps = sorted(int(e) for e in it["episode_ids"])
            out.append((tid, eps))
        return out

    if task_ids:
        tid_list = [int(x.strip()) for x in task_ids.split(",") if x.strip()]
    elif task_range:
        assert len(task_range) == 2
        tid_list = list(range(int(task_range[0]), int(task_range[1])))
    else:
        benchmark_dict = benchmark.get_benchmark_dict()
        num_tasks = benchmark_dict[libero_benchmark]().get_num_tasks()
        tid_list = list(range(num_tasks))

    if episode_ids:
        eps_template = [int(x.strip()) for x in episode_ids.split(",") if x.strip()]
    else:
        eps_template = list(range(int(test_num)))

    out = []
    for tid in tid_list:
        eps = list(eps_template)
        if skip_completed:
            _, total, done_set, _ = _load_existing_stat(out_dir, libero_benchmark, tid)
            if total >= len(eps) and all(e in done_set for e in eps):
                print(f"[plan] task {tid}: already completed ({int(total)} eps), skip.")
                continue
            eps = [e for e in eps if e not in done_set] if done_set else eps
        out.append((tid, eps))
    return out


def run_eval(
    libero_benchmark: str = "libero_10",
    task_id: int = -1,
    task_range: str = "",
    task_ids: str = "",
    episode_ids: str = "",
    plan_file: str = "",
    skip_completed: bool = False,
    test_num: int = 50,
    port: int = 23908,
    host: str = "127.0.0.1",
    out_dir: str = "outputs/libero",
    camera_keys: str = "",
    server_save_root: str = "",
    seed: int | None = None,
    debug: bool = False,
    xlab_data: bool = False,
):
    """Run LIBERO evaluation (aligned with robocasa's typer.run(run_eval) style).

    One invocation handles one or more tasks (plan-file/task-range/task-ids),
    reusing a single WebsocketClientPolicy connection. Supports single-task
    mode via --task-id (robocasa-style). Writes per-task JSON with rich schema
    (episode_manifest) incrementally after each episode.
    """
    # ---- Fix random seed for reproducibility ----
    # Priority: CLI --seed > env SEED > no seed
    seed_val = seed
    if seed_val is None:
        _env_seed = os.environ.get("SEED", "").strip()
        if _env_seed:
            try:
                seed_val = int(_env_seed)
            except ValueError:
                print(f"[eval_policy_client.py] WARN: invalid SEED env '{_env_seed}', ignoring.", flush=True)
    if seed_val is not None:
        set_seed(seed_val)
    else:
        print("[eval_policy_client.py] WARN: no seed specified; results may NOT be reproducible.", flush=True)

    # Resolve xlab_data: CLI flag > env var > default (False)
    if not xlab_data:
        xlab_data = os.environ.get("XLAB_DATA", "0").strip().lower() in ("1", "true", "yes")

    global _XLAB_DATA
    _XLAB_DATA = bool(xlab_data)
    cam_keys = [x.strip() for x in camera_keys.split(",") if x.strip()] if camera_keys.strip() else list(DEFAULT_CAMERA_KEYS)
    print(f"[INFO] cam_keys: {cam_keys}")
    print(f"[INFO] xlab_data: {_XLAB_DATA} (image flip: {'OFF' if _XLAB_DATA else 'ON'}, gripper remap: {'ON' if _XLAB_DATA else 'OFF'})")
    if debug:
        print("[INFO] debug mode: ON")

    # Single-task mode (--task-id >= 0) overrides task_range/task_ids/plan_file
    if task_id >= 0:
        task_range = ""
        task_ids = str(task_id)
        plan_file = ""

    # Parse task_range string "0 10" or "0,10" into list[int]
    task_range_parsed = None
    if task_range.strip():
        parts = task_range.replace(",", " ").split()
        task_range_parsed = [int(x) for x in parts]

    # Backward compat: if neither plan/task-ids/task-range/task-id given, default to full range [0, 10)
    if not plan_file and not task_ids and task_range_parsed is None and task_id < 0:
        task_range_parsed = [0, 10]

    plan = _plan_from_args(
        libero_benchmark=libero_benchmark,
        test_num=test_num,
        task_range=task_range_parsed,
        task_ids=task_ids,
        episode_ids=episode_ids,
        plan_file=plan_file,
        out_dir=out_dir,
        skip_completed=skip_completed,
    )
    total_eps = sum(len(eps) for _, eps in plan)
    print(f"#################### Use benchmark: {libero_benchmark}, tasks={len(plan)}, pending_episodes={total_eps} #############")
    if total_eps == 0:
        print("[run_eval] nothing to do.")
        return

    print("[eval_policy_client.py] initializing WebsocketClientPolicy...", flush=True)
    model = WebsocketClientPolicy(host=host, port=port)
    for task_idx, eps in plan:
        if skip_completed:
            init_succ, init_total, init_done, init_manifest = _load_existing_stat(out_dir, libero_benchmark, task_idx)
            eps = [e for e in eps if e not in init_done]
            if not eps:
                continue
        else:
            init_succ, init_total, init_done, init_manifest = 0.0, 0.0, set(), []

        # Get task_name for JSON metadata
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        task_name = benchmark_instance.get_task(task_idx).language

        _run_episodes_for_task(
            model=model,
            libero_benchmark=libero_benchmark,
            task_idx=task_idx,
            task_name=task_name,
            out_dir=out_dir,
            episode_ids=eps,
            init_succ_num=init_succ,
            init_total_num=init_total,
            init_done_set=init_done,
            init_manifest=init_manifest,
            cam_keys=cam_keys,
            debug=debug,
            seed=seed_val,
            server_save_root=server_save_root,
            host=host,
            port=port,
        )
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    typer.run(run_eval)
