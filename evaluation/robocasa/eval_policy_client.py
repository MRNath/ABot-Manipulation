import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.spaces.utils import flatdim, unflatten
import imageio
import numpy as np
import robocasa
from robocasa.utils.dataset_registry_utils import get_task_horizon
from scipy.spatial.transform import Rotation as R
import typer

def _safe_name(text: str | None) -> str:
    if text is None:
        return "default"
    value = str(text).strip()
    if not value:
        return "default"
    value = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.replace(" ", "_"))
    while "__" in value:
        value = value.replace("__", "_")
    value = value.strip("_")
    return value or "default"


try:
    from wam_client import WebsocketClientPolicy
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from wam_client import WebsocketClientPolicy

# Must match `wam/configs/robocasa_cfg.py` `obs_cam_keys` (ServeSteak / LeRobot).
DEFAULT_OBS_CAM_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]


def flatten_obs_keys(obs, prefix=""):
    keys = []
    if isinstance(obs, Mapping):
        for k, v in obs.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            keys.extend(flatten_obs_keys(v, new_prefix))
    else:
        shape = tuple(obs.shape) if hasattr(obs, "shape") else ()
        dtype = str(obs.dtype) if hasattr(obs, "dtype") else type(obs).__name__
        keys.append({"key": prefix, "shape": shape, "dtype": dtype})
    return keys


def get_by_path(data, path):
    current = data
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def collect_arrays(obs, prefix=""):
    items = []
    if isinstance(obs, Mapping):
        for k, v in obs.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            items.extend(collect_arrays(v, key))
    elif isinstance(obs, np.ndarray):
        items.append((prefix, obs))
    return items


def to_uint8_rgb(image):
    img = np.asarray(image)
    if img.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape={img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
    return img


def pick_camera_images(obs, target_keys, debug=False):
    arrays = collect_arrays(obs)
    image_items = [(k, v) for k, v in arrays if v.ndim == 3 and v.shape[-1] in (1, 3, 4)]
    if not image_items:
        raise RuntimeError("No image-like arrays found in robocasa observation.")
    priority = ["agentview", "left", "right", "eye", "hand", "wrist", "front", "cam", "image", "rgb"]
    # priority = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye", "agentview_left", "agentview_right", "eye_in_hand", "agentview", "left", "right", "eye", "hand", "wrist", "front", "cam", "image", "rgb"]
    ranked = sorted(
        image_items,
        key=lambda kv: (-sum(tag in kv[0].lower() for tag in priority), kv[0]),
    )
    if debug:
        print(f"[debug obs] image-like topics found: {len(image_items)}")
        for key, arr in ranked:
            score = sum(tag in key.lower() for tag in priority)
            print(f"[debug obs] candidate topic={key} score={score} shape={arr.shape} dtype={arr.dtype}")
    selected = [to_uint8_rgb(v) for _, v in ranked[: len(target_keys)]]
    selected_keys = [k for k, _ in ranked[: len(target_keys)]]
    while len(selected) < len(target_keys):
        selected.append(selected[-1].copy())
        selected_keys.append(selected_keys[-1])
    if debug:
        for i, target_key in enumerate(target_keys):
            print(f"[debug obs] picked topic for target={target_key}: {selected_keys[i]}")
    return {target_keys[i]: selected[i] for i in range(len(target_keys))}


def pick_state_vector(obs):
    preferred_keys = [
        "observation.state",
        "state",
        "observation.robot_state",
        "observation.proprio",
        "observation.robot0_proprio-state",
        "observation.robot0_proprio",
    ]
    for key in preferred_keys:
        val = get_by_path(obs, key)
        if val is None:
            continue
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            return arr
    arrays = collect_arrays(obs)
    candidates = []
    for key, arr in arrays:
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            score = sum(tag in key.lower() for tag in ["state", "proprio", "robot", "joint"])
            candidates.append((score, key, arr))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return np.asarray(candidates[0][2], dtype=np.float32)
    numeric = [(k, v) for k, v in arrays if np.issubdtype(v.dtype, np.number)]
    if not numeric:
        return np.zeros((1,), dtype=np.float32)
    numeric.sort(key=lambda kv: kv[0])
    return np.asarray(numeric[0][1], dtype=np.float32).reshape(-1)


def format_obs_for_server(obs, camera_keys, include_state=False, debug=False):
    """Build one frame dict for `wam server`: LeRobot-style keys + optional `observation.state`."""
    out = {}
    source_map = {}
    print(obs.keys())
    for k in camera_keys:
        v = get_by_path(obs, k)
        source = k
        if v is None:
            tail = k.split(".")[-1]
            fallback_key = f"observation.images.{tail}"
            v = get_by_path(obs, fallback_key)
            source = fallback_key
        if v is not None:
            out[k] = to_uint8_rgb(v)
            source_map[k] = source
        elif debug:
            print(f"[debug obs] missing camera key: requested={k}")
    if len(out) < len(camera_keys):
        if debug:
            print(
                f"[debug obs] fallback picker triggered: found={len(out)} requested={len(camera_keys)}; "
                f"auto-selecting from image-like arrays"
            )
        out = pick_camera_images(obs, camera_keys, debug=debug)
        source_map = {k: "pick_camera_images" for k in out.keys()}
    if debug:
        for k in camera_keys:
            img = out.get(k, None)
            if img is None:
                print(f"[debug obs] final camera missing after fallback: key={k}")
                continue
            print(
                f"[debug obs] camera ready: key={k} source={source_map.get(k, 'unknown')} "
                f"shape={img.shape} dtype={img.dtype} min={int(np.min(img))} max={int(np.max(img))}"
            )
    if include_state:
        # Keep this field aligned with training schema key. In robocasa online eval this is mainly
        # for observability/debug; action cache state is provided separately in `compute_kv_cache`.
        out["observation.state"] = pick_state_vector(obs)
    return out


def _resize_image_to_hw(image, target_h, target_w):
    h, w = image.shape[:2]
    if h == target_h and w == target_w:
        return image
    y_idx = np.linspace(0, h - 1, target_h).astype(np.int32)
    x_idx = np.linspace(0, w - 1, target_w).astype(np.int32)
    return image[y_idx][:, x_idx]


def compose_obs_multiview_frame(obs, camera_keys, debug=False):
    frames = format_obs_for_server(obs, camera_keys, debug=debug)
    if len(frames) == 0:
        raise RuntimeError("No camera frames available to compose multiview video frame.")
    ordered = [to_uint8_rgb(frames[k]) for k in camera_keys if k in frames]
    if len(ordered) == 0:
        raise RuntimeError("No matched camera keys available in formatted obs.")
    base_h, base_w = ordered[0].shape[:2]
    aligned = [_resize_image_to_hw(img, base_h, base_w) for img in ordered]
    return np.concatenate(aligned, axis=1)


def _clip_action_to_space(space, x):
    if isinstance(space, gym.spaces.Box):
        arr = np.asarray(x, dtype=np.float32)
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        return np.clip(arr, low, high).astype(arr.dtype)
    if isinstance(space, gym.spaces.Dict):
        return {k: _clip_action_to_space(space.spaces[k], x[k]) for k in space.spaces}
    if isinstance(space, gym.spaces.Tuple):
        return tuple(_clip_action_to_space(s, xi) for s, xi in zip(space.spaces, x))
    return x


def _summarize_action_space(space):
    if isinstance(space, gym.spaces.Box):
        d = int(np.prod(space.shape))
        return f"Box shape={space.shape} flat_dim={d} dtype={getattr(space, 'dtype', None)}"
    if isinstance(space, (gym.spaces.Dict, gym.spaces.Tuple)):
        return f"{type(space).__name__} flat_dim={flatdim(space)}"
    return f"{type(space).__name__}"


def _print_action_space_dict_order(space):
    if isinstance(space, gym.spaces.Dict):
        keys = list(space.spaces.keys())
        print(f"[debug] env.action_space dict key order: {keys}")
        for k, subspace in space.spaces.items():
            shape = getattr(subspace, "shape", None)
            dtype = getattr(subspace, "dtype", None)
            print(f"[debug]   - {k}: {type(subspace).__name__}, shape={shape}, dtype={dtype}")


def convert_action(action):
    """
    Converts input action (np.array) to format expected by gym env (dict)
    """
    action = action.copy()
    output_action = {
        "action.end_effector_position": action[5:8],
        "action.end_effector_rotation": action[8:11],
        "action.gripper_close": action[11:12],
        "action.base_motion": action[0:4],
        "action.control_mode": action[4:5],
    }
    # print(output_action["action.control_mode"])
    # output_action["action.control_mode"] = 1.0
    # output_action["action.base_motion"][2] = 0.0
    # output_action = {
    #     "action.end_effector_position": action[0:3],
    #     "action.end_effector_rotation": action[3:6],
    #     "action.gripper_close": action[6:7],
    #     "action.base_motion": action[7:11],
    #     "action.control_mode": action[11:12],
    # }
    return output_action


def convert_rel_first_action_chunk_to_rel_prev(action_chunk):
    """
    Convert robocasa action chunk from "relative to first frame" to "relative to previous frame".
    The conversion is applied only on pose-related channels in the LeRobot PandaOmron action layout:
    - base_motion:            [0:4]
    - end_effector_position:  [5:8]
    - end_effector_rotation:  [8:11]
    """
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 3:
        raise RuntimeError(f"Unexpected action chunk shape: {chunk.shape}")
    c, _, _ = chunk.shape
    if c < 11:
        assert False
        return chunk
    out = chunk.copy()

    # 1) translation-like channels: simple diff over time, first step = 0
    trans_channels = list(range(0, 4)) + list(range(5, 8))
    for ch in trans_channels:
        flat = out[ch].reshape(-1)
        flat[1:] = flat[1:] - flat[:-1]
        # flat[0, 0] = 0.0
        out[ch] = flat.reshape(out[ch].shape)

    # 2) rotation channels (axis-angle), stored as rotvec in channels 8:11
    # Interpret them as absolute axis-angle wrt first frame, convert to
    # relative axis-angle wrt previous step.
    rot_block = out[8:11]  # shape (3, F, H)
    rot_abs = rot_block.reshape(3, -1).T  # (T, 3)
    if rot_abs.shape[0] > 0:
        rot_rel = np.zeros_like(rot_abs, dtype=np.float32)
        for t in range(1, rot_abs.shape[0]):
            R_t = R.from_rotvec(rot_abs[t])
            R_prev = R.from_rotvec(rot_abs[t - 1])
            R_delta = R_t * R_prev.inv()
            rot_rel[t] = R_delta.as_rotvec()
        # t=0 stays zero (no previous step)
        out[8:11] = rot_rel.T.reshape(rot_block.shape)
    
    # out[8:11] = 0.

    return out


def adapt_action_for_env(action_step, env, debug=False):
    """Map model flat/step vector to `env.step` action (Box or Dict/Tuple composite, common in robosuite/robocasa)."""
    action_flat = np.asarray(action_step, dtype=np.float32).reshape(-1)
    in_dim = int(action_flat.shape[0])
    assert in_dim == 12, f"action_flat.shape[0] = {action_flat.shape[0]}, in_dim = {in_dim}"
    space = env.action_space
    if isinstance(space, gym.spaces.Box):
        target_dim = int(np.prod(space.shape))
        if action_flat.shape[0] > target_dim:
            if debug:
                print(f"[debug adapt] truncate model {in_dim} -> env {target_dim} (Box)")
            action_flat = action_flat[:target_dim].copy()
        elif action_flat.shape[0] < target_dim:
            if debug:
                print(f"[debug adapt] pad model {in_dim} -> env {target_dim} (Box); tail filled with 0")
            action_flat = np.pad(action_flat, (0, target_dim - action_flat.shape[0]), mode="constant", constant_values=0.0)
        low = np.asarray(space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(space.high, dtype=np.float32).reshape(-1)
        action_flat = np.clip(action_flat, low, high)
        out = action_flat.reshape(space.shape)
        if debug:
            print(f"[debug adapt] env_action Box out shape={out.shape} range=[{float(np.min(out)):.4f},{float(np.max(out)):.4f}]")
        return out
    if isinstance(space, (gym.spaces.Dict, gym.spaces.Tuple)):
        n = flatdim(space)
        assert action_flat.shape[0] == n, f"action_flat.shape[0] = {action_flat.shape[0]}, n = {n}"
        # if action_flat.shape[0] > n:
        #     if debug:
        #         print(f"[debug adapt] truncate model {in_dim} -> env flat {n} ({type(space).__name__})")
        #     action_flat = action_flat[:n].copy()
        # elif action_flat.shape[0] < n:
        #     if debug:
        #         print(f"[debug adapt] pad model {in_dim} -> env flat {n} ({type(space).__name__})")
        #     action_flat = np.pad(action_flat, (0, n - action_flat.shape[0]), mode="constant", constant_values=0.0)
        # structured = unflatten(space, action_flat)
        # out = _clip_action_to_space(space, structured)
        out = convert_action(action_flat)
        if debug:
            if isinstance(out, dict):
                parts = [f"{k}:{np.asarray(v).shape}" for k, v in out.items()]
                print(f"[debug adapt] env_action Dict {', '.join(parts)}")
            else:
                print(f"[debug adapt] env_action Tuple len={len(out)}")
        return out
    raise NotImplementedError(f"Unsupported action_space type: {type(space).__name__}")


def run_eval(
    env_name: str = "ServeSteak",
    split: str = "target",
    num_episodes: int = 1,
    max_steps: int = 300,
    seed: int = 0,
    host: str = "127.0.0.1",
    port: int = 29056,
    prompt: str | None = None,
    frame_chunk_size: int = 2,
    camera_keys: str = "",
    save_json: str | None = None,
    save_video_dir: str | None = None,
    server_save_root: str | None = None,
    video_fps: int = 20,
    debug_actions: bool = False,
    horizon_multiplier: float = 1.0,
):
    task_horizon = get_task_horizon(env_name)
    max_steps = max(1, int(round(task_horizon * float(horizon_multiplier))))
    print(f"Task horizon: {task_horizon} x{horizon_multiplier} -> max_steps={max_steps}")
    
    if split not in ["pretrain", "target", "all"]:
        raise ValueError("split must be one of {'pretrain', 'target', 'all'}")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_video_run_dir = None
    if save_video_dir is not None:
        save_video_run_dir = os.path.join(save_video_dir, run_timestamp)
        os.makedirs(save_video_run_dir, exist_ok=True)
    env_name = env_name.replace("robocasa/", "")
    env = gym.make(f"robocasa/{env_name}", split=split, seed=seed)
    if debug_actions:
        print(f"[debug] env.action_space: {_summarize_action_space(env.action_space)}")
        _print_action_space_dict_order(env.action_space)
    model = WebsocketClientPolicy(host=host, port=port)
    cam_keys = [x.strip() for x in camera_keys.split(",") if x.strip()] if camera_keys.strip() else list(DEFAULT_OBS_CAM_KEYS)
    print(f"[debug] cam_keys: {cam_keys}")
    success_count = 0
    first_obs_schema = None
    episode_manifest: list[dict] = []

    try:
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep)
            if ep == 0:
                first_obs_schema = flatten_obs_keys(obs)
            task_lang = get_by_path(obs, "annotation.human.task_description")
            task_lang = obs["annotation.human.task_description"]
            if isinstance(task_lang, str) and task_lang.strip():
                task_lang = task_lang.strip()
            else:
                task_lang = None
            episode_prompt = prompt if prompt is not None else (task_lang if task_lang is not None else env_name)
            episode_tag = f"ep{ep:03d}"
            episode_name = _safe_name(task_lang if task_lang is not None else env_name)
            print(episode_prompt)
            model.infer(dict(reset=True, prompt=episode_prompt, episode_tag=episode_tag, episode_name=episode_name))
            episode_frames = []
            client_video_path = None
            server_pred_video_path = None
            if save_video_dir is not None:
                frame = compose_obs_multiview_frame(obs, cam_keys, debug=debug_actions)
                episode_frames.append(np.asarray(frame))

            step_count = 0
            episode_success = False
            first_chunk = True
            task_horizon = get_task_horizon(env_name)
            max_steps = max(1, int(round(task_horizon * float(horizon_multiplier))))
            print(f"Task horizon: {task_horizon} x{horizon_multiplier} -> max_steps={max_steps}")
            while step_count < max_steps:
                obs_frame = format_obs_for_server(obs, cam_keys, debug=debug_actions)
                ret = model.infer(dict(obs=obs_frame, prompt=episode_prompt))
                action_chunk_raw = np.asarray(ret["action"], dtype=np.float32)
                # if action_chunk_raw.ndim != 3:
                #     raise RuntimeError(f"Unexpected action shape from server: {action_chunk_raw.shape}")
                # action_chunk = convert_rel_first_action_chunk_to_rel_prev(action_chunk_raw)
                action_chunk = action_chunk_raw
                key_frames = []
                if action_chunk.shape[2] % 4 != 0:
                    raise RuntimeError(f"Unexpected action temporal size: {action_chunk.shape[2]}, expected divisible by 4")
                actions_per_frame = action_chunk.shape[2] // 4
                start_frame_idx = 1 if first_chunk else 0
                if debug_actions:
                    c_act, f_n, h_n = action_chunk.shape
                    print(
                        f"[debug] action_chunk shape (C,F,H)={action_chunk.shape} "
                        f"=> model_action_dim={c_act}, chunk_frames={f_n}, sub_steps_per_frame={h_n}, "
                        f"keyframe_stride(actions_per_frame)={actions_per_frame}, "
                        f"first_chunk={first_chunk}, frame_i in [{start_frame_idx},{f_n})"
                    )
                dbg_adapt_once = debug_actions
                chunk_interrupted = False
                terminated = False
                truncated = False
                for i in range(start_frame_idx, action_chunk.shape[1]):
                    for j in range(action_chunk.shape[2]):
                        use_adapt_debug = dbg_adapt_once
                        if dbg_adapt_once:
                            dbg_adapt_once = False
                        env_action = adapt_action_for_env(action_chunk[:, i, j], env, debug=use_adapt_debug)
                        if debug_actions and use_adapt_debug:
                            print(f"[debug] first step of chunk uses slice action_chunk[:, i={i}, j={j}] -> env.step")
                        obs, _, terminated, truncated, info = env.step(env_action)
                        step_count += 1
                        if save_video_dir is not None:
                            frame = compose_obs_multiview_frame(obs, cam_keys, debug=debug_actions)
                            episode_frames.append(np.asarray(frame))
                        if (j + 1) % actions_per_frame == 0:
                            key_frames.append(format_obs_for_server(obs, cam_keys, debug=debug_actions))
                        if bool(info.get("success", False)):
                            episode_success = True
                            success_count += 1
                            chunk_interrupted = True
                            break
                        if terminated or truncated or step_count >= max_steps:
                            chunk_interrupted = True
                            break
                    if episode_success or terminated or truncated or step_count >= max_steps:
                        break

                if key_frames and not chunk_interrupted:
                    model.infer(dict(obs=key_frames, compute_kv_cache=True, state=action_chunk_raw))
                first_chunk = False
                if episode_success or step_count >= max_steps:
                    break

            if save_video_dir is not None and len(episode_frames) > 0:
                suffix = "success" if episode_success else "failure"
                if save_video_run_dir is None:
                    raise RuntimeError("save_video_run_dir should not be None when save_video_dir is set")
                env_video_dir = os.path.join(save_video_run_dir, env_name)
                os.makedirs(env_video_dir, exist_ok=True)
                client_video_path = os.path.join(env_video_dir, f"ep{ep:03d}_{suffix}.mp4")
                imageio.mimwrite(client_video_path, episode_frames, fps=video_fps)
            if server_save_root is not None and server_save_root.strip():
                server_root = server_save_root
                server_episode_dir = os.path.join(server_root, "real", f"{episode_tag}_{episode_name}")
                server_pred_video_path = os.path.join(server_episode_dir, "pred_video.mp4")
            episode_manifest.append(
                {
                    "episode_index": ep,
                    "episode_tag": episode_tag,
                    "episode_name": episode_name,
                    "seed": seed + ep,
                    "success": bool(episode_success),
                    "steps": int(step_count),
                    "client_video": client_video_path,
                    "server_pred_video": server_pred_video_path,
                }
            )
        model.infer(dict(flush_pred_video=True))
    finally:
        env.close()

    result = {
        "env_name": env_name,
        "split": split,
        "num_episodes": num_episodes,
        "max_steps": max_steps,
        "success_count": success_count,
        "success_rate": success_count / max(1, num_episodes),
        "episode_manifest": episode_manifest,
        "obs_schema": first_obs_schema or [],
        "host": host,
        "port": port,
        "frame_chunk_size": frame_chunk_size,
    }
    print(f"Env: {result['env_name']}")
    print(f"Split: {result['split']}")
    print(f"Episodes: {result['num_episodes']}, Max steps: {result['max_steps']}")
    print(f"Success: {result['success_count']}/{result['num_episodes']}")
    print(f"Success rate: {result['success_rate']:.4f}")
    if save_json is not None:
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved result json to: {save_json}")
    if save_video_dir is not None:
        print(f"Saved rollout videos to: {save_video_run_dir}")


if __name__ == "__main__":
    typer.run(run_eval)
