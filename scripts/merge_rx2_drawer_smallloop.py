"""Dedup-merge the "小回环" (small-loop) drawer-medicine batches into one LeRobot dataset.

Source batches (newer "pipeline_*" export format, task "Take the medicine box out of the drawer
and place it on top of the cabinet."):
  - added60  = pipeline_20260909T022844Z_22830eca (60 episodes)
  - added22  = pipeline_20260909T025144Z_9a95d498 (22 episodes)
  - added57  = pipeline_20260909T025306Z_20d86f41 (57 episodes)
  - medicine_added214_0910 = pipeline_20260910T030029Z_47b7ba84 (214 episodes)

added128 (pipeline_...f2de778f) and added156 (pipeline_...752e7244) are EXCLUDED per manual
review: added128 is from robot 12 with a too-small loop, added156 is from robot 13 which is not
approved for training.

These 4 batches were checked for overlap by hashing every episode's video file (no useful
per-sortie label exists in this export format, unlike the older "抽屉取药-9号" snapshots) —
added60 and added57 share 53 identical videos (53 of 57 episodes in added57 are duplicates of
episodes already in added60). added22 and medicine_added214_0910 are clean. Deduping by video
sha256 keeping first occurrence in SOURCES order gives 300 unique episodes (vs 353 raw).

This export schema also lacks the `action.wbc` field the rest of the codebase (Rx101Inputs /
LeRobotRx101DataConfig) expects — instead it has `action.robot_motion_head_gripper_h0` [37]:
cols [0:29] body joints (same absolute-joint-target semantics as the old action.wbc, verified by
comparing magnitude/scale against observation.state), [29:35] root_rot6d (unused), [35:36]/[36:37]
left/right gripper. This script derives action.wbc / action.left_gripper / action.right_gripper
columns from that field so the output is schema-compatible with the existing policy/config code
with no changes needed there.
"""

import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path("/data/nas_ray/dataset/foundation_data/processed/lerobot/rx2_blackbox")

# Chronological order. Earlier snapshots win ties on duplicate video content.
SOURCES = [
    ROOT / "pipeline_20260909T022844Z_22830eca_38obs_68token_37motion",  # added60
    ROOT / "pipeline_20260909T025144Z_9a95d498_38obs_68token_37motion",  # added22
    ROOT / "pipeline_20260909T025306Z_20d86f41_38obs_68token_37motion",  # added57
    ROOT / "pipeline_20260910T030029Z_47b7ba84_38obs_68token_37motion",  # medicine_added214_0910
]

OUT_PARENT = pathlib.Path("/data/nas_ray/home/siyu.luo/data/lerobot/rx2_blackbox_drawer_merged")
OUT_REPO_ID = "rx2_drawer_medicine_smallloop_dedup_v1"
OUT_DIR = OUT_PARENT / OUT_REPO_ID

VIDEO_KEY = "observation.images.ego_view"
MOTION_KEY = "action.robot_motion_head_gripper_h0"


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def video_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_feature_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    keepdims = arr.ndim == 1
    return {
        "min": np.min(arr, axis=0, keepdims=keepdims).tolist(),
        "max": np.max(arr, axis=0, keepdims=keepdims).tolist(),
        "mean": np.mean(arr, axis=0, keepdims=keepdims).tolist(),
        "std": np.std(arr, axis=0, keepdims=keepdims).tolist(),
        "count": [int(len(arr))],
    }


# openpi's Normalize transform never reads image stats, so this stub (matching the
# established (3,1,1)-shaped placeholder pattern used for the rx101 dataset) avoids decoding
# video just to satisfy lerobot's aggregate_stats shape assertion.
IMAGE_STATS_STUB = {
    "min": [[[0.0]], [[0.0]], [[0.0]]],
    "max": [[[1.0]], [[1.0]], [[1.0]]],
    "mean": [[[0.5]], [[0.5]], [[0.5]]],
    "std": [[[0.5]], [[0.5]], [[0.5]]],
}


def episode_stats(df: pd.DataFrame) -> dict:
    stats = {
        key: numeric_feature_stats(np.stack(df[key].to_numpy()))
        for key in (
            "observation.state",
            "observation.left_gripper",
            "observation.right_gripper",
            "action.wbc",
            "action.left_gripper",
            "action.right_gripper",
        )
    }
    stats[VIDEO_KEY] = {**IMAGE_STATS_STUB, "count": [len(df)]}
    return stats


def main() -> None:
    (OUT_DIR / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "videos" / "chunk-000" / VIDEO_KEY).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "meta").mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    plan: list[tuple[pathlib.Path, int, str]] = []
    for src in SOURCES:
        present_indices = sorted(
            int(p.stem.split("_")[-1]) for p in (src / "data" / "chunk-000").glob("episode_*.parquet")
        )
        for old_idx in present_indices:
            video_path = src / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{old_idx:06d}.mp4"
            h = video_sha256(video_path)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            plan.append((src, old_idx, h))

    total_present = sum(len(list((s / "data" / "chunk-000").glob("episode_*.parquet"))) for s in SOURCES)
    print(f"Deduped plan: {len(plan)} unique episodes (from {total_present} raw episode files across {len(SOURCES)} batches, keyed by video sha256)")

    episodes_cache: dict[pathlib.Path, dict[int, dict]] = {}
    for src in SOURCES:
        episodes_cache[src] = {e["episode_index"]: e for e in load_jsonl(src / "meta" / "episodes.jsonl")}

    out_episodes = []
    out_stats = []
    out_sources = []
    global_index = 0
    task_to_index: dict[str, int] = {}

    for new_idx, (src, old_idx, video_hash) in enumerate(plan):
        df = pd.read_parquet(src / "data" / "chunk-000" / f"episode_{old_idx:06d}.parquet")

        motion = np.stack(df[MOTION_KEY].to_numpy())
        df["action.wbc"] = pd.Series(list(motion[:, :29]), index=df.index)
        df["action.left_gripper"] = pd.Series(list(motion[:, 35:36].astype("float32")), index=df.index)
        df["action.right_gripper"] = pd.Series(list(motion[:, 36:37].astype("float32")), index=df.index)

        ep = dict(episodes_cache[src][old_idx])
        ep["episode_index"] = new_idx
        task_str = ep["tasks"][0]
        if task_str not in task_to_index:
            task_to_index[task_str] = len(task_to_index)
            print(f"  new task string encountered: task_index={task_to_index[task_str]!r} {task_str!r} (first seen at {src.name} ep{old_idx})")
        task_idx = task_to_index[task_str]

        df["episode_index"] = pd.Series([new_idx] * len(df), dtype="int64")
        df["index"] = pd.Series(range(global_index, global_index + len(df)), dtype="int64")
        df["task_index"] = pd.Series([task_idx] * len(df), dtype="int64")
        global_index += len(df)
        df.to_parquet(OUT_DIR / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet")

        src_video = src / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{old_idx:06d}.mp4"
        dst_video = OUT_DIR / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{new_idx:06d}.mp4"
        if dst_video.exists():
            dst_video.unlink()
        dst_video.hardlink_to(src_video)

        out_episodes.append(ep)
        out_stats.append({"episode_index": new_idx, "stats": episode_stats(df)})
        out_sources.append(
            {
                "episode_index": new_idx,
                "source_dataset": str(src),
                "source_episode_index": old_idx,
                "video_sha256": video_hash,
                "length": len(df),
            }
        )

        if new_idx % 20 == 0:
            print(f"  merged {new_idx + 1}/{len(plan)}")

    with (OUT_DIR / "meta" / "episodes.jsonl").open("w") as f:
        for ep in out_episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    with (OUT_DIR / "meta" / "episodes_stats.jsonl").open("w") as f:
        for st in out_stats:
            f.write(json.dumps(st, ensure_ascii=False) + "\n")

    with (OUT_DIR / "meta" / "episode_sources.jsonl").open("w") as f:
        for s in out_sources:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with (OUT_DIR / "meta" / "tasks.jsonl").open("w") as f:
        for task_str, task_idx in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": task_idx, "task": task_str}, ensure_ascii=False) + "\n")

    src_info = json.loads((SOURCES[0] / "meta" / "info.json").read_text())
    out_info = dict(src_info)
    out_info["total_episodes"] = len(plan)
    out_info["total_frames"] = global_index
    out_info["total_tasks"] = len(task_to_index)
    out_info["total_videos"] = len(plan)
    out_info["total_chunks"] = 1
    out_info["splits"] = {"train": f"0:{len(plan)}"}
    # Derived fields not present in the source pipeline schema; keep the rest (state/gripper/video
    # feature specs) as-is since they're unchanged.
    out_info["features"]["action.wbc"] = {"dtype": "float64", "shape": [29], "names": None}
    out_info["features"]["action.left_gripper"] = {"dtype": "float32", "shape": [1], "names": None}
    out_info["features"]["action.right_gripper"] = {"dtype": "float32", "shape": [1], "names": None}
    with (OUT_DIR / "meta" / "info.json").open("w") as f:
        json.dump(out_info, f, ensure_ascii=False, indent=2)

    manifest = {
        "schema": "openpi.dataset-merge/v1",
        "video_storage": "hardlink",
        "episodes": len(plan),
        "frames": global_index,
        "note": (
            "Deduped merge of the small-loop drawer-medicine batches (added60/added22/added57/"
            "medicine_added214_0910), keyed by video sha256. added128 (robot 12, loop too small) "
            "and added156 (robot 13, not approved) excluded per manual review. action.wbc/"
            "action.left_gripper/action.right_gripper derived from action.robot_motion_head_gripper_h0."
        ),
        "source_batches": [str(s) for s in SOURCES],
    }
    with (OUT_DIR / "meta" / "merge_manifest.json").open("w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {len(plan)} episodes / {global_index} frames to {OUT_DIR}")


if __name__ == "__main__":
    main()
