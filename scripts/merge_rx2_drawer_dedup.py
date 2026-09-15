"""Dedup-merge multiple rx2_blackbox "抽屉取药" LeRobot export snapshots into one dataset.

The data-collection pipeline periodically re-exports overlapping batches of sorties into new
date-stamped snapshot directories under rx2_blackbox/. Different snapshots can contain the same
underlying sortie recording (same source_label in meta/episode_sources.jsonl) under a different
local episode_index. Naively concatenating snapshots would train on duplicated trajectories.

This script dedups by source_label (sortie id), keeping the first occurrence in SOURCES order,
and writes a single new LeRobot v2.1 dataset with contiguous episode_index / global index.
Video files are hardlinked (same NAS filesystem, content unchanged); parquet files are rewritten
because episode_index/index columns must be renumbered.
"""

import collections
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path("/data/nas_ray/dataset/foundation_data/processed/lerobot/rx2_blackbox")

# Chronological order. Earlier snapshots win ties on duplicate source_label.
SOURCES = [
    ROOT / "20260901_204326_抽屉取药-抽屉一层黑色药盒-9号"
    / "mondoRX-03_task_0016_20260901_201336_lerobot-cdf007d8-09fe-411b-90b2-0368ec7f0f5c",
    ROOT / "20260902_014345_抽屉取药-抽屉一层黑色药盒-9号"
    / "mondoRX-03_task_0016_20260902_005106_lerobot-f3a05a7f-252a-460a-ac70-c2c2e31ce363",
    ROOT / "20260902_193415_抽屉取药-抽屉一层黑色药盒-9号"
    / "mondoRX-03_task_0016_20260902_192027_lerobot-af5fc6a2-4e10-4e27-b27a-9ae600aca518",
    ROOT / "20260902_215815_抽屉取药-抽屉一层黑色药盒-9号"
    / "mondoRX-03_task_0016_20260902_214640_lerobot-a1b363eb-6cda-4191-addb-3bca990ba27e",
]

OUT_PARENT = pathlib.Path("/data/nas_ray/home/siyu.luo/data/lerobot/rx2_blackbox_drawer_merged")
OUT_REPO_ID = "rx2_drawer_medicine_dedup_v1"
OUT_DIR = OUT_PARENT / OUT_REPO_ID

VIDEO_KEY = "observation.images.ego_view"


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    (OUT_DIR / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "videos" / "chunk-000" / VIDEO_KEY).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "meta").mkdir(parents=True, exist_ok=True)

    seen_labels: set[str] = set()
    plan: list[tuple[pathlib.Path, int, str, int]] = []
    for src in SOURCES:
        # episodes.jsonl + on-disk parquet are authoritative for which episodes actually exist;
        # episode_sources.jsonl can be stale (e.g. cdf007d8 lists 2 sorties but only 1 episode
        # file/episodes.jsonl entry is actually present on disk). Use it opportunistically for
        # the dedup label, falling back to a repo-unique label when it disagrees or is missing.
        present_indices = sorted(
            int(p.stem.split("_")[-1]) for p in (src / "data" / "chunk-000").glob("episode_*.parquet")
        )
        by_idx = {e["episode_index"]: e for e in load_jsonl(src / "meta" / "episode_sources.jsonl")}
        for old_idx in present_indices:
            entry = by_idx.get(old_idx)
            label = entry["source_label"] if entry else f"{src.name}#ep{old_idx}"
            declared_length = entry["length"] if entry else None
            if label in seen_labels:
                continue
            seen_labels.add(label)
            plan.append((src, old_idx, label, declared_length))

    total_present = sum(len(list((s / "data" / "chunk-000").glob("episode_*.parquet"))) for s in SOURCES)
    print(f"Deduped plan: {len(plan)} unique sorties (from {total_present} on-disk episode files across {len(SOURCES)} snapshots)")

    episodes_cache: dict[pathlib.Path, dict[int, dict]] = {}
    stats_cache: dict[pathlib.Path, dict[int, dict]] = {}
    for src in SOURCES:
        episodes_cache[src] = {e["episode_index"]: e for e in load_jsonl(src / "meta" / "episodes.jsonl")}
        stats_cache[src] = {e["episode_index"]: e for e in load_jsonl(src / "meta" / "episodes_stats.jsonl")}

    out_episodes = []
    out_stats = []
    out_sources = []
    global_index = 0
    task_to_index: dict[str, int] = {}

    for new_idx, (src, old_idx, label, declared_length) in enumerate(plan):
        df = pd.read_parquet(src / "data" / "chunk-000" / f"episode_{old_idx:06d}.parquet")
        length = len(df)
        if declared_length is not None and declared_length != length:
            print(f"  WARNING: {src.name} ep{old_idx} ({label}): episode_sources.jsonl length={declared_length} != actual {length}, using actual")

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

        st = dict(stats_cache[src][old_idx])
        st["episode_index"] = new_idx
        out_stats.append(st)

        out_sources.append(
            {
                "episode_index": new_idx,
                "source_label": label,
                "source_dataset": str(src),
                "source_episode_index": old_idx,
                "length": length,
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
    with (OUT_DIR / "meta" / "info.json").open("w") as f:
        json.dump(out_info, f, ensure_ascii=False, indent=2)

    manifest = {
        "schema": "openpi.dataset-merge/v1",
        "video_storage": "hardlink",
        "episodes": len(plan),
        "frames": global_index,
        "note": "Deduped merge across overlapping rx2_blackbox drawer-medicine snapshot exports, keyed by episode_sources.source_label.",
        "source_snapshots": [str(s) for s in SOURCES],
    }
    with (OUT_DIR / "meta" / "merge_manifest.json").open("w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {len(plan)} episodes / {global_index} frames to {OUT_DIR}")


if __name__ == "__main__":
    main()
