"""Smoke test for the pi05_rx101_blackbox VLA server.

Constructs a synthetic obs matching the post-repack contract from Rx101Inputs
and hits serve_policy over WebSocket. Prints output shape, dtype, timing,
and first few sampled action values.

Usage (after starting `uv run scripts/serve_policy.py policy:checkpoint ...`):
    uv run scripts/smoke_client.py
    uv run scripts/smoke_client.py --host 127.0.0.1 --port 8000 --n-repeats 3
"""

import dataclasses
import time

import numpy as np
import tyro

from openpi_client import websocket_client_policy


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    n_repeats: int = 3
    prompt: str = "take the book from the bookshelf and hand it to the person"


def make_fake_obs(prompt: str) -> dict:
    """Post-repack obs matching Rx101Inputs contract.

    Rx101Inputs reads: state[27], left_gripper_state[1], right_gripper_state[1],
    images.ego_view[H,W,3], optional prompt. Values here are unit-scale noise —
    they exercise the pipeline but tell us nothing about behavior quality.
    """
    rng = np.random.default_rng(0)
    return {
        "state": rng.standard_normal(27).astype(np.float32),
        "left_gripper_state": rng.standard_normal(1).astype(np.float32),
        "right_gripper_state": rng.standard_normal(1).astype(np.float32),
        "images": {
            "ego_view": (rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)),
        },
        "prompt": prompt,
    }


def main(args: Args) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    meta = client.get_server_metadata()
    print(f"server metadata: {meta}")

    for i in range(args.n_repeats):
        obs = make_fake_obs(args.prompt)
        t0 = time.monotonic()
        out = client.infer(obs)
        dt_ms = (time.monotonic() - t0) * 1000
        actions = np.asarray(out["actions"])
        print(
            f"[{i}] actions shape={actions.shape} dtype={actions.dtype} "
            f"latency={dt_ms:.1f}ms "
            f"min={actions.min():.3f} max={actions.max():.3f} "
            f"mean={actions.mean():.3f} std={actions.std():.3f}"
        )
        if i == 0:
            print(f"    first action step (t=0): {actions[0]}")


if __name__ == "__main__":
    main(tyro.cli(Args))
