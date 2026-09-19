import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Rx101Inputs(transforms.DataTransformFn):
    """Inputs for RX blackbox humanoid (rx_p2_27dof / rx_p2_29dof) policies.

    Canonical inputs (after RepackTransform):
    - images: {"ego_view": image}
    - state: [B]  (B=27 or 29 body joints)
    - left_gripper_state / right_gripper_state: [1] each
    - projected_gravity: [3]  (only if include_projected_gravity; see LeRobotRx101DataConfig)
    - action_wbc: [action_horizon, B]  (only during training)
    - action_left_gripper / action_right_gripper: [action_horizon, 1] each

    The B body joints + 2 grippers are concatenated into a (B+2)-D vector (29 or 31),
    unless include_projected_gravity: then head_yaw/head_pitch are dropped from `state`
    and projected_gravity(3) takes their place — body[:27] + gravity(3) + grippers(2)
    = 32, matching the model's hard-capped proprio width (see LeRobotRx101DataConfig's
    include_projected_gravity docstring for why).
    """

    model_type: _model.ModelType
    include_projected_gravity: bool = False

    def __call__(self, data: dict) -> dict:
        body_state = np.asarray(data["state"], dtype=np.float32)
        left_g = np.asarray(data["left_gripper_state"], dtype=np.float32).reshape(-1)
        right_g = np.asarray(data["right_gripper_state"], dtype=np.float32).reshape(-1)
        if self.include_projected_gravity:
            body_state = body_state[:27]
            gravity = np.asarray(data["projected_gravity"], dtype=np.float32).reshape(-1)
            state = np.concatenate([body_state, gravity, left_g, right_g], axis=-1)
        else:
            state = np.concatenate([body_state, left_g, right_g], axis=-1)

        ego = _parse_image(data["images"]["ego_view"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (ego, np.zeros_like(ego), np.zeros_like(ego))
                image_masks = (np.True_, np.False_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (ego, np.zeros_like(ego), np.zeros_like(ego))
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "action_wbc" in data:
            body_a = np.asarray(data["action_wbc"], dtype=np.float32)
            left_ga = np.asarray(data["action_left_gripper"], dtype=np.float32).reshape(body_a.shape[0], 1)
            right_ga = np.asarray(data["action_right_gripper"], dtype=np.float32).reshape(body_a.shape[0], 1)
            inputs["actions"] = np.concatenate([body_a, left_ga, right_ga], axis=-1)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class Rx101Outputs(transforms.DataTransformFn):
    """Slice the model's 32-D action back to the robot's real action width.

    Set action_dim=29 for rx_p2_27dof, or 31 for rx_p2_29dof.
    """

    action_dim: int = 29

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
