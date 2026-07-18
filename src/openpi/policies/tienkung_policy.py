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
class TienkungInputs(transforms.DataTransformFn):
    """Inputs for Tienkung humanoid dual-arm policies.

    Expected canonical inputs:
    - images: {"camera_head": image}
    - state: [26], ordered as left arm 7 + left hand 6 + right arm 7 + right hand 6
    - actions: [action_horizon, 26], only during training
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data.get("state", data.get("observation.state")))

        if "images" in data:
            head_image = data["images"]["camera_head"]
        else:
            head_image = data["observation.images.camera_head"]
        head_image = _parse_image(head_image)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (head_image, np.zeros_like(head_image), np.zeros_like(head_image))
                image_masks = (np.True_, np.False_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (head_image, np.zeros_like(head_image), np.zeros_like(head_image))
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        elif "action" in data:
            inputs["actions"] = np.asarray(data["action"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class TienkungOutputs(transforms.DataTransformFn):
    """Outputs for Tienkung humanoid dual-arm policies."""

    action_dim: int = 26

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
