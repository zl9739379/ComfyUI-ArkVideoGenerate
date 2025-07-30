"""
ArkVideoGenerate
ComfyUI 节点：火山方舟视频生成 + 本地解码
"""
import base64
import io
import os
import time
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from volcenginesdkarkruntime import Ark
from folder_paths import get_output_directory


# --------------------------------------------------
# 工具：把本地 mp4 解码成 (T, H, W, C) 0-1 Tensor
# --------------------------------------------------
def load_video_to_tensor(path: str, max_frames: int = -1):
    """
    用 OpenCV 把 mp4 解码成 torch.Tensor
    :param path: 本地 mp4 文件绝对路径
    :param max_frames: 最多返回多少帧；-1=全部
    :return: (T, H, W, C)  float32 0-1 Tensor
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb.astype(np.float32) / 255.0)

        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()

    if not frames:
        raise RuntimeError("Empty video")

    return torch.from_numpy(np.stack(frames, axis=0))  # (T,H,W,C)


# --------------------------------------------------
# ComfyUI 节点
# --------------------------------------------------
class ArkVideoGenerate:
    RETURN_TYPES = ("IMAGE", "INT")          # IMAGE = (T,H,W,C) 张量；INT = 帧数
    RETURN_NAMES = ("frames", "frame_count")
    FUNCTION = "generate"
    CATEGORY = "video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "model": (["doubao-seedance-1-0-pro-250528",
                          "wan2-1-14b-flf2v"],),
                "prompt": ("STRING", {"multiline": True, "default": "a cat is dancing"}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "ratio": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "9:21", "keep_ratio", "adaptive"],
                          {"default": "16:9"}),
                "duration": (["5", "10"], {"default": "5"}),
                "fps": (["16", "24"], {"default": "24"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2 ** 32 - 1}),
                "watermark": (["false", "true"], {"default": "false"}),
                "camera_fixed": (["false", "true"], {"default": "false"}),
            },
            "optional": {
                "first_frame_image": ("IMAGE",),
                "last_frame_image": ("IMAGE",),
                "callback_url": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    # --------------------------------------------------
    # 主逻辑
    # --------------------------------------------------
    def generate(self,
                 api_key: str,
                 model: str,
                 prompt: str,
                 resolution: str,
                 ratio: str,
                 duration: str,
                 fps: str,
                 seed: int,
                 watermark: str,
                 camera_fixed: str,
                 first_frame_image=None,
                 last_frame_image=None,
                 callback_url: str = ""):
        # 1. 构造 prompt
        text_prompt = prompt.strip()
        param_suffix = (f" --resolution {resolution} --ratio {ratio} --duration {duration}"
                        f" --fps {fps} --seed {seed} --watermark {watermark} --camera_fixed {camera_fixed}")
        text_prompt += param_suffix

        # 2. Base64 图片
        content_list: List[dict] = [{"type": "text", "text": text_prompt}]
        if first_frame_image is not None:
            img = Image.fromarray(
                (first_frame_image.cpu().squeeze(0).numpy() * 255).astype("uint8")
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            url_first = f"data:image/png;base64,{b64}"
            content_list.append({"type": "image_url", "image_url": {"url": url_first}, "role": "first_frame"})

        if last_frame_image is not None:
            if first_frame_image is None:
                raise ValueError("last_frame_image 必须同时提供 first_frame_image")
            img = Image.fromarray(
                (last_frame_image.cpu().squeeze(0).numpy() * 255).astype("uint8")
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            url_last = f"data:image/png;base64,{b64}"
            content_list.append({"type": "image_url", "image_url": {"url": url_last}, "role": "last_frame"})

        # 3. 创建任务
        client = Ark(api_key=api_key)
        task_resp = client.content_generation.tasks.create(
            model=model,
            content=content_list,
            callback_url=callback_url or None
        )
        task_id = task_resp.id
        print(f"[ArkVideoGenerate] task created: {task_id}")

        # 4. 轮询
        while True:
            resp = client.content_generation.tasks.get(task_id=task_id)
            status = resp.status
            print(f"[ArkVideoGenerate] task {task_id} status: {status}")
            if status == "succeeded":
                video_url = resp.content.video_url
                break
            elif status in ("failed", "cancelled"):
                raise RuntimeError(f"Task {status}: {resp.error.message}")
            time.sleep(3)

        # 5. 下载
        output_root = Path(get_output_directory()) / "ark_video"
        output_root.mkdir(parents=True, exist_ok=True)
        local_path = output_root / f"{task_id}.mp4"

        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"[ArkVideoGenerate] downloaded: {local_path}")

        # 6. 解码为 ComfyUI 张量
        frames = load_video_to_tensor(str(local_path))  # (T,H,W,C)
        return (frames, frames.shape[0])
