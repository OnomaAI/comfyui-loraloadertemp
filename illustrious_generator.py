from .autonode import node_wrapper, get_node_names_mappings, validate
import base64
import io
import numpy as np
import torch
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
from urllib3.exceptions import InsecureRequestWarning

fundamental_classes = []
fundamental_node = node_wrapper(fundamental_classes)

@fundamental_node
class IllustriousGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "1 girl"}),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "worst quality, low quality, lowers, low details, "
                            "bad quality, poorly drawn, bad anatomy, multiple views, "
                            "bad hands, blurry, artist sign, weibo username"
                        ),
                    },
                ),
                "steps": ("INT", {"min": 1, "max": 100, "default": 28}),
                "width": ("INT", {"min": 256, "max": 2048, "default": 1024}),
                "height": ("INT", {"min": 256, "max": 2048, "default": 1024}),
                "cfg_scale": ("FLOAT", {"min": 1.0, "max": 20.0, "default": 7.4}),
                "sampler": (["euler", "euler_a", "ddim", "dpmpp_2m"],),
                "scheduler": (["normal", "klms", "sgm_uniform"],),
                "model_id": ("INT", {"min": 1, "max": 20, "default": 5}),
                "seed": ("INT", {"min": -1, "max": 2**32 - 1, "default": -1}),
                "n_requests": ("INT", {"min": 1, "max": 32, "default": 10}),
                "threads": ("INT", {"min": 1, "max": 32, "default": 10}),
                "url": (
                    "STRING",
                    {
                        "default": "https://api-dev.v1.illustrious-xl.ai/api/text-to-image/generate"
                    },
                ),
                "access_token": ("STRING", {"multiline": False, "default": "my_access_token"}),
                "illustrious_api_key": ("STRING", {"multiline": False, "default": "my key"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    custom_name = "illustrious-generate"
    CATEGORY = "Illustrious"
    @staticmethod
    def _build_headers(access_token: str, api_key: str) -> dict:
        return {
            "User-Agent": "MyCustomAgent/1.0",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json, text/plain, */*",
            "sec-ch-ua": 'Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
            "sec-ch-ua-mobile": "?0",
            "x-illustrious-api-key": api_key,
        }

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """
        PIL.Image (RGB) → torch.Tensor [H,W,C], float32, 0-1
        """
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)

    @staticmethod
    def _b64_to_tensor(data_url_or_b64: str) -> torch.Tensor:
        """Handle both bare base64 strings and full data URLs."""
        if "," in data_url_or_b64:
            data_url_or_b64 = data_url_or_b64.split(",", 1)[1]
        raw = base64.b64decode(data_url_or_b64)
        pil = Image.open(io.BytesIO(raw))
        return IllustriousGenerate._pil_to_tensor(pil)
    def generate(
        self,
        prompt,
        negative_prompt,
        steps,
        width,
        height,
        cfg_scale,
        sampler,
        scheduler,
        model_id,
        seed,
        n_requests,
        threads,
        url,
        access_token,
        illustrious_api_key,
    ):
        base_params = {
            "modelId": model_id,
            "steps": steps,
            "width": width,
            "height": height,
            "prompt": prompt,
            "negativePrompt": negative_prompt,
            "cfgScale": cfg_scale,
            "samplerName": sampler,
            "scheduler": scheduler,
        }

        headers = self._build_headers(access_token, illustrious_api_key)
        warnings.simplefilter("ignore", InsecureRequestWarning)

        def _call(_) -> list[torch.Tensor]:
            p = base_params.copy()
            p["seed"] = random.randint(0, 2**32 - 1) if seed == -1 else seed

            try:
                r = requests.post(url, headers=headers, json=p, verify=False, timeout=120)
                r.raise_for_status()
                imgs64 = r.json().get("images", [])
                return [self._b64_to_tensor(b64) for b64 in imgs64]
            except Exception as e:
                print(f"IllustriousGenerate error: {e}")
                return []
        with ThreadPoolExecutor(max_workers=threads) as pool:
            tensors = [t for sub in pool.map(_call, range(n_requests)) for t in sub]

        if not tensors:
            tensors = [torch.zeros((height, width, 3), dtype=torch.float32)]

        batch = torch.stack(tensors, dim=0)      # shape [B,H,W,C]
        return (batch,)

CLASS_MAPPINGS = get_node_names_mappings(fundamental_classes)


CLASS_MAPPINGS, CLASS_NAMES = get_node_names_mappings(fundamental_classes)
validate(fundamental_classes)