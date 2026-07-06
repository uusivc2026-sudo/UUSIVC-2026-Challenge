# datasets/base_dataset.py

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset


class BaseUSDataset(Dataset):
    def __init__(self, dataset_name: str):
        super().__init__()
        self.dataset_name = dataset_name

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        path = str(path)
        with Image.open(path) as image:
            return np.array(image)

    @staticmethod
    def read_mask(path: str) -> np.ndarray:
        path = str(path)
        with Image.open(path) as mask:
            return np.array(mask)

    @staticmethod
    def read_numpy_image(path: str) -> np.ndarray:
        array = np.load(path, allow_pickle=True)
        if array.ndim == 2:
            return array
        if array.ndim == 3:
            if array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
                return np.transpose(array, (1, 2, 0))
            return array
        raise ValueError(f"Unsupported numpy image shape: {array.shape}")

    @staticmethod
    def read_lines(txt_path: str) -> List[str]:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines

    @staticmethod
    def read_yaml(yaml_path: str) -> Dict:
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def list_sorted_frames(folder: str, suffixes=(".png", ".jpg", ".jpeg", ".bmp")):
        folder = Path(folder)
        files = [p for p in folder.iterdir() if p.suffix.lower() in suffixes]
        files = sorted(files)
        return [str(p) for p in files]

    @staticmethod
    def resolve_path(root: str, value: str, default_subdir: Optional[str] = None) -> str:
        value = value.strip().replace("\\", "/")
        root_path = Path(root)
        candidate = Path(value)

        if candidate.is_absolute() and candidate.exists():
            return str(candidate)

        if candidate.exists():
            return str(candidate.resolve())

        candidate = root_path / value
        if candidate.exists():
            return str(candidate)

        if default_subdir is not None:
            candidate = root_path / default_subdir / value
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(f"Cannot resolve path '{value}' under root '{root}'")

    @staticmethod
    def parse_simple_label_config(config_path: str) -> Dict[int, int]:
        if not Path(config_path).exists():
            return {}

        mapping = {}
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(":")
                if len(parts) < 3:
                    continue
                try:
                    class_idx = int(parts[0].strip())
                    raw_value = int(parts[-1].strip())
                except ValueError:
                    continue
                mapping[raw_value] = class_idx
        return mapping
