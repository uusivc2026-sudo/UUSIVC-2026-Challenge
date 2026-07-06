# datasets/us_image_seg_dataset.py

from pathlib import Path
from typing import Optional

import numpy as np

from .base_dataset import BaseUSDataset
from .transforms import BasicImageTransform
from .task_defs import (
    IMAGE_SEG,
    BUS_IMAGE,
    IMAGE,
    LABEL_CLS,
    LABEL_SEG,
    TASK_TYPE,
    MODALITY,
    DATASET_NAME,
    IS_VIDEO,
    NUM_FRAMES,
    PATIENT_ID,
    CASE_ID,
    IGNORE_CLS_LABEL,
)


class USImageSegDataset(BaseUSDataset):
    """
    The organizers keep this baseline note in English for public release.
        image_path mask_path
    The organizers keep this baseline note in English for public release.
        /data/a.png /data/a_mask.png
        /data/b.png /data/b_mask.png
    """

    def __init__(
        self,
        list_file: str,
        dataset_name: str,
        transform: Optional[BasicImageTransform] = None,
    ):
        super().__init__(dataset_name=dataset_name)
        self.list_file = list_file
        self.root = str(Path(list_file).resolve().parent)
        self.samples = self.read_lines(list_file)
        self.label_mapping = self.parse_simple_label_config(str(Path(self.root) / "config.yaml"))
        self.transform = transform or BasicImageTransform((224, 224), binary_mask=False)

    def _map_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            if mask.shape[-1] in {1, 3, 4}:
                mask = mask[..., :3].max(axis=-1)
            elif mask.shape[0] in {1, 3, 4}:
                mask = mask[:3].max(axis=0)
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")

        if mask.dtype == np.bool_ or mask.max() <= 1:
            return mask.astype(np.uint8)
        if not self.label_mapping:
            return (mask > 0).astype(np.uint8)

        mapped = np.zeros_like(mask, dtype=np.uint8)
        for raw_value, class_idx in self.label_mapping.items():
            mapped[mask == raw_value] = class_idx
        return mapped

    def _resolve_sample_paths(self, line: str):
        parts = line.split("	") if "	" in line else line.split()
        if len(parts) == 1:
            image_rel = parts[0]
            mask_rel = parts[0]
            image_path = self.resolve_path(self.root, image_rel, default_subdir="imgs")
            mask_path = self.resolve_path(self.root, mask_rel, default_subdir="masks")
            return image_path, mask_path
        if len(parts) == 2:
            image_path = self.resolve_path(self.root, parts[0], default_subdir="imgs")
            mask_path = self.resolve_path(self.root, parts[1], default_subdir="masks")
            return image_path, mask_path
        raise ValueError(f"Unsupported segmentation sample format: '{line}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        line = self.samples[idx]
        image_path, mask_path = self._resolve_sample_paths(line)

        image = self.read_image(image_path)
        mask = self.read_mask(mask_path)
        mask = self._map_mask(mask)

        image_tensor, mask_tensor = self.transform(image, mask)

        sample = {
            IMAGE: image_tensor,
            LABEL_CLS: IGNORE_CLS_LABEL,
            LABEL_SEG: mask_tensor,   # [H,W], 0/1
            TASK_TYPE: IMAGE_SEG,
            MODALITY: BUS_IMAGE,
            DATASET_NAME: self.dataset_name,
            IS_VIDEO: False,
            NUM_FRAMES: 1,
            PATIENT_ID: None,
            CASE_ID: Path(image_path).stem,
        }
        return sample
