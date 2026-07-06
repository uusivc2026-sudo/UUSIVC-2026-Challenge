from pathlib import Path
from typing import List, Optional

from .base_dataset import BaseUSDataset
from .transforms import BasicImageTransform
from .task_defs import (
    IMAGE_CLS,
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
    IGNORE_SEG_LABEL,
)


class USImageClsDataset(BaseUSDataset):
    """
    The organizers keep this baseline note in English for public release.
        image_path label
    The organizers keep this baseline note in English for public release.
        /data/a.png 0
        /data/b.png 1
    """

    def __init__(
        self,
        list_file: str,
        dataset_name: str,
        transform: Optional[BasicImageTransform] = None,
    ):
        super().__init__(dataset_name=dataset_name)
        self.list_file = list_file
        self.root = str(Path(list_file).resolve().parent) if Path(list_file).is_file() else str(Path(list_file).resolve())
        self.samples = self._build_samples(list_file)
        self.transform = transform or BasicImageTransform((224, 224))

    def _scan_two_class_dir(self, root: Path) -> List[str]:
        samples = []
        for class_name in ("0", "1"):
            class_dir = root / class_name
            if not class_dir.is_dir():
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}:
                    continue
                samples.append(f"{image_path} {class_name}")
        if not samples:
            raise RuntimeError(f"No image classification samples found under {root}")
        return samples

    def _build_samples(self, source: str) -> List[str]:
        source_path = Path(source)
        if source_path.is_file():
            return self.read_lines(source)
        if source_path.is_dir():
            return self._scan_two_class_dir(source_path)
        raise FileNotFoundError(f"Image classification source not found: {source}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        line = self.samples[idx]
        if "	" in line:
            parts = line.rsplit("	", 1)
        else:
            parts = line.rsplit(maxsplit=1)
        if len(parts) == 2 and parts[1].strip().lstrip("-").isdigit():
            image_path = self.resolve_path(self.root, parts[0])
            label_cls = int(parts[1])
        elif len(parts) == 1:
            image_path = self.resolve_path(self.root, parts[0])
            label_cls = int(Path(parts[0]).parent.name)
        else:
            raise ValueError(f"Unsupported classification sample format: '{line}'")

        if image_path.endswith(".npy"):
            image = self.read_numpy_image(image_path)
        else:
            image = self.read_image(image_path)
        image_tensor, _ = self.transform(image, None)

        sample = {
            IMAGE: image_tensor,
            LABEL_CLS: label_cls,
            LABEL_SEG: IGNORE_SEG_LABEL,
            TASK_TYPE: IMAGE_CLS,
            MODALITY: BUS_IMAGE,
            DATASET_NAME: self.dataset_name,
            IS_VIDEO: False,
            NUM_FRAMES: 1,
            PATIENT_ID: None,
            CASE_ID: Path(image_path).stem,
        }
        return sample
