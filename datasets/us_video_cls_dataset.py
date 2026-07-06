from pathlib import Path
import json
import subprocess
from typing import List, Optional, Tuple
import numpy as np

from .base_dataset import BaseUSDataset
from .transforms import BasicVideoTransform
from .task_defs import (
    VIDEO_CLS,
    BUS_VIDEO,
    CEUS_VIDEO,
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


class USVideoClsDataset(BaseUSDataset):
    """
    Supported inputs:
    - txt manifest: `video_path label`
    - split dir with `0/` and `1/` subfolders containing `.npy`, `.mp4`, or frame folders
    """

    def __init__(
        self,
        list_file: str,
        dataset_name: str,
        modality: str,
        num_frames: int = 8,
        transform: Optional[BasicVideoTransform] = None,
    ):
        super().__init__(dataset_name=dataset_name)
        assert modality in {BUS_VIDEO, CEUS_VIDEO}
        self.root = str(Path(list_file).resolve().parent) if Path(list_file).is_file() else str(Path(list_file).resolve())
        self.samples = self._build_samples(list_file)
        self.modality = modality
        self.num_frames = num_frames
        self.transform = transform or BasicVideoTransform((224, 224))

    def _build_samples(self, source: str) -> List[Tuple[str, int]]:
        source_path = Path(source)
        if source_path.is_file():
            samples = []
            for line in self.read_lines(source):
                path_str = line
                label = None

                if "\t" in line:
                    left, right = line.rsplit("\t", 1)
                    if right.isdigit():
                        path_str = left
                        label = int(right)
                else:
                    parts = line.rsplit(maxsplit=1)
                    if len(parts) == 2 and parts[1].isdigit():
                        path_str, label_str = parts
                        label = int(label_str)

                path = self.resolve_path(self.root, path_str)
                if label is None:
                    label = int(Path(path).parent.name)
                samples.append((path, label))
            return samples

        if source_path.is_dir():
            samples = []
            for class_name in ("0", "1"):
                class_dir = source_path / class_name
                if not class_dir.is_dir():
                    continue
                for item in sorted(class_dir.iterdir()):
                    if item.is_dir() or item.suffix.lower() in {".npy", ".mp4", ".avi", ".mov"}:
                        samples.append((str(item.resolve()), int(class_name)))
            if not samples:
                raise RuntimeError(f"No video classification samples found under {source}")
            return samples

        raise FileNotFoundError(f"Video classification source not found: {source}")

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, num_frames_total: int) -> List[int]:
        if num_frames_total <= 0:
            raise ValueError("Video contains no frames.")
        if num_frames_total >= self.num_frames:
            return np.linspace(0, num_frames_total - 1, self.num_frames).astype(int).tolist()
        indices = list(range(num_frames_total))
        while len(indices) < self.num_frames:
            indices.append(indices[-1])
        return indices

    @staticmethod
    def _to_thwc(video: np.ndarray) -> np.ndarray:
        if video.ndim == 4 and video.shape[-1] in {1, 3}:
            return video
        if video.ndim == 4 and video.shape[0] in {1, 3}:
            return np.transpose(video, (1, 2, 3, 0))
        if video.ndim == 3:
            return video[..., None]
        raise ValueError(f"Unsupported video array shape: {video.shape}")

    @staticmethod
    def _load_video_via_ffmpeg(video_path: str) -> np.ndarray:
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            video_path,
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        meta = json.loads(probe.stdout)
        stream = meta["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])

        read_cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-",
        ]
        proc = subprocess.run(read_cmd, capture_output=True, check=True)
        frame_size = width * height * 3
        raw = np.frombuffer(proc.stdout, dtype=np.uint8)
        if raw.size % frame_size != 0:
            raise ValueError(f"Unexpected raw frame buffer size for {video_path}")
        return raw.reshape(-1, height, width, 3)

    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        path = Path(video_path)
        suffix = path.suffix.lower()

        if path.is_dir():
            frame_paths = self.list_sorted_frames(video_path)
            indices = self._sample_indices(len(frame_paths))
            return [self.read_image(frame_paths[i]) for i in indices]

        if suffix == ".npy":
            video = self._to_thwc(np.load(video_path, allow_pickle=True))
            indices = self._sample_indices(video.shape[0])
            return [video[i] for i in indices]

        if suffix in {".mp4", ".avi", ".mov"}:
            video = self._load_video_via_ffmpeg(video_path)
            indices = self._sample_indices(video.shape[0])
            return [video[i] for i in indices]

        raise ValueError(f"Unsupported video classification sample: {video_path}")

    def __getitem__(self, idx: int):
        video_path, label_cls = self.samples[idx]
        frames = self._load_video_frames(video_path)
        video_tensor, _ = self.transform(frames, None)

        sample = {
            IMAGE: video_tensor,
            LABEL_CLS: label_cls,
            LABEL_SEG: IGNORE_SEG_LABEL,
            TASK_TYPE: VIDEO_CLS,
            MODALITY: self.modality,
            DATASET_NAME: self.dataset_name,
            IS_VIDEO: True,
            NUM_FRAMES: self.num_frames,
            PATIENT_ID: None,
            CASE_ID: Path(video_path).stem,
        }
        return sample
