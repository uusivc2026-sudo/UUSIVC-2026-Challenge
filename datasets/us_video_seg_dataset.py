from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base_dataset import BaseUSDataset
from .transforms import BasicVideoTransform
from .task_defs import (
    VIDEO_SEG,
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
    IGNORE_CLS_LABEL,
)


VISUAL_IMAGE = "visual_image"
VISUAL_GT_MASK = "visual_gt_mask"
OFFICIAL_GT_MASK = "official_gt_mask"
CEUS_RESTORE_META = "ceus_restore_meta"


class USVideoSegDataset(BaseUSDataset):
    """
    Supported inputs:
    - txt manifest: `video_path mask_path`
    - canonical directory mode: `videos/<split>` paired with `annotations/<split>`
    """

    def __init__(
        self,
        list_file: str,
        dataset_name: str,
        modality: str,
        num_frames: int = 8,
        transform: Optional[BasicVideoTransform] = None,
        ceus_dualview_fusion: bool = True,
        ceus_fusion_mode: str = "baseline_difference",
        ceus_midline_offset: int = 0,
        ceus_crop_top_ratio: float = 0.10,
        ceus_crop_bottom_ratio: float = 0.04,
        ceus_crop_side_ratio: float = 0.04,
        ceus_crop_center_ratio: float = 0.02,
        ceus_gt_side_mode: str = "auto",
        ceus_baseline_frames: int = 5,
    ):
        super().__init__(dataset_name=dataset_name)
        assert modality in {BUS_VIDEO, CEUS_VIDEO}
        self.list_file = list_file
        self.modality = modality
        self.num_frames = num_frames
        self.transform = transform or BasicVideoTransform((224, 224), binary_mask=True)
        self.ceus_dualview_fusion = ceus_dualview_fusion
        self.ceus_fusion_mode = ceus_fusion_mode
        self.ceus_midline_offset = ceus_midline_offset
        self.ceus_crop_top_ratio = ceus_crop_top_ratio
        self.ceus_crop_bottom_ratio = ceus_crop_bottom_ratio
        self.ceus_crop_side_ratio = ceus_crop_side_ratio
        self.ceus_crop_center_ratio = ceus_crop_center_ratio
        self.ceus_gt_side_mode = ceus_gt_side_mode
        self.ceus_baseline_frames = ceus_baseline_frames
        self.samples = self._build_samples(list_file)

    def _build_samples(self, source: str) -> List[Tuple[str, str]]:
        source_path = Path(source)
        if source_path.is_file():
            root = str(source_path.resolve().parent)
            samples = []
            for line in self.read_lines(source):
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) == 1:
                    stem = parts[0]
                    video_path = self.resolve_path(root, stem, default_subdir="ceus")
                    mask_path = self.resolve_path(root, stem.replace(".npy", ".npz"), default_subdir="mask")
                elif len(parts) == 2:
                    video_path = self.resolve_path(root, parts[0])
                    mask_path = self.resolve_path(root, parts[1])
                else:
                    raise ValueError(f"Unsupported video segmentation sample format: '{line}'")
                samples.append((video_path, mask_path))
            return samples

        if source_path.is_dir():
            video_files = sorted(source_path.glob("*.npy"))
            if not video_files:
                raise RuntimeError(f"No .npy files found under {source}")

            source_str = str(source_path)
            if "\\videos\\" in source_str or "/videos/" in source_str:
                annotation_dir = Path(str(source_path).replace("\\videos\\", "\\annotations\\").replace("/videos/", "/annotations/"))
            else:
                raise ValueError("Directory-mode video segmentation expects a videos/<split> directory.")

            samples = []
            for video_file in video_files:
                ann_path = annotation_dir / f"{video_file.stem}.npz"
                if not ann_path.exists():
                    raise FileNotFoundError(f"Annotation file not found for {video_file}: {ann_path}")
                samples.append((str(video_file), str(ann_path)))
            return samples

        raise FileNotFoundError(f"Video segmentation source not found: {source}")

    def __len__(self):
        return len(self.samples)

    def _sample_pair_paths(self, frame_paths: List[str], mask_paths: List[str]):
        if len(frame_paths) == 0 or len(mask_paths) == 0:
            raise ValueError("Empty frame folder or mask folder.")
        if len(frame_paths) != len(mask_paths):
            raise ValueError(f"Frame/mask length mismatch: {len(frame_paths)} vs {len(mask_paths)}")

        indices = self._build_sampling_indices(len(frame_paths))
        frame_paths = [frame_paths[i] for i in indices]
        mask_paths = [mask_paths[i] for i in indices]
        return frame_paths, mask_paths

    def _build_sampling_indices(self, total: int) -> List[int]:
        if total <= 0:
            raise ValueError("Video contains no frames.")
        if total >= self.num_frames:
            return np.linspace(0, total - 1, self.num_frames).astype(int).tolist()
        indices = list(range(total))
        while len(indices) < self.num_frames:
            indices.append(indices[-1])
        return indices

    def _sample_aligned_video_and_mask(
        self,
        video: np.ndarray,
        masks: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        video_len = int(video.shape[0])
        mask_len = int(masks.shape[0])
        if video_len <= 0 or mask_len <= 0:
            raise ValueError(f"Invalid video/mask lengths: {video_len}, {mask_len}")

        if self.num_frames == 1:
            positions = np.array([0.0], dtype=np.float32)
        else:
            positions = np.linspace(0.0, 1.0, self.num_frames, dtype=np.float32)

        video_idx = np.clip(np.round(positions * max(video_len - 1, 0)).astype(int), 0, max(video_len - 1, 0))
        mask_idx = np.clip(np.round(positions * max(mask_len - 1, 0)).astype(int), 0, max(mask_len - 1, 0))

        frames = [video[i] for i in video_idx.tolist()]
        masks = [masks[i] for i in mask_idx.tolist()]
        return frames, masks

    @staticmethod
    def _resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return np.array(Image.fromarray(image).resize(size, resample=Image.BILINEAR))

    @staticmethod
    def _rgb_to_gray(frame: np.ndarray) -> np.ndarray:
        frame = frame.astype(np.float32)
        if frame.max() <= 1.0:
            frame = frame * 255.0
        gray = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        return np.clip(gray, 0, 255).astype(np.uint8)

    @staticmethod
    def _crop_half(
        image: np.ndarray,
        crop_top_ratio: float,
        crop_bottom_ratio: float,
        crop_left_ratio: float,
        crop_right_ratio: float,
    ) -> np.ndarray:
        h, w = image.shape[:2]
        top = int(round(h * crop_top_ratio))
        bottom = h - int(round(h * crop_bottom_ratio))
        left = int(round(w * crop_left_ratio))
        right = w - int(round(w * crop_right_ratio))

        top = max(0, min(top, h - 1))
        bottom = max(top + 1, min(bottom, h))
        left = max(0, min(left, w - 1))
        right = max(left + 1, min(right, w))
        return image[top:bottom, left:right]

    @staticmethod
    def _make_ceus_baseline(
        ceus_frames: List[np.ndarray],
        baseline_frames: int,
    ) -> np.ndarray:
        if not ceus_frames:
            raise ValueError("Cannot build CEUS baseline from an empty frame list.")
        end = min(len(ceus_frames), max(int(baseline_frames), 1))
        return np.stack(ceus_frames[:end], axis=0).astype(np.float32).mean(axis=0).astype(np.uint8)

    @staticmethod
    def _ceus_colorfulness_score(frame: np.ndarray) -> float:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] < 3:
            return 0.0
        rgb = frame[..., :3].astype(np.float32)
        rg = np.abs(rgb[..., 0] - rgb[..., 1])
        rb = np.abs(rgb[..., 0] - rgb[..., 2])
        gb = np.abs(rgb[..., 1] - rgb[..., 2])
        return float((rg + rb + gb).mean())

    def _resolve_ceus_side_from_content(self, frames: List[np.ndarray]) -> str:
        left_score = 0.0
        right_score = 0.0
        for frame in frames:
            h, w = frame.shape[:2]
            mid = int(round(w / 2.0)) + int(self.ceus_midline_offset)
            mid = max(8, min(w - 8, mid))
            left_half = self._crop_half(
                frame[:, :mid],
                crop_top_ratio=self.ceus_crop_top_ratio,
                crop_bottom_ratio=self.ceus_crop_bottom_ratio,
                crop_left_ratio=self.ceus_crop_side_ratio,
                crop_right_ratio=self.ceus_crop_center_ratio,
            )
            right_half = self._crop_half(
                frame[:, mid:],
                crop_top_ratio=self.ceus_crop_top_ratio,
                crop_bottom_ratio=self.ceus_crop_bottom_ratio,
                crop_left_ratio=self.ceus_crop_center_ratio,
                crop_right_ratio=self.ceus_crop_side_ratio,
            )
            left_score += self._ceus_colorfulness_score(left_half)
            right_score += self._ceus_colorfulness_score(right_half)
        return "right" if right_score >= left_score else "left"

    def _select_ceus_sides(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
        gt_side: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mode = gt_side or self.ceus_gt_side_mode
        if mode == "right" or mode == "fixed_right":
            return right_frame, left_frame, right_mask
        if mode == "left" or mode == "fixed_left":
            return left_frame, right_frame, left_mask
        if mode != "auto":
            raise ValueError(f"Unsupported CEUS gt side mode: {mode}")

        left_sum = int((left_mask > 0).sum())
        right_sum = int((right_mask > 0).sum())
        use_right = right_sum >= left_sum
        if use_right:
            return right_frame, left_frame, right_mask
        return left_frame, right_frame, left_mask

    def _resolve_ceus_gt_side(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> str:
        if self.ceus_gt_side_mode == "fixed_right":
            return "right"
        if self.ceus_gt_side_mode == "fixed_left":
            return "left"
        if self.ceus_gt_side_mode != "auto":
            raise ValueError(f"Unsupported CEUS gt side mode: {self.ceus_gt_side_mode}")

        left_sum = 0
        right_sum = 0
        for frame, mask in zip(frames, masks):
            w = frame.shape[1]
            mid = int(round(w / 2.0)) + int(self.ceus_midline_offset)
            mid = max(8, min(w - 8, mid))
            left_sum += int((mask[:, :mid] > 0).sum())
            right_sum += int((mask[:, mid:] > 0).sum())
        if left_sum == 0 and right_sum == 0:
            ceus_side = self._resolve_ceus_side_from_content(frames)
            return "left" if ceus_side == "right" else "right"
        return "right" if right_sum >= left_sum else "left"

    def _split_ceus_dualview_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        ceus_baseline: Optional[np.ndarray] = None,
        gt_side: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        mid = int(round(w / 2.0)) + int(self.ceus_midline_offset)
        mid = max(8, min(w - 8, mid))

        left_half = frame[:, :mid]
        right_half = frame[:, mid:]
        left_mask = mask[:, :mid]
        right_mask = mask[:, mid:]

        left_half = self._crop_half(
            left_half,
            crop_top_ratio=self.ceus_crop_top_ratio,
            crop_bottom_ratio=self.ceus_crop_bottom_ratio,
            crop_left_ratio=self.ceus_crop_side_ratio,
            crop_right_ratio=self.ceus_crop_center_ratio,
        )
        right_half = self._crop_half(
            right_half,
            crop_top_ratio=self.ceus_crop_top_ratio,
            crop_bottom_ratio=self.ceus_crop_bottom_ratio,
            crop_left_ratio=self.ceus_crop_center_ratio,
            crop_right_ratio=self.ceus_crop_side_ratio,
        )
        left_mask = self._crop_half(
            left_mask,
            crop_top_ratio=self.ceus_crop_top_ratio,
            crop_bottom_ratio=self.ceus_crop_bottom_ratio,
            crop_left_ratio=self.ceus_crop_side_ratio,
            crop_right_ratio=self.ceus_crop_center_ratio,
        )
        right_mask = self._crop_half(
            right_mask,
            crop_top_ratio=self.ceus_crop_top_ratio,
            crop_bottom_ratio=self.ceus_crop_bottom_ratio,
            crop_left_ratio=self.ceus_crop_center_ratio,
            crop_right_ratio=self.ceus_crop_side_ratio,
        )

        bmode_half, ceus_half, bmode_mask = self._select_ceus_sides(
            left_frame=left_half,
            right_frame=right_half,
            left_mask=left_mask,
            right_mask=right_mask,
            gt_side=gt_side,
        )

        target_h, target_w = bmode_mask.shape[:2]
        ceus_gray = self._rgb_to_gray(ceus_half)
        bmode_gray = self._rgb_to_gray(bmode_half)
        ceus_gray = self._resize_image(ceus_gray, (target_w, target_h))
        bmode_gray = self._resize_image(bmode_gray, (target_w, target_h))

        if self.ceus_fusion_mode == "difference_enhanced":
            fusion = np.abs(ceus_gray.astype(np.int16) - bmode_gray.astype(np.int16)).astype(np.uint8)
        elif self.ceus_fusion_mode == "baseline_difference":
            if ceus_baseline is None:
                fusion = np.zeros_like(ceus_gray, dtype=np.uint8)
            else:
                ceus_baseline = self._resize_image(ceus_baseline, (target_w, target_h))
                fusion = np.clip(
                    ceus_gray.astype(np.float32) - ceus_baseline.astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)
        elif self.ceus_fusion_mode == "mean_overlap":
            fusion = np.clip(
                0.5 * ceus_gray.astype(np.float32) + 0.5 * bmode_gray.astype(np.float32),
                0,
                255,
            ).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported CEUS fusion mode: {self.ceus_fusion_mode}")

        fused_frame = np.stack([bmode_gray, ceus_gray, fusion], axis=-1)
        fused_mask = self._to_binary_mask_255(bmode_mask)
        return fused_frame, fused_mask

    def _build_ceus_restore_meta(self, frame_shape: Tuple[int, ...], gt_side: str) -> Dict[str, Any]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        mid = int(round(w / 2.0)) + int(self.ceus_midline_offset)
        mid = max(8, min(w - 8, mid))
        top = int(round(h * self.ceus_crop_top_ratio))
        bottom = h - int(round(h * self.ceus_crop_bottom_ratio))

        if gt_side == "right":
            half_left = mid
            half_w = w - mid
            crop_left_ratio = self.ceus_crop_center_ratio
            crop_right_ratio = self.ceus_crop_side_ratio
        else:
            half_left = 0
            half_w = mid
            crop_left_ratio = self.ceus_crop_side_ratio
            crop_right_ratio = self.ceus_crop_center_ratio

        left = half_left + int(round(half_w * crop_left_ratio))
        right = half_left + half_w - int(round(half_w * crop_right_ratio))
        top = max(0, min(top, h - 1))
        bottom = max(top + 1, min(bottom, h))
        left = max(0, min(left, w - 1))
        right = max(left + 1, min(right, w))
        return {"shape": (h, w), "crop_box": (top, bottom, left, right), "gt_side": gt_side}

    def _build_ceus_visual_reference(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        gt_side: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        frame_idx = int(len(frames) // 2)
        frame = frames[frame_idx]
        mask = masks[frame_idx]
        h, w = frame.shape[:2]
        mid = int(round(w / 2.0)) + int(self.ceus_midline_offset)
        mid = max(8, min(w - 8, mid))

        left_half = frame[:, :mid]
        right_half = frame[:, mid:]
        left_mask = mask[:, :mid]
        right_mask = mask[:, mid:]

        left_half = self._crop_half(
            left_half,
            crop_top_ratio=self.ceus_crop_top_ratio,
            crop_bottom_ratio=self.ceus_crop_bottom_ratio,
            crop_left_ratio=self.ceus_crop_side_ratio,
            crop_right_ratio=self.ceus_crop_center_ratio,
        )
        right_half = self._crop_half(
            right_half,
            crop_top_ratio=self.ceus_crop_top_ratio,
            crop_bottom_ratio=self.ceus_crop_bottom_ratio,
            crop_left_ratio=self.ceus_crop_center_ratio,
            crop_right_ratio=self.ceus_crop_side_ratio,
        )
        left_mask = self._crop_half(
            left_mask,
            crop_top_ratio=self.ceus_crop_top_ratio,
            crop_bottom_ratio=self.ceus_crop_bottom_ratio,
            crop_left_ratio=self.ceus_crop_side_ratio,
            crop_right_ratio=self.ceus_crop_center_ratio,
        )
        right_mask = self._crop_half(
            right_mask,
            crop_top_ratio=self.ceus_crop_top_ratio,
            crop_bottom_ratio=self.ceus_crop_bottom_ratio,
            crop_left_ratio=self.ceus_crop_center_ratio,
            crop_right_ratio=self.ceus_crop_side_ratio,
        )

        bmode_half, _, bmode_mask = self._select_ceus_sides(
            left_frame=left_half,
            right_frame=right_half,
            left_mask=left_mask,
            right_mask=right_mask,
            gt_side=gt_side,
        )
        return bmode_half, self._to_binary_mask_255(bmode_mask)

    def _fuse_sampled_ceus_video_and_mask(
        self,
        sampled_frames: List[np.ndarray],
        sampled_masks: List[np.ndarray],
        gt_side: str,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        baseline_frames = []
        if self.ceus_fusion_mode == "baseline_difference":
            for frame, mask in zip(sampled_frames, sampled_masks):
                h, w = frame.shape[:2]
                mid = int(round(w / 2.0)) + int(self.ceus_midline_offset)
                mid = max(8, min(w - 8, mid))
                left_half = frame[:, :mid]
                right_half = frame[:, mid:]
                left_mask = mask[:, :mid]
                right_mask = mask[:, mid:]
                left_half = self._crop_half(
                    left_half,
                    crop_top_ratio=self.ceus_crop_top_ratio,
                    crop_bottom_ratio=self.ceus_crop_bottom_ratio,
                    crop_left_ratio=self.ceus_crop_side_ratio,
                    crop_right_ratio=self.ceus_crop_center_ratio,
                )
                right_half = self._crop_half(
                    right_half,
                    crop_top_ratio=self.ceus_crop_top_ratio,
                    crop_bottom_ratio=self.ceus_crop_bottom_ratio,
                    crop_left_ratio=self.ceus_crop_center_ratio,
                    crop_right_ratio=self.ceus_crop_side_ratio,
                )
                left_mask = self._crop_half(
                    left_mask,
                    crop_top_ratio=self.ceus_crop_top_ratio,
                    crop_bottom_ratio=self.ceus_crop_bottom_ratio,
                    crop_left_ratio=self.ceus_crop_side_ratio,
                    crop_right_ratio=self.ceus_crop_center_ratio,
                )
                right_mask = self._crop_half(
                    right_mask,
                    crop_top_ratio=self.ceus_crop_top_ratio,
                    crop_bottom_ratio=self.ceus_crop_bottom_ratio,
                    crop_left_ratio=self.ceus_crop_center_ratio,
                    crop_right_ratio=self.ceus_crop_side_ratio,
                )
                _, ceus_half, _ = self._select_ceus_sides(
                    left_half,
                    right_half,
                    left_mask,
                    right_mask,
                    gt_side=gt_side,
                )
                baseline_frames.append(self._rgb_to_gray(ceus_half))

        fused_frames: List[np.ndarray] = []
        fused_masks: List[np.ndarray] = []
        ceus_baseline = None
        if baseline_frames:
            ceus_baseline = self._make_ceus_baseline(
                ceus_frames=baseline_frames,
                baseline_frames=self.ceus_baseline_frames,
            )

        for frame, mask in zip(sampled_frames, sampled_masks):
            baseline = ceus_baseline if ceus_baseline is not None else None
            fused_frame, fused_mask = self._split_ceus_dualview_frame(frame, mask, baseline, gt_side=gt_side)
            fused_frames.append(fused_frame)
            fused_masks.append(fused_mask)
        return fused_frames, fused_masks

    def _fuse_ceus_video_and_mask(
        self,
        video: np.ndarray,
        masks: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        sampled_frames, sampled_masks = self._sample_aligned_video_and_mask(video, masks)
        gt_side = self._resolve_ceus_gt_side(sampled_frames, sampled_masks)
        baseline_frames = []
        if self.ceus_fusion_mode == "baseline_difference":
            for frame, mask in zip(sampled_frames, sampled_masks):
                h, w = frame.shape[:2]
                mid = int(round(w / 2.0)) + int(self.ceus_midline_offset)
                mid = max(8, min(w - 8, mid))
                left_half = frame[:, :mid]
                right_half = frame[:, mid:]
                left_mask = mask[:, :mid]
                right_mask = mask[:, mid:]
                left_half = self._crop_half(
                    left_half,
                    crop_top_ratio=self.ceus_crop_top_ratio,
                    crop_bottom_ratio=self.ceus_crop_bottom_ratio,
                    crop_left_ratio=self.ceus_crop_side_ratio,
                    crop_right_ratio=self.ceus_crop_center_ratio,
                )
                right_half = self._crop_half(
                    right_half,
                    crop_top_ratio=self.ceus_crop_top_ratio,
                    crop_bottom_ratio=self.ceus_crop_bottom_ratio,
                    crop_left_ratio=self.ceus_crop_center_ratio,
                    crop_right_ratio=self.ceus_crop_side_ratio,
                )
                left_mask = self._crop_half(
                    left_mask,
                    crop_top_ratio=self.ceus_crop_top_ratio,
                    crop_bottom_ratio=self.ceus_crop_bottom_ratio,
                    crop_left_ratio=self.ceus_crop_side_ratio,
                    crop_right_ratio=self.ceus_crop_center_ratio,
                )
                right_mask = self._crop_half(
                    right_mask,
                    crop_top_ratio=self.ceus_crop_top_ratio,
                    crop_bottom_ratio=self.ceus_crop_bottom_ratio,
                    crop_left_ratio=self.ceus_crop_center_ratio,
                    crop_right_ratio=self.ceus_crop_side_ratio,
                )
                _, ceus_half, _ = self._select_ceus_sides(
                    left_half,
                    right_half,
                    left_mask,
                    right_mask,
                    gt_side=gt_side,
                )
                baseline_frames.append(self._rgb_to_gray(ceus_half))

        fused_frames: List[np.ndarray] = []
        fused_masks: List[np.ndarray] = []
        ceus_baseline = None
        if baseline_frames:
            ceus_baseline = self._make_ceus_baseline(
                ceus_frames=baseline_frames,
                baseline_frames=self.ceus_baseline_frames,
            )

        for frame, mask in zip(sampled_frames, sampled_masks):
            baseline = None
            if ceus_baseline is not None:
                baseline = ceus_baseline
            fused_frame, fused_mask = self._split_ceus_dualview_frame(frame, mask, baseline, gt_side=gt_side)
            fused_frames.append(fused_frame)
            fused_masks.append(fused_mask)
        return fused_frames, fused_masks

    @staticmethod
    def _load_video_npy(video_path: str) -> np.ndarray:
        video = np.load(video_path, allow_pickle=True)
        if video.ndim == 4 and video.shape[-1] in {1, 3}:
            return video
        if video.ndim == 4 and video.shape[0] in {1, 3}:
            return np.transpose(video, (1, 2, 3, 0))
        if video.ndim == 3:
            return video[..., None]
        raise ValueError(f"Unsupported video array shape: {video.shape}")

    @staticmethod
    def _to_binary_mask_255(masks: np.ndarray) -> np.ndarray:
        return (masks > 0).astype(np.uint8) * 255

    @staticmethod
    def _load_mask_array(mask_path: str) -> np.ndarray:
        ann = np.load(mask_path, allow_pickle=True)
        if "mask" in ann.files:
            masks = ann["mask"]
            if masks.ndim == 2:
                masks = masks[None, ...]
            return USVideoSegDataset._to_binary_mask_255(masks)
        if "fnum_mask" in ann.files:
            frame_mask_dict = ann["fnum_mask"].item()
            ordered_keys = sorted(frame_mask_dict.keys(), key=lambda x: int(x))
            masks = [np.asarray(frame_mask_dict[k]) for k in ordered_keys]
            masks = np.stack(masks, axis=0)
            return USVideoSegDataset._to_binary_mask_255(masks)
        raise KeyError(f"No supported mask key found in {mask_path}; expected 'mask' or 'fnum_mask'")

    def __getitem__(self, idx: int):
        video_path, mask_path = self.samples[idx]

        visual_image = None
        visual_gt_mask = None
        official_gt_mask = None
        ceus_restore_meta = None

        if video_path.endswith(".npy") and mask_path.endswith(".npz"):
            video = self._load_video_npy(video_path)
            masks = self._load_mask_array(mask_path)
            if self.modality == CEUS_VIDEO and self.ceus_dualview_fusion:
                sampled_frames, sampled_masks = self._sample_aligned_video_and_mask(video, masks)
                gt_side = self._resolve_ceus_gt_side(sampled_frames, sampled_masks)
                visual_image, visual_gt_mask = self._build_ceus_visual_reference(
                    sampled_frames,
                    sampled_masks,
                    gt_side=gt_side,
                )
                official_gt_mask = self._to_binary_mask_255(masks[0])
                ceus_restore_meta = self._build_ceus_restore_meta(sampled_frames[0].shape, gt_side)
                frames, masks = self._fuse_sampled_ceus_video_and_mask(sampled_frames, sampled_masks, gt_side)
            else:
                frames, masks = self._sample_aligned_video_and_mask(video, masks)
        else:
            frame_paths = self.list_sorted_frames(video_path)
            mask_paths = self.list_sorted_frames(mask_path)
            frame_paths, mask_paths = self._sample_pair_paths(frame_paths, mask_paths)
            frames = [self.read_image(p) for p in frame_paths]
            masks = [self.read_mask(p) for p in mask_paths]

        video_tensor, mask_tensor = self.transform(frames, masks)

        sample = {
            IMAGE: video_tensor,
            LABEL_CLS: IGNORE_CLS_LABEL,
            LABEL_SEG: mask_tensor,
            TASK_TYPE: VIDEO_SEG,
            MODALITY: self.modality,
            DATASET_NAME: self.dataset_name,
            IS_VIDEO: True,
            NUM_FRAMES: self.num_frames,
            PATIENT_ID: None,
            CASE_ID: Path(video_path).stem,
        }
        if visual_image is not None and visual_gt_mask is not None:
            sample[VISUAL_IMAGE] = visual_image
            sample[VISUAL_GT_MASK] = visual_gt_mask
        if official_gt_mask is not None and ceus_restore_meta is not None:
            sample[OFFICIAL_GT_MASK] = official_gt_mask
            sample[CEUS_RESTORE_META] = ceus_restore_meta
        return sample
