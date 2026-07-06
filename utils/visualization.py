from pathlib import Path
from typing import Dict, List, Optional
import csv

import numpy as np
from PIL import Image


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
        image = np.transpose(image, (1, 2, 0))
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape for visualization: {image.shape}")

    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
    return image


def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def _resize_mask_to_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    target_h, target_w = image.shape[:2]
    if mask.shape[:2] == (target_h, target_w):
        return mask
    pil_mask = Image.fromarray(_mask_to_uint8(mask))
    pil_mask = pil_mask.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
    return np.asarray(pil_mask)


def save_segmentation_visuals(
    output_dir: str,
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    case_id: Optional[str],
    loader_name: str,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = _ensure_uint8_rgb(image)
    pred_mask = _resize_mask_to_image(pred_mask, image)
    gt_mask = _resize_mask_to_image(gt_mask, image)
    pred_mask_u8 = _mask_to_uint8(pred_mask)
    gt_mask_u8 = _mask_to_uint8(gt_mask)

    pred_overlay = image.copy()
    gt_overlay = image.copy()
    overlap_overlay = image.copy()

    pred_region = pred_mask_u8 > 0
    gt_region = gt_mask_u8 > 0
    tp_region = pred_region & gt_region
    fp_region = pred_region & (~gt_region)
    fn_region = (~pred_region) & gt_region

    pred_overlay[pred_region] = (0.6 * pred_overlay[pred_region] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
    gt_overlay[gt_region] = (0.6 * gt_overlay[gt_region] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    overlap_overlay[tp_region] = (0.6 * overlap_overlay[tp_region] + 0.4 * np.array([255, 255, 0])).astype(np.uint8)
    overlap_overlay[fp_region] = (0.6 * overlap_overlay[fp_region] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
    overlap_overlay[fn_region] = (0.6 * overlap_overlay[fn_region] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

    prefix = f"{loader_name}__{case_id or 'sample'}"
    Image.fromarray(image).save(out_dir / f"{prefix}__image.png")
    Image.fromarray(gt_mask_u8).save(out_dir / f"{prefix}__gt_mask.png")
    Image.fromarray(pred_mask_u8).save(out_dir / f"{prefix}__pred_mask.png")
    Image.fromarray(gt_overlay).save(out_dir / f"{prefix}__gt_overlay.png")
    Image.fromarray(pred_overlay).save(out_dir / f"{prefix}__pred_overlay.png")
    Image.fromarray(overlap_overlay).save(out_dir / f"{prefix}__overlap.png")


def save_classification_csv(output_path: str, rows: List[Dict[str, object]]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["loader_name", "case_id", "gt", "pred", "score_0", "score_1"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
