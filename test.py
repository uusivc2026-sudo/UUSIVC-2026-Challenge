import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets.build_loader import build_stage1_seg_loaders, build_stage2_cls_loaders
from datasets.uusivc2026_paths import expand_fixed_uusivc_data_cfg
from models.build_model import build_model
from utils.auc_utils import binary_accuracy_from_logits, binary_auc_from_logits, stack_numpy
from utils.checkpoint import resolve_checkpoint_reference
from utils.metrics import compute_binary_seg_score_from_logits, compute_ceus_official_score_from_logits
from utils.visualization import save_classification_csv, save_segmentation_visuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate on the labeled local holdout split created from TRAIN data.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to yaml config. Defaults to configs/stage1_seg.yaml or configs/stage2_cls.yaml.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["stage1_seg", "stage2_cls"],
        help="Override run.stage from config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Evaluation device",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="UUSIVC2026 public data root containing TRAIN/ and VAL/ packages. If omitted, uses UUSIVC2026_DATA_ROOT.",
    )
    parser.add_argument(
        "--local-val-fraction",
        type=float,
        default=None,
        help="Fraction of labeled TRAIN data held out for local validation. Use the same value as training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, resolve from --which under trainer.save_dir.",
    )
    parser.add_argument(
        "--which",
        type=str,
        default="best",
        choices=["best", "latest"],
        help="Which checkpoint to load when --checkpoint is not provided.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which labeled local split to evaluate. Public VAL has no labels; use predict.py for VAL submissions.",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=20,
        help="Maximum number of segmentation samples to visualize across all loaders. Set 0 to disable.",
    )
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint_path(cfg: Dict[str, Any], stage: str, explicit_path: Optional[str], which: str) -> str:
    prefix = "best_stage1_seg" if stage == "stage1_seg" else "best_stage2_cls"
    if explicit_path:
        return resolve_checkpoint_reference(explicit_path, prefix=prefix)
    else:
        save_dir = Path(cfg["trainer"]["save_dir"])
        filename = f"{prefix}.pth"
        if which == "latest":
            filename = "latest_stage1_seg.pth" if stage == "stage1_seg" else "latest_stage2_cls.pth"
        ckpt_path = save_dir / filename

    if not ckpt_path.exists() and which == "best":
        return resolve_checkpoint_reference(save_dir, prefix=prefix)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return str(ckpt_path.resolve())


def _resolve_split_entry(data_cfg: Dict[str, Any], base_name: str, split: str) -> Optional[Dict[str, Any]]:
    preferred_key = f"{base_name}_{split}"
    if preferred_key in data_cfg:
        return data_cfg[preferred_key]

    if split == "test":
        fallback_key = f"{base_name}_val"
        if fallback_key in data_cfg:
            return data_cfg[fallback_key]
    return None


def build_eval_data_cfg(cfg: Dict[str, Any], stage: str, split: str) -> Dict[str, Any]:
    src = expand_fixed_uusivc_data_cfg(cfg["data"])
    eval_cfg = {
        "num_workers": src.get("num_workers", 0),
        "num_frames": src.get("num_frames", 8),
        "batch_size_image_seg": src.get("batch_size_image_seg", 4),
        "batch_size_video_seg": src.get("batch_size_video_seg", 1),
        "batch_size_image_cls": src.get("batch_size_image_cls", 16),
        "batch_size_video_cls": src.get("batch_size_video_cls", 4),
        "use_balanced_sampler_image_cls": False,
        "use_balanced_sampler_video_cls": False,
    }
    for key in ("data_root", "root", "manifest_cache_dir", "resolved_data_root"):
        if key in src:
            eval_cfg[key] = src[key]
    for key in (
        "ceus_dualview_fusion",
        "ceus_fusion_mode",
        "ceus_midline_offset",
        "ceus_crop_top_ratio",
        "ceus_crop_bottom_ratio",
        "ceus_crop_side_ratio",
        "ceus_crop_center_ratio",
        "ceus_gt_side_mode",
        "ceus_baseline_frames",
    ):
        if key in src:
            eval_cfg[key] = src[key]

    if stage == "stage1_seg":
        key_pairs = [
            ("image_seg", "image_seg_val"),
            ("cardiac_video_seg", "cardiac_video_seg_val"),
            ("ceus_video_seg", "ceus_video_seg_val"),
        ]
    else:
        key_pairs = [
            ("image_cls", "image_cls_val"),
            ("standard_video_cls", "standard_video_cls_val"),
            ("ceus_video_cls", "ceus_video_cls_val"),
        ]

    for base_name, dst_key in key_pairs:
        value = _resolve_split_entry(src, base_name, split)
        if value is not None:
            eval_cfg[dst_key] = value

    return eval_cfg


def build_eval_loaders(cfg: Dict[str, Any], stage: str, split: str) -> Dict[str, Any]:
    eval_data_cfg = build_eval_data_cfg(cfg, stage, split)
    if stage == "stage1_seg":
        loaders = build_stage1_seg_loaders(eval_data_cfg)
    else:
        loaders = build_stage2_cls_loaders(eval_data_cfg)

    if not loaders["val"]:
        raise ValueError(f"No evaluation loaders built for stage={stage}, split={split}.")
    return loaders["val"]


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return moved


def load_checkpoint_model(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


def tensor_to_numpy_image(image_tensor: torch.Tensor) -> np.ndarray:
    image_np = image_tensor.detach().cpu().numpy()
    if image_np.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got {image_np.shape}")
    return np.transpose(image_np, (1, 2, 0))


def get_visual_reference(raw_batch: Dict[str, Any], sample_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    visual_images = raw_batch.get("visual_image")
    visual_gt_masks = raw_batch.get("visual_gt_mask")
    if visual_images is None or visual_gt_masks is None:
        return None, None
    return visual_images[sample_idx], visual_gt_masks[sample_idx]


def extract_case_ids(case_id_value: Any, batch_size: int) -> List[str]:
    if isinstance(case_id_value, (list, tuple)):
        values = [str(x) for x in case_id_value]
        if len(values) < batch_size:
            values.extend([""] * (batch_size - len(values)))
        return values[:batch_size]
    if case_id_value is None:
        return [""] * batch_size
    return [str(case_id_value)] * batch_size


@torch.no_grad()
def evaluate_segmentation(
    model: torch.nn.Module,
    val_loaders: Dict[str, Any],
    device: torch.device,
    output_dir: str,
    max_visualizations: int,
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, float] = {}
    mean_score_list = []
    seg_out_dir = os.path.join(output_dir, "segmentation")
    saved_visuals = 0

    for loader_name, loader in val_loaders.items():
        dsc_scores = []
        nsd_scores = []
        official_scores = []
        for raw_batch in tqdm(loader, desc=f"Eval-{loader_name}", leave=False):
            batch = move_batch_to_device(raw_batch, device)
            outputs = model(batch)
            logits = outputs["seg_logits"]
            if logits is None:
                raise ValueError(f"seg_logits is None during segmentation evaluation for {loader_name}")

            if raw_batch.get("official_gt_mask") is not None:
                score_dict = compute_ceus_official_score_from_logits(logits, raw_batch)
            else:
                target = batch["label_seg"]
                score_dict = compute_binary_seg_score_from_logits(logits, target)
            dsc_scores.append(score_dict["dsc"])
            nsd_scores.append(score_dict["nsd"])
            official_scores.append(score_dict["score"])

            if max_visualizations > 0 and saved_visuals < max_visualizations:
                batch_size = logits.shape[0]
                case_ids = extract_case_ids(raw_batch.get("case_id"), batch_size)
                for sample_idx in range(batch_size):
                    if saved_visuals >= max_visualizations:
                        break
                    if logits.ndim == 4:
                        image = tensor_to_numpy_image(batch["image"][sample_idx])
                        gt_mask = batch["label_seg"][sample_idx].detach().cpu().numpy()
                        pred_mask = torch.argmax(logits[sample_idx], dim=0).detach().cpu().numpy()
                    elif logits.ndim == 5:
                        frame_idx = int(logits.shape[1] // 2)
                        image = tensor_to_numpy_image(batch["image"][sample_idx, frame_idx])
                        gt_mask = batch["label_seg"][sample_idx, frame_idx].detach().cpu().numpy()
                        pred_mask = torch.argmax(logits[sample_idx, frame_idx], dim=0).detach().cpu().numpy()
                    else:
                        raise ValueError(f"Unsupported seg logits shape: {tuple(logits.shape)}")

                    visual_image, visual_gt_mask = get_visual_reference(raw_batch, sample_idx)
                    if visual_image is not None and visual_gt_mask is not None:
                        image = visual_image
                        gt_mask = visual_gt_mask

                    save_segmentation_visuals(
                        output_dir=seg_out_dir,
                        image=image,
                        pred_mask=pred_mask,
                        gt_mask=gt_mask,
                        case_id=case_ids[sample_idx],
                        loader_name=loader_name,
                    )
                    saved_visuals += 1

        mean_dsc = sum(dsc_scores) / max(len(dsc_scores), 1)
        mean_nsd = sum(nsd_scores) / max(len(nsd_scores), 1)
        mean_score = sum(official_scores) / max(len(official_scores), 1)
        metrics[f"{loader_name}_dsc"] = mean_dsc
        metrics[f"{loader_name}_nsd"] = mean_nsd
        metrics[f"{loader_name}_score"] = mean_score
        mean_score_list.append(mean_score)

    metrics["mean_score"] = sum(mean_score_list) / max(len(mean_score_list), 1)
    return metrics


@torch.no_grad()
def evaluate_classification(
    model: torch.nn.Module,
    val_loaders: Dict[str, Any],
    device: torch.device,
    output_dir: str,
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, float] = {}
    scores = []
    csv_rows: List[Dict[str, object]] = []

    for loader_name, loader in val_loaders.items():
        logits_list = []
        labels_list = []

        for raw_batch in tqdm(loader, desc=f"Eval-{loader_name}", leave=False):
            batch = move_batch_to_device(raw_batch, device)
            outputs = model(batch)
            logits = outputs["cls_logits"]
            if logits is None:
                raise ValueError(f"cls_logits is None during classification evaluation for {loader_name}")

            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels = batch["label_cls"].detach().cpu().numpy()
            case_ids = extract_case_ids(raw_batch.get("case_id"), probs.shape[0])

            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(labels)

            for idx in range(probs.shape[0]):
                csv_rows.append(
                    {
                        "loader_name": loader_name,
                        "case_id": case_ids[idx],
                        "gt": int(labels[idx]),
                        "pred": int(preds[idx]),
                        "score_0": float(probs[idx, 0]),
                        "score_1": float(probs[idx, 1]),
                    }
                )

        logits_np = stack_numpy(logits_list)
        labels_np = stack_numpy(labels_list)
        acc = binary_accuracy_from_logits(logits_np, labels_np)
        auc = binary_auc_from_logits(logits_np, labels_np)
        score = 0.5 * (acc + auc)

        metrics[f"{loader_name}_acc"] = acc
        metrics[f"{loader_name}_auc"] = auc
        metrics[f"{loader_name}_score"] = score
        scores.append(score)

    metrics["mean_score"] = float(np.mean(scores)) if scores else 0.0
    csv_path = os.path.join(output_dir, "classification", "predictions.csv")
    save_classification_csv(csv_path, csv_rows)
    return metrics


def build_result_dir(cfg: Dict[str, Any], stage: str, split: str, checkpoint_path: str) -> str:
    base_dir = Path(cfg["trainer"]["save_dir"]) / "test_results"
    ckpt_stem = Path(checkpoint_path).stem
    out_dir = base_dir / f"{stage}_{split}_{ckpt_stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


def save_metrics_json(output_dir: str, metrics: Dict[str, float], meta: Dict[str, Any]) -> str:
    out_path = Path(output_dir) / "metrics.json"
    payload = {"meta": meta, "metrics": metrics}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(out_path)


def main() -> None:
    args = parse_args()
    stage_hint = args.stage or "stage2_cls"
    if args.config is None:
        config_name = "stage1_seg.yaml" if stage_hint == "stage1_seg" else "stage2_cls.yaml"
        args.config = str(Path(__file__).resolve().parent / "configs" / config_name)
    cfg = load_yaml(args.config)
    stage = args.stage or cfg.get("run", {}).get("stage", stage_hint)
    if args.data_root:
        cfg.setdefault("data", {})["data_root"] = args.data_root
    if args.local_val_fraction is not None:
        cfg.setdefault("data", {})["local_val_fraction"] = args.local_val_fraction
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint_path(cfg, stage, args.checkpoint, args.which)

    print(f"[Info] Using device: {device}")
    print(f"[Info] Loading config: {args.config}")
    print(f"[Info] Running stage: {stage}")
    print(f"[Info] Evaluating split: {args.split}")
    print(f"[Info] Loading checkpoint: {checkpoint_path}")

    val_loaders = build_eval_loaders(cfg, stage, args.split)
    print(f"[Info] Eval loaders: {list(val_loaders.keys())}")

    model = build_model(cfg).to(device)
    checkpoint_meta = load_checkpoint_model(model, checkpoint_path, device)
    output_dir = build_result_dir(cfg, stage, args.split, checkpoint_path)

    if stage == "stage1_seg":
        metrics = evaluate_segmentation(
            model=model,
            val_loaders=val_loaders,
            device=device,
            output_dir=output_dir,
            max_visualizations=max(args.max_visualizations, 0),
        )
    else:
        metrics = evaluate_classification(
            model=model,
            val_loaders=val_loaders,
            device=device,
            output_dir=output_dir,
        )

    metrics_path = save_metrics_json(
        output_dir=output_dir,
        metrics=metrics,
        meta={
            "stage": stage,
            "split": args.split,
            "checkpoint_path": checkpoint_path,
            "checkpoint_epoch": checkpoint_meta.get("epoch"),
            "checkpoint_metrics": checkpoint_meta.get("metrics"),
        },
    )

    print("[Info] Evaluation finished.")
    print(f"[Info] Results saved to: {output_dir}")
    print(f"[Info] Metrics json: {metrics_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

