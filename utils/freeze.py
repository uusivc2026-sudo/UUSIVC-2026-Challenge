from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn as nn

from utils.checkpoint import resolve_checkpoint_reference


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _matches_prefix(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _resolve_existing_modules(model: nn.Module, module_paths: Iterable[str]) -> Dict[str, nn.Module]:
    modules: Dict[str, nn.Module] = {}
    for module_path in module_paths:
        try:
            modules[module_path] = model.get_submodule(module_path)
        except AttributeError:
            continue
    return modules


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    map_location: torch.device,
    strict: bool = True,
) -> Dict[str, Any]:
    resolved_path = Path(resolve_checkpoint_reference(checkpoint_path)).expanduser().resolve()

    checkpoint = torch.load(str(resolved_path), map_location=map_location)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    incompatible = model.load_state_dict(state_dict, strict=strict)

    return {
        "path": str(resolved_path),
        "strict": strict,
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "metrics": checkpoint.get("metrics") if isinstance(checkpoint, dict) else None,
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
    }


def freeze_modules_by_keywords(model: nn.Module, keywords: Iterable[str]) -> Dict[str, Any]:
    freeze_keywords = _as_list(keywords)
    existing_modules = _resolve_existing_modules(model, freeze_keywords)
    frozen_param_names = set()

    for module_path, module in existing_modules.items():
        for local_name, param in module.named_parameters(recurse=True):
            param.requires_grad = False
            full_name = f"{module_path}.{local_name}" if local_name else module_path
            frozen_param_names.add(full_name)

    for name, param in model.named_parameters():
        if _matches_prefix(name, freeze_keywords):
            param.requires_grad = False
            frozen_param_names.add(name)

    return {
        "requested_keywords": freeze_keywords,
        "matched_modules": sorted(existing_modules.keys()),
        "unmatched_keywords": [
            keyword
            for keyword in freeze_keywords
            if keyword not in existing_modules
            and not any(_matches_prefix(name, [keyword]) for name, _ in model.named_parameters())
        ],
        "frozen_parameter_count": len(frozen_param_names),
    }


def set_modules_eval_by_keywords(model: nn.Module, keywords: Iterable[str]) -> List[str]:
    eval_keywords = _as_list(keywords)
    existing_modules = _resolve_existing_modules(model, eval_keywords)
    seen_ids = set()
    applied = []
    for module_path, module in existing_modules.items():
        module_id = id(module)
        if module_id in seen_ids:
            continue
        module.eval()
        seen_ids.add(module_id)
        applied.append(module_path)
    return applied


def apply_freeze_config(model: nn.Module, cfg: Dict[str, Any], stage: str) -> Dict[str, Any]:
    freeze_root = cfg.get("freeze", {}) or {}
    stage_cfg = freeze_root.get(stage, {}) or {}

    freeze_keywords = _as_list(stage_cfg.get("freeze_keywords", []))
    frozen_eval_keywords = _as_list(stage_cfg.get("frozen_eval_keywords", freeze_keywords))

    freeze_result = freeze_modules_by_keywords(model, freeze_keywords)
    counts = count_parameters(model)

    return {
        "stage": stage,
        "freeze_keywords": freeze_keywords,
        "frozen_eval_keywords": frozen_eval_keywords,
        "freeze_result": freeze_result,
        "parameter_counts": counts,
    }
