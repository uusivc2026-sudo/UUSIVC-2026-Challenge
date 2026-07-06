# Metrics Reference

This directory contains standalone reference implementations of the baseline metrics.
These files are for participant reference only and are not imported by the training or inference pipeline.

- `segmentation_metrics_reference.py`: DSC, lightweight NSD, and `0.7 * DSC + 0.3 * NSD`.
- `classification_metrics_reference.py`: ACC, AUC, and `0.5 * ACC + 0.5 * AUC`.

The active project implementations remain in `utils/metrics.py` and `utils/auc_utils.py`.
The training and local evaluation scripts call those project implementations.
