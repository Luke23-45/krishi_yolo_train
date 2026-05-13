"""
yoloml.training
---------------
Training pipelines and class-imbalance mitigation strategies.
"""

from yoloml.training.train import (
    compute_class_weights,
    compute_repeat_factors,
    create_balanced_dataset,
    effective_number,
    inject_class_weights,
    main,
    scan_class_distribution,
    train,
)

__all__ = [
    "train",
    "main",
    "train_module",
    "scan_class_distribution",
    "compute_repeat_factors",
    "create_balanced_dataset",
    "effective_number",
    "compute_class_weights",
    "inject_class_weights",
]


class _TrainModuleStub:
    main = main


train_module = _TrainModuleStub()
