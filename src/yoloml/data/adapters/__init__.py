"""
yoloml/data/adapters/__init__.py
---------------------------------
Adapter registry for the Krishi Vaidya bouncer data pipeline.
"""

from yoloml.data.adapters.base import BaseAdapter
from yoloml.data.adapters.yolo import YOLOAdapter
from yoloml.data.adapters.coco import COCOAdapter
from yoloml.data.adapters.classifier import ClassifierAdapter

ADAPTER_REGISTRY = {
    "yolo": YOLOAdapter,
    "coco": COCOAdapter,
    "classification": ClassifierAdapter,
}


def get_adapter(format_name: str):
    """Return the adapter class for a given format string."""
    adapter_cls = ADAPTER_REGISTRY.get(format_name)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown format '{format_name}'. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return adapter_cls
