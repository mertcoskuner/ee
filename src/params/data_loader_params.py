"""MNIST loading, batching, and validation split settings."""

from dataclasses import dataclass

from src.utils.helper_params import option


@dataclass
class DataLoaderParams:
    """Data: paths, batch sizes, workers, and subset sizes."""

    data_dir: str = option("data", "MNIST download directory")
    num_workers: int = option(2, "DataLoader worker processes", ge=0)
    batch_size: int = option(50, "training batch size", gt=0)
    test_batch_size: int = option(1000, "evaluation batch size", gt=0)
    num_samples: int | None = option(
        None, "evaluate on the first N test images only", gt=0
    )
    validation_size: int = option(
        5000, "training images held out for validation", gt=0, lt=60000
    )
