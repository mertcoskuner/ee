"""MNIST loading, batching, and validation split settings."""

from dataclasses import dataclass


@dataclass
class DataLoaderParams:
    """Store data paths, batch sizes, workers, and subset sizes."""

    data_dir: str = "data"
    num_workers: int = 2
    batch_size: int = 50
    test_batch_size: int = 1000
    num_samples: int | None = None
    validation_size: int = 5000


def get_data_loader_params(args) -> DataLoaderParams:
    """Build MNIST loading settings from validated CLI arguments."""
    return DataLoaderParams(
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_samples=args.num_samples,
        validation_size=args.validation_size,
    )
