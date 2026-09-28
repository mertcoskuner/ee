"""Federated learning: clients, server, data partitioning, attacks, and aggregation."""

from .client import Client
from .partition import partition
from .server import Server, assumed_attackers, build_server_optimizer, num_byzantine

__all__ = [
    "Client",
    "Server",
    "assumed_attackers",
    "build_server_optimizer",
    "num_byzantine",
    "partition",
]
