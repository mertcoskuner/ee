"""Federated learning: clients, server, data partitioning, attacks, and aggregation."""

from .client import Client
from .partition import partition
from .server import Server, build_server_optimizer

__all__ = ["Client", "Server", "build_server_optimizer", "partition"]
