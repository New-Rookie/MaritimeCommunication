"""
Mobility model dispatcher.

Each node subclass implements its own ``update_position(dt)`` method,
so this module simply iterates over all nodes and calls it.
"""

from __future__ import annotations
from typing import List


def update_all_positions(nodes: List, dt: float) -> None:
    for node in nodes:
        node.update_position(dt)
