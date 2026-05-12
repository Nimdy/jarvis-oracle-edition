"""Telemetry API -- stable contract for dashboard data shapes.

Every subsystem that exposes data to the dashboard must return one of these
4 shapes.  The cache builder in app.py never transforms or reshapes data;
it only collects pre-shaped outputs.

Adding new panels never requires inventing new backend one-offs -- only new
data sources that conform to the existing shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimeseriesPoint:
    """Single data point in a time series."""
    timestamp: float
    value: float
    label: str = ""


@dataclass
class HistogramBin:
    """Single bin in a histogram / distribution."""
    bin_start: float
    bin_end: float
    count: int


@dataclass
class HeatmapCell:
    """Single cell in a matrix / heatmap."""
    row: str
    col: str
    value: float


@dataclass
class TopologyNode:
    """Node in a network topology graph."""
    id: str
    label: str
    size: float = 1.0
    type: str = "default"


@dataclass
class TopologyEdge:
    """Edge in a network topology graph."""
    source: str
    target: str
    weight: float = 1.0


@dataclass
class TopologyGraph:
    """Full network topology visualization."""
    nodes: list[TopologyNode] = field(default_factory=list)
    edges: list[TopologyEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Serialization helpers (for JSON cache)
# ---------------------------------------------------------------------------


def timeseries_to_dict(points: list[TimeseriesPoint]) -> list[dict[str, Any]]:
    return [{"t": p.timestamp, "v": p.value, "l": p.label} for p in points]


def histogram_to_dict(bins: list[HistogramBin]) -> list[dict[str, Any]]:
    return [{"start": b.bin_start, "end": b.bin_end, "count": b.count} for b in bins]


def heatmap_to_dict(cells: list[HeatmapCell]) -> list[dict[str, Any]]:
    return [{"row": c.row, "col": c.col, "value": c.value} for c in cells]


def topology_to_dict(graph: TopologyGraph) -> dict[str, Any]:
    return {
        "nodes": [{"id": n.id, "label": n.label, "size": n.size, "type": n.type} for n in graph.nodes],
        "edges": [{"source": e.source, "target": e.target, "weight": e.weight} for e in graph.edges],
    }
