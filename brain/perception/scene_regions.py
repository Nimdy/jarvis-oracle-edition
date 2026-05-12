"""Layer 3B Scene Regions — maps raw bboxes to semantic spatial anchors.

Semantic regions let Jarvis reason about "desk_left" instead of "bbox x=412"
and enable region-aware permanence decay (occluded regions decay slowly,
visible-empty regions decay fast).
"""

from __future__ import annotations

from perception.scene_types import BBox

REGIONS = (
    "desk_left", "desk_center", "desk_right",
    "monitor_zone", "background",
)


def infer_region(bbox: BBox | None, frame_w: int, frame_h: int) -> str:
    """Map a bounding box center to a semantic region."""
    if not bbox or frame_w <= 0 or frame_h <= 0:
        return "unknown"

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    rx = cx / frame_w
    ry = cy / frame_h

    if ry > 0.60:
        if rx < 0.33:
            return "desk_left"
        if rx < 0.66:
            return "desk_center"
        return "desk_right"

    if ry > 0.35:
        return "monitor_zone"

    return "background"


def estimate_region_visibility(
    person_bboxes: list[BBox],
    frame_w: int,
    frame_h: int,
) -> dict[str, float]:
    """Estimate how visible each semantic region is, given person occlusions.

    Returns a dict of region -> visibility [0, 1] where 1.0 means fully
    observable and 0.0 means fully occluded by a person.
    """
    visibility: dict[str, float] = {r: 1.0 for r in REGIONS}

    if not person_bboxes or frame_w <= 0 or frame_h <= 0:
        return visibility

    region_bounds = _region_bounds(frame_w, frame_h)

    for region, (rx1, ry1, rx2, ry2) in region_bounds.items():
        region_area = max(1, (rx2 - rx1) * (ry2 - ry1))
        total_occluded = 0

        for pbbox in person_bboxes:
            ix1 = max(rx1, pbbox[0])
            iy1 = max(ry1, pbbox[1])
            ix2 = min(rx2, pbbox[2])
            iy2 = min(ry2, pbbox[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            total_occluded += inter

        occluded_frac = min(1.0, total_occluded / region_area)
        visibility[region] = round(1.0 - occluded_frac, 3)

    return visibility


def _region_bounds(frame_w: int, frame_h: int) -> dict[str, tuple[int, int, int, int]]:
    """Return approximate pixel bounds for each semantic region."""
    w3 = frame_w // 3
    desk_top = int(frame_h * 0.60)
    monitor_top = int(frame_h * 0.35)

    return {
        "desk_left": (0, desk_top, w3, frame_h),
        "desk_center": (w3, desk_top, w3 * 2, frame_h),
        "desk_right": (w3 * 2, desk_top, frame_w, frame_h),
        "monitor_zone": (0, monitor_top, frame_w, desk_top),
        "background": (0, 0, frame_w, monitor_top),
    }
