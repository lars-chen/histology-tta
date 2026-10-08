"""Shared utilities for parsing Camelyon17 ASAP lesion annotation XMLs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon


def parse_lesion_polygons(xml_path: Path) -> MultiPolygon | None:
    """Parse ASAP XML → shapely MultiPolygon at level-0 coordinates."""
    tree = ET.parse(xml_path)
    polys = []
    for ann in tree.getroot().findall(".//Annotation"):
        if ann.get("PartOfGroup", "").lower() != "metastases":
            continue
        coords = [
            (float(c.get("X")), float(c.get("Y")))
            for c in ann.findall(".//Coordinate")
        ]
        if len(coords) >= 3:
            try:
                polys.append(Polygon(coords))
            except Exception:
                pass
    return MultiPolygon(polys) if polys else None


def patch_labels(
    coords: np.ndarray,            # (N, 2) level-0 (x, y) top-left corners
    patch_size: int,               # patch size at level 0
    tumor_region: MultiPolygon | None,
) -> np.ndarray:                   # (N,) int {0, 1}
    """Return 1 if patch centre is inside a tumor polygon, else 0."""
    if tumor_region is None:
        return np.zeros(len(coords), dtype=np.int32)
    half = patch_size / 2
    centres = coords.astype(float) + half
    labels = np.array(
        [int(tumor_region.contains(Point(cx, cy))) for cx, cy in centres],
        dtype=np.int32,
    )
    return labels
