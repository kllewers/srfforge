"""
Search NASA EarthData for spatially overlapping NEON AOP and EMIT granules.

Install search dependencies:
    pip install 'srfforge[search]'

Examples
--------
Single site and year::

    pairs = find_overlapping(site="NIWO", year=2023)

Multiple sites::

    pairs = find_overlapping(site=["NIWO", "HARV"], year=2023)

Explicit list of years::

    pairs = find_overlapping(site="NIWO", year=[2021, 2022, 2023])

Inclusive year range (tuple of start, end)::

    pairs = find_overlapping(site="NIWO", year=(2021, 2023))  # 2021, 2022, 2023
"""

# ─────────────────────────────────────────────────────────────────────────────
# Module: search/__init__.py  —  public API for EarthData colocation search
#
#   called by ◄── user code
#   calls     ──► _emit.py::search_emit_granules
#                 _neon.py::get_neon_tile_bounds
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from ._emit import search_emit_granules, EmitGranule
from ._neon import get_neon_tile_bounds, NeonTile


@dataclass
class OverlapResult:
    """A matched NEON tile and EMIT granule that overlap spatially."""

    neon: NeonTile
    emit: EmitGranule


#--------------------------------------
# Called by: user code
# Calls:     _neon.get_neon_tile_bounds, _emit.search_emit_granules
#--------------------------------------
def find_overlapping(
    site: str | list[str],
    year: int | list[int] | tuple[int, int],
    date_range: tuple[str, str] | None = None,
    max_cloud: float = 30.0,
    dpid: str = "DP3.30006.002",
    emit_collection: str = "EMITL2ARFL",
) -> list[OverlapResult]:
    """
    Find EMIT granules that overlap with NEON AOP tiles.

    Requires: pip install 'srfforge[search]'

    Parameters
    ----------
    site : NEON site code or list of site codes, e.g. "NIWO" or ["NIWO", "HARV"]
    year : one of:
        - int         — single year, e.g. 2023
        - list[int]   — explicit years, e.g. [2021, 2022, 2023]
        - tuple[int, int] — inclusive range, e.g. (2021, 2023) → 2021, 2022, 2023
    date_range : optional (start, end) ISO date strings — overrides the year-derived
                 range when provided, e.g. ("2023-06-01", "2023-09-30")
    max_cloud : maximum EMIT cloud cover percent (0–100)
    dpid : NEON data product ID (default DP3.30006.002 — bidirectional reflectance)
    emit_collection : CMR short name for EMIT (default EMITL2ARFL)

    Returns
    -------
    list of OverlapResult, each holding a NeonTile and a matched EmitGranule
    """
    sites = [site] if isinstance(site, str) else list(site)
    years = _parse_years(year)

    if date_range is None:
        date_range = (f"{min(years)}-01-01", f"{max(years)}-12-31")

    results: list[OverlapResult] = []

    for s in sites:
        for y in years:
            tiles = get_neon_tile_bounds(site=s, year=str(y), dpid=dpid)
            if not tiles:
                continue

            lon_min = min(t.bbox_latlon[0] for t in tiles)
            lat_min = min(t.bbox_latlon[1] for t in tiles)
            lon_max = max(t.bbox_latlon[2] for t in tiles)
            lat_max = max(t.bbox_latlon[3] for t in tiles)

            year_range = date_range if date_range is not None else (f"{y}-01-01", f"{y}-12-31")
            emit_granules = search_emit_granules(
                bbox=(lon_min, lat_min, lon_max, lat_max),
                date_range=year_range,
                max_cloud=max_cloud,
                collection_shortname=emit_collection,
            )

            for tile in tiles:
                for granule in emit_granules:
                    if _bbox_overlaps(tile.bbox_latlon, granule.bbox_latlon):
                        results.append(OverlapResult(neon=tile, emit=granule))

    return results


def _parse_years(year: int | list[int] | tuple[int, int]) -> list[int]:
    """Normalise the year argument to a sorted list of ints."""
    if isinstance(year, int):
        return [year]
    if isinstance(year, tuple):
        if len(year) != 2:
            raise ValueError(
                f"year tuple must be (start, end) inclusive, got {year!r}"
            )
        start, end = int(year[0]), int(year[1])
        if start > end:
            raise ValueError(f"year range start {start} > end {end}")
        return list(range(start, end + 1))
    # list / range / any iterable
    return sorted(int(y) for y in year)


def _bbox_overlaps(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """Return True if bounding boxes (lon_min, lat_min, lon_max, lat_max) overlap."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


__all__ = ["find_overlapping", "OverlapResult", "EmitGranule", "NeonTile"]
