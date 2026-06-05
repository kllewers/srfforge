"""EMIT L2A granule search via earthaccess / NASA CMR."""

# ─────────────────────────────────────────────────────────────────────────────
# Module: search/_emit.py  —  EMIT granule search
#
#   called by ◄── search/__init__.py::find_overlapping
#   calls     ──► earthaccess.search_data (NASA CMR)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import datetime
from dataclasses import dataclass, field


@dataclass
class EmitGranule:
    """Metadata for a single EMIT L2A reflectance granule."""

    granule_id: str
    datetime: datetime.datetime
    cloud_cover: float
    bbox_latlon: tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)
    download_url: str = ""


#--------------------------------------
# Called by: search/__init__.py::find_overlapping
# Calls:     earthaccess.login, earthaccess.search_data
#--------------------------------------
def search_emit_granules(
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str],
    max_cloud: float = 30.0,
    collection_shortname: str = "EMITL2ARFL",
    count: int = 500,
) -> list[EmitGranule]:
    """
    Search for EMIT L2A granules intersecting a bounding box and date range.

    Parameters
    ----------
    bbox : (lon_min, lat_min, lon_max, lat_max) in WGS84 degrees
    date_range : (start_iso, end_iso) e.g. ("2023-06-01", "2023-09-30")
    max_cloud : maximum cloud cover percent (0–100)
    collection_shortname : CMR short name — "EMITL2ARFL" for L2A reflectance
    count : max granules to return from CMR

    Returns
    -------
    list of EmitGranule, filtered to <= max_cloud
    """
    try:
        import earthaccess
    except ImportError as e:
        raise ImportError(
            "earthaccess is required for EMIT search: pip install 'srfforge[search]'"
        ) from e

    earthaccess.login(strategy="netrc")

    raw = earthaccess.search_data(
        short_name=collection_shortname,
        temporal=date_range,
        bounding_box=bbox,
        count=count,
    )

    granules: list[EmitGranule] = []
    for r in raw:
        cloud = _extract_cloud(r)
        if cloud is not None and cloud > max_cloud:
            continue

        granule_id = r["meta"].get("native-id", "")
        dt = _parse_datetime(r)
        granule_bbox = _extract_bbox(r)
        url = _extract_url(r)

        granules.append(EmitGranule(
            granule_id=granule_id,
            datetime=dt,
            cloud_cover=cloud or 0.0,
            bbox_latlon=granule_bbox,
            download_url=url,
        ))

    return granules


def _extract_cloud(granule) -> float | None:
    """Pull cloud cover from CMR AdditionalAttributes list."""
    attrs = granule.get("umm", {}).get("AdditionalAttributes", [])
    for attr in attrs:
        if attr.get("Name") == "CloudCover":
            vals = attr.get("Values", [])
            if vals:
                try:
                    return float(vals[0])
                except (ValueError, TypeError):
                    pass
    return None


def _parse_datetime(granule) -> datetime.datetime:
    """Parse the granule start datetime from CMR UMM."""
    try:
        dt_str = (
            granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        )
        return datetime.datetime.fromisoformat(dt_str.rstrip("Z"))
    except (KeyError, ValueError):
        return datetime.datetime(1970, 1, 1)


def _extract_bbox(granule) -> tuple[float, float, float, float]:
    """Extract a (lon_min, lat_min, lon_max, lat_max) bbox from CMR UMM geometry."""
    try:
        spatial = granule["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
        if "BoundingRectangles" in spatial:
            rect = spatial["BoundingRectangles"][0]
            return (
                rect["WestBoundingCoordinate"],
                rect["SouthBoundingCoordinate"],
                rect["EastBoundingCoordinate"],
                rect["NorthBoundingCoordinate"],
            )
        if "GPolygons" in spatial:
            points = spatial["GPolygons"][0]["Boundary"]["Points"]
            lons = [p["Longitude"] for p in points]
            lats = [p["Latitude"] for p in points]
            return (min(lons), min(lats), max(lons), max(lats))
    except (KeyError, IndexError):
        pass
    return (-180.0, -90.0, 180.0, 90.0)


def _extract_url(granule) -> str:
    """Extract the first HTTPS download URL from CMR RelatedUrls."""
    try:
        for link in granule["umm"].get("RelatedUrls", []):
            if link.get("Type") == "GET DATA" and link.get("URL", "").startswith("https"):
                return link["URL"]
    except (KeyError, TypeError):
        pass
    return ""
