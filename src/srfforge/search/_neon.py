"""NEON AOP tile metadata via neonutilities and UTM → lat/lon conversion."""

# ─────────────────────────────────────────────────────────────────────────────
# Module: search/_neon.py  —  NEON tile footprint lookup
#
#   called by ◄── search/__init__.py::find_overlapping
#   calls     ──► neonutilities.get_aop_tile_extents
#                 pyproj.Transformer  (UTM → WGS84)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NeonTile:
    """Spatial and collection metadata for one NEON AOP tile."""

    site: str
    year: str
    easting: int
    northing: int
    utm_zone: int
    bbox_latlon: tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)


#--------------------------------------
# Called by: search/__init__.py::find_overlapping
# Calls:     neonutilities.get_aop_tile_extents, _utm_tile_to_latlon
#--------------------------------------
def get_neon_tile_bounds(
    site: str,
    year: str,
    dpid: str = "DP3.30006.002",
    tile_size_m: int = 1000,
) -> list[NeonTile]:
    """
    Return lat/lon bounding boxes for all NEON AOP tiles available at a site/year.

    Parameters
    ----------
    site : NEON site code, e.g. "NIWO"
    year : four-digit year string, e.g. "2023"
    dpid : NEON data product ID (default DP3.30006.002)
    tile_size_m : tile edge length in metres (NEON default is 1000 m)

    Returns
    -------
    list of NeonTile — one per available tile
    """
    try:
        import neonutilities as nu
    except ImportError as e:
        raise ImportError(
            "neonutilities is required for NEON tile search: pip install 'srfforge[search]'"
        ) from e

    # returns list of (easting, northing) tuples
    result = nu.get_aop_tile_extents(dpid, site, year)
    if not result:
        return []

    utm_zone, hemisphere = _neon_site_utm_zone(site)

    tiles: list[NeonTile] = []
    for easting, northing in result:
        easting  = int(easting)
        northing = int(northing)
        utm_zone_int = int(utm_zone)

        bbox = _utm_tile_to_latlon(
            easting=easting,
            northing=northing,
            utm_zone=utm_zone_int,
            tile_size_m=tile_size_m,
            hemisphere=hemisphere,
        )
        tiles.append(NeonTile(
            site=site,
            year=year,
            easting=easting,
            northing=northing,
            utm_zone=utm_zone_int,
            bbox_latlon=bbox,
        ))

    return tiles


#--------------------------------------
# Called by: get_neon_tile_bounds
# Calls:     NEON REST API (urllib, stdlib only)
#--------------------------------------
def _neon_site_utm_zone(site: str) -> tuple[int, str]:
    """
    Return (utm_zone, hemisphere) for a NEON site by querying the NEON REST API.

    Uses only stdlib (urllib) — no extra dependency.
    """
    import json
    import urllib.request

    url = f"https://data.neonscience.org/api/v0/sites/{site}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())

    lat = data["data"]["siteLatitude"]
    lon = data["data"]["siteLongitude"]

    zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    return zone, hemisphere


#--------------------------------------
# Called by: get_neon_tile_bounds
# Calls:     pyproj.Transformer
#--------------------------------------
def _utm_tile_to_latlon(
    easting: int,
    northing: int,
    utm_zone: int,
    tile_size_m: int = 1000,
    hemisphere: str = "north",
) -> tuple[float, float, float, float]:
    """
    Convert a UTM tile corner (SW corner) to a WGS84 bounding box.

    Parameters
    ----------
    easting, northing : SW corner of the tile in UTM metres
    utm_zone : UTM zone number (1–60)
    tile_size_m : tile edge length in metres

    Returns
    -------
    (lon_min, lat_min, lon_max, lat_max) in WGS84 degrees
    """
    try:
        from pyproj import Transformer
    except ImportError as e:
        raise ImportError(
            "pyproj is required for coordinate conversion: pip install 'srfforge[search]'"
        ) from e

    epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    # SW and NE corners
    lon_sw, lat_sw = transformer.transform(easting, northing)
    lon_ne, lat_ne = transformer.transform(easting + tile_size_m, northing + tile_size_m)

    return (
        min(lon_sw, lon_ne),
        min(lat_sw, lat_ne),
        max(lon_sw, lon_ne),
        max(lat_sw, lat_ne),
    )
