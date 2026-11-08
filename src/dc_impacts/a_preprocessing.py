import difflib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd
import pyogrio

# Setup logging
logger = logging.getLogger(__name__)

# EJI 2024 attribute columns (confirmed from EJI_DATADICTIONARY_2024)
EJI_LAYER = None              # None = first layer in the .gdb
GEOID_COL = "GEOID"           # 11-digit tract GEOID. Used for BOTH lookup and filtering:
                              # CT uses 2022 planning-region FIPS here (GEOID_2020 has the old ones)
COUNTY_COL = "COUNTY"         # county name
STATE_NAME_COL = "StateDesc"  # full state name
STATE_ABBR_COL = "STATEABBR"  # 2-letter state abbreviation

# Stripped so "Rockland" matches "Rockland County". Longest first.
# "city" is NOT stripped, so "Baltimore city" and "Baltimore County" stay distinct.
COUNTY_SUFFIXES = (" city and borough", " census area", " municipality", " borough", " parish", " county")

# Data center CSV coordinate columns (matched case-insensitively)
LON_CANDIDATES = ["longitude", "lon", "lng", "long", "x"]
LAT_CANDIDATES = ["latitude", "lat", "y"]

# Vector formats searched for when a folder is given (e.g., Rockland_Land_Use_2021/)
VECTOR_EXTENSIONS = (".shp", ".geojson", ".gpkg")


# ---------------------------------------------------------------------------
# File finders
# ---------------------------------------------------------------------------
def find_gdb(path: Union[str, Path]) -> Path:
    """Returns the .gdb at `path`, or the first .gdb found inside it."""
    path = Path(path)
    if path.suffix.lower() == ".gdb" and path.exists():
        return path
    if path.is_dir():
        found = sorted(path.rglob("*.gdb"))
        if found:
            return found[0]
        if any(path.glob("*.gdbtable")):
            raise FileNotFoundError(
                f"{path.name} looks like a geodatabase but its name doesn't end in .gdb. "
                f"Rename it to {path.name}.gdb"
            )
    raise FileNotFoundError(f"No .gdb found at or inside {path}")


def find_vector_file(path: Union[str, Path]) -> Path:
    """Returns `path` if it's a file, or the first vector file found inside a folder."""
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        found = [p for ext in VECTOR_EXTENSIONS for p in sorted(path.rglob(f"*{ext}"))]
        if not found:
            raise FileNotFoundError(f"No {', '.join(VECTOR_EXTENSIONS)} file found inside {path.name}")
        if len(found) > 1:
            logger.warning(
                f"Multiple vector files in {path.name}; using {found[0].name}. "
                "Point --land_use_file at the exact file to choose another."
            )
        return found[0]
    raise FileNotFoundError(f"Land use file/folder not found: {path}")


# ---------------------------------------------------------------------------
# Catchment lookup: county + state name -> GEOID (from the EJI attributes)
# ---------------------------------------------------------------------------
def _normalize(text) -> str:
    """Lowercase, drop periods ('St. Lawrence' == 'St Lawrence'), collapse spaces."""
    text = str(text).lower().replace(".", "")
    return re.sub(r"\s+", " ", text).strip()


def _strip_suffix(text: str) -> str:
    for suffix in COUNTY_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)].strip()
    return text


def _clean_geoid(series: pd.Series) -> pd.Series:
    """GEOIDs can load as numbers -> clean 11-character strings with leading zeros."""
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(11)


def load_county_lookup(eji_path: Path) -> pd.DataFrame:
    """
    Builds a county lookup table from the EJI attributes only (no geometry -> fast).
    One row per county: county GEOID (first 5 digits of the tract GEOID), names, state.
    """
    cols = [GEOID_COL, COUNTY_COL, STATE_NAME_COL, STATE_ABBR_COL]
    df = gpd.read_file(eji_path, layer=EJI_LAYER, columns=cols, ignore_geometry=True)
    df["county_geoid"] = _clean_geoid(df[GEOID_COL]).str[:5]

    lookup = df.drop_duplicates("county_geoid").copy()
    lookup["county_full"] = lookup[COUNTY_COL].map(_normalize)
    lookup["county_base"] = lookup["county_full"].map(_strip_suffix)
    lookup["state_name"] = lookup[STATE_NAME_COL].map(_normalize)
    lookup["state_abbr"] = lookup[STATE_ABBR_COL].map(_normalize)
    return lookup.reset_index(drop=True)


def resolve_county(lookup: pd.DataFrame, county: str, state: str) -> Dict[str, str]:
    """Finds the single county matching the user's county + state input."""
    # 1. State: accept full name or abbreviation
    st = _normalize(state)
    in_state = lookup[(lookup["state_name"] == st) | (lookup["state_abbr"] == st)]
    if in_state.empty:
        options = sorted(lookup[STATE_NAME_COL].unique())
        close = difflib.get_close_matches(state.title(), options, n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(f"State '{state}' not found in EJI data.{hint}")

    # 2. County: match on the name without suffix ("rockland")
    full = _normalize(county)
    base = _strip_suffix(full)
    matches = in_state[in_state["county_base"] == base]

    # Several hits -> try exact full-name match, otherwise ask the user to be specific
    if len(matches) > 1:
        exact = matches[matches["county_full"] == full]
        if len(exact) == 1:
            matches = exact
        else:
            raise ValueError(
                f"'{county}' is ambiguous in {state}: {', '.join(matches[COUNTY_COL])}. "
                "Please use the full name."
            )

    # No hits -> suggest close matches
    if matches.empty:
        close = difflib.get_close_matches(base, in_state["county_base"].tolist(), n=3)
        suggestions = in_state.loc[in_state["county_base"].isin(close), COUNTY_COL].tolist()
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"County '{county}' not found in {state}.{hint}")

    row = matches.iloc[0]
    slug = re.sub(r"[^a-z0-9]+", "_", row["county_base"]).strip("_")
    return {
        "geoid": row["county_geoid"],
        "county": row[COUNTY_COL],
        "state": row[STATE_NAME_COL],
        "label": f"{slug}_{row['state_abbr']}",  # used in output filenames, e.g. "rockland_ny"
    }


def resolve_catchment(eji_path: Path, counties: List[str], states: List[str]) -> List[Dict[str, str]]:
    """Resolves each (county, state) pair to a GEOID. Counties and states must already be paired."""
    logger.info("Building county lookup from EJI attributes...")
    lookup = load_county_lookup(eji_path)
    return [resolve_county(lookup, c, s) for c, s in zip(counties, states)]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------
def _find_column(columns, candidates) -> Optional[str]:
    """Case-insensitive column lookup."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _fix_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Repairs invalid geometries and drops empty/null ones."""
    invalid = ~gdf.is_valid
    if invalid.any():
        logger.info(f"Repairing {invalid.sum()} invalid geometries...")
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].make_valid()
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]


def _require_crs(gdf: gpd.GeoDataFrame, name: str):
    if gdf.crs is None:
        raise ValueError(f"{name} has no CRS (missing .prj file?). Cannot reproject.")


def _save_geojson(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    path.unlink(missing_ok=True)  # GeoJSON driver can refuse to overwrite
    gdf.to_file(path, driver="GeoJSON")
    logger.info(f"Successfully saved {path.name} ({len(gdf)} features)")
    return path


def _crs_matches(path: Path, target_epsg: int) -> bool:
    """Checks whether an existing output was written in the requested projection."""
    crs = gpd.read_file(path, rows=1).crs
    return crs is not None and crs.to_epsg() == target_epsg


# ---------------------------------------------------------------------------
# 1) EJI tracts: filter by GEOID, then reproject
# ---------------------------------------------------------------------------
def load_eji_tracts(eji_path: Path, geoids: List[str]) -> gpd.GeoDataFrame:
    """
    Reads only the catchment tracts from the national EJI geodatabase.
    Tries a fast attribute filter at read time; falls back to a full read if needed.
    """
    where = " OR ".join(f"{GEOID_COL} LIKE '{g}%'" for g in geoids)
    try:
        gdf = gpd.read_file(eji_path, layer=EJI_LAYER, where=where)
    except Exception as e:
        logger.warning(f"Read-time filter failed ({e}).")
        gdf = gpd.GeoDataFrame()

    if gdf.empty:  # e.g., GEOID stored as a number -> LIKE doesn't work
        logger.warning("Reading the full national EJI file instead (slower)...")
        gdf = gpd.read_file(eji_path, layer=EJI_LAYER)

    # Exact filter on the cleaned GEOID (makes the result correct either way)
    gdf[GEOID_COL] = _clean_geoid(gdf[GEOID_COL])
    gdf = gdf[gdf[GEOID_COL].str[:5].isin(geoids)].copy()

    if gdf.empty:
        raise ValueError(f"No EJI tracts found for GEOIDs {geoids}.")
    logger.info(f"Filtered EJI to {len(gdf)} tracts (source CRS: {gdf.crs})")
    return gdf


def transform_eji_tracts(eji_path: Path, geoids: List[str], target_epsg: int) -> gpd.GeoDataFrame:
    """Filters EJI to the catchment, then reprojects. All attributes are kept (-999 handled in b)."""
    gdf = load_eji_tracts(eji_path, geoids)
    _require_crs(gdf, "EJI")
    logger.info(f"Adjusting projection to EPSG:{target_epsg}...")
    gdf = gdf.to_crs(epsg=target_epsg)
    return _fix_geometries(gdf)


# ---------------------------------------------------------------------------
# 2) Catchment boundary: dissolve the EJI tracts
# ---------------------------------------------------------------------------
def create_catchment_boundary(tracts: gpd.GeoDataFrame, area_name: str) -> gpd.GeoDataFrame:
    """Dissolves the catchment tracts into one boundary polygon (2020 TIGER/Line tract geometry)."""
    boundary = tracts[["geometry"]].dissolve().reset_index(drop=True)
    boundary["area_name"] = area_name
    boundary["n_tracts"] = len(tracts)
    return boundary


# ---------------------------------------------------------------------------
# 3) Data centers: clip to (buffered) boundary, then reproject
# ---------------------------------------------------------------------------
def load_data_centers(dc_path: Path) -> gpd.GeoDataFrame:
    """Loads data centers from a CSV with lat/lon columns or from a vector file."""
    if dc_path.suffix.lower() == ".csv":
        df = pd.read_csv(dc_path)
        lon = _find_column(df.columns, LON_CANDIDATES)
        lat = _find_column(df.columns, LAT_CANDIDATES)
        if lon is None or lat is None:
            raise ValueError(f"No lat/lon columns found in {dc_path.name}. Columns: {list(df.columns)}")
        n_missing = df[[lon, lat]].isna().any(axis=1).sum()
        if n_missing:
            logger.warning(f"Dropping {n_missing} data centers without coordinates.")
            df = df.dropna(subset=[lon, lat])
        # Assumes coordinates are WGS84 decimal degrees -> check against the source documentation
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")
    return gpd.read_file(dc_path)


def transform_data_centers(dc_path: Path, clip_area: gpd.GeoSeries, target_epsg: int) -> gpd.GeoDataFrame:
    gdf = load_data_centers(dc_path)
    _require_crs(gdf, "Data centers")
    gdf = gpd.clip(gdf, clip_area.to_crs(gdf.crs))  # filter in source CRS
    logger.info(f"Clipped data centers to catchment: {len(gdf)} features")
    logger.info(f"Adjusting projection to EPSG:{target_epsg}...")
    return gdf.to_crs(epsg=target_epsg)


# ---------------------------------------------------------------------------
# 4) Land use: read only the catchment extent, clip, then reproject
# ---------------------------------------------------------------------------
def transform_land_use(land_use_path: Path, clip_area: gpd.GeoSeries, target_epsg: int) -> gpd.GeoDataFrame:
    src_crs = pyogrio.read_info(land_use_path)["crs"]
    if src_crs is None:
        raise ValueError(f"{land_use_path.name} has no CRS (missing .prj file?). Cannot reproject.")

    mask_geom = clip_area.to_crs(src_crs)
    gdf = gpd.read_file(land_use_path, mask=mask_geom)  # reads only features touching the catchment
    if gdf.empty:
        return gdf

    gdf = _fix_geometries(gdf)  # clipping fails on invalid polygons
    gdf = gpd.clip(gdf, mask_geom, keep_geom_type=True)
    logger.info(f"Clipped land use to catchment: {len(gdf)} features")
    logger.info(f"Adjusting projection to EPSG:{target_epsg}...")
    return _fix_geometries(gdf.to_crs(epsg=target_epsg))


# ---------------------------------------------------------------------------
# Main preprocessing function (called from __main__)
# ---------------------------------------------------------------------------
def filter_and_transform(
    eji_path: Path,
    dc_path: Optional[Path],
    land_use_path: Optional[Path],
    transformed_dir: Union[str, Path],
    geoids: List[str],
    area_name: str,
    target_epsg: int,
    buffer_m: float = 0,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """
    Filters all layers to the catchment FIRST, then reprojects to target_epsg,
    and saves them as GeoJSON in data/transformed/.
    Returns a dict of layer name -> output path for the layers that exist.
    """
    transformed_path = Path(transformed_dir)
    transformed_path.mkdir(parents=True, exist_ok=True)

    outputs = {
        "eji": transformed_path / f"eji_tracts_{area_name}.geojson",
        "boundary": transformed_path / f"catchment_boundary_{area_name}.geojson",
        "data_centers": transformed_path / f"data_centers_{area_name}.geojson",
        "land_use": transformed_path / f"land_use_{area_name}.geojson",
    }

    def _skip(key: str) -> bool:
        path = outputs[key]
        if path.exists() and not overwrite:
            if _crs_matches(path, target_epsg):
                logger.info(f"Skipping {path.name} (exists; use --overwrite)")
                return True
            logger.info(f"{path.name} exists in a different CRS -> re-processing.")
        return False

    # 1. EJI tracts (required: everything else depends on the boundary)
    if _skip("eji"):
        tracts = gpd.read_file(outputs["eji"])
    else:
        logger.info(f"Processing EJI tracts from {eji_path.name}...")
        tracts = transform_eji_tracts(eji_path, geoids, target_epsg)
        _save_geojson(tracts, outputs["eji"])

    # 2. Catchment boundary (cheap -> always rebuilt so it matches the tracts)
    boundary = create_catchment_boundary(tracts, area_name)
    _save_geojson(boundary, outputs["boundary"])
    clip_area = boundary.geometry.buffer(buffer_m) if buffer_m > 0 else boundary.geometry
    if buffer_m > 0:
        logger.info(f"Context layers clipped to boundary + {buffer_m:,.0f} m buffer.")

    # 3. Data centers (optional)
    if dc_path is None or not Path(dc_path).exists():
        logger.warning(f"Data center file not found ({dc_path}). Skipping.")
    elif not _skip("data_centers"):
        try:
            logger.info(f"Processing data centers from {Path(dc_path).name}...")
            dc = transform_data_centers(Path(dc_path), clip_area, target_epsg)
            if dc.empty:
                logger.warning("No data centers inside the catchment. Nothing saved.")
            else:
                _save_geojson(dc, outputs["data_centers"])
        except Exception as e:
            logger.error(f"Failed to process data centers: {e}")

    # 4. Land use (optional)
    if land_use_path is None:
        logger.warning("No land use file. Skipping.")
    elif not _skip("land_use"):
        try:
            logger.info(f"Processing land use from {land_use_path.name}...")
            lu = transform_land_use(land_use_path, clip_area, target_epsg)
            if lu.empty:
                logger.warning(
                    "Land use file has no features inside the catchment "
                    "(does it cover this county?). Nothing saved."
                )
            else:
                _save_geojson(lu, outputs["land_use"])
        except Exception as e:
            logger.error(f"Failed to process land use: {e}")

    logger.info("Preprocessing complete!")
    return {k: v for k, v in outputs.items() if v.exists()}