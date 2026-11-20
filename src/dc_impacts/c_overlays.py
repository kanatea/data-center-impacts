"""
c_overlays.py: attribute cleaning for the overlay layers (NOT part of the SEVI).
  1) Data centers: column names, text cleanup, duplicates, shared locations
  2) Land use: group original categories into LU_CLASS
Inputs come from data/transformed/ (a_preprocessing), outputs go to data/cleaned/overlays/.
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) Data centers
# ---------------------------------------------------------------------------
DC_ID_COL = "dc_id"
SPATIAL_COLS = ["dc_id", "in_study_area", "dist_to_area_m", "n_at_location"]

# TODO: fill in once the CSV structure is confirmed (names AFTER snake_case conversion)
DC_RENAME: Dict[str, str] = {}       # e.g. {"facility_name": "name", "capacity_mw": "mw"}
DC_KEEP: Optional[List[str]] = None  # None = keep all columns


def _snake_case(name) -> str:
    """'Facility Name' / 'FacilityName' -> 'facility_name'."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def clean_data_center_attributes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    geom = gdf.geometry.name

    # Column names -> snake_case (consistent names for the web app popups)
    new_names = {c: _snake_case(c) for c in gdf.columns if c != geom}
    dupes = pd.Series(list(new_names.values()))
    if dupes.duplicated().any():
        raise ValueError(f"Column names collide after snake_case: {dupes[dupes.duplicated()].tolist()}")
    gdf = gdf.rename(columns=new_names)

    # Text: strip whitespace, empty strings -> missing
    # (no automatic text->number conversion: ZIP codes etc. would lose leading zeros)
    for c in gdf.columns:
        if c != geom and (gdf[c].dtype == object or pd.api.types.is_string_dtype(gdf[c])):
            gdf[c] = gdf[c].map(lambda v: (v.strip() or None) if isinstance(v, str) else v)

    # Exact duplicates: same attributes AND same location (dc_id ignored)
    attr_cols = [c for c in gdf.columns if c not in (geom, DC_ID_COL)]
    key = gdf[attr_cols].astype(str)
    key["_wkb"] = gdf.geometry.to_wkb().values
    dup = key.duplicated()
    if dup.any():
        logger.warning(f"Dropping {int(dup.sum())} exact duplicate records (dc_id: {gdf.loc[dup, DC_ID_COL].tolist()})")
        gdf = gdf[~dup].copy()

    # Shared locations (campuses, several operators at one site): flag, don't drop
    loc = gdf.geometry.to_wkb()
    gdf["n_at_location"] = loc.map(loc.value_counts()).values
    n_shared = int((gdf["n_at_location"] > 1).sum())
    if n_shared:
        logger.info(f"{n_shared} records share a location with another record (see n_at_location).")

    # Optional rename + column selection
    if DC_RENAME:
        gdf = gdf.rename(columns=DC_RENAME)
    if DC_KEEP is not None:
        missing = [c for c in DC_KEEP if c not in gdf.columns]
        if missing:
            logger.warning(f"DC_KEEP columns not found: {missing}")
        keep = list(dict.fromkeys(c for c in SPATIAL_COLS + DC_KEEP if c in gdf.columns))
        gdf = gdf[keep + [geom]]
    return gdf


def summarize_data_centers(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per-column completeness table (methods appendix + decides what goes in the app popups)."""
    df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "n_missing": df.isna().sum().values,
        "pct_missing": (100 * df.isna().mean()).round(1).values,
        "n_unique": df.nunique().values,
        "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })


def clean_data_centers(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    summary_path: Union[str, Path],
) -> Path:
    """Loads transformed data centers, cleans attributes, saves GeoJSON + CSV + column summary."""
    output_path = Path(output_path)
    gdf = gpd.read_file(input_path)
    n_in = int(gdf["in_study_area"].sum()) if "in_study_area" in gdf.columns else len(gdf)
    logger.info(f"Loaded {len(gdf)} data centers ({n_in} inside the study area, {len(gdf) - n_in} nearby)")

    gdf = clean_data_center_attributes(gdf)
    summarize_data_centers(gdf).to_csv(summary_path, index=False)
    logger.info(f"Data center column summary saved to {Path(summary_path).name}")

    output_path.unlink(missing_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    pd.DataFrame(gdf.drop(columns=gdf.geometry.name)).to_csv(output_path.with_suffix(".csv"), index=False)
    logger.info(f"Cleaned data centers saved to {output_path.name} ({len(gdf)} features)")
    return output_path


# ---------------------------------------------------------------------------
# 2) Land use reclassification
# ---------------------------------------------------------------------------

LAND_USE_COL = "LANDUSE"          # original category column (Rockland file)
LAND_USE_CLASS_COL = "LU_CLASS"   # new column with the grouped category
UNCLASSIFIED = "Unclassified"

# Grouped category -> original LANDUSE values (exact names from the Rockland 2021 land use file).
# Matching ignores case, punctuation and extra spaces. Values not listed -> "Unclassified".
LAND_USE_CATEGORIES: Dict[str, List[str]] = {
    "Residential": [
        "One Family Residential", "Two Family Residential", "Three Family Residential",
        "Multi-Family Residential", "Multi-Family (Senior Housing)",
        "Mixed Use (Res. - Com.)",
    ],
    "Secondary Areas and Third Spaces": [
        "LocalPark - Open Space", "Private Recreation", "Public Park - Open Space",
        "Regional Commercial", "Institutional", "Local Neighborhood", "Office",
        "General Bus. - Community Com.",
    ],
    "Agriculture": ["Agricultural"],
    "Transportation": ["Railroad", "Road - Commuter Parking"],
    "Industrial": ["Heavy Industrial", "Light Industrial - Warehouse", "Utilities"],
    "Other": ["Vacant", "Not Yet Classified", "Under Water"],
}


def _normalize_label(text) -> str:
    """'Public Park - Open Space' -> 'public park open space' (case/punctuation-insensitive matching)."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _resolve_column(gdf: gpd.GeoDataFrame, name: str) -> str:
    """Case-insensitive column lookup that raises a clear error."""
    col = next((c for c in gdf.columns if c.lower() == name.lower()), None)
    if col is None:
        raise ValueError(f"Column '{name}' not found. Columns: {list(gdf.columns)}")
    return col


def classify_land_use(
    gdf: gpd.GeoDataFrame,
    land_use_col: str = LAND_USE_COL,
    categories: Dict[str, List[str]] = LAND_USE_CATEGORIES,
) -> gpd.GeoDataFrame:
    """Adds LU_CLASS with the grouped category. Unmatched values are logged and set to 'Unclassified'."""
    col = _resolve_column(gdf, land_use_col)

    # Reverse lookup: normalized original value -> grouped category
    lookup = {}
    for category, values in categories.items():
        for value in values:
            key = _normalize_label(value)
            if key in lookup and lookup[key] != category:
                logger.warning(f"'{value}' is listed under both {lookup[key]} and {category}; using {lookup[key]}.")
                continue
            lookup[key] = category

    gdf = gdf.copy()
    gdf[LAND_USE_CLASS_COL] = gdf[col].map(lambda v: lookup.get(_normalize_label(v), UNCLASSIFIED))

    # Report values in the data that aren't in the dictionary
    unmatched = gdf.loc[gdf[LAND_USE_CLASS_COL] == UNCLASSIFIED, col].value_counts(dropna=False)
    if len(unmatched):
        logger.warning(
            f"{int(unmatched.sum())} parcels not matched to a category (add these values to LAND_USE_CATEGORIES):\n"
            f"{unmatched.to_string()}"
        )

    # Report dictionary values that don't appear in the data (likely spelling differences)
    present = set(gdf[col].dropna().map(_normalize_label))
    unused = [v for values in categories.values() for v in values if _normalize_label(v) not in present]
    if unused:
        logger.info(f"Listed values not found in the data (check spelling): {unused}")
    return gdf


def summarize_land_use(gdf: gpd.GeoDataFrame, land_use_col: str = LAND_USE_COL) -> pd.DataFrame:
    """Crosswalk table: grouped category x original value, with parcel counts and area (methods appendix)."""
    col = _resolve_column(gdf, land_use_col)
    df = pd.DataFrame({
        "category": gdf[LAND_USE_CLASS_COL],
        "original": gdf[col],
        "area_km2": gdf.geometry.area / 1e6,  # projected CRS in meters
    })
    crosswalk = (
        df.groupby(["category", "original"], dropna=False)
        .agg(n_parcels=("area_km2", "size"), area_km2=("area_km2", "sum"))
        .reset_index()
    )
    crosswalk["pct_area"] = (100 * crosswalk["area_km2"] / crosswalk["area_km2"].sum()).round(2)

    totals = (
        crosswalk.groupby("category")[["n_parcels", "area_km2", "pct_area"]]
        .sum()
        .sort_values("area_km2", ascending=False)
    )
    logger.info(f"Land use categories:\n{totals.round(2).to_string()}")
    return crosswalk.sort_values(["category", "area_km2"], ascending=[True, False])


def classify_land_use_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    summary_path: Union[str, Path],
    land_use_col: str = LAND_USE_COL,
    categories: Dict[str, List[str]] = LAND_USE_CATEGORIES,
) -> Path:
    """Loads the transformed land use, adds grouped categories, saves the result and a crosswalk CSV."""
    output_path = Path(output_path)
    gdf = gpd.read_file(input_path)
    logger.info(f"Loaded {len(gdf)} land use parcels from {Path(input_path).name}")

    gdf = classify_land_use(gdf, land_use_col, categories)
    summarize_land_use(gdf, land_use_col).to_csv(summary_path, index=False)
    logger.info(f"Land use crosswalk saved to {Path(summary_path).name}")

    output_path.unlink(missing_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    logger.info(f"Classified land use saved to {output_path.name}")
    return output_path


# ---------------------------------------------------------------------------
# Main function (called from main)
# ---------------------------------------------------------------------------
def clean_overlays(
    dc_input: Path,
    dc_output: Path,
    dc_summary: Path,
    land_use_input: Path,
    land_use_output: Path,
    land_use_summary: Path,
    land_use_col: str = "LANDUSE",
) -> Dict[str, Path]:
    """Cleans both overlay layers. Each one is optional: missing inputs are skipped with a warning."""
    outputs = {}
    dc_output.parent.mkdir(parents=True, exist_ok=True)

    if dc_input.exists():
        logger.info("Cleaning data center attributes...")
        outputs["data_centers"] = clean_data_centers(dc_input, dc_output, dc_summary)
    else:
        logger.warning(f"{dc_input.name} not found. Skipping data centers.")

    if land_use_input.exists():
        logger.info("Classifying land use...")
        outputs["land_use"] = classify_land_use_file(
            land_use_input, land_use_output, land_use_summary, land_use_col=land_use_col
        )
    else:
        logger.warning(f"{land_use_input.name} not found. Skipping land use classification.")

    logger.info("Overlay cleaning complete!")
    return outputs