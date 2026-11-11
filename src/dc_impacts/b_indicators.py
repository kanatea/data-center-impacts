import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

NULL_VALUE = -999            # EJI code for "no data"
HVM_FLAG_THRESHOLD = 0.6666  # HVM: indicator flagged if in the top third (percentile rank >= 0.6666)
LOCAL_SUFFIX = "_LOCAL"      # suffix for catchment-specific columns

# EJI 2024 module/domain structure (indicator names without the E_/EPL_ prefix),
# taken from the order of the EJI 2024 data dictionary.
EJI_MODULES: Dict[str, Dict[str, List[str]]] = {
    "SVM": {  # Social Vulnerability Module
        "DOM1": ["MINRTY"],
        "DOM2": ["POV200", "NOHSDP", "UNEMP", "RENTER", "HOUBDN", "UNINSUR", "NOINT"],
        "DOM3": ["AGE65", "AGE17", "DISABL", "LIMENG"],
        "DOM4": ["MOBILE", "GROUPQ"],
    },
    "EBM": {  # Environmental Burden Module
        "DOM1": ["OZONE", "PM", "DSLPM", "TOTCR"],
        "DOM2": ["NPL", "TRI", "TSD", "RMP", "COAL", "LEAD"],
        "DOM3": ["PARK", "HOUAGE", "WLKIND"],
        "DOM4": ["RAIL", "ROAD", "AIRPRT"],
        "DOM5": ["IMPWTR"],
    },
    "CBM": {  # Climate Burden Module
        "DOM1": ["NEHD"],
        "DOM2": ["SMOKE", "BURN"],
        "DOM3": ["CFLD", "DRGT", "HRCN", "RFLD", "SWND", "TRND"],
    },
    "HVM": {  # Health Vulnerability Module (flag-based, no domains in EJI)
        "FLAGS": ["ASTHMA", "CANCER", "CHD", "MHLTH", "DIABETES"],
    },
}
FLAG_MODULES = {"HVM"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def percent_rank(series: pd.Series) -> pd.Series:
    """
    Percentile rank in [0, 1], ties share the lowest rank (like Excel PERCENTRANK.INC).
    NaNs stay NaN. Constant columns get 0 everywhere (no relative difference).
    """
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    n = len(valid)
    if n == 0:
        return out
    if n == 1:
        out[valid.index] = 0.0
        return out
    ranks = valid.rank(method="min")
    out[valid.index] = ((ranks - 1) / (n - 1)).round(4)
    return out


def _uppercase_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """EJI capitalization isn't consistent (e.g., StateAbbr vs STATEABBR) -> uppercase all attribute names."""
    geom = gdf.geometry.name
    gdf.columns = [c if c == geom else c.upper() for c in gdf.columns]
    return gdf


def all_indicators(modules: Dict[str, Dict[str, List[str]]] = EJI_MODULES) -> List[str]:
    return [ind for domains in modules.values() for inds in domains.values() for ind in inds]


# ---------------------------------------------------------------------------
# 1) Null handling
# ---------------------------------------------------------------------------
def replace_nulls(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Replaces EJI -999 codes with NaN in all numeric columns (flag columns F_ excluded)."""
    num_cols = [c for c in gdf.select_dtypes(include="number").columns if not c.startswith("F_")]
    n_nulls = int((gdf[num_cols] == NULL_VALUE).sum().sum())
    gdf[num_cols] = gdf[num_cols].replace(NULL_VALUE, np.nan)
    logger.info(f"Replaced {n_nulls} '-999' values with NaN across {len(num_cols)} numeric columns.")
    return gdf


# ---------------------------------------------------------------------------
# 2) Verification: does the data follow the assumed EJI aggregation logic?
# ---------------------------------------------------------------------------
def verify_domains(gdf: gpd.GeoDataFrame, modules=EJI_MODULES, tol: float = 0.01) -> bool:
    """
    Domain level: SPL_<MOD>_<DOM> == sum of national EPL_ in that domain,
    and F_HVM == number of HVM indicators with EPL >= threshold.
    """
    ok = True
    for mod, domains in modules.items():
        for dom, inds in domains.items():
            epl_cols = [f"EPL_{i}" for i in inds]
            if not all(c in gdf.columns for c in epl_cols):
                logger.warning(f"{mod} {dom}: missing national EPL columns, cannot verify.")
                continue

            if mod in FLAG_MODULES:
                if "F_HVM" not in gdf.columns:
                    logger.warning("F_HVM not found, cannot verify HVM.")
                    continue
                expected = (gdf[epl_cols] >= HVM_FLAG_THRESHOLD).sum(axis=1)
                diff = (gdf["F_HVM"] - expected).abs().max()
                label = "F_HVM"
            else:
                spl = f"SPL_{mod}_{dom}"
                if spl not in gdf.columns:
                    logger.warning(f"{spl} not found, cannot verify.")
                    continue
                expected = gdf[epl_cols].sum(axis=1, skipna=False)
                diff = (gdf[spl] - expected).abs().max()
                label = spl

            if pd.isna(diff):
                logger.warning(f"{label}: could not verify (all values missing).")
            elif diff <= tol:
                logger.info(f"Verified {label} (max diff {diff:.4f}).")
            else:
                ok = False
                logger.warning(f"{label}: national values DON'T match the assumed logic (max diff {diff:.4f}).")
    return ok


def detect_module_aggregation(gdf: gpd.GeoDataFrame, modules=EJI_MODULES, tol: float = 0.01) -> Dict[str, str]:
    """
    Module level: is SPL_<MOD> the sum of the domain SPLs or of the domain RPLs?
    Tests both against the national values and returns the matching method per module.
    """
    methods = {}
    for mod, domains in modules.items():
        if mod in FLAG_MODULES:
            continue
        methods[mod] = "SPL"  # default
        spl = f"SPL_{mod}"
        if spl not in gdf.columns:
            logger.warning(f"{spl} not found. Defaulting to sum of domain SPLs.")
            continue

        diffs = {}
        for method in ("SPL", "RPL"):
            cols = [f"{method}_{mod}_{d}" for d in domains]
            if all(c in gdf.columns for c in cols):
                diffs[method] = (gdf[spl] - gdf[cols].sum(axis=1, skipna=False)).abs().max()

        matches = [m for m, d in diffs.items() if pd.notna(d) and d <= tol]
        if matches:
            methods[mod] = matches[0]
            logger.info(f"Verified {spl} = sum of domain {matches[0]}s (max diff {diffs[matches[0]]:.4f}).")
        else:
            logger.warning(f"{spl} matches neither sum of domain SPLs nor RPLs {diffs}. Defaulting to SPL sum.")
    return methods


# ---------------------------------------------------------------------------
# 3) Direction check: is a higher E_ value worse (as for most indicators)?
# ---------------------------------------------------------------------------
def detect_directions(gdf: gpd.GeoDataFrame, indicators: List[str]) -> Dict[str, int]:
    """
    Uses the national EPL_ rank as the reference: if E_x and EPL_x are negatively
    correlated, EJI ranks that indicator in reverse (higher E_ = less burden).
    Returns +1 (higher = worse) or -1 (reversed) per indicator.
    """
    directions = {}
    for ind in indicators:
        e, epl = f"E_{ind}", f"EPL_{ind}"
        directions[ind] = 1
        if e not in gdf.columns or epl not in gdf.columns:
            continue
        both = gdf[[e, epl]].dropna()
        if both[e].nunique() < 2:
            continue  # constant -> direction irrelevant
        rho = both[e].rank().corr(both[epl].rank())  # Spearman correlation
        if rho < 0:
            directions[ind] = -1
            logger.info(f"{ind}: reversed (Spearman rho = {rho:.2f}) -> lower E_ value = higher burden.")
    return directions


# ---------------------------------------------------------------------------
# 4-5) Local ranks, domain and module scores
# ---------------------------------------------------------------------------
def compute_local_scores(
    gdf: gpd.GeoDataFrame,
    directions: Dict[str, int],
    module_methods: Dict[str, str],
    modules=EJI_MODULES,
) -> gpd.GeoDataFrame:
    """Recalculates EJI ranks within the catchment, following the (verified) EJI aggregation logic."""
    gdf = gdf.copy()
    sfx = LOCAL_SUFFIX

    # Indicator-level local percentile ranks
    for ind in all_indicators(modules):
        e = f"E_{ind}"
        if e not in gdf.columns:
            logger.warning(f"{e} not found in EJI data. Skipping indicator.")
            continue
        gdf[f"EPL_{ind}{sfx}"] = percent_rank(gdf[e] * directions.get(ind, 1))

    for mod, domains in modules.items():
        # HVM: flag indicators in the local top third, count flags, rank the count
        if mod in FLAG_MODULES:
            flag_cols = []
            for ind in domains["FLAGS"]:
                epl = f"EPL_{ind}{sfx}"
                if epl in gdf.columns:
                    flag = (gdf[epl] >= HVM_FLAG_THRESHOLD).astype(float)
                    gdf[f"F_{ind}{sfx}"] = flag.where(gdf[epl].notna())
                    flag_cols.append(f"F_{ind}{sfx}")
            gdf[f"F_{mod}{sfx}"] = gdf[flag_cols].sum(axis=1, skipna=False)
            gdf[f"RPL_{mod}{sfx}"] = percent_rank(gdf[f"F_{mod}{sfx}"])
            logger.info(f"Computed local {mod}: {len(flag_cols)} health flags.")
            continue

        # Other modules: sum ranks per domain -> rank; aggregate domains per module -> rank
        dom_spl_cols, dom_rpl_cols = [], []
        for dom, inds in domains.items():
            epl_cols = [f"EPL_{i}{sfx}" for i in inds if f"EPL_{i}{sfx}" in gdf.columns]
            spl, rpl = f"SPL_{mod}_{dom}{sfx}", f"RPL_{mod}_{dom}{sfx}"
            gdf[spl] = gdf[epl_cols].sum(axis=1, skipna=False)  # tracts with nulls -> NaN (as in EJI)
            gdf[rpl] = percent_rank(gdf[spl])
            dom_spl_cols.append(spl)
            dom_rpl_cols.append(rpl)

        method = module_methods.get(mod, "SPL")
        agg_cols = dom_rpl_cols if method == "RPL" else dom_spl_cols
        gdf[f"SPL_{mod}{sfx}"] = gdf[agg_cols].sum(axis=1, skipna=False)
        gdf[f"RPL_{mod}{sfx}"] = percent_rank(gdf[f"SPL_{mod}{sfx}"])
        logger.info(f"Computed local {mod}: {len(domains)} domains (module = sum of domain {method}s).")
    return gdf


# ---------------------------------------------------------------------------
# 6) Indicator summary table (methods appendix)
# ---------------------------------------------------------------------------
def summarize_indicators(gdf: gpd.GeoDataFrame, directions: Dict[str, int], modules=EJI_MODULES) -> pd.DataFrame:
    rows = []
    for mod, domains in modules.items():
        for dom, inds in domains.items():
            for ind in inds:
                e = f"E_{ind}"
                if e not in gdf.columns:
                    continue
                s = gdf[e]
                rows.append({
                    "module": mod, "domain": dom, "indicator": ind,
                    "direction": "reversed" if directions.get(ind, 1) == -1 else "higher = worse",
                    "min": s.min(), "median": s.median(), "max": s.max(),
                    "n_missing": int(s.isna().sum()),
                    "n_unique": int(s.nunique()),
                    "zero_variance": s.nunique() <= 1,
                })
    summary = pd.DataFrame(rows)
    zero_var = summary.loc[summary["zero_variance"], "indicator"].tolist()
    if zero_var:
        logger.info(f"No variation within catchment (local rank 0 everywhere): {zero_var}")
    return summary


# ---------------------------------------------------------------------------
# 7) Console preview: national vs local module ranks
# ---------------------------------------------------------------------------
def preview_local_scores(gdf: gpd.GeoDataFrame, n_rows: int = 5, modules=EJI_MODULES):
    """Logs the first n tracts with national and local module ranks side by side, plus a per-module comparison."""
    if n_rows <= 0:
        return

    id_cols = [c for c in ["GEOID", "E_TOTPOP"] if c in gdf.columns]
    rank_cols = []
    for mod in modules:
        rank_cols += [c for c in (f"RPL_{mod}", f"RPL_{mod}{LOCAL_SUFFIX}") if c in gdf.columns]

    table = pd.DataFrame(gdf[id_cols + rank_cols]).head(n_rows)
    logger.info(f"Preview of {n_rows} tracts (national RPL_ vs local RPL_..._LOCAL):\n{table.to_string(index=False)}")

    rows = []
    for mod in modules:
        nat, loc = f"RPL_{mod}", f"RPL_{mod}{LOCAL_SUFFIX}"
        if nat in gdf.columns and loc in gdf.columns:
            rows.append({
                "module": mod,
                "national_min": gdf[nat].min(), "national_max": gdf[nat].max(),
                "local_min": gdf[loc].min(), "local_max": gdf[loc].max(),
                "spearman_nat_vs_local": gdf[nat].rank().corr(gdf[loc].rank()),
            })
    if rows:
        logger.info(f"Module comparison across all tracts:\n{pd.DataFrame(rows).round(3).to_string(index=False)}")


# ---------------------------------------------------------------------------
# Main function (called from __main__)
# ---------------------------------------------------------------------------
def calculate_local_indicators(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    summary_path: Union[str, Path],
    n_preview: int = 5,
) -> Path:
    """
    Cleans EJI nulls, verifies the EJI aggregation logic, and recalculates all four
    modules (SVM, EBM, HVM, CBM) relative to the catchment. National columns are kept.
    """
    output_path = Path(output_path)
    gdf = gpd.read_file(input_path)
    gdf = _uppercase_columns(gdf)
    logger.info(f"Loaded {len(gdf)} tracts from {Path(input_path).name}")

    if "E_TOTPOP" in gdf.columns and (gdf["E_TOTPOP"] == 0).any():
        n = int((gdf["E_TOTPOP"] == 0).sum())
        logger.warning(f"{n} tract(s) with zero population: their scores are NaN in EJI and should be excluded from maps.")

    gdf = replace_nulls(gdf)

    domains_ok = verify_domains(gdf)
    module_methods = detect_module_aggregation(gdf)
    if domains_ok:
        logger.info("EJI domain aggregation verified against national values.")
    else:
        logger.warning("Some national domain values didn't match the assumed logic. Check the warnings above.")

    directions = detect_directions(gdf, all_indicators())
    gdf = compute_local_scores(gdf, directions, module_methods)

    summary = summarize_indicators(gdf, directions)
    summary.to_csv(summary_path, index=False)
    logger.info(f"Indicator summary saved to {Path(summary_path).name}")

    preview_local_scores(gdf, n_preview)

    output_path.unlink(missing_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    logger.info(f"Local indicators saved to {output_path.name}")

    # Full attribute table without geometry, for checking in Excel
    csv_path = output_path.with_suffix(".csv")
    pd.DataFrame(gdf.drop(columns=gdf.geometry.name)).to_csv(csv_path, index=False)
    logger.info(f"Attribute table saved to {csv_path.name}")
    return output_path



# ---------------------------------------------------------------------------
# 8) Land use reclassification
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