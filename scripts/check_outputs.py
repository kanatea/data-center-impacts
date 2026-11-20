"""
scripts/check_outputs.py: sanity checks on the pipeline outputs (steps a-c + maps).

Run from the project root AFTER `uv run dc_impacts`:
    uv run python scripts/check_outputs.py                          # auto-detects the study area
    uv run python scripts/check_outputs.py -a rockland_ny_bergen_nj # pick one if several exist

Statuses: PASS = as expected | WARN = check manually | FAIL = broken | SKIP = layer not produced
Exit code 1 if anything FAILs.
"""
import json
import re
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import pyogrio
import typer
from pyproj import CRS
from rich.console import Console
from rich.table import Table

from dc_impacts.b_sevi import EJI_MODULES, LOCAL_SUFFIX, NULL_VALUE
from dc_impacts.c_overlays import LAND_USE_CLASS_COL, UNCLASSIFIED
from dc_impacts.__main__ import DC_SEARCH_BUFFER_M

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMED = PROJECT_ROOT / "data/transformed"
SEVI = PROJECT_ROOT / "data/cleaned/SEVI"
OVERLAYS = PROJECT_ROOT / "data/cleaned/overlays"
VIZ = PROJECT_ROOT / "visualizations"

DIST_TOLERANCE_M = 1.0        # rounding tolerance for dist_to_area_m
AREA_TOLERANCE_PCT = 1.0      # boundary area vs. metadata
LU_COVERAGE_WARN = (80, 105)  # % of boundary covered by land use parcels outside this range -> WARN

console = Console()
app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------
class Report:
    STYLES = {"PASS": "green", "WARN": "yellow", "FAIL": "bold red", "SKIP": "dim"}

    def __init__(self):
        self.rows = []

    def add(self, status: str, section: str, item: str, detail: str = ""):
        self.rows.append((status, section, item, detail))

    def ok(self, *a):   self.add("PASS", *a)
    def warn(self, *a): self.add("WARN", *a)
    def fail(self, *a): self.add("FAIL", *a)
    def skip(self, *a): self.add("SKIP", *a)

    def print(self):
        table = Table(show_lines=False, header_style="bold")
        for col in ("Status", "Section", "Item", "Detail"):
            table.add_column(col, overflow="fold")
        for status, section, item, detail in self.rows:
            table.add_row(f"[{self.STYLES[status]}]{status}[/]", section, item, detail)
        console.print(table)
        counts = pd.Series([r[0] for r in self.rows]).value_counts()
        console.print("  ".join(f"[{self.STYLES[s]}]{s}: {counts.get(s, 0)}[/]" for s in self.STYLES))

    @property
    def failed(self) -> bool:
        return any(r[0] == "FAIL" for r in self.rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_area(area: Optional[str]) -> str:
    """Uses --area if given, otherwise the single study_area_*.json in data/transformed/."""
    if area:
        return area
    found = sorted(TRANSFORMED.glob("study_area_*.json"))
    if not found:
        raise typer.BadParameter("No study_area_*.json in data/transformed/. Run `uv run dc_impacts` first.")
    if len(found) > 1:
        names = [p.stem.removeprefix("study_area_") for p in found]
        raise typer.BadParameter(f"Several study areas found: {names}. Pick one with --area.")
    return found[0].stem.removeprefix("study_area_")


def check_layer_file(r: Report, section: str, path: Path, epsg: int, required: bool = True) -> bool:
    """Exists, has features, and is in the study area CRS. Reads metadata only (fast, even for land use)."""
    if not path.exists():
        (r.fail if required else r.skip)(section, path.name, "missing")
        return False
    info = pyogrio.read_info(path, force_feature_count=True)
    file_epsg = CRS.from_user_input(info["crs"]).to_epsg() if info["crs"] else None
    if info["features"] == 0:
        r.fail(section, path.name, "0 features")
        return False
    if file_epsg != epsg:
        r.fail(section, path.name, f"CRS is EPSG:{file_epsg}, expected EPSG:{epsg}")
        return False
    r.ok(section, path.name, f"{info['features']} features, {info['geometry_type']}, EPSG:{file_epsg}")
    return True


# ---------------------------------------------------------------------------
# a) Preprocessing
# ---------------------------------------------------------------------------
def check_study_area(r: Report, area: str) -> Optional[dict]:
    sec, path = "a: study area", TRANSFORMED / f"study_area_{area}.json"
    if not path.exists():
        r.fail(sec, path.name, "missing")
        return None
    meta = json.loads(path.read_text())
    missing = [k for k in ("epsg", "crs_name", "crs_selection", "bounds_wgs84", "county_geoids") if k not in meta]
    if missing:
        r.fail(sec, path.name, f"missing keys: {missing}")
        return None
    crs = CRS.from_epsg(meta["epsg"])
    if not crs.is_projected or crs.axis_info[0].unit_name != "metre":
        r.fail(sec, "projection", f"EPSG:{meta['epsg']} is not a projected CRS in metres")
    else:
        r.ok(sec, "projection", f"EPSG:{meta['epsg']} ({meta['crs_name']}): {meta['crs_selection']}")
    states = sorted(set(meta.get("states", [])))
    if len(states) == 1 and states[0].lower() not in meta["crs_name"].lower():
        r.warn(sec, "projection vs state", f"single-state study area ({states[0]}) but CRS is {meta['crs_name']}")
    return meta


def check_transformed_layers(r: Report, area: str, epsg: int):
    sec = "a: transformed"
    check_layer_file(r, sec, TRANSFORMED / f"eji_tracts_{area}.geojson", epsg)
    check_layer_file(r, sec, TRANSFORMED / f"catchment_boundary_{area}.geojson", epsg)
    check_layer_file(r, sec, TRANSFORMED / f"data_centers_{area}.geojson", epsg, required=False)
    check_layer_file(r, sec, TRANSFORMED / f"land_use_{area}.geojson", epsg, required=False)

    context = sorted((TRANSFORMED / "context").glob(f"*_{area}.geojson"))
    if not context:
        r.skip("a: context", "context layers", "none found")
    for p in context:
        check_layer_file(r, "a: context", p, epsg, required=False)


def check_tracts_and_boundary(r: Report, area: str, meta: dict):
    sec = "a: tracts/boundary"
    tracts_path = TRANSFORMED / f"eji_tracts_{area}.geojson"
    boundary_path = TRANSFORMED / f"catchment_boundary_{area}.geojson"
    if not tracts_path.exists() or not boundary_path.exists():
        return

    tracts = gpd.read_file(tracts_path)
    geoid = tracts["GEOID"].astype(str)
    outside = ~geoid.str[:5].isin(meta["county_geoids"])
    if outside.any():
        r.fail(sec, "GEOID filter", f"{int(outside.sum())} tracts outside counties {meta['county_geoids']}")
    else:
        r.ok(sec, "GEOID filter", f"{len(tracts)} tracts, all in {meta['county_geoids']}")

    dupes = geoid.duplicated().sum()
    (r.fail if dupes else r.ok)(sec, "unique GEOIDs", f"{dupes} duplicates")

    invalid = (~tracts.is_valid).sum()
    (r.fail if invalid else r.ok)(sec, "valid geometries", f"{invalid} invalid tract geometries")

    boundary = gpd.read_file(boundary_path)
    if len(boundary) != 1:
        r.fail(sec, "boundary", f"{len(boundary)} features, expected 1 (dissolved)")
    elif int(boundary["n_tracts"].iloc[0]) != len(tracts):
        r.fail(sec, "boundary", f"n_tracts={boundary['n_tracts'].iloc[0]} but {len(tracts)} tracts on disk")
    else:
        area_km2 = boundary.area.sum() / 1e6
        diff = abs(area_km2 - meta["area_km2"]) / meta["area_km2"] * 100
        (r.warn if diff > AREA_TOLERANCE_PCT else r.ok)(
            sec, "boundary", f"1 feature, {area_km2:,.1f} km² (metadata: {meta['area_km2']:,.1f} km²)"
        )


def check_data_centers_transformed(r: Report, area: str):
    sec = "a: data centers"
    path = TRANSFORMED / f"data_centers_{area}.geojson"
    issues = TRANSFORMED / "data_centers_coord_issues.csv"

    if issues.exists():
        n = len(pd.read_csv(issues))
        r.warn(sec, issues.name, f"{n} national records dropped for bad coordinates (review once)")
    if not path.exists():
        r.skip(sec, path.name, "no data centers in or near the study area")
        return

    dc = gpd.read_file(path)
    for col in ("dc_id", "in_study_area", "dist_to_area_m"):
        if col not in dc.columns:
            r.fail(sec, col, "column missing")
            return

    dupes = dc["dc_id"].duplicated().sum()
    (r.fail if dupes else r.ok)(sec, "unique dc_id", f"{dupes} duplicates")

    inside = dc["in_study_area"].astype(bool)
    dist = dc["dist_to_area_m"]
    too_far = (dist > DC_SEARCH_BUFFER_M + DIST_TOLERANCE_M).sum()
    (r.fail if too_far else r.ok)(
        sec, "search buffer", f"max distance {dist.max():,.0f} m (limit {DC_SEARCH_BUFFER_M:,} m)"
    )

    inconsistent = ((inside & (dist > DIST_TOLERANCE_M)) | (~inside & (dist == 0))).sum()
    (r.fail if inconsistent else r.ok)(sec, "in_study_area vs distance", f"{inconsistent} inconsistent records")

    n_in = int(inside.sum())
    (r.warn if n_in == 0 else r.ok)(
        sec, "counts", f"{n_in} inside the study area, {len(dc) - n_in} within {DC_SEARCH_BUFFER_M / 1000:g} km"
    )


# ---------------------------------------------------------------------------
# b) SEVI indicators
# ---------------------------------------------------------------------------
def check_sevi(r: Report, area: str):
    sec = "b: sevi"
    path = SEVI / f"eji_local_{area}.geojson"
    summary = SEVI / f"indicator_summary_{area}.csv"
    if not path.exists():
        r.fail(sec, path.name, "missing")
        return
    (r.ok if summary.exists() else r.warn)(sec, summary.name, "exists" if summary.exists() else "missing")

    gdf = gpd.read_file(path)
    num_cols = [c for c in gdf.select_dtypes(include="number").columns if not c.startswith("F_")]
    n999 = int((gdf[num_cols] == NULL_VALUE).sum().sum())
    (r.fail if n999 else r.ok)(sec, "null codes", f"{n999} remaining {NULL_VALUE} values")

    if "E_TOTPOP" in gdf.columns:
        zero_pop = int((gdf["E_TOTPOP"] == 0).sum())
        (r.warn if zero_pop else r.ok)(sec, "zero-population tracts", f"{zero_pop} (exclude from maps)")

    for mod in EJI_MODULES:
        col, nat = f"RPL_{mod}{LOCAL_SUFFIX}", f"RPL_{mod}"
        if col not in gdf.columns:
            r.fail(sec, col, "column missing")
            continue
        s = gdf[col]
        if s.notna().sum() == 0:
            r.fail(sec, col, "all NaN")
        elif s.min() < 0 or s.max() > 1:
            r.fail(sec, col, f"out of [0, 1]: {s.min():.3f} to {s.max():.3f}")
        else:
            rho = gdf[nat].rank().corr(s.rank()) if nat in gdf.columns else float("nan")
            r.ok(sec, col, f"{s.min():.2f} to {s.max():.2f}, {int(s.isna().sum())} NaN, "
                           f"Spearman vs national = {rho:.2f}")


# ---------------------------------------------------------------------------
# c) Overlays
# ---------------------------------------------------------------------------
def check_overlays(r: Report, area: str):
    # Data centers
    sec = "c: data centers"
    dc_path = OVERLAYS / f"data_centers_{area}.geojson"
    if dc_path.exists():
        dc = gpd.read_file(dc_path)
        bad_names = [c for c in dc.columns if c != dc.geometry.name and not re.fullmatch(r"[a-z0-9_]+", c)]
        (r.fail if bad_names else r.ok)(sec, "snake_case columns", f"not converted: {bad_names}" if bad_names else "all OK")
        if "n_at_location" in dc.columns:
            shared = int((dc["n_at_location"] > 1).sum())
            (r.warn if shared else r.ok)(sec, "shared locations", f"{shared} records share coordinates")
        summary = OVERLAYS / f"data_center_summary_{area}.csv"
        (r.ok if summary.exists() else r.warn)(sec, summary.name, "exists" if summary.exists() else "missing")
        r.ok(sec, dc_path.name, f"{len(dc)} features, {len(dc.columns) - 1} attribute columns")
    else:
        r.skip(sec, dc_path.name, "not produced")

    # Land use
    sec = "c: land use"
    lu_path = OVERLAYS / f"land_use_classified_{area}.geojson"
    boundary_path = TRANSFORMED / f"catchment_boundary_{area}.geojson"
    if not lu_path.exists():
        r.skip(sec, lu_path.name, "not produced")
        return
    lu = gpd.read_file(lu_path, columns=[LAND_USE_CLASS_COL])
    if LAND_USE_CLASS_COL not in lu.columns:
        r.fail(sec, LAND_USE_CLASS_COL, "column missing")
        return

    n_unclass = int((lu[LAND_USE_CLASS_COL] == UNCLASSIFIED).sum())
    (r.warn if n_unclass else r.ok)(
        sec, "classification", f"{n_unclass} '{UNCLASSIFIED}' parcels of {len(lu)}"
        + (" (add values to LAND_USE_CATEGORIES)" if n_unclass else "")
    )

    if boundary_path.exists():
        coverage = 100 * lu.area.sum() / gpd.read_file(boundary_path).area.sum()
        lo, hi = LU_COVERAGE_WARN
        (r.ok if lo <= coverage <= hi else r.warn)(
            sec, "coverage", f"parcels cover {coverage:.1f}% of the study area"
            + (" (gaps or overlapping parcels?)" if not lo <= coverage <= hi else "")
        )


def check_separation(r: Report, area: str):
    """Land use moved out of processed/ (SEVI) into overlays/: flag leftovers from older runs."""
    leftovers = sorted(SEVI.glob(f"land_use*_{area}.*"))
    if leftovers:
        r.warn("c: separation", "sevi/", f"old land use files (safe to delete): {[p.name for p in leftovers]}")
    else:
        r.ok("c: separation", "sevi/", "SEVI folder contains no land use files")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def check_visualizations(r: Report, area: str):
    sec = "viz"
    expected = [
        VIZ / "eji_modules" / f"eji_modules_local_{area}.png",
        VIZ / "eji_modules" / f"eji_modules_national_{area}.png",
    ]
    if (OVERLAYS / f"data_centers_{area}.geojson").exists():
        expected.append(VIZ / "data_centers" / f"data_centers_{area}.png")
    has_context = any((TRANSFORMED / "context").glob(f"*_{area}.geojson"))
    if has_context:
        expected.append(VIZ / f"check_preprocessing_context_{area}.png")
    if (OVERLAYS / f"land_use_classified_{area}.geojson").exists():
        expected += [
            VIZ / "land_use" / f"land_use_original_{area}.png",
            VIZ / "land_use" / f"land_use_categories_{area}.png",
            VIZ / "land_use" / f"land_use_focus_{area}.png",
        ]
        if has_context:
            expected.append(VIZ / "land_use" / f"land_use_context_{area}.png")

    for p in expected:
        rel = p.relative_to(VIZ).as_posix()
        (r.ok if p.exists() else r.warn)(sec, rel, "exists" if p.exists() else "missing (was --skip-viz used?)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@app.command()
def main(
    area: Optional[str] = typer.Option(
        None, "--area", "-a",
        help="Study area name used in the filenames (e.g. rockland_ny). Auto-detected if only one exists.",
    ),
):
    area = detect_area(area)
    console.rule(f"[bold]Output checks: {area}")
    r = Report()

    meta = check_study_area(r, area)
    if meta is not None:
        check_transformed_layers(r, area, meta["epsg"])
        check_tracts_and_boundary(r, area, meta)
    check_data_centers_transformed(r, area)
    check_sevi(r, area)
    check_overlays(r, area)
    check_separation(r, area)
    check_visualizations(r, area)

    r.print()
    if r.failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()