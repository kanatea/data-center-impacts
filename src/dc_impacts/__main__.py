import logging
import sys
from pathlib import Path
from typing import List

import typer

from dc_impacts.a_preprocessing import (
    filter_and_transform,
    find_gdb,
    find_vector_file,
    resolve_catchment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("pyogrio").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})

# Raw input files in data/raw/ (hard coded, like the hospital paths in the old project)
EJI_FILENAME = "EJI_2024_United_States.gdb"        # .gdb folder (or folder containing the .gdb)
DATA_CENTER_FILENAME = "data_centers.csv"      # TODO: update to actual name (.csv or .geojson)
LAND_USE_DEFAULT = "Rockland_Land_Use_2021"    # Rockland County Dept. of Planning, GIS Division (2021)
LAND_USE_DEFAULT_AREA = "rockland_ny"          # catchment the default land use file covers


@app.command()
def main(
    counties: List[str] = typer.Option(
        ["Rockland"],
        "--county",
        "-c",
        help="County name(s). Repeatable: -c Rockland -c Orange. Default: Rockland.",
    ),
    states: List[str] = typer.Option(
        ["New York"],
        "--state",
        "-s",
        help="State name or abbreviation. Give one (applies to all counties) or one per county, in order. Default: New York.",
    ),
    land_use_file: str = typer.Option(
        LAND_USE_DEFAULT,
        "--land_use_file",
        "-lu",
        help="Land use file or folder inside data/raw/ (or a full path). Folders are searched for .shp/.geojson/.gpkg. Default: Rockland_Land_Use_2021.",
    ),
    target_epsg: int = typer.Option(
        32115,
        "--epsg_projection",
        "-proj",
        help="EPSG projection. Default: 32115 (NAD83 / New York East, meters). Use 26918 for larger extents.",
    ),
    buffer_m: float = typer.Option(
        0,
        "--buffer",
        "-b",
        help="Buffer (meters) around the catchment for context layers (data centers, land use). EJI tracts are always filtered exactly.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="Re-process layers that already exist in data/transformed/.",
    ),
    skip_preprocessing: bool = typer.Option(
        False,
        "--skip-preprocessing",
        "-sp",
        help="Skip filtering + reprojection (use already transformed data).",
    ),
):
    # Loading configuration
    project_root = Path(__file__).resolve().parents[2]

    # Resolve paths relative to project root
    raw_dir = project_root / "data/raw"
    transformed_dir = project_root / "data/transformed"
    cleaned_dir = project_root / "data/cleaned"
    processed_dir = cleaned_dir / "processed"
    spatial_dir = cleaned_dir / "spatial_analysis"
    viz_dir = project_root / "visualizations"

    dc_path = raw_dir / DATA_CENTER_FILENAME

    # Create output folders if they don't exist
    for d in [transformed_dir, processed_dir, spatial_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 0. CATCHMENT: pair counties with states, then look up GEOIDs in the EJI data
    if len(states) == 1:
        states = states * len(counties)
    elif len(states) != len(counties):
        raise typer.BadParameter(
            f"Got {len(counties)} counties but {len(states)} states. "
            "Give one state for all counties, or one state per county."
        )

    try:
        eji_path = find_gdb(raw_dir / EJI_FILENAME)
        catchment = resolve_catchment(eji_path, counties, states)
    except (FileNotFoundError, ValueError) as e:
        logger.error(e)
        raise typer.Exit(code=1)

    geoids = [c["geoid"] for c in catchment]
    area_name = "_".join(c["label"] for c in catchment)  # e.g., "rockland_ny"
    for c in catchment:
        logger.info(f"Catchment: {c['county']}, {c['state']} -> GEOID {c['geoid']}")

    # Land use: relative names are looked up in data/raw/, full paths are used as given
    land_use_path = Path(land_use_file)
    if not land_use_path.is_absolute():
        land_use_path = raw_dir / land_use_path
    try:
        land_use_path = find_vector_file(land_use_path)
        logger.info(f"Land use file: {land_use_path.name}")
    except FileNotFoundError as e:
        logger.warning(f"{e} -> land use layer will be skipped.")
        land_use_path = None

    if land_use_file == LAND_USE_DEFAULT and area_name != LAND_USE_DEFAULT_AREA:
        logger.warning(
            "The default land use file only covers Rockland County. "
            "Use --land_use_file to provide one for this catchment."
        )

    # 1. PREPROCESSING: filter EJI tracts to catchment, derive boundary by dissolving them,
    #    clip data centers + land use to (buffered) boundary, THEN reproject to target_epsg
    if not skip_preprocessing:
        logger.info("Starting preprocessing...")
        filter_and_transform(
            eji_path=eji_path,
            dc_path=dc_path,
            land_use_path=land_use_path,
            transformed_dir=transformed_dir,
            geoids=geoids,
            area_name=area_name,
            target_epsg=target_epsg,
            buffer_m=buffer_m,
            overwrite=overwrite,
        )
    else:
        logger.info("Skipping preprocessing.")

    # 2. INDICATORS        (b_indicators.py)       -> EJI -999 -> NaN, local re-ranking of all 4 modules
    # 3. MCDA WEIGHTS      (c_apply_weights.py)
    # 4. SPATIAL ANALYSIS  (d_spatial_analysis.py)
    # 5. DC BUFFERS        (e_dc_buffers.py)

    logger.info("Complete! :)")


if __name__ == "__main__":
    app()