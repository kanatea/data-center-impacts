import logging
import sys
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import typer

from dc_impacts.a_preprocessing import (
    filter_and_transform,
    find_gdb,
    find_vector_file,
    load_study_area_metadata,
    resolve_catchment,
)
from dc_impacts.b_sevi import calculate_local_indicators
from dc_impacts.c_overlays import clean_overlays
from dc_impacts.visualization import (
    plot_context_layers,
    plot_data_centers,
    plot_eji_modules,
    plot_land_use_categories,
    plot_land_use_overlay,
    plot_land_use_with_context,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("pyogrio").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})

# Raw input files in data/raw/ (hard coded, like the hospital paths in the old project)
EJI_FILENAME = "EJI_2024_United_States.gdb"                                       # CDC/ATSDR EJI 2024
DATA_CENTER_FILE = Path("Data_Center_Information") / "Data_Centers_Database.csv"  # FracTracker US Data Center Repository
LAND_USE_DEFAULT = "Rockland_Land_Use_2021"    # Rockland County Dept. of Planning, GIS Division (2021)
LAND_USE_DEFAULT_AREA = "rockland_ny"          # study area the default land use file covers
LAND_USE_COL = "LANDUSE"                       # land use category column (Rockland file)
LAND_USE_FOCUS = ["Residential", "Secondary Areas and Third Spaces", "Industrial"]  # categories for the focus map
CONTEXT_DIR = "context"                        # folder in data/raw/; every vector file inside is processed

# Pipeline settings
PREVIEW_ROWS = 5             # tracts in the national vs local console preview (0 = off)
DC_SEARCH_BUFFER_M = 10_000  # keep data centers within this distance of the study area: impacts cross
                             # county lines. Must be >= the largest impact buffer in f_dc_buffers.py.

@app.command()
def main(
    # Defines the study area, defaults to Rockland
    counties: List[str] = typer.Option(
        ["Rockland"],
        "--county",
        "-c",
        help="County name(s). Repeatable: -c Rockland -c Orange. Default: Rockland.",
    ),
    # Defines the study area, defaults to New York
    states: List[str] = typer.Option(
        ["New York"],
        "--state",
        "-s",
        help="State name or abbreviation. Give one (applies to all counties) or one per county, in order. Default: New York.",
    ),
    # Points to a land use file for non-default (non-Rockland) areas
    land_use_file: str = typer.Option(
        LAND_USE_DEFAULT,
        "--land_use_file",
        "-lu",
        help="Land use file or folder inside data/raw/ (or a full path). Folders are searched for .shp/.geojson/.gpkg. Default: Rockland_Land_Use_2021.",
    ),
    # --- Development flags
    epsg_override: Optional[int] = typer.Option(
        None, "--epsg", hidden=True,
        help="Override the automatic projection (sensitivity checks only).",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite/--no-overwrite", hidden=True,
        help="Re-process land use + context layers even if they exist in data/transformed/.",
    ),
    skip_preprocessing: bool = typer.Option(False, "--skip-preprocessing", "-sp", hidden=True),
    skip_sevi: bool = typer.Option(False, "--skip-sevi", "-ss", hidden=True),
    skip_overlays: bool = typer.Option(False, "--skip-overlays", "-so", hidden=True),
    skip_viz: bool = typer.Option(False, "--skip-viz", "-sv", hidden=True),
):
    # Loading configuration and project paths
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data/raw"
    transformed_dir = project_root / "data/transformed"
    cleaned_dir = project_root / "data/cleaned"
    sevi_dir = cleaned_dir / "SEVI"        # SEVI layers
    overlays_dir = cleaned_dir / "overlays"          # data centers + land use
    spatial_dir = cleaned_dir / "spatial_analysis"   # Moran's I / LISA
    viz_dir = project_root / "visualizations"


    dc_path = raw_dir / DATA_CENTER_FILE

    # Create output folders if they don't exist
    for d in [transformed_dir, sevi_dir, overlays_dir, spatial_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 0. STUDY AREA: pair counties with states, look up GEOIDs in the EJI data
    # -----------------------------------------------------------------------
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
    area_name = "_".join(c["label"] for c in catchment)                         # e.g., "rockland_ny"
    area_label = " + ".join(f"{c['county']}, {c['state']}" for c in catchment)  # for map titles
    for c in catchment:
        logger.info(f"Study area: {c['county']}, {c['state']} -> GEOID {c['geoid']}")

    # Land use: relative names are looked up in data/raw/, full paths used as given
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
            "Use --land_use_file to provide one for this study area."
        )

    # Context layers (parks, lakes, streams, ...)
    context_path = raw_dir / CONTEXT_DIR
    if not context_path.is_dir():
        logger.info(f"No context folder at {context_path}; context layers skipped.")
        context_path = None

    # -----------------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------------
    # a) transformed/ (geometry only)
    study_area_path = transformed_dir / f"study_area_{area_name}.json"
    eji_transformed = transformed_dir / f"eji_tracts_{area_name}.geojson"
    boundary_path = transformed_dir / f"catchment_boundary_{area_name}.geojson"
    dc_transformed = transformed_dir / f"data_centers_{area_name}.geojson"
    land_use_transformed = transformed_dir / f"land_use_{area_name}.geojson"

    # b) cleaned/processed/ (SEVI inputs)
    indicators_output = sevi_dir / f"eji_local_{area_name}.geojson"
    indicators_summary = sevi_dir / f"indicator_summary_{area_name}.csv"

    # c) cleaned/overlays/ (data centers + land use)
    dc_cleaned = overlays_dir / f"data_centers_{area_name}.geojson"
    dc_summary = overlays_dir / f"data_center_summary_{area_name}.csv"
    land_use_classified = overlays_dir / f"land_use_classified_{area_name}.geojson"
    land_use_summary = overlays_dir / f"land_use_summary_{area_name}.csv"

    # -----------------------------------------------------------------------
    # 1. PREPROCESSING (a): filter EJI tracts -> choose projection -> boundary
    #    -> clip + reproject data centers, land use, context layers
    # -----------------------------------------------------------------------
    if not skip_preprocessing:
        logger.info("Starting preprocessing...")
        filter_and_transform(
            eji_path=eji_path,
            dc_path=dc_path,
            land_use_path=land_use_path,
            transformed_dir=transformed_dir,
            geoids=geoids,
            area_name=area_name,
            epsg_override=epsg_override,
            dc_search_buffer_m=DC_SEARCH_BUFFER_M,
            overwrite=overwrite,
            context_dir=context_path,
            state_names=sorted({c["state"] for c in catchment}),
        )
    else:
        logger.info("Skipping preprocessing.")

    if study_area_path.exists():
        meta = load_study_area_metadata(study_area_path)
        logger.info(f"Study area CRS: EPSG:{meta['epsg']} ({meta['crs_name']})")
    else:
        logger.warning(f"{study_area_path.name} not found. Run preprocessing first.")

    # -----------------------------------------------------------------------
    # 2. SEVI INDICATORS (b): -999 -> NaN, verify EJI logic,
    #    recalculate all four modules within the study area
    # -----------------------------------------------------------------------
    if not skip_sevi:
        if eji_transformed.exists():
            logger.info("Starting local indicator calculation...")
            calculate_local_indicators(
                eji_transformed, indicators_output, indicators_summary, n_preview=PREVIEW_ROWS
            )
        else:
            logger.error(f"{eji_transformed.name} not found. Run preprocessing first.")
            raise typer.Exit(code=1)
    else:
        logger.info("Skipping indicators.")

    # -----------------------------------------------------------------------
    # 3. OVERLAYS (c): data center + land use attribute cleaning (separate from the SEVI)
    # -----------------------------------------------------------------------
    if not skip_overlays:
        clean_overlays(
            dc_input=dc_transformed, dc_output=dc_cleaned, dc_summary=dc_summary,
            land_use_input=land_use_transformed, land_use_output=land_use_classified,
            land_use_summary=land_use_summary, land_use_col=LAND_USE_COL,
        )
    else:
        logger.info("Skipping overlays.")

    # 4. MCDA WEIGHTS       (d_apply_weights.py)
    # 5. SPATIAL ANALYSIS   (e_spatial_analysis.py)
    # 6. DC IMPACT BUFFERS  (f_dc_buffers.py)
    # 7. BUFFER LAND USE %  (g_buffer_land_use.py)

    # -----------------------------------------------------------------------
    # VISUALIZATION: EJI modules, data centers, land use, context layers
    # -----------------------------------------------------------------------
    if not skip_viz:
        if not indicators_output.exists() or not boundary_path.exists():
            logger.error("Indicator or boundary file missing. Run preprocessing + indicators first.")
        else:
            logger.info("Generating test maps...")
            gdf_local = gpd.read_file(indicators_output)
            boundary = gpd.read_file(boundary_path)
            module_viz_dir = viz_dir / "eji_modules"
            land_use_viz_dir = viz_dir / "land_use"
            dc_viz_dir = viz_dir / "data_centers"

            # EJI modules: same 0-1 color scale -> local and national figures are directly comparable
            plot_eji_modules(
                gdf=gdf_local, boundary=boundary, scope="LOCAL",
                title=f"EJI modules, ranked within {area_label}",
                output_path=module_viz_dir / f"eji_modules_local_{area_name}.png",
            )
            plot_eji_modules(
                gdf=gdf_local, boundary=boundary, scope="NATIONAL",
                title=f"EJI modules, ranked nationally ({area_label})",
                output_path=module_viz_dir / f"eji_modules_national_{area_name}.png",
            )

            # Data centers: inside the study area vs. within the search buffer
            if dc_cleaned.exists():
                plot_data_centers(
                    tracts=gdf_local, boundary=boundary, data_centers=gpd.read_file(dc_cleaned),
                    search_buffer_m=DC_SEARCH_BUFFER_M,
                    title=f"Data centers in and near {area_label}",
                    output_path=dc_viz_dir / f"data_centers_{area_name}.png",
                )
            else:
                logger.warning("No cleaned data center file. Skipping data center map.")

            # Original land use categories (loaded once, reused as base for the context map)
            land_use_raw = None
            if land_use_transformed.exists():
                logger.info("Loading land use (large file, may take a moment)...")
                land_use_raw = gpd.read_file(land_use_transformed)
                plot_land_use_overlay(
                    tracts=gdf_local, land_use=land_use_raw, boundary=boundary,
                    land_use_col=LAND_USE_COL,
                    title=f"Original land use categories: {area_label}",
                    output_path=land_use_viz_dir / f"land_use_original_{area_name}.png",
                )

            # Context layers on top of the preprocessing check map
            context_layers = {
                p.name.removesuffix(f"_{area_name}.geojson"): gpd.read_file(p)
                for p in sorted((transformed_dir / "context").glob(f"*_{area_name}.geojson"))
            }
            if context_layers:
                plot_context_layers(
                    tracts=gdf_local, boundary=boundary,
                    context_layers=context_layers, land_use=land_use_raw,
                    title=f"Preprocessing check with context layers: {area_label}",
                    output_path=viz_dir / f"check_preprocessing_context_{area_name}.png",
                )
            else:
                logger.warning("No transformed context layers found. Skipping context map.")

            # Grouped land use categories: all categories, focus categories, with context
            if land_use_classified.exists():
                lu_classified = gpd.read_file(land_use_classified)
                plot_land_use_categories(
                    land_use=lu_classified, boundary=boundary, tracts=gdf_local,
                    title=f"Land use categories: {area_label}",
                    output_path=land_use_viz_dir / f"land_use_categories_{area_name}.png",
                )
                plot_land_use_categories(
                    land_use=lu_classified, boundary=boundary, tracts=gdf_local,
                    categories=LAND_USE_FOCUS,
                    title=f"Residential, secondary areas/third spaces and industrial land use: {area_label}",
                    output_path=land_use_viz_dir / f"land_use_focus_{area_name}.png",
                )
                if context_layers:
                    plot_land_use_with_context(
                        land_use=lu_classified, boundary=boundary,
                        context_layers=context_layers, tracts=gdf_local,
                        title=f"Land use categories with parks and water features: {area_label}",
                        output_path=land_use_viz_dir / f"land_use_context_{area_name}.png",
                    )
            else:
                logger.warning("No classified land use file. Skipping land use category maps.")
    else:
        logger.info("Skipping visualization.")

    logger.info("Complete! :)")


if __name__ == "__main__":
    app()