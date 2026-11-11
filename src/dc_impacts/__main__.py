import logging
import sys
from pathlib import Path
from typing import List

import geopandas as gpd
import typer

from dc_impacts.a_preprocessing import (
    filter_and_transform,
    find_gdb,
    find_vector_file,
    resolve_catchment,
)
from dc_impacts.b_indicators import calculate_local_indicators, classify_land_use_file
from dc_impacts.visualization import (
    plot_context_layers,
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
EJI_FILENAME = "EJI_2024_United_States.gdb"    # EJI geodatabase folder
DATA_CENTER_FILENAME = "data_centers.csv"      # TODO: update to actual name (.csv or .geojson)
LAND_USE_DEFAULT = "Rockland_Land_Use_2021"    # Rockland County Dept. of Planning, GIS Division (2021)
LAND_USE_DEFAULT_AREA = "rockland_ny"          # catchment the default land use file covers
LAND_USE_COL = "LANDUSE"                       # land use category column (Rockland file)
LAND_USE_FOCUS = ["Residential", "Secondary Areas and Third Spaces", "Industrial"]  # categories for the focus map
CONTEXT_DEFAULT = "context"                    # folder in data/raw/ with parks, lakes, streams, ...


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
    context_dir: str = typer.Option(
        CONTEXT_DEFAULT,
        "--context_dir",
        "-cx",
        help="Folder inside data/raw/ (or a full path) with context layers (parks, lakes, streams). Every vector file inside is clipped + reprojected. Default: context.",
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
        help="Buffer (meters) around the catchment for context layers (data centers, land use, parks, water). EJI tracts are always filtered exactly.",
    ),
    preview_rows: int = typer.Option(
        5,
        "--preview_rows",
        "-pr",
        help="Number of tracts to print in the console preview of national vs local module ranks. 0 = no preview.",
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
    skip_indicators: bool = typer.Option(
        False,
        "--skip-indicators",
        "-si",
        help="Skip local recalculation of the EJI modules.",
    ),
    skip_viz: bool = typer.Option(
        False,
        "--skip-viz",
        "-sv",
        help="Skip the test maps (EJI modules, land use overlay).",
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
    area_label = " + ".join(f"{c['county']}, {c['state']}" for c in catchment)  # for map titles
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

    # Context layers folder (parks, lakes, streams, ...)
    context_path = Path(context_dir)
    if not context_path.is_absolute():
        context_path = raw_dir / context_path
    if not context_path.is_dir():
        logger.info(f"No context folder at {context_path}; context layers skipped.")
        context_path = None

    # 1. PREPROCESSING: filter EJI tracts to catchment, derive boundary by dissolving them,
    #    clip data centers, land use + context layers to (buffered) boundary, THEN reproject
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
            context_dir=context_path,
        )
    else:
        logger.info("Skipping preprocessing.")

    # 2. INDICATORS: -999 -> NaN, verify EJI logic, recalculate all modules within the catchment,
    #    then group the land use categories
    eji_transformed = transformed_dir / f"eji_tracts_{area_name}.geojson"
    boundary_path = transformed_dir / f"catchment_boundary_{area_name}.geojson"
    land_use_transformed = transformed_dir / f"land_use_{area_name}.geojson"
    indicators_output = processed_dir / f"eji_local_{area_name}.geojson"
    indicators_summary = processed_dir / f"indicator_summary_{area_name}.csv"
    land_use_classified = processed_dir / f"land_use_classified_{area_name}.geojson"
    land_use_summary = processed_dir / f"land_use_summary_{area_name}.csv"

    if not skip_indicators:
        if eji_transformed.exists():
            logger.info("Starting local indicator calculation...")
            calculate_local_indicators(eji_transformed, indicators_output, indicators_summary, n_preview=preview_rows)
        else:
            logger.error(f"{eji_transformed.name} not found. Run preprocessing first.")
            raise typer.Exit(code=1)

        if land_use_transformed.exists():
            logger.info("Classifying land use...")
            classify_land_use_file(land_use_transformed, land_use_classified, land_use_summary, land_use_col=LAND_USE_COL)
        else:
            logger.warning(f"{land_use_transformed.name} not found. Skipping land use classification.")
    else:
        logger.info("Skipping indicators.")


    # 3. MCDA WEIGHTS      (c_apply_weights.py)
    # 4. SPATIAL ANALYSIS  (d_spatial_analysis.py)
    # 5. DC BUFFERS        (e_dc_buffers.py)

# VISUALIZATION: EJI modules, land use (original + grouped), context layers
    if not skip_viz:
        if not indicators_output.exists() or not boundary_path.exists():
            logger.error("Indicator or boundary file missing. Run preprocessing + indicators first.")
        else:
            logger.info("Generating test maps...")
            gdf_local = gpd.read_file(indicators_output)
            boundary = gpd.read_file(boundary_path)
            module_viz_dir = viz_dir / "eji_modules"
            land_use_viz_dir = viz_dir / "land_use"

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

            # Grouped land use categories: all categories + focus categories
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