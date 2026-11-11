import logging
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Setup logging
logger = logging.getLogger(__name__)

MODULE_LABELS = {
    "SVM": "Social Vulnerability",
    "EBM": "Environmental Burden",
    "HVM": "Health Vulnerability",
    "CBM": "Climate Burden",
}

# Fixed colors -> each category looks the same on every map
LAND_USE_COLORS = {
    "Residential": "#f2c14e",
    "Secondary Areas and Third Spaces": "#e66101",   # orange
    "Industrial": "#7b3294",
    "Transportation": "#8c8c8c",
    "Agriculture": "#a6d96a",
    "Other": "#e3e3e3",
    "Unclassified": "#ff00ff",
}
BACKGROUND_COLOR = "#f0f0f0"

# Context layer styles: (keyword in layer name, color, legend label, draw order).
# Checked top to bottom, first match wins ("major_lake" before "lake").
# Draw order: parks first, then lakes on top of parks, streams last.
CONTEXT_STYLES = [
    ("state_park", "#1b7837", "State parks", 0),
    ("county_park", "#7fbf7b", "County parks", 1),
    ("park", "#5aae61", None, 1),
    ("major_lake", "#08519c", "Major lakes", 3),
    ("lake", "#6baed6", "Lakes", 2),
    ("stream", "#4292c6", "Streams", 4),
]


def _save(fig, output_path: Optional[Path]):
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Visualization saved to {output_path}")
    plt.close(fig)


def _pretty(name: str) -> str:
    return name.replace("_", " ").title()


def _context_style(name: str, fallback_index: int) -> dict:
    for keyword, color, label, order in CONTEXT_STYLES:
        if keyword in name:
            return {"color": color, "label": label or _pretty(name), "order": order}
    return {"color": plt.cm.tab10(fallback_index % 10), "label": _pretty(name), "order": len(CONTEXT_STYLES)}


# ---------------------------------------------------------------------------
# EJI modules
# ---------------------------------------------------------------------------
def plot_eji_modules(
    gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    scope: str = "LOCAL",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    cmap: str = "YlOrRd",
):
    """
    2x2 grid with one map per EJI module (RPL_ percentile ranks).
    scope='LOCAL'    -> ranks relative to the catchment (RPL_<MOD>_LOCAL)
    scope='NATIONAL' -> original EJI ranks relative to the US (RPL_<MOD>)
    Same 0-1 color scale in both, so the two figures are directly comparable.
    """
    suffix = "_LOCAL" if scope.upper() == "LOCAL" else ""
    scope_label = "compared within catchment" if suffix else "compared to the US"

    fig, axes = plt.subplots(2, 2, figsize=(12, 13))
    for ax, (mod, label) in zip(axes.flat, MODULE_LABELS.items()):
        col = f"RPL_{mod}{suffix}"
        ax.set_axis_off()
        if col not in gdf.columns:
            ax.set_title(f"{label}\n({col} missing)", fontsize=11)
            continue

        gdf.plot(
            column=col, ax=ax, cmap=cmap, vmin=0, vmax=1,
            edgecolor="white", linewidth=0.3,
            legend=True,
            legend_kwds={"label": "Percentile rank (0 = lowest, 1 = highest)", "shrink": 0.6},
            missing_kwds={"color": "lightgrey", "hatch": "///", "label": "No data"},
        )
        boundary.boundary.plot(ax=ax, color="black", linewidth=0.8)
        ax.set_title(f"{label}\n{col}", fontsize=11)

    fig.suptitle(title or f"EJI modules ({scope_label})", fontsize=14)
    plt.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Land use
# ---------------------------------------------------------------------------
def plot_land_use_overlay(
    tracts: gpd.GeoDataFrame,
    land_use: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    land_use_col: str = "LANDUSE",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
):
    """Original land use categories (unclassified) on top of the census tracts."""
    col = next((c for c in land_use.columns if c.lower() == land_use_col.lower()), None)
    if col is None:
        logger.error(f"Land use column '{land_use_col}' not found. Columns: {list(land_use.columns)}")
        return

    n_cat = land_use[col].nunique()
    if n_cat > 20:
        logger.warning(f"{n_cat} land use categories: colors will repeat (tab20 has 20).")

    fig, ax = plt.subplots(figsize=(12, 13))
    tracts.plot(ax=ax, color="whitesmoke", edgecolor="none")
    land_use.plot(
        ax=ax, column=col, categorical=True, cmap="tab20",
        linewidth=0, alpha=0.85, legend=True,
        legend_kwds={"loc": "upper left", "bbox_to_anchor": (1.01, 1),
                     "fontsize": 7, "title": "Land use", "frameon": False},
    )
    tracts.boundary.plot(ax=ax, color="dimgrey", linewidth=0.4)
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.2)

    ax.set_title(title or f"Land use and census tracts ({len(tracts)} tracts)", fontsize=13)
    ax.set_axis_off()
    _save(fig, output_path)


def plot_land_use_categories(
    land_use: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    tracts: Optional[gpd.GeoDataFrame] = None,
    class_col: str = "LU_CLASS",
    categories: Optional[List[str]] = None,
    colors: Dict[str, str] = LAND_USE_COLORS,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
):
    """
    Grouped land use categories with fixed colors.
    categories=None -> all categories
    categories=[...] -> only these are colored; everything else is drawn light grey for context
    Legend shows each category's share of the total parcel area.
    """
    if class_col not in land_use.columns:
        logger.error(f"'{class_col}' not found. Run the land use classification first.")
        return

    areas = land_use.geometry.area
    total_area = areas.sum()
    in_data = set(land_use[class_col])
    if categories is None:
        categories = [c for c in colors if c in in_data] + sorted(in_data - set(colors))

    fig, ax = plt.subplots(figsize=(12, 13))
    handles = []

    if tracts is not None:
        tracts.plot(ax=ax, color="white", edgecolor="none")

    # Non-selected categories as grey background
    rest = land_use[~land_use[class_col].isin(categories)]
    if not rest.empty:
        rest.plot(ax=ax, color=BACKGROUND_COLOR, linewidth=0)

    for cat in categories:
        mask = land_use[class_col] == cat
        if not mask.any():
            logger.warning(f"No parcels in category '{cat}'.")
            continue
        color = colors.get(cat, "#000000")
        land_use[mask].plot(ax=ax, color=color, linewidth=0)
        share = 100 * areas[mask].sum() / total_area
        handles.append(Patch(facecolor=color, label=f"{cat} ({share:.0f}% of parcel area)"))

    if not rest.empty:
        share = 100 * areas[rest.index].sum() / total_area
        handles.append(Patch(facecolor=BACKGROUND_COLOR, edgecolor="lightgrey",
                             label=f"Other land uses ({share:.0f}%)"))

    if tracts is not None:
        tracts.boundary.plot(ax=ax, color="dimgrey", linewidth=0.3)
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.2)

    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False, title="Land use", fontsize=9)
    ax.set_title(title or "Land use categories", fontsize=13)
    ax.set_axis_off()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Context layers (parks, lakes, streams) on the preprocessing check map
# ---------------------------------------------------------------------------
def plot_context_layers(
    tracts: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    context_layers: Dict[str, gpd.GeoDataFrame],
    land_use: Optional[gpd.GeoDataFrame] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
):
    """
    Same base as check_preprocessing (grey parcels, tract lines, boundary) with every
    context layer on top. Polygons are filled, lines drawn as lines; styles by layer name.
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    handles = []

    if land_use is not None:
        land_use.plot(ax=ax, color="lightgrey", linewidth=0)
        handles.append(Patch(facecolor="lightgrey", label="Land use parcels"))

    styled = [(name, gdf, _context_style(name, i)) for i, (name, gdf) in enumerate(context_layers.items())]
    for name, gdf, style in sorted(styled, key=lambda x: x[2]["order"]):
        if gdf.empty:
            continue
        geom_types = set(gdf.geom_type)
        if geom_types <= {"LineString", "MultiLineString"}:
            gdf.plot(ax=ax, color=style["color"], linewidth=0.6)
            handles.append(Line2D([0], [0], color=style["color"], lw=1.5, label=style["label"]))
        elif geom_types <= {"Point", "MultiPoint"}:
            gdf.plot(ax=ax, color=style["color"], markersize=8)
            handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=style["color"],
                                  markersize=6, label=style["label"]))
        else:
            gdf.plot(ax=ax, color=style["color"], alpha=0.8, edgecolor="none")
            handles.append(Patch(facecolor=style["color"], alpha=0.8, label=style["label"]))
        logger.info(f"Context layer '{name}': {len(gdf)} features ({', '.join(sorted(geom_types))})")

    tracts.boundary.plot(ax=ax, color="dimgrey", linewidth=0.4)
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.5)
    handles += [
        Line2D([0], [0], color="dimgrey", lw=0.8, label="Census tracts"),
        Line2D([0], [0], color="black", lw=1.5, label="Catchment boundary"),
    ]

    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    ax.set_title(title or f"Preprocessing check with context layers ({len(tracts)} tracts)", fontsize=13)
    ax.set_axis_off()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Land use categories + context layers
# ---------------------------------------------------------------------------
def plot_land_use_with_context(
    land_use: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    context_layers: Dict[str, gpd.GeoDataFrame],
    tracts: Optional[gpd.GeoDataFrame] = None,
    class_col: str = "LU_CLASS",
    categories: Optional[List[str]] = None,
    colors: Dict[str, str] = LAND_USE_COLORS,
    context_alpha: float = 0.9,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
):
    """
    Grouped land use categories with all context layers drawn as filled polygons on top.
    Draw order follows CONTEXT_STYLES: parks first, then lakes, then streams (water stays visible).
    categories=None -> all categories; categories=[...] -> only these colored, rest grey.
    Two legends: land use categories and context layers.
    """
    if class_col not in land_use.columns:
        logger.error(f"'{class_col}' not found. Run the land use classification first.")
        return

    areas = land_use.geometry.area
    total_area = areas.sum()
    in_data = set(land_use[class_col])
    if categories is None:
        categories = [c for c in colors if c in in_data] + sorted(in_data - set(colors))

    fig, ax = plt.subplots(figsize=(12, 13))

    # --- Land use base ---
    if tracts is not None:
        tracts.plot(ax=ax, color="white", edgecolor="none")

    lu_handles = []
    rest = land_use[~land_use[class_col].isin(categories)]
    if not rest.empty:
        rest.plot(ax=ax, color=BACKGROUND_COLOR, linewidth=0)

    for cat in categories:
        mask = land_use[class_col] == cat
        if not mask.any():
            continue
        color = colors.get(cat, "#000000")
        land_use[mask].plot(ax=ax, color=color, linewidth=0)
        share = 100 * areas[mask].sum() / total_area
        lu_handles.append(Patch(facecolor=color, label=f"{cat} ({share:.0f}%)"))

    if not rest.empty:
        share = 100 * areas[rest.index].sum() / total_area
        lu_handles.append(Patch(facecolor=BACKGROUND_COLOR, edgecolor="lightgrey",
                                label=f"Other land uses ({share:.0f}%)"))

    # --- Context layers: filled, drawn in CONTEXT_STYLES order (parks -> lakes -> streams) ---
    styled = [(name, gdf, _context_style(name, i)) for i, (name, gdf) in enumerate(context_layers.items())]
    styled.sort(key=lambda x: x[2]["order"])

    ctx_handles = []
    for name, gdf, style in styled:
        if gdf.empty:
            continue
        geom_types = set(gdf.geom_type)
        color, label = style["color"], style["label"]

        if geom_types <= {"LineString", "MultiLineString"}:
            gdf.plot(ax=ax, color=color, linewidth=0.6)
            ctx_handles.append(Line2D([0], [0], color=color, lw=1.5, label=label))
        elif geom_types <= {"Point", "MultiPoint"}:
            gdf.plot(ax=ax, color=color, markersize=8)
            ctx_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                                      markersize=6, label=label))
        else:
            gdf.plot(ax=ax, color=color, alpha=context_alpha, edgecolor="none")
            ctx_handles.append(Patch(facecolor=color, alpha=context_alpha, label=label))

    # --- Boundaries ---
    if tracts is not None:
        tracts.boundary.plot(ax=ax, color="dimgrey", linewidth=0.3)
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.2)
    ctx_handles.append(Line2D([0], [0], color="black", lw=1.5, label="Catchment boundary"))

    # --- Two legends ---
    lu_legend = ax.legend(handles=lu_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                          frameon=False, title="Land use (% of parcel area)", fontsize=9)
    ax.add_artist(lu_legend)
    ax.legend(handles=ctx_handles, loc="lower left", bbox_to_anchor=(1.01, 0),
              frameon=False, title="Context layers", fontsize=9)

    ax.set_title(title or "Land use categories with parks and water features", fontsize=13)
    ax.set_axis_off()
    _save(fig, output_path)