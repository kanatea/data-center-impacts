"""Sanity check of the preprocessing outputs, including context layers. Run from the project root."""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "data" / "transformed"
AREA = "rockland_ny"

# (keyword in file name, color, legend label, draw order); first match wins
STYLES = [
    ("state_park", "#1b7837", "State parks", 0),
    ("county_park", "#7fbf7b", "County parks", 1),
    ("major_lake", "#08519c", "Major lakes", 3),
    ("lake", "#6baed6", "Lakes", 2),
    ("stream", "#4292c6", "Streams", 4),
]


def style_for(name):
    for key, color, label, order in STYLES:
        if key in name:
            return color, label, order
    return "#e66101", name.replace("_", " ").title(), 9


tracts = gpd.read_file(T / f"eji_tracts_{AREA}.geojson")
boundary = gpd.read_file(T / f"catchment_boundary_{AREA}.geojson")
land_use = gpd.read_file(T / f"land_use_{AREA}.geojson")
context = {
    p.name.removesuffix(f"_{AREA}.geojson"): gpd.read_file(p)
    for p in sorted((T / "context").glob(f"*_{AREA}.geojson"))
}

# --- Basic checks ---
for name, gdf in [("tracts", tracts), ("boundary", boundary), ("land_use", land_use), *context.items()]:
    print(f"{name:18s} | {len(gdf):>6} features | CRS: {gdf.crs.to_epsg()} | {', '.join(sorted(set(gdf.geom_type)))}")

print(f"\nCatchment area: {boundary.area.sum() / 1e6:,.0f} km2")
pop = tracts["E_TOTPOP"].replace(-999, pd.NA)
print(f"Total population: {pop.sum():,.0f}")

# --- Map 1: original check (parcels + tracts + boundary) ---
fig, ax = plt.subplots(figsize=(8, 10))
land_use.plot(ax=ax, color="lightgrey", linewidth=0)
tracts.boundary.plot(ax=ax, color="steelblue", linewidth=0.5)
boundary.boundary.plot(ax=ax, color="black", linewidth=1.5)
ax.set_title(f"Preprocessing check: {AREA} ({len(tracts)} tracts)")
ax.set_axis_off()
fig.savefig(ROOT / "visualizations" / f"check_preprocessing_{AREA}.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# --- Map 2: same base + context layers ---
fig, ax = plt.subplots(figsize=(10, 12))
land_use.plot(ax=ax, color="lightgrey", linewidth=0)
handles = [Patch(facecolor="lightgrey", label="Land use parcels")]

for name, gdf in sorted(context.items(), key=lambda kv: style_for(kv[0])[2]):
    color, label, _ = style_for(name)
    if set(gdf.geom_type) <= {"LineString", "MultiLineString"}:
        gdf.plot(ax=ax, color=color, linewidth=0.6)
        handles.append(Line2D([0], [0], color=color, lw=1.5, label=label))
    else:
        gdf.plot(ax=ax, color=color, alpha=0.8, edgecolor="none")
        handles.append(Patch(facecolor=color, alpha=0.8, label=label))

tracts.boundary.plot(ax=ax, color="steelblue", linewidth=0.5)
boundary.boundary.plot(ax=ax, color="black", linewidth=1.5)
handles += [Line2D([0], [0], color="steelblue", lw=0.8, label="Census tracts"),
            Line2D([0], [0], color="black", lw=1.5, label="Catchment boundary")]

ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
ax.set_title(f"Preprocessing check with context layers: {AREA}")
ax.set_axis_off()
out = ROOT / "visualizations" / f"check_preprocessing_context_{AREA}.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nMaps saved to {out.parent}")