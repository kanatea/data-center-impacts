"""Quick sanity check of the preprocessing outputs. Run from the project root."""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "data" / "transformed"
AREA = "rockland_ny"

tracts = gpd.read_file(T / f"eji_tracts_{AREA}.geojson")
boundary = gpd.read_file(T / f"catchment_boundary_{AREA}.geojson")
land_use = gpd.read_file(T / f"land_use_{AREA}.geojson")

# --- Basic checks ---
for name, gdf in [("tracts", tracts), ("boundary", boundary), ("land_use", land_use)]:
    print(f"{name:9s} | {len(gdf):>6} features | CRS: {gdf.crs.to_epsg()}")

print(f"\nCatchment area: {boundary.area.sum() / 1e6:,.0f} km²")

# --- Population (EJI E_TOTPOP; Rockland's 2020 census population was ~338,000) ---
pop_col = next(c for c in tracts.columns if c.upper() == "E_TOTPOP")
pop = tracts[pop_col].replace(-999, pd.NA)
print(f"Total population: {pop.sum():,.0f}")
print(f"Tracts with zero population: {(pop == 0).sum()}")

# --- Missing data: -999 counts per estimate column ---
est_cols = [c for c in tracts.columns if c.upper().startswith("E_")]
missing = (tracts[est_cols] == -999).sum()
print(f"\n-999 values in E_ columns ({len(est_cols)} columns checked):")
print(missing[missing > 0].to_string() if missing.any() else "  none")

# --- Land use attributes (needed for step e) ---
print("\nLand use columns:", [c for c in land_use.columns if c != "geometry"])
print(land_use.drop(columns="geometry").head(3).to_string())

# --- Overlay map for a visual check ---
fig, ax = plt.subplots(figsize=(8, 10))
land_use.plot(ax=ax, color="lightgrey", linewidth=0)
tracts.boundary.plot(ax=ax, color="steelblue", linewidth=0.5)
boundary.boundary.plot(ax=ax, color="black", linewidth=1.5)
ax.set_title(f"Preprocessing check: {AREA} ({len(tracts)} tracts)")
ax.set_axis_off()
out = ROOT / "visualizations" / f"check_preprocessing_{AREA}.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nMap saved to {out}")