import logging
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

# PySAL imports for ESDA
import libpysal
from esda.moran import Moran, Moran_Local

logger = logging.getLogger(__name__)

# EXACT MAPPING based on indicator_summary_rockland_ny.csv
MODULE_VARIABLE_MAP = {
    "SVM": [
        "MINRTY", "POV200", "NOHSDP", "UNEMP", "RENTER", "HOUBDN", 
        "UNINSUR", "NOINT", "AGE65", "AGE17", "DISABL", "LIMENG", "MOBILE", "GROUPQ"
    ],
    "EBM": [
        "OZONE", "PM", "DSLPM", "TOTCR", "NPL", "TRI", "TSD", "RMP", 
        "COAL", "LEAD", "PARK", "HOUAGE", "WLKIND", "RAIL", "ROAD", "AIRPRT", "IMPWTR"
    ],
    "HVM": [
        "ASTHMA", "CANCER", "CHD", "MHLTH", "DIABETES"
    ],
    "CBM": [
        "NEHD", "SMOKE", "BURN", "CFLD", "DRGT", "HRCN", "RFLD", "SWND", "TRND"
    ]
}

def calculate_module_correlations(gdf, area_name, output_dir):
    """
    Calculates 4 separate Spearman correlation matrices (one for each module)
    using the exact indicator codes.
    """
    for module, indicators in MODULE_VARIABLE_MAP.items():
        # 1. Find columns that match the indicator codes
        matched_cols = [c for c in gdf.columns if any(ind in c for ind in indicators)]
        
        if not matched_cols:
            logger.warning(f"No columns found for module {module}. Check column names.")
            continue
            
        # 2. Subset the data
        module_data = gdf[matched_cols]
        
        # --- Keep only numeric columns ---
        # This removes any columns that might have slipped in but contain strings
        module_data = module_data.select_dtypes(include=[np.number])
        
        if module_data.empty:
            logger.warning(f"No numeric columns found for module {module}. Skipping.")
            continue
            
        # 3. Drop NaNs and Compute Spearman Correlation
        module_data = module_data.dropna()
        if module_data.empty:
            logger.warning(f"No valid data for {module} after dropping NaNs.")
            continue
            
        try:
            corr_matrix = module_data.corr(method='spearman')
            out_path = output_dir / f"module_corr_{module.lower()}_{area_name}.csv"
            corr_matrix.to_csv(out_path)
            logger.info(f"Saved {module} module correlations to {out_path}")
        except Exception as e:
            logger.error(f"Failed to compute correlation for {module}: {e}")

def calculate_eji_correlations(gdf, area_name, output_dir):
    """
    Calculates Spearman correlation between the final local ranks (RPL_..._LOCAL).
    """
    modules = ["SVM", "EBM", "CBM", "HVM"]
    cols = [f"RPL_{m}_LOCAL" for m in modules]
    existing_cols = [c for c in cols if c in gdf.columns]
    
    if len(existing_cols) < 2:
        logger.error("Not enough local rank columns to compute EJI-module correlations.")
        return

    data = gdf[existing_cols].dropna()
    corr_matrix = data.corr(method='spearman')
    
    out_path = output_dir / f"eji_corr_{area_name}.csv"
    corr_matrix.to_csv(out_path)
    logger.info(f"Saved EJI correlations to {out_path}")

def run_spatial_analysis(
    gdf: gpd.GeoDataFrame,
    target_col: str,
    area_name: str,
    output_dir: Path,
    contiguity_type: str = "queen",
    k_neighbors: int = 4,
    overwrite: bool = False
):
    """
    Performs Global Moran's I and LISA analysis on a specified column.
    Can be used for individual EJI modules or the composite SEVI.
    """
    if target_col not in gdf.columns:
        logger.error(f"Column {target_col} not found in GeoDataFrame. Skipping.")
        return None

    # Extract values for the specific analysis
    y = gdf[target_col].values
    
    # Handle potential NaNs in the target column to prevent PySAL from crashing
    if np.isnan(y).any():
        logger.warning(f"NaNs found in {target_col}. Dropping rows with NaNs for this analysis.")
        gdf_clean = gdf.dropna(subset=[target_col])
        y = gdf_clean[target_col].values
    else:
        gdf_clean = gdf.copy()

    # 1. Flexible Spatial Weights
    if contiguity_type == "queen":
        w = libpysal.weights.Queen.from_dataframe(gdf_clean)
    elif contiguity_type == "rook":
        w = libpysal.weights.Rook.from_dataframe(gdf_clean)
    elif contiguity_type == "k_nearest":
        w = libpysal.weights.KNN.from_dataframe(gdf_clean, k=k_neighbors)
    else:
        raise ValueError("contiguity_type must be 'queen', 'rook', or 'k_nearest'")
    
    w.transform = 'r' # Row-standardized

    # 2. Global Moran's I
    mi = Moran(y, w)
    
    # 3. LISA (Local Moran's I)
    li = Moran_Local(y, w)
    lisa_labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    
    # We create a copy of the cleaned GDF to store results for this specific variable
    result_gdf = gdf_clean.copy()
    result_gdf["lisa_cluster"] = [lisa_labels[val] for val in li.q]
    result_gdf["lisa_p_value"] = li.p_sim
    result_gdf["lisa_sig"] = result_gdf["lisa_p_value"] < 0.05
    result_gdf.loc[~result_gdf["lisa_sig"], "lisa_cluster"] = "NS"

    # 4. Save Results
    # Filename is now dynamic: e.g., lisa_hotspots_RPL_SVM_LOCAL_rockland_ny.geojson
    output_file = output_dir / f"lisa_hotspots_{target_col}_{area_name}.geojson"
    
    if output_file.exists() and not overwrite:
        logger.info(f"LISA output for {target_col} already exists. Skipping.")
        return output_file

    result_gdf.to_file(output_file, driver="GeoJSON")
    
    # 5. Save a variable-specific summary file
    summary_path = output_dir / f"spatial_stats_{target_col}_{area_name}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Variable: {target_col}\nArea: {area_name}\n")
        f.write(f"Contiguity: {contiguity_type}\n")
        f.write(f"Global Moran's I: {mi.I:.4f}\np-value: {mi.p_sim:.4f}\nZ-score: {mi.z_sim:.4f}\n")

    logger.info(f"LISA Hotspots for {target_col} saved to {output_file}")
    return output_file