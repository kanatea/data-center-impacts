
import logging
from pathlib import Path
import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

MODULES = ["SVM", "EBM", "CBM", "HVM"]
LOCAL_SUFFIX = "_LOCAL"

def apply_weights(
    area_name: str,
    input_dir: Path,
    output_dir: Path,
    w_svm: float = 0.25,
    w_ebm: float = 0.25,
    w_cbm: float = 0.25,
    w_hvm: float = 0.25,
    overwrite: bool = False,
):
    """
    Calculates a Composite SEVI and classifies tracts into Low, Moderate, and High risk.
    """
    input_file = input_dir / f"eji_local_{area_name}.geojson"
    output_file = output_dir / f"composite_sevi_{area_name}.geojson"
    summary_file = output_dir / f"weight_summary_{area_name}.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if output_file.exists() and not overwrite:
        logger.info(f"Composite SEVI already exists. Skipping.")
        return output_file

    output_dir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(input_file)

    # Normalize Weights
    raw_weights = {"SVM": w_svm, "EBM": w_ebm, "CBM": w_cbm, "HVM": w_hvm}
    total_w = sum(raw_weights.values())
    norm_weights = {k: v / total_w for k, v in raw_weights.items()}
    
    # Calculate Continuous Composite SEVI (0 to 1)
    sevi_cols = [f"RPL_{mod}{LOCAL_SUFFIX}" for mod in MODULES]
    weights_array = [norm_weights[mod] for mod in MODULES]
    # matrix multiplication for efficiency
    gdf["composite_sevi"] = (gdf[sevi_cols].values @ weights_array).round(4)

    # Calculate Categorical Risk (4 Quartiles)
    try:
        # q=4 creates quartiles (25% each)
        gdf["risk_level"] = pd.qcut(
            gdf["composite_sevi"], 
            q=4, 
            labels=["Very Low Risk", "Low-Moderate Risk", "Moderate-High Risk", "High Risk"]
        )
        # Map to 1, 2, 3, 4
        gdf["risk_rank"] = gdf["risk_level"].map({
            "Very Low Risk": 1, 
            "Low-Moderate Risk": 2, 
            "Moderate-High Risk": 3, 
            "High Risk": 4
        })
    except ValueError as e:
        logger.warning(f"Quantile cut failed: {e}. Falling back to fixed thresholds.")
        # Fixed thresholds for 4 levels
        gdf["risk_level"] = gdf["composite_sevi"].apply(
            lambda x: "High Risk" if x > 0.75 else 
                      ("Moderate-High Risk" if x > 0.50 else 
                       ("Low-Moderate Risk" if x > 0.25 else "Very Low Risk"))
        )
        gdf["risk_rank"] = gdf["risk_level"].map({
            "Very Low Risk": 1, "Low-Moderate Risk": 2, "Moderate-High Risk": 3, "High Risk": 4
        })
        
    # Save
    output_file.unlink(missing_ok=True)
    gdf.to_file(output_file, driver="GeoJSON")
    #summary of weights
    pd.DataFrame([{"area": area_name, **norm_weights}]).to_csv(summary_file, index=False)

    logger.info(f"Successfully saved composite SEVI to {output_file}")
    return output_file