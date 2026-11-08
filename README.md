# GIS-Based Tool to Understand Data Center Impacts on Community
Thesis Project by Kana Tateishi, "Contributing to Community Knowledge on Potential Data Center Impacts Using Spatial Decision Support Systems: A New York Case Study"

## Project Structure

```
dc-impact/
├── data/
│   ├── raw/                  # untouched downloads
│   ├── transformed/          # geometric operations only
│   └── cleaned/              # attribute cleaning
│       ├── processed/         # Reconfigured layer for SEVI
│       └── spatial_analysis/  # Moran's I or something like this
├── visualizations/           
│   ├── spatial_analysis_viz/          
│   └── sevi_map/   		  
├── src/dc-impact/
│   ├── __init__.py
│   ├── __main__.py           
│   ├── a_preprocessing.py      # reproject + filter to Rockland
│   ├── b_indicators.py         # build/normalize SEVI indicators   
│   ├── c_apply_weights.py 	   # MCDA weighting + aggregation
│   ├── d_spatial_analysis.py   # Moran's I / LISA on the composite index 
│   ├── e_dc_buffers.py         # data center impact buffers
│   └── visualization.py      
├── pyproject.toml
└── README.md

```
