# GIS-Based Tool to Understand Data Center Impacts on Community
Thesis Project by Kana Tateishi, "Contributing to Community Knowledge on Potential Data Center Impacts Using Spatial Decision Support Systems: A New York Case Study"

## Project Structure

```
dc_impacts/
├── data/
│   ├── raw/                                 # untouched downloads
|   |   ├── context/                         # additional data for spatial context - all shapefiles
|   |   ├── Data_Center_Information/         # csv files - point location data
|   |   ├── EJI_2024_United_States.gdb/      # geodatabase file, county geometry is taken from this
|   |   └── Rockland_Land_Use_2021/          # land use data - shapefiles
│   ├── transformed/                         # geometric operations only, transforms to geojson files
|   |   └── context/
│   └── cleaned/                             # attribute cleaning
│       ├── SEVI/                            # Reconfigured layer for SEVI
│       ├── overlays/                        # Reconfigured layer for land use and data center buffer overlays
│       └── spatial_analysis/                # correlation analyses and LISA results
├── src/dc_impacts/
│   ├── __init__.py
│   ├── __main__.py           
│   ├── a_preprocessing.py                   # filter and clip study area + reproject geometry + geojson
│   ├── b_sevi.py                            # clean/normalize EJI SEVI attributes   
│   ├── c_overlays.py                        # clean data center + land use attributes   
│   ├── d_apply_weights.py 	                 # MCDA weighting + aggregation
│   ├── e_spatial_analysis.py                # Moran's I / LISA on the individual indices and overall composite index
│   ├── f_dc_buffers.py                      # build data center impact buffers
│   ├── g_land_use.py                        # land use percentage inside the buffer
│   └── visualization.py   
├── visualizations/    
│   ├── eji_modules/                         # descriptive, visualize all 4 modules
│   ├── land_use/                            # descriptive, visualize land use layer
│   ├── spatial_analysis_viz/                # LISA hotspot maps
│   └── sevi_map/   		    
├── scripts/
│   └── check_outputs.py   
├── pyproject.toml
└── README.md

```
