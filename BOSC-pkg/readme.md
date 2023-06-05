################################################
#	Author: Shaun Eisner
#	Project: Blended Ocean Surface Currents
################################################



BOSC is a blended ocean surface currents product that synthesizes
geostrophic currents from NOAA CW/OW Satellite Altimetry, Wind Stress,
Stokes Drift, and SST imagery for feature-tracking. The datasets used for BOSC
are listed in the config.json.


Currently BOSC is generated in years, and can be run with:

./bosc.sh

This runs the entire existing BOSC workflow which are stored as python files in BOSC-pkg/src. In order to modify the chosen names of output files, input data files, start/stop years or main directories,
changes must be made to the config.json file. Currently these changes are done manually.

To Be Added:

- Refs Directory

- Optional Modularity of Workflow

- Sea Ice Correction in Workflow

- Automatic update of main directory

About the Software: The source code software is temporal and spatial resolution agnostic. Temporal resolution is set by default to daily but is ultimately determined by the Sea Level Anomaly temporal resolution. Spatial resolution can be defined independently by placing a CDO grid-mapping file into the grids directory. Note however that choosing a grid of resolution greater than any of the input datasets is simply equivalent to a bilinear interpolation and does not improve observational resolution.
