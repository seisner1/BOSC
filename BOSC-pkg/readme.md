################################################
#	Author: Shaun Eisner
#	Project: Blended Ocean Surface Currents
################################################



BOSC is a blended ocean surface currents product that synthesizes
geostrophic currents from NOAA CW/OW Satellite Altimetry, Wind Stress,
Stokes Drift, and SST imagery for feature-tracking. The datasets used for BOSC
are in the /aosc/iceland1/seisner1/bosc folder.


Currently BOSC is generated in years that are processed in months. In order to generate BOSC,
The following files must be run:

1. ./BOSC-generate.sh -version -message -start_year -final_year

BOSC-generate.sh prestages the required datasets for BOSC by running run_prestage_bosc.sh.
BOSC-generate.sh then computes the 1/4 degree gridded geostrophic, Ekman, and Stokes components and outputs them
as monthly files into a temporary directory called bosc-temp. The Geos, Ekman, and Stokes components are computed
by the semi_geos_calc.py file

2. ./run_mean_correct.sh -version -message -year

run_mean_correct.sh corrects the climatological mean of the Geos + Ekman + Stokes components to align with the 
Laurindo et al. (2017) drifter climatology. NOTE: in order to perform the mean correction, the monthly BOSC files must
be remerged into yearly files and a 'means' file must be generated for each year by taking the time average of the BOSC currents for
all years in the run. The means file is located at /aosc/iceland1/seisner1/bosc-temp/products/means

3. /aosc/iceland1/seisner1/bosc-temp/products/prestage/run_prestage_sst.sh -version -start_year -final_year

run_prestage_sst.sh performs the prestaging of the newly mean_corrected files. This includes making necessary copies before SST correction
as well as remapping to the 1/6 degree grid.

4. matlab SST_corr.m (args: year, vers)

This code performs the SST correction by feature-tracking on a given year and version number. The final output is stored in the bosc-temp directory
at 1/6 degree daily resolution.
