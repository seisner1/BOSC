#!/bin/bash


$conf="config.json"


################# INSERT AUTO DIRECTORY ASSIGN HERE #####################


#########################################################################



################ INSERT FILE MAPPING ONTO GRID CHOICE HERE ##############





#########################################################################




################ MAIN ###################################################


python gen_semi_geos.py #Create Semi-Geos Component from SLA

python gen_geos_filt.py #Filter/QC Semi-Geos Component

python gen_geos_adv.py #Include 3 way momentum balance

python gen_ek_st.py #Generate Ekman-Stokes Drift from Wind Stress

python SST_corr.py #Generate SST correction with SST field
