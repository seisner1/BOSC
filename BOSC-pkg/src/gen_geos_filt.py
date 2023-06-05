import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import h5py
import scipy
import pandas
import xarray as xr
import json
import pathlib
import requests
import os

with open("config.json", "r") as f:
    config = json.load(f)

syear = int(config['header']['syear'])
fyear = int(config['header']['fyear'])

def filt_geos(year,config,plot_flag):
#############################Data Opening/Formatting###########################
    
    vers = config['header']['vers'];
    dir_main = config['input']['dir_main']
    dir_out = config['output']['dir_out']
    geos_out = config['output']['geos_out'].format(vers=vers,year=year)


    geos_file = os.path.join(dir_out,geos_out)
    
    print(geos_file)
    
    #------ File Keys
    
    bosc_keys = config['bosc_file_keys']
    
    ug_key = bosc_keys['u_g']
    vg_key = bosc_keys['v_g']
    lat_key = bosc_keys['lat']
    lon_key = bosc_keys['lon']
    time_key = bosc_keys['time']
    
    print(year)


    geos = xr.open_dataset(geos_file);
    

    ugsg = geos.ugsg.transpose(time_key,lat_key,lon_key)
    vgsg = geos.vgsg.transpose(time_key,lat_key,lon_key)
    
    # ugsg = ugsg.rename({"latitude":"lat","longitude":"lon"})
    # vgsg = vgsg.rename({"latitude":"lat","longitude":"lon"})
    
    print("check 1")
    
    
    
    #############################Fill Missing Data#################################
    
    speed3 = np.sqrt(ugsg**2 + vgsg**2)
    
    ugsg = ugsg.interpolate_na(dim=time_key,use_coordinate=False,max_gap=5)
    vgsg = vgsg.interpolate_na(dim=time_key,use_coordinate=False,max_gap=5)
    
    
    
    ############################## Equatorial Filtering############################
    
    out_bnds = [-7,7]
    in_bnds = [-5,5]
    
    out_win = 3;
    in_win = 5;
    
    ugsg = xr.where(np.logical_and(ugsg.lat<=out_bnds[1],ugsg.lat>=out_bnds[0]),ugsg.rolling(time=out_win,center=True,min_periods=1).mean(skipna=True),ugsg)
    vgsg = xr.where(np.logical_and(vgsg.lat<=out_bnds[1],vgsg.lat>=out_bnds[0]),vgsg.rolling(time=out_win,center=True,min_periods=1).mean(skipna=True),vgsg)
    
    
    ugsg = xr.where(np.logical_and(ugsg.lat<=in_bnds[1],ugsg.lat>=in_bnds[0]),ugsg.rolling(time=in_win,center=True,min_periods=1).mean(skipna=True),ugsg)
    vgsg = xr.where(np.logical_and(vgsg.lat<=in_bnds[1],vgsg.lat>=in_bnds[0]),vgsg.rolling(time=in_win,center=True,min_periods=1).mean(skipna=True),vgsg)
    
    
    
    ###############################################################################
    
    
    ugsg = ugsg.transpose(time_key,lat_key,lon_key)
    vgsg = vgsg.transpose(time_key,lat_key,lon_key)
    print("check 2")
    
    # speed2 = np.sqrt(ugsg**2 + vgsg**2)
    
    # plt.figure(dpi=1200)
    # speed2.isel(time=180).plot(cmap=plt.cm.jet,vmin=0,vmax=0.6)
    
    
    
    ###############################Outlier Filtering w/ tanh#######################
    
    
    # mean_ugsg = ugsg.mean(dim="time")
    # mean_vgsg = vgsg.mean(dim="time")
    # std_vgsg = vgsg.std(dim="time",skipna=True)
    # std_ugsg = ugsg.std(dim="time",skipna=True)
    
    ugsg = xr.where(np.abs(ugsg)>=1,(2/np.sqrt(2))*np.tanh(2*ugsg/ugsg.max()),ugsg)
    vgsg = xr.where(np.abs(vgsg)>=1,(2/np.sqrt(2))*np.tanh(2*vgsg/vgsg.max()),vgsg)
    
    
    
    ###################################Plotting and Viz############################
    
    if plot_flag:
        
        
        # plt.figure(dpi=1200)
        # speed.sel(lat=slice(-20,20),lon=slice(100,150)).max(dim="time").plot(cmap=plt.cm.jet,vmin=4,vmax=6)
        
        # plt.figure(dpi=1200)
        # speed2.sel(lat=slice(-20,20),lon=slice(100,150)).max(dim="time").plot(cmap=plt.cm.jet,vmin=4,vmax=6)
        
        plt.figure(dpi=1200)
        ugsg.max(dim=time_key).plot(cmap=plt.cm.jet,vmin=1,vmax=3)
        
        plt.figure(dpi=1200)
        vgsg.std(dim=time_key,skipna=True).plot(cmap=plt.cm.jet,vmin=0,vmax=0.6)


    ##########################file writing#########################################
    
    geos_out = geos.drop_vars([ug_key,vg_key])
    
    geos.close()
    
    geos_out = geos_out.assign({ug_key : ugsg.astype(np.float32)})
    geos_out = geos_out.assign({vg_key : vgsg.astype(np.float32)})
    geos_out.ugsg.attrs['standard_name'] = 'surface_geostrophic_eastward_seawater_velocity(m s-1)'
    geos_out.ugsg.attrs['long_name'] = 'Geostrophic Zonal Velocity'
    geos_out.ugsg.attrs['units'] = 'm s-1'
    geos_out.vgsg.attrs['standard_name'] = 'surface_geostrophic_northward_seawater_velocity(m s-1)'
    geos_out.vgsg.attrs['long_name'] = 'Geostrophic Meridional Velocity'
    geos_out.vgsg.attrs['units'] = 'm s-1'

    out_file = geos_file
    
    geos_out.ugsg.to_netcdf(out_file,mode='a',unlimited_dims='time')
    geos_out.vgsg.to_netcdf(out_file,mode='a',unlimited_dims='time')
    
    print("done")
    
for year in range(syear,fyear):    
    filt_geos(year,config,False)