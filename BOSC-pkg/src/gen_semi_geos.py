#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:19:48 2023

@author: seisner
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import netCDF4
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

def semi_geos(year,config):#SG_CALC


############################################################
#       semi_geos component calculation                    #
#       input: altimetry file (netcdf)                     #
#       output: write semi-geos to write out file          #
############################################################

####################### File Import Header ####################################

    vers = config['header']['vers'];
    dir_main = config['input']['dir_main']
    sla_file = config['input']['sla_file']
    dir_out = config['output']['dir_out']
    geos_out = config['output']['geos_out'].format(vers=vers,year=year)


    sla_file = os.path.join(dir_main,sla_file)
    out_file = os.path.join(dir_out,geos_out)
    
    
    #------Keys
    
    sla_keys = config['SLA_file_keys']
    
    sla_key = sla_keys['sla']
    time_key = sla_keys['time']
    lat_key = sla_keys['lat']
    lon_key = sla_keys['lon']
    
    bosc_keys = config['bosc_file_keys']
    
    ug_key = bosc_keys['u_g']
    vg_key = bosc_keys['v_g']
    
    
    print(sla_file)
    print(out_file)
        
###############################################################################

    
    time_bnd = ['{year}-01-01'.format(year=year),'{year}-12-31'.format(year=year)]


    sla_ds = xr.open_mfdataset(sla_file, parallel=True)
    
    sla_ds = sla_ds.sel(time=slice(time_bnd[0],time_bnd[1]))
    
    sla = sla_ds[sla_key];
    time = sla_ds[time_key];
    lat = sla_ds[lat_key];
    lon = sla_ds[lon_key];
    
################################## NaN Sanitizing #############################

    mask_sla = (sla.isnull()).to_numpy()
    
    sla = xr.where(sla.isnull(),np.nan,sla)
    
    nan_msk = xr.where(np.isnan(sla),1,0)
    
    plt.figure(dpi=1200)
    nan_msk.mean(dim='time').plot()
    
    sla = sla.interpolate_na(dim='time')
    
    sla_lon_int = sla.interpolate_na(dim='lon',method='nearest', limit=2)
    sla_lat_int = sla.interpolate_na(dim='lat',method='nearest', limit=2)
    
    sla = 0.5*sla_lon_int + 0.5*sla_lat_int



###############################################################################
    
    Time,Lat,Lon = xr.broadcast(time,lat,lon)
    omega = 7.2921e-5
    a = 6.371e6
    g = 9.80665
    dtr = np.pi/180
    f = 2*omega*np.sin(dtr*Lat)
    beta = (1/a)*2*omega*np.cos(dtr*Lat)
    dx = (a*np.cos(dtr*Lat)*dtr*Lon.differentiate(coord="lon")).to_numpy()
    dy = (a*dtr*Lat.differentiate(coord="lat")).to_numpy()
    
    
    dsladx = sla.differentiate(coord='lon')
    dsladx = dsladx/dx;
    
    dslady = sla.differentiate(coord='lat')
    dslady = dslady/dy;
    
    ddslady = dslady.differentiate(coord='lat')
    ddslady = ddslady/dy;
    
    ddsladxdy = dslady.differentiate(coord='lon')
    ddsladxdy = ddsladxdy/dx;
    
    
    u_geo_oe = -g*dslady/f;
    v_geo_oe = g*dsladx/f;
    
    
    
    n = 3;

    for i in range(0,n):
        ddslady = ddslady.rolling(lon=3,center=True,min_periods=1).mean(skipna=True)
        ddslady = ddslady.rolling(lat=3,center=True,min_periods=1).mean(skipna=True)
        
        ddsladxdy = ddsladxdy.rolling(lon=3,center=True,min_periods=1).mean(skipna=True)
        ddsladxdy = ddsladxdy.rolling(lat=3,center=True,min_periods=1).mean(skipna=True)
        
    u_sg_eq = -g*ddslady/beta;
    v_sg_eq = g*ddsladxdy/beta;
    
    
    amp = 0.7
    waist = 2.2
    gauss_kernel = np.exp(-Lat**2/(waist**2))
    comp_gauss_kernel = 1 - gauss_kernel
    gauss_kernel = amp*gauss_kernel
    
    
    
    u_sg = u_geo_oe*comp_gauss_kernel + u_sg_eq*gauss_kernel
    v_sg = v_geo_oe*comp_gauss_kernel + v_sg_eq*gauss_kernel
    
    u_sg = xr.where(mask_sla,np.nan,u_sg)
    v_sg = xr.where(mask_sla,np.nan,v_sg)
    
    
    
    plt.figure(dpi=1200)
    (np.sqrt(u_sg**2 + v_sg**2)).isel(time=100).plot(cmap=plt.cm.jet,vmin=0,vmax=1)
    
############################ Out File Writing #################################
        


    geos = sla_ds.drop_vars(['sla'])
    geos = geos.assign({ug_key : u_sg.astype(np.float32)})
    geos = geos.assign({vg_key : v_sg.astype(np.float32)})
    geos.ugsg.attrs['standard_name'] = 'surface_geostrophic_eastward_seawater_velocity(m s-1)'
    geos.ugsg.attrs['long_name'] = 'Geostrophic Zonal Velocity'
    geos.ugsg.attrs['units'] = 'm s-1'
    geos.vgsg.attrs['standard_name'] = 'surface_geostrophic_northward_seawater_velocity(m s-1)'
    geos.vgsg.attrs['long_name'] = 'Geostrophic Meridional Velocity'
    geos.vgsg.attrs['units'] = 'm s-1'
    
    out_file = out_file
    
    geos.ugsg.to_netcdf(out_file,mode='a',unlimited_dims="time")
    geos.vgsg.to_netcdf(out_file,mode='a',unlimited_dims="time")
    
    print("done")
    
    
for year in range(syear,fyear):    
    semi_geos(year,config)