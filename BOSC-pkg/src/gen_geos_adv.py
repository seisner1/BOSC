#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:14:14 2023

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

def geos_adv(year,config,plot_flag):
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
    ug_adv_key = bosc_keys['ug_adv']
    vg_adv_key = bosc_keys['vg_adv']
    lat_key = bosc_keys['lat']
    lon_key = bosc_keys['lon']
    time_key = bosc_keys['time']


    time_bnd = ['{year}-01-01'.format(year=year),'{year}-12-31'.format(year=year)]

    print(year)


    geos = xr.open_mfdataset(geos_file,parallel=True);
    
    geos = geos.sel(time=slice(time_bnd[0],time_bnd[1]))
    
    
    lon = geos[lon_key]
    lat = geos[lat_key]
    time = geos[time_key]
    ugsg = geos[ug_key].transpose(time_key,lat_key,lon_key)
    vgsg = geos[vg_key].transpose(time_key,lat_key,lon_key)
    
    # ugsg = ugsg.rename({"latitude":"lat","longitude":"lon"})
    # vgsg = vgsg.rename({"latitude":"lat","longitude":"lon"})
    # lon = lon.rename({"longitude":"lon"})
    # lat = lat.rename({"latitude":"lat"})
    
    print("check 1")
    
    
    ############################## Nan sanitizing##############################
    
    
    ugsg_san = xr.where(ugsg.isnull(),0,ugsg)
    vgsg_san = xr.where(vgsg.isnull(),0,vgsg)
    
    
    
    ############################## Equatorial Filtering############################
    
    out_bnds = [-7,7]
    in_bnds = [-5,5]
    
    out_win = 3;
    in_win = 5;
    
    
    Time,Lat,Lon = xr.broadcast(time,lat,lon)
    omega = 7.2921e-5
    a = 6.371e6
    dtr = np.pi/180
    f = 2*omega*np.sin(dtr*Lat)
    dx = a*np.cos(dtr*Lat)*dtr*Lon.differentiate(coord="lon")
    dy = a*dtr*Lat.differentiate(coord="lat")
    
    # print(np.shape(dx))
    
    
    dudx = ugsg_san.differentiate(coord="lon")
    
    # print(dudx)
    
    # print(np.shape(dudx))
    
    dudx = dudx/dx;
    
    dvdy = vgsg_san.differentiate(coord='lat')
    dvdy = dvdy/dy;

    
    
    
    
    ug_adv = ugsg - (1/f)*(ugsg*dudx + vgsg*dvdy)
    vg_adv = vgsg - (1/f)*(ugsg*dudx + vgsg*dvdy)
    
    
    ug_adv = xr.where(np.logical_and(ug_adv.lat<=out_bnds[1],ug_adv.lat>=out_bnds[0]),ugsg,ug_adv)
    vg_adv = xr.where(np.logical_and(vg_adv.lat<=out_bnds[1],vg_adv.lat>=out_bnds[0]),vgsg,vg_adv)
    
    
    ug_adv = xr.where(np.logical_and(ug_adv.lat<=in_bnds[1],ug_adv.lat>=in_bnds[0]),ugsg,ug_adv)
    vg_adv = xr.where(np.logical_and(vg_adv.lat<=in_bnds[1],vg_adv.lat>=in_bnds[0]),vgsg,vg_adv)
    
    
    
    ###############################################################################
    
    
    ug_adv = ug_adv.transpose(time_key,lat_key,lon_key)
    vg_adv = vg_adv.transpose(time_key,lat_key,lon_key)
    
    # print(np.shape(ugsg))
    print("check 2")
    
    speed2 = np.sqrt(ugsg**2 + vgsg**2)
    
    # plt.figure(dpi=1200)
    # speed2.isel(time=180).plot(cmap=plt.cm.jet,vmin=0,vmax=0.6)
    
    
    
    ###############################Outlier Filtering w/ tanh#######################
    
    
    # mean_ugsg = ugsg.mean(dim="time")
    # mean_vgsg = vgsg.mean(dim="time")
    # std_vgsg = vgsg.std(dim="time",skipna=True)
    # std_ugsg = ugsg.std(dim="time",skipna=True)
    
    ug_adv = xr.where(np.abs(ug_adv)>=1,(2/np.sqrt(2))*np.tanh(2*ug_adv/ug_adv.max()),ug_adv)
    vg_adv = xr.where(np.abs(vg_adv)>=1,(2/np.sqrt(2))*np.tanh(2*vg_adv/vg_adv.max()),vg_adv)
    
    
    
    ###################################Plotting and Viz############################
    
    if plot_flag:
        speed = np.sqrt(ug_adv**2 + vg_adv**2)
        
        
        plt.figure(dpi=1200)
        speed.isel(time=180).plot(cmap=plt.cm.jet,vmin=0,vmax=0.6)
        
        plt.figure(dpi=1200)
        speed.mean(dim="time",skipna=False).plot(cmap=plt.cm.jet,vmin=0,vmax=0.6)
        
        plt.figure(dpi=1200)
        speed2.mean(dim="time",skipna=False).plot(cmap=plt.cm.jet,vmin=0,vmax=0.6)
        
        plt.figure(dpi=1200)
        (np.abs(speed-speed2)).isel(time=180).plot(cmap=plt.cm.jet,vmin=0,vmax=0.0)
        
        plt.figure(dpi=1200)
        speed.max(dim="time").plot(cmap=plt.cm.jet,vmin=1,vmax=3)

        plt.figure(dpi=1200)
        speed2.max(dim="time").plot(cmap=plt.cm.jet,vmin=1,vmax=3)
        
        plt.figure(dpi=1200)
        ugsg.max(dim="time").plot(cmap=plt.cm.jet,vmin=1,vmax=3)
        
        plt.figure(dpi=1200)
        vgsg.std(dim="time",skipna=True).plot(cmap=plt.cm.jet,vmin=0,vmax=0.6)
        
        
        plt.figure(dpi=1200)
        speed2.std(dim="time",skipna=True).plot(cmap=plt.cm.jet,vmin=0,vmax=0.6)

    ##########################file writing#########################################
    
    # ug_adv = ug_adv.rename({"lat":"latitude","lon":"longitude"})
    # vg_adv = vg_adv.rename({"lat":"latitude","lon":"longitude"})
    
    geo_adv = geos.assign({ug_adv_key : ug_adv.astype(np.float32)})
    geo_adv = geo_adv.assign({vg_adv_key : vg_adv.astype(np.float32)})
    
    geos.close()
    
    geo_adv.ug_adv.attrs['standard_name'] = 'surface_geostrophic_eastward_seawater_velocity(m s-1)'
    geo_adv.ug_adv.attrs['long_name'] = 'Geostrophic and Advective Zonal Velocity'
    geo_adv.ug_adv.attrs['units'] = 'm s-1'
    geo_adv.vg_adv.attrs['standard_name'] = 'surface_geostrophic_northward_seawater_velocity(m s-1)'
    geo_adv.vg_adv.attrs['long_name'] = 'Geostrophic and Advective Meridional Velocity'
    geo_adv.vg_adv.attrs['units'] = 'm s-1'

    out_file = geos_file
    
    geo_adv.ug_adv.to_netcdf(out_file,mode='a',unlimited_dims=time_key)
    geo_adv.vg_adv.to_netcdf(out_file,mode='a',unlimited_dims=time_key)
    
    print("done")
    
for year in range(syear,fyear):    
    geos_adv(year,config,False)