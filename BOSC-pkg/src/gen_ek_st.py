#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:03:18 2023

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


def ek_st(year,config,plot_flag):
#############################Data Opening/Formatting###########################
    
    vers = config['header']['vers'];
    dir_main = config['input']['dir_main']
    dir_out = config['output']['dir_out']
    geos_end = config['output']['geos_out'].format(vers=vers,year=year)
    beta_15 = config['input']['beta15_file'].format(year=year)
    beta_02 = config['input']['beta02_file'].format(year=year)
    theta_15 = config['input']['theta15_file'].format(year=year)
    theta_02 = config['input']['theta02_file'].format(year=year)
    stress = config['input']['stress_file'].format(year=year)
    
    beta_15 = os.path.join(dir_main,beta_15)
    beta_02 = os.path.join(dir_main,beta_02)
    
    theta_15 = os.path.join(dir_main,theta_15)
    theta_02 = os.path.join(dir_main,theta_02)
    
    stress_file = os.path.join(dir_main,stress)
    
    geos_end_in = '_'.join(geos_end.split('_')[0:-2]) + '*.nc'
    
    geos_file = os.path.join(dir_out,geos_end_in)
    geos_out = os.path.join(dir_out,geos_end)


    #--File Keys------
    
    ekst_keys = config['Ekst_file_keys']
    
    taux_key = ekst_keys['taux']
    tauy_key = ekst_keys['tauy']
    beta_key = ekst_keys['beta']
    theta_key = ekst_keys['theta']
    
    bosc_keys = config['bosc_file_keys']
    
    ug_adv_key = bosc_keys['ug_adv']
    vg_adv_key = bosc_keys['vg_adv']
    u_20cm_key = bosc_keys['u_20cm']
    v_20cm_key = bosc_keys['v_20cm']
    u_15m_key = bosc_keys['u_15m']
    v_15m_key = bosc_keys['v_15m']
    lat_key = bosc_keys['lat']
    lon_key = bosc_keys['lon']
    time_key = bosc_keys['time']


    time_bnd = ['{year}-01-01'.format(year=year),'{year}-12-31'.format(year=year)]
    
    
    print(year)
    print(beta_15)
    print(beta_02)
    print(theta_15)
    print(theta_02)
    print(stress_file)
    print(geos_file)
    print(geos_out)


    geos = xr.open_mfdataset(geos_file, parallel=True);
    beta_15m = xr.open_dataset(beta_15,decode_times=False);
    beta_20cm = xr.open_dataset(beta_02,decode_times=False);
    theta_15m = xr.open_dataset(theta_15,decode_times=False);
    theta_20cm = xr.open_dataset(theta_02,decode_times=False);
    
    stress = xr.open_dataset(stress_file,decode_times=False);
    
    geos = geos.sel(time=slice(time_bnd[0],time_bnd[1]))
    # beta_15m = beta_15m.sel(time=slice(time_bnd[0],time_bnd[1]))
    # beta_20cm = beta_20cm.sel(time=slice(time_bnd[0],time_bnd[1]))
    # theta_15m = theta_15m.sel(time=slice(time_bnd[0],time_bnd[1]))
    # theta_20cm = theta_20cm.sel(time=slice(time_bnd[0],time_bnd[1]))
    # stress = stress.sel(time=slice(time_bnd[0],time_bnd[1]))
    
    
    lon = geos[lon_key]
    lat = geos[lat_key]
    ug_adv = geos[ug_adv_key]
    vg_adv = geos[vg_adv_key]
    
    # print(np.shape(ug_adv))
    # print(stress)
    # print(beta_15m)

    
    # ug_adv = ug_adv.rename({"latitude":"lat","longitude":"lon"})
    # vg_adv = vg_adv.rename({"latitude":"lat","longitude":"lon"})
    # lon = lon.rename({"longitude":"lon"})
    # lat = lat.rename({"latitude":"lat"})
    
    beta_15 = beta_15m[beta_key];
    beta_02 = beta_20cm[beta_key];
    theta_15 = theta_15m[theta_key];
    theta_02 = theta_20cm[theta_key];
    
    taux = stress[taux_key];
    tauy = stress[tauy_key];
    
    print("check 1")
    
    ############################## Nan sanitizing##############################

    beta_02 = xr.where(beta_02.isnull(),0,beta_02);
    beta_15 = xr.where(beta_15.isnull(),0,beta_15);
    theta_02 = xr.where(theta_02.isnull(),0,theta_02);
    theta_15 = xr.where(theta_15.isnull(),0,theta_15);
    
    ############################## Time-Shaping################################
    
    time_len = len(ug_adv)
    
    beta_02 = beta_02.isel(time=slice(0,time_len))
    beta_15 = beta_15.isel(time=slice(0,time_len))
    theta_02 = theta_02.isel(time=slice(0,time_len))
    theta_15 = theta_15.isel(time=slice(0,time_len))
    taux = taux.isel(time=slice(0,time_len))
    tauy = tauy.isel(time=slice(0,time_len))
    
    ############################## Ekman-Stokes Gen############################
    
    # plt.figure(dpi=1200)
    # beta_02.mean(dim='time',skipna=True).plot(cmap=plt.cm.jet,vmin=0,vmax=5)
    
    # plt.figure(dpi=1200)
    # beta_15.mean(dim='time',skipna=True).plot(cmap=plt.cm.jet,vmin=0,vmax=3)
    
    u_geks_20cm = 0*ug_adv;
    v_geks_20cm = 0*vg_adv;
    u_geks_15m = 0*ug_adv;
    v_geks_15m = 0*vg_adv;
    
    
    ug_adv = ug_adv.to_numpy()
    vg_adv = vg_adv.to_numpy()
    
    Tau = np.empty(np.shape(taux),dtype=np.complex128)
    Tau.real = taux
    Tau.imag = tauy
    
    # plt.figure(dpi=1200)
    # (beta_02.sel(lat=0,method='nearest').mean(dim='lon') - beta_02.sel(lat=-80,method='nearest').mean(dim='lon')).plot()
    # plt.title('Beta2 mean values over 2017, NH')
    
    # print(beta_02.mean())
    
##########################Ekman-Stokes Computation#############################

    vel_eks_20cm = beta_02*(Tau)*np.exp(1j*(np.pi/180)*theta_02);
    
    vel_eks_20cm = vel_eks_20cm.to_numpy()
    
    u_eks_20cm = np.real(vel_eks_20cm)
    v_eks_20cm = np.imag(vel_eks_20cm)
    
    
    u_geks_20cm_np = ug_adv + u_eks_20cm
    v_geks_20cm_np = vg_adv + v_eks_20cm
    
    u_geks_20cm[:,:,:] = u_geks_20cm_np
    v_geks_20cm[:,:,:] = v_geks_20cm_np
    
    
    vel_eks_15m = beta_15*(Tau)*np.exp(1j*(np.pi/180)*theta_15);
    
    vel_eks_15m = vel_eks_15m.to_numpy()
    
    u_eks_15m = np.real(vel_eks_15m)
    v_eks_15m = np.imag(vel_eks_15m)
    
    
    u_geks_15m_np = ug_adv + u_eks_15m
    v_geks_15m_np = vg_adv + v_eks_15m
    
    u_geks_15m[:,:,:] = u_geks_15m_np
    v_geks_15m[:,:,:] = v_geks_15m_np
    
    
    ###############################################################################

    print("check 2")
    
    # speed2 = np.sqrt(ug_adv_copy**2 + vg_adv_copy**2)
    
    # plt.figure(dpi=1200)
    # speed2.isel(time=180).plot(cmap=plt.cm.jet,vmin=0,vmax=0.6)
    
    print(np.shape(u_geks_20cm))
    
    
    ###################################Plotting and Viz############################
    
    if plot_flag:
        speed = np.sqrt(u_geks_15m**2 + v_geks_15m**2)
        speed2 = np.sqrt(u_geks_20cm**2 + v_geks_20cm**2)
        
        
        plt.figure(dpi=1200)
        speed.isel(time=180).plot(cmap=plt.cm.jet,vmin=0,vmax=0.6)
        
        plt.figure(dpi=1200)
        speed.mean(dim="time",skipna=False).plot(cmap=plt.cm.jet,vmin=0,vmax=0.6, cbar_kwargs={'label':'m/s'})
        plt.title('Ekman-Stokes Means over {year}, beta2'.format(year=year))
        
        plt.figure(dpi=1200)
        speed2.mean(dim="time",skipna=False).plot(cmap=plt.cm.jet,vmin=0,vmax=0.6)
        
        plt.figure(dpi=1200)
        speed2.max(dim="time").plot(cmap=plt.cm.jet,vmin=-5,vmax=5)
        
        plt.figure(dpi=1200)
        (np.abs(speed-speed2)).isel(time=180).plot(cmap=plt.cm.jet,vmin=0,vmax=0.5)
        
        plt.figure(dpi=1200)
        speed.max(dim="time").plot(cmap=plt.cm.bwr,vmin=-5,vmax=5,cbar_kwargs={'label':'m/s'})
        plt.title('Ekman-Stokes Maximums over {year}, beta2'.format(year=year));

    ##########################file writing#########################################
    
    # u_geks_20cm = u_geks_20cm.rename({"lat":"latitude","lon":"longitude"})
    # v_geks_20cm = v_geks_20cm.rename({"lat":"latitude","lon":"longitude"})
    # u_geks_15m = u_geks_15m.rename({"lat":"latitude","lon":"longitude"})
    # v_geks_15m = v_geks_15m.rename({"lat":"latitude","lon":"longitude"})
    
    geoek = geos.assign({u_20cm_key : u_geks_20cm.astype(np.float32)})
    geoek = geoek.assign({v_20cm_key : v_geks_20cm.astype(np.float32)})
    geoek = geoek.assign({u_15m_key : u_geks_15m.astype(np.float32)})
    geoek = geoek.assign({v_15m_key : v_geks_15m.astype(np.float32)})
    
    geos.close()
    
    geoek.u_20cm.attrs['standard_name'] = 'surface_eastward_seawater_velocity(m s-1)'
    geoek.u_20cm.attrs['long_name'] = 'Geostrophic, Advective and Ekman-Stokes Zonal Velocity at 20cm depth'
    geoek.u_20cm.attrs['units'] = 'm s-1'
    geoek.v_20cm.attrs['standard_name'] = 'surface_northward_seawater_velocity(m s-1)'
    geoek.v_20cm.attrs['long_name'] = 'Geostrophic, Advective and Ekman-Stokes Meridional Velocity at 20cm depth'
    geoek.v_20cm.attrs['units'] = 'm s-1'
    geoek.u_15m.attrs['standard_name'] = 'eastward_seawater_velocity(m s-1)'
    geoek.u_15m.attrs['long_name'] = 'Geostrophic, Advective and Ekman-Stokes Zonal Velocity at 15m depth'
    geoek.u_15m.attrs['units'] = 'm s-1'
    geoek.v_15m.attrs['standard_name'] = 'northward_seawater_velocity(m s-1)'
    geoek.v_15m.attrs['long_name'] = 'Geostrophic, Advective and Ekman-Stokes Meridional Velocity at 15m depth'
    geoek.v_15m.attrs['units'] = 'm s-1'

    out_file = geos_out
    
    geoek.u_20cm.to_netcdf(out_file,mode='a',unlimited_dims="time")
    geoek.v_20cm.to_netcdf(out_file,mode='a',unlimited_dims="time")
    geoek.u_15m.to_netcdf(out_file,mode='a',unlimited_dims="time")
    geoek.v_15m.to_netcdf(out_file,mode='a',unlimited_dims="time")
    
    print("done")
    
for year in range(syear,fyear):    
    ek_st(year,config,False)