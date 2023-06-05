#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:57:31 2023

@author: seisner
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import netCDF4
import scipy
import pandas
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import json
import pathlib
import requests
import os

with open("config.json", "r") as f:
    config = json.load(f)
    
syear = int(config['header']['syear'])
fyear = int(config['header']['fyear'])

def sst_corr(year,config,plot_flag):
#############################Data Opening/Formatting###########################


    vers = config['header']['vers'];
    sst_vers = config['header']['sst_vers']
    dir_main = config['input']['dir_main']
    dir_out = config['output']['dir_out']
    corr_end = config['output']['corr_out'].format(vers=vers,year=year)
    sst = config['input']['sst_file']
    sst_out = config['output']['sst_out'].format(vers=vers,sst_vers=sst_vers,year=year)

    sst_file = os.path.join(dir_main,sst)
    
    corr_end = '_'.join(corr_end.split('_')[0:-2]) + '*.nc'

    sst_out = os.path.join(dir_out,sst_out)
    bosc_file = os.path.join(dir_out,corr_end)
    
    
    #- File Keys ----------------
    
    sst_keys = config['SST_file_keys']
    
    sst_key = sst_keys['sst']
    
    bosc_keys = config['bosc_file_keys']
    
    u_sst20cm_key = bosc_keys['u_sst_20cm']
    v_sst20cm_key = bosc_keys['v_sst_20cm']
    u_sst15m_key = bosc_keys['u_sst_15m']
    v_sst15m_key = bosc_keys['v_sst_15m']
    u_20cm_key = bosc_keys['u_20cm']
    v_20cm_key = bosc_keys['v_20cm']
    u_15m_key = bosc_keys['u_15m']
    v_15m_key = bosc_keys['v_15m']
    lat_key = bosc_keys['lat']
    lon_key = bosc_keys['lon']
    time_key = bosc_keys['time']
    

    sst = xr.open_mfdataset(sst_file, parallel=True);
    bosc = xr.open_mfdataset(bosc_file, parallel=True);
    
    
    time_bnd = ['{year}-01-01'.format(year=year),'{year}-12-31'.format(year=year)]
    
    bosc = bosc.sel(time=slice(time_bnd[0],time_bnd[1]))
    sst = sst.sel(time=slice(time_bnd[0],time_bnd[1]))

    sst = sst[sst_key];
    u_20cm = bosc[u_20cm_key]
    v_20cm = bosc[v_20cm_key]
    u_15m = bosc[u_15m_key]
    v_15m = bosc[v_15m_key]
    time = bosc[time_key]
    lat = bosc[lat_key]
    lon = bosc[lon_key]

    # sst = sst.sel(time=slice(time_bnd[0],time_bnd[1]))
    # u_20cm = u_20cm.sel(time=slice(time_bnd[0],time_bnd[1]))
    # v_20cm = v_20cm.sel(time=slice(time_bnd[0],time_bnd[1]))
    # u_15m = u_15m.sel(time=slice(time_bnd[0],time_bnd[1]))
    # v_15m = v_15m.sel(time=slice(time_bnd[0],time_bnd[1]))
    # time = time.sel(time=slice(time_bnd[0],time_bnd[1]))
    # bosc = bosc.sel(time=slice(time_bnd[0],time_bnd[1]))

    ############################Nan Sanitizing#####################################


    mask = (sst.isnull()).to_numpy()
    mask_bkg = (u_20cm*0!=0).to_numpy()


    sst = xr.where(sst.isnull(),0,sst)
    u_20cm = xr.where(u_20cm.isnull(),0,u_20cm)
    v_20cm = xr.where(v_20cm.isnull(),0,v_20cm)
    u_15m = xr.where(u_15m.isnull(),0,u_15m)
    v_15m = xr.where(v_15m.isnull(),0,v_15m)


    #############################Differentiation###################################


    Time,Lat,Lon = xr.broadcast(time,lat,lon)
    omega = 7.2921e-5
    a = 6.371e6
    dtr = np.pi/180
    f = 2*omega*np.sin(dtr*Lat)
    dx = (a*np.cos(dtr*Lat)*dtr*Lon.differentiate(coord="lon")).to_numpy()
    dy = (a*dtr*Lat.differentiate(coord="lat")).to_numpy()

    dt = 86470;



    dsstdt = sst.differentiate(coord='time',datetime_unit='s').assign_coords(time=time,lat=lat,lon=lon);

    dsstdx = sst.differentiate(coord='lon').assign_coords(time=time,lat=lat,lon=lon)
    dsstdx = dsstdx/dx;

    dsstdy = sst.differentiate(coord='lat').assign_coords(time=time,lat=lat,lon=lon)
    dsstdy = dsstdy/dy;

    norm_grad = np.sqrt(dsstdx**2 + dsstdy**2)

    ##########################Feature-Tracking#####################################
    win = 39;

    Q = dsstdt.rolling(lat=win,lon=win,min_periods=1,center=True).mean()

    print('check 1')



    dsstdx = xr.where(mask,np.nan,dsstdx);
    dsstdy = xr.where(mask,np.nan,dsstdy);
    dsstdt = xr.where(mask,np.nan,dsstdt);


    norm_grad = np.sqrt(dsstdx**2 + dsstdy**2)

    zeta = norm_grad/np.abs(f.to_numpy())


    eps = 100;
    gamma = 12;
    kappa = 1;

    alpha = kappa/(1 + eps*np.exp(-gamma*zeta));



    print('check 2')

    # speed_surf = np.sqrt(u_20cm**2 + v_20cm**2)

    # alpha = xr.where(speed_surf==0,1,alpha)
    
    alpha = xr.where(norm_grad<=1e-8,0,alpha)


    adv_surf = u_20cm*dsstdx + v_20cm*dsstdy
    adv_15m = u_15m*dsstdx + v_15m*dsstdy

    adv_surf = xr.where(mask_bkg,np.nan,adv_surf)
    adv_15m = xr.where(mask_bkg,np.nan,adv_15m)

    

    #-------------------- SST Param ---------------------------------
    
    grad_norm_param = norm_grad.rolling(time=5,center=True).mean()

    grad_norm_param = grad_norm_param.rolling(lon=6,center=True).mean()
    grad_norm_param = grad_norm_param.rolling(lat=6,center=True).mean()
    
    
    
    sst_param = (8e-6)/grad_norm_param;
    


    #---------------------Contains the advective component subtraction

    u_sst_20_oeq = (alpha)*(dsstdx/(norm_grad**2))*(Q - adv_surf - dsstdt);
    v_sst_20_oeq = (alpha)*(dsstdy/(norm_grad**2))*(Q - adv_surf - dsstdt);

    u_sst_15_oeq = (alpha)*(dsstdx/(norm_grad**2))*(Q - adv_15m - dsstdt);
    v_sst_15_oeq = (alpha)*(dsstdy/(norm_grad**2))*(Q - adv_15m - dsstdt);

    ###############################################################################

    u_sst_eq = (alpha)*(dsstdx/(norm_grad**2))*(Q - dsstdt);
    v_sst_eq = (alpha)*(dsstdy/(norm_grad**2))*(Q - dsstdt);
    
    u_sst_eq = u_sst_eq*sst_param;
    v_sst_eq = v_sst_eq*sst_param;


    u_sst_20_oeq = xr.where((u_sst_20_oeq*0!=0 | u_sst_20_oeq.isnull()),u_sst_eq,u_sst_20_oeq)
    v_sst_20_oeq = xr.where((v_sst_20_oeq*0!=0 | v_sst_20_oeq.isnull()),v_sst_eq,v_sst_20_oeq)

    u_sst_15_oeq = xr.where((u_sst_15_oeq*0!=0 | u_sst_15_oeq.isnull()),u_sst_eq,u_sst_15_oeq)
    v_sst_15_oeq = xr.where((v_sst_15_oeq*0!=0 | v_sst_15_oeq.isnull()),v_sst_eq,v_sst_15_oeq)



    sigma = 3.5

    gauss = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(Lat**2)/(2*sigma**2))

    u_sst_20 = (1-gauss)*u_sst_20_oeq + gauss*u_sst_eq
    v_sst_20 = (1-gauss)*v_sst_20_oeq + gauss*v_sst_eq

    u_sst_15 = (1-gauss)*u_sst_15_oeq + gauss*u_sst_eq
    v_sst_15 = (1-gauss)*v_sst_15_oeq + gauss*v_sst_eq
    
    
    # u_sst_20 = (u_sst_20.rolling(time=5,center=True,min_periods=1).mean())
    # v_sst_20 = (v_sst_20.rolling(time=5,center=True,min_periods=1).mean())
    
    # u_sst_15 = (u_sst_15.rolling(time=5,center=True,min_periods=1).mean())
    # v_sst_15 = (v_sst_15.rolling(time=5,center=True,min_periods=1).mean())
    
    # n = 3;

    # for i in range(0,n):
    #     u_sst_20 = u_sst_20.rolling(lon=3,center=True,min_periods=1).mean(skipna=True)
    #     u_sst_20 = u_sst_20.rolling(lat=3,center=True,min_periods=1).mean(skipna=True)
        
    #     v_sst_20 = v_sst_20.rolling(lon=3,center=True,min_periods=1).mean(skipna=True)
    #     v_sst_20 = v_sst_20.rolling(lat=3,center=True,min_periods=1).mean(skipna=True)
        
        
    #     u_sst_15 = u_sst_15.rolling(lon=3,center=True,min_periods=1).mean(skipna=True)
    #     u_sst_15 = u_sst_15.rolling(lat=3,center=True,min_periods=1).mean(skipna=True)
        
    #     v_sst_15 = v_sst_15.rolling(lon=3,center=True,min_periods=1).mean(skipna=True)
    #     v_sst_15 = v_sst_15.rolling(lat=3,center=True,min_periods=1).mean(skipna=True)
    
    
    u_20cm = xr.where(mask_bkg,np.nan,u_20cm)
    v_20cm = xr.where(mask_bkg,np.nan,v_20cm)
    u_15m = xr.where(mask_bkg,np.nan,u_15m)
    v_15m = xr.where(mask_bkg,np.nan,v_15m)
    
    
    print('check 3')
    
    if plot_flag==True:
        
        # plt.figure(dpi=1200)
        # p = (norm_grad).sel(time='{year}-02-20'.format(year=year)).plot(cmap=plt.cm.jet,vmin=0,vmax=8e-4,subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)),transform=ccrs.PlateCarree())
        # p.axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
        # plt.title('Norm grad Test')
        

        # plt.figure(dpi=1200)
        # p = (dsstdt).sel(time='{year}-02-20'.format(year=year)).plot(cmap=plt.cm.jet,vmin=-5e-6,vmax=5e-6,subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)),transform=ccrs.PlateCarree())
        # p.axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
        # plt.title('dsstdt Test')
        
        
        # plt.figure(dpi=1200)
        # p = (Q).sel(time='{year}-02-20'.format(year=year)).plot(cmap=plt.cm.jet,vmin=-5e-6,vmax=5e-6,subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)),transform=ccrs.PlateCarree())
        # p.axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
        # plt.title('Q Test')
        
        # plt.figure(dpi=1200)
        # p = (norm_grad).sel(time='{year}-02-20'.format(year=year)).plot(cmap=plt.cm.jet,vmin=0,vmax=8e-4,subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)),transform=ccrs.PlateCarree())
        # # p.axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
        # plt.title('Norm grad Test 2')

        # plt.figure(dpi=1200)
        # p = (dsstdt).sel(time='{year}-02-20'.format(year=year)).plot(cmap=plt.cm.jet,vmin=-5e-6,vmax=5e-6,subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)),transform=ccrs.PlateCarree())
        # # p.axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
        # plt.title('dsstdt Test 2')
        
        # plt.figure(dpi=1200)
        # p = (alpha).sel(time='{year}-02-20'.format(year=year)).plot(cmap=plt.cm.OrRd,vmin=0,vmax=1,subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)),transform=ccrs.PlateCarree())
        # p.axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
        # plt.title('Alpha Test')

        # plt.figure(dpi=1200)
        # p = (alpha).sel(time='{year}-02-20'.format(year=year)).plot(cmap=plt.cm.OrRd,vmin=0,vmax=1,subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)),transform=ccrs.PlateCarree())
        # # p.axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
        # plt.title('Alpha Test 2')
        
        # plt.figure(dpi=1200)
        # p = (np.sqrt(u_sst_20_oeq**2 + v_sst_20_oeq**2)).sel(time='{year}-02-20'.format(year=year)).plot(cmap=plt.cm.jet,vmin=0,vmax=0.8,subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)),transform=ccrs.PlateCarree())
        # p.axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
        # plt.title('SST OEQ Test')
        
        plt.figure(dpi=1200)
        p = (np.sqrt(u_sst_20**2 + v_sst_20**2)).sel(time='{year}-02-20'.format(year=year)).plot(cmap=plt.cm.jet,vmin=0,vmax=0.8,subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)),transform=ccrs.PlateCarree())
        p.axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
        plt.title('merged SST Test - Surf')

        plt.figure(dpi=1200)
        p = (np.sqrt(u_sst_15**2 + v_sst_15**2)).sel(time='{year}-02-20'.format(year=year)).plot(cmap=plt.cm.jet,vmin=0,vmax=0.8,subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)),transform=ccrs.PlateCarree())
        p.axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
        plt.title('merged SST Test - 15m')
        
        plt.figure(dpi=1200)
        p = (np.sqrt(u_20cm**2 + v_20cm**2)).sel(time='{year}-02-20'.format(year=year)).plot(cmap=plt.cm.jet,vmin=0,vmax=0.8,subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)),transform=ccrs.PlateCarree())
        p.axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
        plt.title('Bkg speed surf')

        plt.figure(dpi=1200)
        p = (np.sqrt(u_15m**2 + v_15m**2)).sel(time='{year}-02-20'.format(year=year)).plot(cmap=plt.cm.jet,vmin=0,vmax=0.8,subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)),transform=ccrs.PlateCarree())
        p.axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
        plt.title('Bkg speed 15m')
    
        # plt.figure(dpi=1200)
        # speed_surf_2.isel(time=120).plot(cmap=plt.cm.jet, vmin=0, vmax=1)
        # plt.title('Speed SST at 20cm, DD 120, 2017')
        
        # plt.figure(dpi=1200)
        # (speed_surf_2 - speed_surf_3).std(dim='time').plot(cmap=plt.cm.jet,vmin=0,vmax=0.6)
        # plt.title('STD Speed SST and background diff, 2017')
        
        # plt.figure(dpi=1200)
        # speed_surf_2.max(dim='time').plot(cmap=plt.cm.jet, vmin=1, vmax=5)
        # plt.title('Speed SST max over 2017')
     



    ############################# Writing out to files ############################
    
    
    u_20cm = u_20cm
    v_20cm = v_20cm
    u_15m = u_15m
    v_15m = v_15m
    
    u_sst_20 = u_sst_20
    v_sst_20 = v_sst_20
    
    u_sst_15 = u_sst_15
    v_sst_15 = v_sst_15
    
    geoek = bosc.assign({u_20cm_key     : u_20cm.astype(np.float32)})
    geoek = geoek.assign({v_20cm_key    : v_20cm.astype(np.float32)})
    geoek = geoek.assign({u_15m_key     : u_15m.astype(np.float32)})
    geoek = geoek.assign({v_15m_key     : v_15m.astype(np.float32)})
    geoek = geoek.assign({u_sst20cm_key : u_sst_20.astype(np.float32)})
    geoek = geoek.assign({v_sst20cm_key : v_sst_20.astype(np.float32)})
    geoek = geoek.assign({u_sst15m_key  : u_sst_15.astype(np.float32)})
    geoek = geoek.assign({v_sst15m_key  : v_sst_15.astype(np.float32)})
    
    bosc.close()
    
    geoek.u_20cm.attrs['standard_name'] = 'surface_eastward_seawater_velocity(m s-1)'
    geoek.u_20cm.attrs['long_name'] = 'Geostrophic, Advective, and Ekman-Stokes Zonal Velocity at 20cm depth'
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
    
    geoek.u_SST20cm.attrs['standard_name'] = 'eastward_seawater_velocity(m s-1)'
    geoek.u_SST20cm.attrs['long_name'] = 'Surface Zonal SST Velocity Correction'
    geoek.u_SST20cm.attrs['units'] = 'm s-1'
    geoek.v_SST20cm.attrs['standard_name'] = 'northward_seawater_velocity(m s-1)'
    geoek.v_SST20cm.attrs['long_name'] = 'Surface Meridional SST Velocity Correction'
    geoek.v_SST20cm.attrs['units'] = 'm s-1'
    geoek.u_SST15m.attrs['standard_name'] = 'eastward_seawater_velocity(m s-1)'
    geoek.u_SST15m.attrs['long_name'] = '15m Zonal SST Velocity Correction'
    geoek.u_SST15m.attrs['units'] = 'm s-1'
    geoek.v_SST15m.attrs['standard_name'] = 'northward_seawater_velocity(m s-1)'
    geoek.v_SST15m.attrs['long_name'] = '15m Meridional SST Velocity Correction'
    geoek.v_SST15m.attrs['units'] = 'm s-1'
    
    
    out_file = sst_out
    
    geoek.u_20cm.to_netcdf(out_file,mode='w',unlimited_dims="time")
    geoek.v_20cm.to_netcdf(out_file,mode='w',unlimited_dims="time")
    geoek.u_15m.to_netcdf(out_file,mode='w',unlimited_dims="time")
    geoek.v_15m.to_netcdf(out_file,mode='w',unlimited_dims="time")
    geoek.u_SST20cm.to_netcdf(out_file,mode='w',unlimited_dims="time")
    geoek.v_SST20cm.to_netcdf(out_file,mode='w',unlimited_dims="time")
    geoek.u_SST15m.to_netcdf(out_file,mode='w',unlimited_dims="time")
    geoek.v_SST15m.to_netcdf(out_file,mode='w',unlimited_dims="time")
    
    print("done")


for year in range(syear,fyear):
    sst_corr(year,config,False)
    
    