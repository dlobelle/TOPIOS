import xarray as xr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np 
from glob import glob
import os, fnmatch 
import pickle

dirread_bgc = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA_BGC/ORCA0083-N006/means/' 
dirread = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/means/'
dirwrite = '/home/dlobelle/Kooi_data/non_parcels_output/'
dirfigs = '/home/dlobelle/Kooi_figures/non_parcels_output/'

file_list_bgc = os.listdir(dirread_bgc)
file_list_ts = os.listdir(dirread)

year =['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010']
ppfiles_or=[]
tsfiles_or = []
for s in ['MAM','JJA','SON']:
    print(s)
    if s == 'MAM':
        months = ['03','04','05'] 
    elif s == 'JJA':
        months = ['06','07','08']         
    elif s == 'SON':
        months = ['09','10','11']        
    for yr in year:
        for mon in months:        
            pattern_pp = '*'+yr+mon+'*D.nc'
            pattern_ts = '*'+yr+mon+'*T.nc'
            for file_name in file_list_bgc:
                if fnmatch.fnmatch(file_name, pattern_pp):
                    ppfiles_or = np.concatenate((ppfiles_or,np.array(file_name)),axis=None)
            for file_name in file_list_ts:       
                if fnmatch.fnmatch(file_name, pattern_ts):
                    tsfiles_or = np.concatenate((tsfiles_or,np.array(file_name)),axis=None)

    for id_,v in enumerate(ppfiles_or):
        print(v)
        ppfiles = xr.open_dataset(dirread_bgc+v)     
        if id_ == 0:
            euph_z = np.array(ppfiles.variables['MED_XZE'])
            prd = np.array(ppfiles.variables['ML_PRD'])
            prn = np.array(ppfiles.variables['ML_PRN'])
        else:
            euph_z = np.vstack((np.array(euph_z),ppfiles.variables['MED_XZE']))
            prd = np.vstack((np.array(prd),ppfiles.variables['ML_PRD']))
            prn = np.vstack((np.array(prn),ppfiles.variables['ML_PRN']))
    for id_,v in enumerate(tsfiles_or):
        print(v)
        tsfiles = xr.open_dataset(dirread+v)     
        if id_ == 0:
            mld = np.array(tsfiles.variables['mldr10_1'])
        else:
            mld = np.vstack((np.array(mld),tsfiles.variables['mldr10_1']))
            
    if s == 'MAM':
        MAM_euph_z_mean = np.nanmean(euph_z,axis=0)
        MAM_PP_mean = np.nanmean(prd+prn,axis=0)
        MAM_MLD_mean = np.nanmean(mld,axis=0)
        with open(dirwrite+'MAM_av_phys_params.pickle', 'wb') as f:
            pickle.dump([MAM_euph_z_mean,MAM_PP_mean,MAM_MLD_mean], f)
        
        fig, axs = plt.subplots(1, 3, figsize=(20,4))
        im1 = axs[0].pcolormesh(MAM_euph_z_mean[0:2700,:],cmap = 'jet')
        fig.colorbar(im1, ax=axs[0])
        axs[0].set_title(s+' Euphotic Layer Depth [m]')
        im2 = axs[1].pcolormesh(MAM_PP_mean[0:2700,:],vmin= 0, vmax = 25,cmap = 'jet')
        fig.colorbar(im2, ax=axs[1])
        axs[1].set_title(s+' Primary productivity [mmolN/m2/d]')
        im3 = axs[2].pcolormesh(MAM_MLD_mean[0:2700,:],vmin = 0, vmax = 200,cmap = 'jet')
        fig.colorbar(im3, ax=axs[2])
        axs[2].set_title(s+' Mixed Layer Depth [m]')
        
        plt.savefig(dirfigs+s+'_average_MLD_PP_MLD.pdf', format='pdf')      
        
    elif s == 'JJA':
        JJA_euph_z_mean = np.nanmean(euph_z,axis=0)
        JJA_PP_mean = np.nanmean(prd+prn,axis=0)
        JJA_MLD_mean = np.nanmean(mld,axis=0)
        with open(dirwrite+'JJA_av_phys_params.pickle', 'wb') as f:
            pickle.dump([JJA_euph_z_mean,JJA_PP_mean,JJA_MLD_mean], f)
            
        fig, axs = plt.subplots(1, 3, figsize=(20,4))
        im1 = axs[0].pcolormesh(JJA_euph_z_mean[0:2700,:],cmap = 'jet')
        fig.colorbar(im1, ax=axs[0])
        axs[0].set_title(s+' Euphotic Layer Depth [m]')
        im2 = axs[1].pcolormesh(JJA_PP_mean[0:2700,:],vmin= 0, vmax = 25,cmap = 'jet')
        fig.colorbar(im2, ax=axs[1])
        axs[1].set_title(s+' Primary productivity [mmolN/m2/d]')
        im3 = axs[2].pcolormesh(JJA_MLD_mean[0:2700,:],vmin = 0, vmax = 200,cmap = 'jet')
        fig.colorbar(im3, ax=axs[2])
        axs[2].set_title(s+' Mixed Layer Depth [m]')
        
        plt.savefig(dirfigs+s+'_average_MLD_PP_MLD.pdf', format='pdf')
    elif s == 'SON':
        SON_euph_z_mean = np.nanmean(euph_z,axis=0)
        SON_PP_mean = np.nanmean(prd+prn,axis=0)
        SON_MLD_mean = np.nanmean(mld,axis=0)        
        with open(dirwrite+'SON_av_phys_params.pickle', 'wb') as f:
            pickle.dump([SON_euph_z_mean,SON_PP_mean,SON_MLD_mean], f)
            
        fig, axs = plt.subplots(1, 3, figsize=(20,4))
        im1 = axs[0].pcolormesh(SON_euph_z_mean[0:2700,:],cmap = 'jet')
        fig.colorbar(im1, ax=axs[0])
        axs[0].set_title(s+' Euphotic Layer Depth [m]')
        im2 = axs[1].pcolormesh(SON_PP_mean[0:2700,:],vmin= 0, vmax = 25,cmap = 'jet')
        fig.colorbar(im2, ax=axs[1])
        axs[1].set_title(s+' Primary productivity [mmolN/m2/d]')
        im3 = axs[2].pcolormesh(SON_MLD_mean[0:2700,:],vmin = 0, vmax = 200,cmap = 'jet')
        fig.colorbar(im3, ax=axs[2])
        axs[2].set_title(s+' Mixed Layer Depth [m]')
        
        plt.savefig(dirfigs+s+'_average_MLD_PP_MLD.pdf', format='pdf')
    