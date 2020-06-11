# 10/06/20- seasonal averages using up to 10 yrs of data, 
    # output plots: global maps of settling onset time (how many days it takes for particles to start sinking)

import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import numpy as np 
import math
from pylab import *
import cmocean
import os, fnmatch 

rho = '920' # [kgm-3]: density of the plastic 
res = '2x2' # [deg]: resolution of the global release of particles
size = 'r1e-04' # [m]: size of plastic
loc = 'global'

dirread = '/home/dlobelle/Kooi_data/data_output/rho_'+rho+'kgm-3/res_'+res+'/'+size+'/'
dirwrite = '/home/dlobelle/Kooi_figures/rho_'+rho+'kgm-3/res_'+res+'/'+size+'/'

t_set_all = []
z_max_all = []

list_aux = os.listdir(dirread) # complete list of all files in dirread

fig, axs = plt.subplots(2,2, figsize=(20,10)) 

axs = axs.ravel() # required for subplots 

for idx,s in enumerate(['DJF', 'MAM', 'JJA', 'SON']): # the 4 seasons 
    season = '*'+s+'*'
    for file_name in list_aux:
        if fnmatch.fnmatch(file_name, season): # since I don't yet have the output from all runs (for all 10 years)           
            data = Dataset(dirread+file_name,'r')  
            time = data.variables['time'][1,:]/86400
            lons=data.variables['lon'][:] 
            lats=data.variables['lat'][:] 
            depths =data.variables['z']

            time = time - time[0] # if the start month is in March, the time starts on day ±90, but I need start time = 0

            ''' find the first time step particles sink (below 1 m, since that's the initial release depth) '''  
            z_set = []
            z_max = []
            for i in range (depths.shape[0]):
                z0 = np.array(depths[i,:])
                z1 = (np.where(z0 > 1.))
                z2 = z1[0] # find the first index where 
                if not z2.any(): 
                    z_set.append(0)
                else:
                    z_set.append(z2[0])


            t_set = time[z_set]
            t_set[t_set == 0] = np.nan 
            t_set_all.append(np.array(t_set).T)
            
    n = np.nanmean(t_set_all,axis=0)  # computing average settling time for each grid point  (the average over however many yrs I have for that season)
    i_noset = np.argwhere(np.isnan(n)) # determines where particles don't sink after 90 days - to plot in grey below

    cmap = cm.get_cmap('magma_r', 11) 

    m = Basemap(projection='robin',lon_0=-180,resolution='l', ax=axs[idx])
    m.drawparallels(np.array([-60,-30,0,30,60]), labels=[True, False, False, True], linewidth=2.0, size=20,zorder=0)
    m.drawmeridians(np.array([50,150,250,350]), labels=[False, False, False, True], linewidth=2.0, size=20,zorder=0)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey',zorder=3)

    xs, ys = m(lons[:,0], lats[:,0]) # for particles that settle
    xn, yn = m(lons[i_noset,0],lats[i_noset,0]) # for particles that don't settle
    
    m.scatter(xs, ys, marker='.', c=n,cmap = cmap, s = 100,zorder=1) 
    #m.colorbar() # I CAN'T GET THE COLORBAR TO WORK (IT WORKED WITHOUT A LOOP)
    m.scatter(xn,yn, marker= '.', c = 'grey',s = 100,zorder=2)
    axs[idx].title.set_text(s)
    
fig.suptitle('Average settling onset time [days], rho ='+rho+', size ='+size ,size = 20)
