# 23/09/20- Modifying global_Kooi+NEMO_NPacific_3D.py (adding a lot from Kooi+NEMO_NPacific_3D.py), to release particles just in N Pacific for 3 years to see if they ever sink

from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4_3D, AdvectionRK4, ErrorCode, ParticleFile, Variable, Field, NestedField, VectorField, timer 
from parcels.kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta as delta
from datetime import  datetime
import numpy as np
import math
from glob import glob
import os
import xarray as xr
import sys
import time as timelib
import matplotlib.pyplot as plt
import warnings
import pickle                                                      
import matplotlib.ticker as mtick
import pandas as pd 
import operator
from numpy import *
import scipy.linalg
import math as math
from argparse import ArgumentParser
warnings.filterwarnings("ignore")

# CHOOSE
region = 'NAtl_SPG' # 'NPac_SPG' 'NPac_STG'
sizes = 'allsizes' #1to10mm

#------ Fieldset grid is 20x30 deg in North Pacific ------
if region == 'NPac_SPG':
    minlat = 40 
    maxlat = 60
    minlon = -160 
    maxlon = -130
elif region == 'NPac_STG':
    minlat = 10 
    maxlat = 40
    minlon = -160 
    maxlon = -130
elif region == 'NAtl_SPG':
    minlat = 40 
    maxlat = 60
    minlon = -40 
    maxlon = -10
     

#------ Release particles on a 10x10 deg grid ------
if region == 'NPac_SPG':
    lat_release0 = np.tile(np.linspace(45,53,5),[5,1]) 
    lon_release = np.tile(np.linspace(-140,-148,5),[5,1])
elif region == 'NPac_STG':
    lat_release0 = np.tile(np.linspace(20,28,5),[5,1]) 
    lon_release = np.tile(np.linspace(-140,-148,5),[5,1])
elif region == 'NAtl_SPG':
    lat_release0 = np.tile(np.linspace(45,53,5),[5,1]) 
    lon_release = np.tile(np.linspace(-20,-28,5),[5,1])

lat_release = lat_release0.T 
z_release = np.tile(0.6,[5,5])
#------ Choose ------:
simdays = 30 #20
secsdt = 30 #60 
hrsoutdt = 6 #2

"""functions and kernels"""

def Kooi(particle,fieldset,time):  
    """
    Kernel to compute the vertical velocity (Vs) of particles due to changes in ambient algal concentrations, growth and death of attached algae based on Kooi et al. 2017 model 
    """
    
    #------ Nitrogen to cell ratios for ambient algal concentrations ('aa') and algal growth ('mu_aa') from NEMO output (no longer using N:C:AA (Redfield ratio), directly N:AA from Menden-Deuer and Lessard 2000)     
    min_N2cell = 2656.0e-09 #[mgN cell-1] (from Menden-Deuer and Lessard 2000)
    max_N2cell = 11.0e-09   #[mgN cell-1] 
    med_N2cell = 356.04e-09 #[mgN cell-1] median value is used below (as done in Kooi et al. 2017)
      
    #------ Ambient algal concentration from MEDUSA's non-diatom + diatom phytoplankton 
    n0 = particle.nd_phy+particle.d_phy # [mmol N m-3] in MEDUSA
    n = n0*14.007                       # conversion from [mmol N m-3] to [mg N m-3] (atomic weight of 1 mol of N = 14.007 g)   
    n2 = n/med_N2cell                   # conversion from [mg N m-3] to [no. m-3]
    
    if n2<0.: 
        aa = 0.
    else:
        aa = n2                        # [no m-3] to compare to Kooi model    
    
    #------ Primary productivity (algal growth) from MEDUSA TPP3 (no longer condition of only above euphotic zone, since not much diff in results)
    tpp0 = particle.tpp3              # [mmol N m-3 d-1]
    mu_n0 = tpp0*14.007               # conversion from [mmol N m-3 d-1] to [mg N m-3 d-1] (atomic weight of 1 mol of N = 14.007 g) 
    mu_n = mu_n0/med_N2cell           # conversion from [mg N m-3 d-1] to [no. m-3 d-1]
    mu_n2 = mu_n/aa                   # conversion from [no. m-3 d-1] to [d-1]
    
    if mu_n2<0.:
        mu_aa = 0.
    else:
        mu_aa = mu_n2/86400.          # conversion from d-1 to s-1
    
    #------ Profiles from MEDUSA or Kooi theoretical profiles -----
    z = particle.depth           # [m]
    t = particle.temp            # [oC]
    sw_visc = particle.sw_visc   # [kg m-1 s-1]
    kin_visc = particle.kin_visc # [m2 s-1]
    rho_sw = particle.density    # [kg m-3]       
    a = particle.a               # [no. m-2 s-1]
    vs = particle.vs             # [m s-1]   

    #------ Constants and algal properties -----
    g = 7.32e10/(86400.**2.)    # gravitational acceleration (m d-2), now [s-2]
    k = 1.0306E-13/(86400.**2.) # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
    rho_bf = 1388.              # density of biofilm ([g m-3]
    v_a = 2.0E-16               # Volume of 1 algal cell [m-3]
    m_a = 0.39/86400.           # mortality rate, now [s-1]
    r20 = 0.1/86400.            # respiration rate, now [s-1] 
    q10 = 2.                    # temperature coefficient respiration [-]
    gamma = 1.728E5/86400.      # shear [d-1], now [s-1]
    
    #------ Volumes -----
    v_pl = (4./3.)*math.pi*particle.r_pl**3.             # volume of plastic [m3]
    theta_pl = 4.*math.pi*particle.r_pl**2.              # surface area of plastic particle [m2]
    r_a = ((3./4.)*(v_a/math.pi))**(1./3.)      # radius of algae [m]
    
    v_bf = (v_a*a)*theta_pl                           # volume of biofilm [m3]
    v_tot = v_bf + v_pl                               # volume of total [m3]
    t_bf = ((v_tot*(3./(4.*math.pi)))**(1./3.))-particle.r_pl  # biofilm thickness [m] 
    
    #------ Diffusivity -----
    r_tot = particle.r_pl + t_bf                               # total radius [m]
    rho_tot = (particle.r_pl**3. * particle.rho_pl + ((particle.r_pl + t_bf)**3. - particle.r_pl**3.)*rho_bf)/(particle.r_pl + t_bf)**3. # total density [kg m-3]
    theta_tot = 4.*math.pi*r_tot**2.                          # surface area of total [m2]
    d_pl = k * (t + 273.16)/(6. * math.pi * sw_visc * r_tot)  # diffusivity of plastic particle [m2 s-1]
    d_a = k * (t + 273.16)/(6. * math.pi * sw_visc * r_a)     # diffusivity of algal cells [m2 s-1] 
    
    #------ Encounter rates -----
    beta_abrown = 4.*math.pi*(d_pl + d_a)*(r_tot + r_a)       # Brownian motion [m3 s-1] 
    beta_ashear = 1.3*gamma*((r_tot + r_a)**3.)               # advective shear [m3 s-1]
    beta_aset = (1./2.)*math.pi*r_tot**2. * abs(vs)           # differential settling [m3 s-1]
    beta_a = beta_abrown + beta_ashear + beta_aset            # collision rate [m3 s-1]
    
    #------ Attached algal growth (Eq. 11 in Kooi et al. 2017) -----
    a_coll = (beta_a*aa)/theta_pl
    a_growth = mu_aa*a
    a_mort = m_a*a
    a_resp = (q10**((t-20.)/10.))*r20*a     
    
    particle.a += (a_coll + a_growth - a_mort - a_resp) * particle.dt

    dn = 2. * (r_tot)                             # equivalent spherical diameter [m]
    delta_rho = (rho_tot - rho_sw)/rho_sw         # normalised difference in density between total plastic+bf and seawater[-]        
    dstar = ((rho_tot - rho_sw) * g * dn**3.)/(rho_sw * kin_visc**2.) # [-]
    
        
    if dstar > 5e9:
        w = 1000.
    elif dstar <0.05:
        w = (dstar**2.) *1.71E-4
    else:
        w = 10.**(-3.76715 + (1.92944*math.log10(dstar)) - (0.09815*math.log10(dstar)**2.) - (0.00575*math.log10(dstar)**3.) + (0.00056*math.log10(dstar)**4.))
    
    #------ Settling of particle -----
#     if z >= 4000.: 
#         vs = 0 
#     elif z<=0.6 and delta_rho < 0:# for rising particles above 0.6 m (negative delta_rho = rho_tot < rho_sw)
#         vs = 0
#         particle.depth = 0.6
    if delta_rho > 0: # sinks 
        vs = (g * kin_visc * w * delta_rho)**(1./3.)
    else: #rises 
        a_del_rho = delta_rho*-1.
        vs = -1.*(g * kin_visc * w * a_del_rho)**(1./3.)  # m s-1
    
    particle.vs_init = vs
    
    z0 = z + vs * particle.dt 
    if z0 <=0.6 or z0 >= 4000.: # NEMO's 'surface depth'
        vs = 0
        particle.depth = 0.6
    else:          
        particle.depth += vs * particle.dt 

    particle.vs = vs
    particle.rho_tot = rho_tot
    particle.r_tot = r_tot
    particle.dstar = dstar
    particle.t_bf = t_bf
    particle.beta_a = beta_a
    particle.beta_abrown = beta_abrown
    particle.beta_ashear = beta_ashear
    particle.beta_aset = beta_aset
    particle.delta_rho = delta_rho
    
def DeleteParticle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted at lon = '+str(particle.lon)+', lat ='+str(particle.lat)+', depth ='+str(particle.depth)) #print(particle.lon, particle.lat, particle.depth)
    particle.delete() 
    
def getclosest_ij(lats,lons,latpt,lonpt):     
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_sq = (lats-latpt)**2 + (lons-lonpt)**2                 # find squared distance of every point on grid
    minindex_flattened = dist_sq.argmin()                       # 1D index of minimum dist_sq element
    return np.unravel_index(minindex_flattened, lats.shape)     # Get 2D index for latvals and lonvals arrays from 1D index

def periodicBC(particle, fieldset, time):
    if particle.lon < 0.:
        particle.lon += 360.
    elif particle.lon >= 360.:
        particle.lon -= 360.
        
def kernel_kukulka_mixing(particle, fieldset, time):  
    """
    :Kernel that randomly distributes particles along the vertical according to an expovariate distribution.
    :Parameterization according to Kukulka et al. (2012): The effect of wind mixing on the vertical distribution of buoyant plastic debris
    :Comment on dimensions: tau needs to be in Pa
    """
    stress =  particle.tau #fieldset.tau[particle.time,particle.depth,particle.lat,particle.lon]
    A0=0.31 * math.pow(stress,1.5)
    l=particle.vs/A0 #fieldset.wrise/A0
    d=random.expovariate(l) #Kukulka formula. Used depths of [0 ... 120.] m
    if d>120.:
        particle.depth=120.
    else:
        particle.depth=d
        
def Profiles(particle, fieldset, time):  
    particle.temp = fieldset.cons_temperature[time, particle.depth,particle.lat,particle.lon]  
    particle.d_phy= fieldset.d_phy[time, particle.depth,particle.lat,particle.lon]  
    particle.nd_phy= fieldset.nd_phy[time, particle.depth,particle.lat,particle.lon] 
    particle.tpp3 = fieldset.tpp3[time,particle.depth,particle.lat,particle.lon]
    particle.euph_z = fieldset.euph_z[time,particle.depth,particle.lat,particle.lon]
    particle.kin_visc = fieldset.KV[time,particle.depth,particle.lat,particle.lon] 
    particle.sw_visc = fieldset.SV[time,particle.depth,particle.lat,particle.lon] 
    particle.w = fieldset.W[time,particle.depth,particle.lat,particle.lon]
    particle.tau = fieldset.tau[time,particle.depth,particle.lat,particle.lon]
    
""" Defining the particle class """

class plastic_particle(ScipyParticle): #JITParticle): #
    u = Variable('u', dtype=np.float32,to_write=True)
    v = Variable('v', dtype=np.float32,to_write=True)
    w = Variable('w', dtype=np.float32,to_write=True)
    tau = Variable('tau', dtype=np.float32,to_write=False)
    temp = Variable('temp',dtype=np.float32,to_write=False)
    density = Variable('density',dtype=np.float32,to_write=True)
    tpp3 = Variable('tpp3',dtype=np.float32,to_write=False)
    euph_z = Variable('euph_z',dtype=np.float32,to_write=False)
    d_phy = Variable('d_phy',dtype=np.float32,to_write=False)
    nd_phy = Variable('nd_phy',dtype=np.float32,to_write=False)    
    kin_visc = Variable('kin_visc',dtype=np.float32,to_write=False)
    sw_visc = Variable('sw_visc',dtype=np.float32,to_write=False)    
    a = Variable('a',dtype=np.float32,to_write=True)
    vs = Variable('vs',dtype=np.float32,to_write=True) 
    rho_tot = Variable('rho_tot',dtype=np.float32,to_write=True) 
    r_tot = Variable('r_tot',dtype=np.float32,to_write=True)
    dstar = Variable('dstar',dtype=np.float32,to_write=True)
    t_bf = Variable('t_bf',dtype=np.float32,to_write=True)
    beta_a = Variable('beta_a',dtype=np.float32,to_write=True)
    beta_aset = Variable('beta_aset',dtype=np.float32,to_write=True)
    beta_ashear = Variable('beta_ashear',dtype=np.float32,to_write=True)
    beta_abrown = Variable('beta_abrown',dtype=np.float32,to_write=True)
    delta_rho = Variable('delta_rho',dtype=np.float32,to_write=True)
    vs_init = Variable('vs_init',dtype=np.float32,to_write=True)
    r_pl = Variable('r_pl',dtype=np.float32,to_write='once')   
    rho_pl = Variable('rho_pl',dtype=np.float32,to_write='once')   

    
""" Defining the fieldset""" 

dirread = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/means/'
dirread_bgc = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA_BGC/ORCA0083-N006/means/'  
dirread_mesh = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/domain/'  

res = '2x2' 
mon = '03'
yr1 = '2004'

ufiles = sorted(glob(dirread+'ORCA0083-N06_'+yr1+'*d05U.nc'))
vfiles = sorted(glob(dirread+'ORCA0083-N06_'+yr1+'*d05V.nc'))
wfiles = sorted(glob(dirread+'ORCA0083-N06_'+yr1+'*d05W.nc'))
pfiles = sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr1+'*d05P.nc'))
ppfiles = sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr1+'*d05D.nc'))
tsfiles = sorted(glob(dirread+'ORCA0083-N06_'+yr1+'*d05T.nc'))

        
mesh_mask = dirread_mesh+'coordinates.nc'

filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles}, #'depth': wfiles,
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles},
                 'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': wfiles},
                 'd_phy': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': pfiles},
                 'nd_phy': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': pfiles},  
                 'euph_z': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ppfiles},
                 'tpp3': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ppfiles},
                 'cons_temperature': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles},
                 'abs_salinity': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles},
                 'tau': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles}}

variables = {'U': 'uo',
                 'V': 'vo',
                 'W': 'wo',
                 'd_phy': 'PHD',
                 'nd_phy': 'PHN',
                 'euph_z': 'MED_XZE',
                 'tpp3': 'TPP3', # units: mmolN/m3/d 
                 'cons_temperature': 'potemp',
                 'abs_salinity': 'salin',
                 'tau': 'taum'}

dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}, #time_centered
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'd_phy': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'nd_phy': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'euph_z': {'lon': 'glamf', 'lat': 'gphif','time': 'time_counter'},
                  'tpp3': {'lon': 'glamf', 'lat': 'gphif','depth': 'depthw', 'time': 'time_counter'},
                  'cons_temperature': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'abs_salinity': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'tau': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'}}
    
initialgrid_mask = dirread+'ORCA0083-N06_20070105d05U.nc'
mask = xr.open_dataset(initialgrid_mask, decode_times=False)
Lat, Lon = mask.variables['nav_lat'], mask.variables['nav_lon']
latvals = Lat[:]; lonvals = Lon[:] # extract lat/lon values to numpy arrays
                                                                                               
iy_min, ix_min = getclosest_ij(latvals, lonvals, minlat, minlon) #minlat-5
iy_max, ix_max = getclosest_ij(latvals, lonvals, maxlat, maxlon) #minlat+5

indices = {'lat': range(iy_min, iy_max),'lon': range(ix_min, ix_max)}  # 'depth': range(0, 2000)  


chs = {'time_counter': 1, 'depthu': 25, 'depthv': 25, 'depthw': 25, 'deptht': 25, 'y': 200, 'x': 200} # for Parcels 2.1.5, can now define chunksize instead of indices in fieldset

fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=False, field_chunksize=chs, indices=indices) 


    #lons = fieldset.U.lon
    #lats = fieldset.U.lat
depths = fieldset.U.depth

    #------ Kinematic viscosity and dynamic viscosity not available in MEDUSA so replicating Kooi's profiles at all grid points ------
with open('/home/dlobelle/Kooi_data/data_input/profiles.pickle', 'rb') as f:
        depth,T_z,S_z,rho_z,upsilon_z,mu_z = pickle.load(f)

KV = Field('KV', np.array(upsilon_z), lon=0, lat=0, depth=depths, mesh='spherical') #np.empty(1)
SV = Field('SV', np.array(mu_z), lon=0, lat=0, depth=depths, mesh='spherical')
fieldset.add_field(KV, 'KV')
fieldset.add_field(SV, 'SV')
    
    
""" Defining the particle set """   
if sizes == '1to10mm':    
    rho_pls = [920, 920, 920, 920, 920, 920, 920, 920, 920, 920]  # add/remove here if more needed
    r_pls = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010]# add/remove here if more needed 
elif sizes == 'allsizes':    
    rho_pls = [920, 920, 920, 920, 920, 920]  # add/remove here if more needed
    r_pls = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]# add/remove here if more needed 
    

pset = ParticleSet.from_list(fieldset=fieldset,         # the fields on which the particles are advected
                                 pclass=plastic_particle,   # the type of particles (JITParticle or ScipyParticle)
                                 lon= lon_release, #-160.,  # a vector of release longitudes 
                                 lat= lat_release, #36., 
                                 time = np.datetime64('%s-%s-05' % (yr1, mon)),
                                 depth = z_release,
                                 r_pl = r_pls[0] * np.ones(np.array(lon_release).size),
                                 rho_pl = rho_pls[0] * np.ones(np.array(lon_release).size),
                                 r_tot = r_pls[0] * np.ones(np.array(lon_release).size),
                                 rho_tot = rho_pls[0] * np.ones(np.array(lon_release).size))

for r_pl, rho_pl in zip(r_pls[1:], rho_pls[1:]):
    pset.add(ParticleSet.from_list(fieldset=fieldset,         # the fields on which the particles are advected
                        pclass=plastic_particle,   # the type of particles (JITParticle or ScipyParticle)
                        lon= lon_release, #-160.,  # a vector of release longitudes 
                        lat= lat_release, #36., 
                        time = np.datetime64('%s-%s-05' % (yr1, mon)),
                        depth = z_release,
                        r_pl = r_pl * np.ones(np.array(lon_release).size),
                        rho_pl = rho_pl * np.ones(np.array(lon_release).size),
                        r_tot = r_pl * np.ones(np.array(lon_release).size),
                        rho_tot = rho_pl * np.ones(np.array(lon_release).size)))


""" Kernal + Execution"""

kernels = pset.Kernel(AdvectionRK4_3D) + pset.Kernel(PolyTEOS10_bsq) + pset.Kernel(Profiles) + pset.Kernel(Kooi) + pset.Kernel(kernel_kukulka_mixing)#pset.Kernel(periodicBC) +    

outfile = '/home/dlobelle/Kooi_data/data_output/allrho/res_'+res+'/allr/tests/kukulka_mix'+region+'_'+sizes+'_checkrho_3D_grid'+res+'_920kgm-3_'+str(round(simdays,2))+'days_startmon_'+mon+'_'+str(secsdt)+'dtsecs_'+str(round(hrsoutdt,2))+'hrsoutdt' 

pfile= ParticleFile(outfile, pset, outputdt=delta(hours = hrsoutdt))

pset.execute(kernels, runtime=delta(days=simdays), dt=delta(seconds = secsdt), output_file=pfile, verbose_progress=True, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle, ErrorCode.ErrorInterpolation: DeleteParticle})

pfile.close()

print('Execution finished')
