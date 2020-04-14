# 29/01/20- Based on Kooi+NEMO_North Pacific_1D.py but using NEMO-MEDUSA profiles (now grid size is no longer 2x2 since 1/12 deg resolution so can use min lons and lats and then index + 1 for 2x2 grid, or 10x10 grid)

from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4_3D, AdvectionRK4, ErrorCode, ParticleFile, Variable, Field, NestedField, VectorField, timer 
#from parcels.kernels import seawaterdensity
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
warnings.filterwarnings("ignore")

# Fieldset grid is 30x30 deg in North Pacific
minlat = 20 
maxlat = 50 
minlon = -175 
maxlon = -145 

# Release particles on a 10x10 deg grid in middle of the 30x30 fieldset grid and 1m depth
lat_release0 = np.tile(np.linspace(30,39,10),[10,1]) 
lat_release = lat_release0.T 
lon_release = np.tile(np.linspace(-165,-156,10),[10,1]) 
z_release = np.tile(1,[10,10])

# Choose:
simdays = 5
time0 = 0
simhours = 1
simmins = 30
secsdt = 30
hrsoutdt = 5

#--------- Choose below: NOTE- MUST ALSO MANUALLY CHANGE IT IN THE KOOI KERNAL BELOW -----
rho_pl = 920.                 # density of plastic (kg m-3): DEFAULT FOR FIG 1 in Kooi: 920 but full range is: 840, 920, 940, 1050, 1380 (last 2 are initially non-buoyant)
r_pl = "1e-04"                # radius of plastic (m): DEFAULT FOR FIG 1: 10-3 to 10-6 included but full range is: 10 mm to 0.1 um or 10-2 to 10-7


def Kooi(particle,fieldset,time):  
    #------ CHOOSE -----
    rho_pl = 920.                 # density of plastic (kg m-3): DEFAULT FOR FIG 1: 920 but full range is: 840, 920, 940, 1050, 1380 (last 2 are initially non-buoyant)
    r_pl = 1e-04                  # radius of plastic (m): DEFAULT FOR FIG 1: 10-3 to 10-6 included but full range is: 10 mm to 0.1 um or 10-2 to 10-7   
    
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
        aa = n2   # [no m-3] to compare to Kooi model    
    
    #------ Primary productivity (algal growth) only above euphotic zone, condition same as in Kooi et al. 2017
    if particle.depth<particle.euph_z:
        tpp0 = particle.tpp3 # (particle.nd_tpp + particle.d_tpp)/particle.euph_z # 2D nondiatom + diatom PP couldn't be converted to a vertical profile (better with TPP3)
    else:
        tpp0 = 0.    
    
    mu_n0 = tpp0*14.007               # conversion from mmol N m-3 d-1 to mg N m-3 d-1 (atomic weight of 1 mol of N = 14.007 g) 
    mu_n = mu_n0/med_N2cell           # conversion from mg N m-3 d-1 to no. m-3 d-1
    mu_n2 = mu_n/aa                   # conversion from no. m-3 d-1 to d-1
    
    if mu_n2<0.:
        mu_aa = 0.
    else:
        mu_aa = mu_n2/86400.          # conversion from d-1 to s-1
    
    #------ Profiles from MEDUSA or Kooi theoretical profiles -----
    z = particle.depth           # [m]
    t = particle.temp            # [oC]
    sw_visc = particle.sw_visc   # [kg m-1 s-1]
    kin_visc = particle.kin_visc # [m2 s-1]
    rho_sw = particle.density    # [kg m-3]   #rho_sw     
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
    v_pl = (4./3.)*math.pi*r_pl**3.             # volume of plastic [m3]
    theta_pl = 4.*math.pi*r_pl**2.              # surface area of plastic particle [m2]
    r_a = ((3./4.)*(v_a/math.pi))**(1./3.)      # radius of algae [m]
    
    v_bf = (v_a*a)*theta_pl                           # volume of biofilm [m3]
    v_tot = v_bf + v_pl                               # volume of total [m3]
    t_bf = ((v_tot*(3./(4.*math.pi)))**(1./3.))-r_pl  # biofilm thickness [m] 
    
    
    r_tot = r_pl + t_bf                               # total radius [m]
    rho_tot = (r_pl**3. * rho_pl + ((r_pl + t_bf)**3. - r_pl**3.)*rho_bf)/(r_pl + t_bf)**3. # total density [kg m-3]
    rho_tot = rho_tot
    theta_tot = 4.*math.pi*r_tot**2.                          # surface area of total [m2]
    d_pl = k * (t + 273.16)/(6. * math.pi * sw_visc * r_tot)  # diffusivity of plastic particle [m2 s-1]
    d_a = k * (t + 273.16)/(6. * math.pi * sw_visc * r_a)     # diffusivity of algal cells [m2 s-1] 
    beta_abrown = 4.*math.pi*(d_pl + d_a)*(r_tot + r_a)       # Brownian motion [m3 s-1] 
    beta_ashear = 1.3*gamma*((r_tot + r_a)**3.)               # advective shear [m3 s-1]
    beta_aset = (1./2.)*math.pi*r_tot**2. * abs(vs)           # differential settling [m3 s-1]
    beta_a = beta_abrown + beta_ashear + beta_aset            # collision rate [m3 s-1]
    
    a_coll = (beta_a*aa)/theta_pl
    a_growth = mu_aa*a
    a_mort = m_a*a
    a_resp = (q10**((t-20.)/10.))*r20*a     
    
    particle.a += (a_coll + a_growth - a_mort - a_resp) * particle.dt

    dn = 2. * (r_tot)                             # equivalent spherical diameter [m]
    delta_rho = (rho_tot - rho_sw)/rho_sw         # normalised difference in density between total plastic+bf and seawater[-]        
    d = ((rho_tot - rho_sw) * g * dn**3.)/(rho_sw * kin_visc**2.) # [-]
    
    if dn > 5e9:
        w = 1000.
    elif dn <0.05:
        w = (d**2.) *1.71E-4
    else:
        w = 10.**(-3.76715 + (1.92944*math.log10(d)) - (0.09815*math.log10(d)**2.) - (0.00575*math.log10(d)**3.) + (0.00056*math.log10(d)**4.))
    
    if z >= 4000.: 
        vs = 0
    elif z < 1. and delta_rho < 0:
        vs = 0  
    elif delta_rho > 0:
        vs = (g * kin_visc * w * delta_rho)**(1./3.)
    else: 
        a_del_rho = delta_rho*-1.
        vs = -1.*(g * kin_visc * w * a_del_rho)**(1./3.)  # m s-1

    particle.depth += vs * particle.dt 
    particle.vs = vs
    z = particle.depth
    dt = particle.dt

""" Defining the particle class """

class plastic_particle(JITParticle): #ScipyParticle): #
    u = Variable('u', dtype=np.float32,to_write=False)
    v = Variable('v', dtype=np.float32,to_write=False)
    w = Variable('w', dtype=np.float32,to_write=False)
    temp = Variable('temp',dtype=np.float32,to_write=False)
    density = Variable('density',dtype=np.float32,to_write=True)
    #aa = Variable('aa',dtype=np.float32,to_write=True)
    #d_tpp = Variable('d_tpp',dtype=np.float32,to_write=False) # mu_aa
    #nd_tpp = Variable('nd_tpp',dtype=np.float32,to_write=False)
    tpp3 = Variable('tpp3',dtype=np.float32,to_write=False)
    euph_z = Variable('euph_z',dtype=np.float32,to_write=False)
    d_phy = Variable('d_phy',dtype=np.float32,to_write=False)
    nd_phy = Variable('nd_phy',dtype=np.float32,to_write=False)    
    kin_visc = Variable('kin_visc',dtype=np.float32,to_write=False)
    sw_visc = Variable('sw_visc',dtype=np.float32,to_write=False)    
    a = Variable('a',dtype=np.float32,to_write=False)
    vs = Variable('vs',dtype=np.float32,to_write=True)    
    
"""functions and kernals"""

def DeleteParticle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted') #print(particle.lon, particle.lat, particle.depth)
    particle.delete()

def getclosest_ij(lats,lons,latpt,lonpt):     
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_sq = (lats-latpt)**2 + (lons-lonpt)**2                 # find squared distance of every point on grid
    minindex_flattened = dist_sq.argmin()                       # 1D index of minimum dist_sq element
    return np.unravel_index(minindex_flattened, lats.shape)     # Get 2D index for latvals and lonvals arrays from 1D index
    
def Profiles(particle, fieldset, time):  
    particle.temp = fieldset.cons_temperature[time, particle.depth,particle.lat,particle.lon]  
    particle.d_phy= fieldset.d_phy[time, particle.depth,particle.lat,particle.lon]  
    particle.nd_phy= fieldset.nd_phy[time, particle.depth,particle.lat,particle.lon] 
    #particle.d_tpp = fieldset.d_tpp[time,particle.depth,particle.lat,particle.lon]
    #particle.nd_tpp = fieldset.nd_tpp[time,particle.depth,particle.lat,particle.lon]
    particle.tpp3 = fieldset.tpp3[time,particle.depth,particle.lat,particle.lon]
    particle.euph_z = fieldset.euph_z[time,particle.depth,particle.lat,particle.lon]
    particle.kin_visc = fieldset.KV[time,particle.depth,particle.lat,particle.lon] 
    particle.sw_visc = fieldset.SV[time,particle.depth,particle.lat,particle.lon] 
    particle.w = fieldset.W[time,particle.depth,particle.lat,particle.lon]

""" Defining the fieldset""" 

dirread = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/means/'
dirread_bgc = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA_BGC/ORCA0083-N006/means/'  
dirread_mesh = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/domain/'  

ufiles = sorted(glob(dirread+'ORCA0083-N06_2007*d05U.nc')) #0105d05
vfiles = sorted(glob(dirread+'ORCA0083-N06_2007*d05V.nc'))
wfiles = sorted(glob(dirread+'ORCA0083-N06_2007*d05W.nc'))
pfiles = sorted(glob(dirread_bgc+'ORCA0083-N06_2007*d05P.nc'))
ppfiles = sorted(glob(dirread_bgc+'ORCA0083-N06_2007*d05D.nc'))
tsfiles = sorted(glob(dirread+'ORCA0083-N06_2007*d05T.nc'))
mesh_mask = dirread_mesh+'coordinates.nc'

filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles}, #'depth': wfiles,
             'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles},
             'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': wfiles},
             'd_phy': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': pfiles},
             'nd_phy': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': pfiles},  
             'euph_z': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ppfiles},
             #'d_tpp': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ppfiles}, # 'depth': wfiles,
             #'nd_tpp': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ppfiles},
             'tpp3': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ppfiles},
             'cons_temperature': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles},
             'abs_salinity': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles}}


variables = {'U': 'uo',
             'V': 'vo',
             'W': 'wo',
             'd_phy': 'PHD',
             'nd_phy': 'PHN',
             'euph_z': 'MED_XZE',
             #'d_tpp': 'ML_PRD', # units: mmolN/m2/d 
             #'nd_tpp': 'ML_PRN', # units: mmolN/m2/d
             'tpp3': 'TPP3', # units: mmolN/m3/d 
             'cons_temperature': 'potemp',
             'abs_salinity': 'salin'}

dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}, #time_centered
              'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
              'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
              'd_phy': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
              'nd_phy': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
              'euph_z': {'lon': 'glamf', 'lat': 'gphif','time': 'time_counter'},
              #'d_tpp': {'lon': 'glamf', 'lat': 'gphif','time': 'time_counter'}, # 'depth': 'depthw',
              #'nd_tpp': {'lon': 'glamf', 'lat': 'gphif','time': 'time_counter'},
              'tpp3': {'lon': 'glamf', 'lat': 'gphif','depth': 'depthw', 'time': 'time_counter'},
              'cons_temperature': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
              'abs_salinity': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'}}

initialgrid_mask = dirread+'ORCA0083-N06_20070105d05U.nc'
mask = xr.open_dataset(initialgrid_mask, decode_times=False)
Lat, Lon, Depth = mask.variables['nav_lat'], mask.variables['nav_lon'], mask.variables['depthu']
latvals = Lat[:]; lonvals = Lon[:] # extract lat/lon values to numpy arrays

iy_min, ix_min = getclosest_ij(latvals, lonvals, minlat, minlon)
iy_max, ix_max = getclosest_ij(latvals, lonvals, maxlat, maxlon)
indices = {'lon': range(ix_min, ix_max), 'lat': range(iy_min, iy_max)}
mask.close()

fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=False, indices=indices) #allow_time_extrapolation=True or False

depths = fieldset.U.depth

# Kinematic viscosity and dynamic viscosity not available in MEDUSA so replicating Kooi's profiles at all grid points
with open('/home/dlobelle/Kooi_data/data_input/profiles.pickle', 'rb') as f:
    depth,T_z,S_z,rho_z,upsilon_z,mu_z = pickle.load(f)

v_lon = np.array([minlon,maxlon]) 
v_lat = np.array([minlat,maxlat]) 

kv_or = np.transpose(np.tile(np.array(upsilon_z),(len(v_lon),len(v_lat),1)), (2,0,1))   # kinematic viscosity
sv_or = np.transpose(np.tile(np.array(mu_z),(len(v_lon),len(v_lat),1)), (2,0,1))        # dynamic viscosity of seawater    
KV = Field('KV',kv_or,lon=v_lon,lat=v_lat,depth = depths, mesh='spherical')#,transpose="True") #,fieldtype='U')
SV = Field('SV',sv_or,lon=v_lon,lat=v_lat,depth = depths, mesh='spherical')#,transpose="True") #,fieldtype='U')
fieldset.add_field(KV, 'KV')
fieldset.add_field(SV, 'SV')

""" Defining the particle set """

pset = ParticleSet.from_list(fieldset=fieldset,         # the fields on which the particles are advected
                             pclass=plastic_particle,   # the type of particles (JITParticle or ScipyParticle)
                             lon= lon_release, #-160.,  # a vector of release longitudes 
                             lat= lat_release, #36., 
                             time = time0,
                             depth = z_release) #[1.]

""" Kernal + Execution"""

kernels = pset.Kernel(AdvectionRK4_3D) + pset.Kernel(polyTEOS10_bsq) + pset.Kernel(Profiles) + pset.Kernel(Kooi) # pset.Kernel(AdvectionRK4_3D_vert) +pset.Kernel(polyTEOS10_bsq) +

outfile = '/home/dlobelle/Kooi_data/data_output/tests/rho_'+str(int(rho_pl))+'kgm-3/Kooi+NEMO_3D_grid10by10_rho'+str(int(rho_pl))+'_r'+ r_pl+'_'+str(simdays)+'days_'+str(secsdt)+'dtsecs_'+str(hrsoutdt)+'hrsoutdt'

pfile= ParticleFile(outfile, pset, outputdt=delta(hours = hrsoutdt))

pset.execute(kernels, runtime=delta(days=simdays), dt=delta(seconds = secsdt), output_file=pfile, verbose_progress=True, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}) # recovery={ErrorCode.ErrorOutOfBounds: Delete, ErrorThroughSurface: PushBack}

pfile.close()

print('Execution finished')