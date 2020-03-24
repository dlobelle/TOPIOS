# 29/01/20- Based on Kooi_North Pacific_1D.py but using NEMO-MEDUSA profiles (now grid size is no longer 2x2 since 1/12 so can use min lons and lats and then index + 1 for 2x2 grid)

from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4_3D, AdvectionRK4, ErrorCode, ParticleFile, Variable, Field, NestedField, VectorField, timer #polyTEOS10_bsq #seawaterdensity
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

minlat = 30 #22 #36
maxlat = 40 #48
minlon = -165 #172 #-160
maxlon = -155 #136

lon_release = np.linspace(minlon+1,maxlon,10) #np.tile(np.linspace(minlon+1,maxlon,10),[10,1])
lat_release =  np.linspace(minlat+1,maxlat,10)# np.tile(np.linspace(minlat+1,maxlat,10),[10,1])
#lat_release = lat_release0.T
z_release = [1]*10 #np.tile(1,[10,10])

simdays = 10
time0 = 0
simhours = 1
simmins = 30
secsdt = 10
hrsoutdt = 10


#------ Choose below: NOTE- MUST ALSO MANUALLY CHANGE IT IN THE KOOI KERNAL -----
rho_pl = 920.                 # density of plastic (kg m-3): DEFAULT FOR FIG 1: 920 but full range is: 840, 920, 940, 1050, 1380 (last 2 are initially non-buoyant)
r_pl = "1e-04"                # radius of plastic (m): DEFAULT FOR FIG 1: 10-3 to 10-6 included but full range is: 10 mm to 0.1 um or 10-2 to 10-7

""" Defining the particle class """

class plastic_particle(JITParticle): #ScipyParticle): #
    u = Variable('u', dtype=np.float32,to_write=False)
    v = Variable('v', dtype=np.float32,to_write=False)
    w = Variable('w', dtype=np.float32,to_write=True)
    temp = Variable('temp',dtype=np.float32,to_write=False)
    density = Variable('density',dtype=np.float32,to_write=True)
    #aa = Variable('aa',dtype=np.float32,to_write=True)
    tpp = Variable('tpp',dtype=np.float32,to_write=False) # mu_aa
    #euph_z = Variable('euph_z',dtype=np.float32,to_write=False)
    d_phy = Variable('d_phy',dtype=np.float32,to_write=False)
    nd_phy = Variable('nd_phy',dtype=np.float32,to_write=False)    
    kin_visc = Variable('kin_visc',dtype=np.float32,to_write=False)
    sw_visc = Variable('sw_visc',dtype=np.float32,to_write=False)    
    a = Variable('a',dtype=np.float32,to_write=False)
    vs = Variable('vs',dtype=np.float32,to_write=True)    
    
"""functions and kernals"""

def DeleteParticle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted') 
    #print(particle.lon, particle.lat, particle.depth)
    particle.delete()

def getclosest_ij(lats,lons,latpt,lonpt):     
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_sq = (lats-latpt)**2 + (lons-lonpt)**2                 # find squared distance of every point on grid
    minindex_flattened = dist_sq.argmin()                       # 1D index of minimum dist_sq element
    return np.unravel_index(minindex_flattened, lats.shape)     # Get 2D index for latvals and lonvals arrays from 1D index

def AdvectionRK4_3D_vert(particle, fieldset, time): # adapting AdvectionRK4_3D kernal to only vertical velocity 
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.
    Function needs to be converted to Kernel object before execution"""
    (w1) = fieldset.W[time, particle.depth, particle.lat, particle.lon]
    #lon1 = particle.lon + u1*.5*particle.dt
    #lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    (w2) = fieldset.W[time + .5 * particle.dt, dep1, particle.lat, particle.lon]
    #lon2 = particle.lon + u2*.5*particle.dt
    #lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    (w3) = fieldset.W[time + .5 * particle.dt, dep2, particle.lat, particle.lon]
    #lon3 = particle.lon + u3*particle.dt
    #lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    (w4) = fieldset.W[time + particle.dt, dep3, particle.lat, particle.lon]
    #particle.lon += particle.lon #(u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    #particle.lat += particle.lat #lats[1,1] #(v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt

def polyTEOS10_bsq(particle, fieldset, time):
    # calculates density based on the polyTEOS10-bsq algorithm from Appendix A.2 of
    # https://www.sciencedirect.com/science/article/pii/S1463500315000566
    # requires fieldset.abs_salinity and fieldset.cons_temperature Fields in the fieldset
    # and a particle.density Variable in the ParticleSet
    #
    # References:
    #  Roquet, F., Madec, G., McDougall, T. J., Barker, P. M., 2014: Accurate
    #   polynomial expressions for the density and specific volume of
    #   seawater using the TEOS-10 standard. Ocean Modelling.
    #  McDougall, T. J., D. R. Jackett, D. G. Wright and R. Feistel, 2003:
    #   Accurate and computationally efficient algorithms for potential
    #   temperature and density of seawater.  Journal of Atmospheric and
    #   Oceanic Technology, 20, 730-741.

    Z = - particle.depth  # note: use negative depths!
    SA = fieldset.abs_salinity[time, particle.depth, particle.lat, particle.lon]
    CT = fieldset.cons_temperature[time, particle.depth, particle.lat, particle.lon]

    SAu = 40 * 35.16504 / 35
    CTu = 40
    Zu = 1e4
    deltaS = 32
    R000 = 8.0189615746e+02
    R100 = 8.6672408165e+02
    R200 = -1.7864682637e+03
    R300 = 2.0375295546e+03
    R400 = -1.2849161071e+03
    R500 = 4.3227585684e+02
    R600 = -6.0579916612e+01
    R010 = 2.6010145068e+01
    R110 = -6.5281885265e+01
    R210 = 8.1770425108e+01
    R310 = -5.6888046321e+01
    R410 = 1.7681814114e+01
    R510 = -1.9193502195e+00
    R020 = -3.7074170417e+01
    R120 = 6.1548258127e+01
    R220 = -6.0362551501e+01
    R320 = 2.9130021253e+01
    R420 = -5.4723692739e+00
    R030 = 2.1661789529e+01
    R130 = -3.3449108469e+01
    R230 = 1.9717078466e+01
    R330 = -3.1742946532e+00
    R040 = -8.3627885467e+00
    R140 = 1.1311538584e+01
    R240 = -5.3563304045e+00
    R050 = 5.4048723791e-01
    R150 = 4.8169980163e-01
    R060 = -1.9083568888e-01
    R001 = 1.9681925209e+01
    R101 = -4.2549998214e+01
    R201 = 5.0774768218e+01
    R301 = -3.0938076334e+01
    R401 = 6.6051753097e+00
    R011 = -1.3336301113e+01
    R111 = -4.4870114575e+00
    R211 = 5.0042598061e+00
    R311 = -6.5399043664e-01
    R021 = 6.7080479603e+00
    R121 = 3.5063081279e+00
    R221 = -1.8795372996e+00
    R031 = -2.4649669534e+00
    R131 = -5.5077101279e-01
    R041 = 5.5927935970e-01
    R002 = 2.0660924175e+00
    R102 = -4.9527603989e+00
    R202 = 2.5019633244e+00
    R012 = 2.0564311499e+00
    R112 = -2.1311365518e-01
    R022 = -1.2419983026e+00
    R003 = -2.3342758797e-02
    R103 = -1.8507636718e-02
    R013 = 3.7969820455e-01
    ss = math.sqrt((SA + deltaS) / SAu)
    tt = CT / CTu
    zz = -Z / Zu
    rz3 = R013 * tt + R103 * ss + R003
    rz2 = (R022 * tt + R112 * ss + R012) * tt + (R202 * ss + R102) * ss + R002
    rz1 = (((R041 * tt + R131 * ss + R031) * tt + (R221 * ss + R121) * ss + R021) * tt + ((R311 * ss + R211) * ss + R111) * ss + R011) * tt + (((R401 * ss + R301) * ss + R201) * ss + R101) * ss + R001
    rz0 = (((((R060 * tt + R150 * ss + R050) * tt + (R240 * ss + R140) * ss + R040) * tt + ((R330 * ss + R230) * ss + R130) * ss + R030) * tt + (((R420 * ss + R320) * ss + R220) * ss + R120) * ss + R020) 
           * tt + ((((R510 * ss + R410) * ss + R310) * ss + R210) * ss + R110) * ss + R010) * tt + (((((R600 * ss + R500) * ss + R400) * ss + R300) * ss + R200) * ss + R100) * ss + R000
    particle.density = ((rz3 * zz + rz2) * zz + rz1) * zz + rz0
    
def Profiles(particle, fieldset, time):  
    particle.temp = fieldset.cons_temperature[time, particle.depth,particle.lat,particle.lon]  
    particle.d_phy= fieldset.d_phy[time, particle.depth,particle.lat,particle.lon]  
    particle.nd_phy= fieldset.nd_phy[time, particle.depth,particle.lat,particle.lon] 
    #particle.tpp = fieldset.tpp[time,particle.lat,particle.lon]
    #particle.euph_z = fieldset.euph_z[time,particle.lat,particle.lon]
    particle.kin_visc = fieldset.KV[time,particle.depth,particle.lat,particle.lon] 
    particle.sw_visc = fieldset.SV[time,particle.depth,particle.lat,particle.lon] 
    particle.w = fieldset.W[time,particle.depth,particle.lat,particle.lon]
    
def Kooi(particle,fieldset,time):  
    # 30/01/20- for aa and mu_aa, using ratios to get ambient algal concentrations and algal growth (N:C:AA using Redfield ratio... C:N = 6.625, so N*6.625)
     
    min_N2cell = 2656.0e-09 #[mgN cell-1] 35339e-09 [mgC cell-1]
    max_N2cell = 11.0e-09   #[mgN cell-1] 47.67e-09 [mgC cell-1]
    med_N2cell = 356.04e-09
    
#   n0 = particle.nd_phy # mmol N m-3 
#   n = n0*14.007 # conversion from mmol N m-3 to mg N m-3 (atomic weight of 1 mol of N = 14.007 g, so same from mmol to mg)    
    #c = n*6.625 # conversion from mg N m-3 to mg C m-3 (Redfield ratio)
    #c2 = c/(47.76*1e-09)
    
    #print(c2)
    #if c2<0:# conversion from mg C m-3 to no. m-3
    #    aa = 0
    #else:
    #    aa = c2   # should be [no m-3] to compare to Kooi model    
    #particle.aa = aa
    
    n0 = particle.nd_phy+particle.d_phy # mmol N m-3 
    n = n0*14.007       # conversion from mmol N m-3 to mg N m-3 (atomic weight of 1 mol of N = 14.007 g)   
    n2 = n/med_N2cell   # conversion from mg N m-3 to no. m-3
    
    if n2<0.: 
        aa = 0.
    else:
        aa = n2   # [no m-3] to compare to Kooi model    

    mu_n0 = particle.tpp/aa    
    mu_n = mu_n0*14.007               # conversion from mmol N m-3 d-1 to mg N m-3 d-1 (atomic weight of 1 mol of N = 14.007 g) 
    mu_n2 = mu_n/med_N2cell           # conversion from mg N m-3 d-1 to d-1

    if mu_n2<0.:
        mu_aa = 0.
    else:
        mu_aa = mu_n2/86400. # conversion from d-1 to s-1
        
    z = particle.depth           # [m]
    t = particle.temp            # [oC]
    sw_visc = particle.sw_visc   # [kg m-1 s-1]
    kin_visc = particle.kin_visc # [m2 s-1]
    rho_sw = particle.density    # [kg m-3]   #rho_sw     
    a = particle.a               # [no. m-2 s-1]
    vs = particle.vs #particle.depth # [m s-1]

    
    #------ CHOOSE -----
    rho_pl = 920.                 # density of plastic (kg m-3): DEFAULT FOR FIG 1: 920 but full range is: 840, 920, 940, 1050, 1380 (last 2 are initially non-buoyant)
    r_pl = 1e-04                  # radius of plastic (m): DEFAULT FOR FIG 1: 10-3 to 10-6 included but full range is: 10 mm to 0.1 um or 10-2 to 10-7

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

""" Defining the fieldset""" # FOR NOW: only 1 day (05 01 2007), and time_extrapolation = True

dirread = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/means/'
dirread_bgc = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA_BGC/ORCA0083-N006/means/'  
dirread_mesh = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/domain/'  

ufiles = (dirread+'ORCA0083-N06_20070105d05U.nc')
vfiles = (dirread+'ORCA0083-N06_20070105d05V.nc')
wfiles = (dirread+'ORCA0083-N06_20070105d05W.nc')
pfiles = (dirread_bgc+'ORCA0083-N06_20070105d05P.nc')
ppfiles = (dirread_bgc+'ORCA0083-N06_20070105d05D.nc')
tsfiles = (dirread+'ORCA0083-N06_20070105d05T.nc')
mesh_mask = dirread_mesh+'coordinates.nc'

filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles, 'data': [ufiles]},
             'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles, 'data': [vfiles]},
             'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles, 'data': [wfiles]},
             'd_phy': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles, 'data': pfiles},
             'nd_phy': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles, 'data': pfiles},  
             #'euph_z': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ppfiles},
             #'tpp': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ppfiles}, # 'depth': wfiles,
             'cons_temperature': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles, 'data': tsfiles},
             'abs_salinity': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles, 'data': tsfiles}}


variables = {'U': 'uo',
             'V': 'vo',
             'W': 'wo',
             'd_phy': 'PHD',
             'nd_phy': 'PHN',
             #'euph_z': 'MED_XZE',
             #'tpp': 'PRN', #TPP3', # AAmu
             'cons_temperature': 'potemp',
             'abs_salinity': 'salin'}

dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_centered'},
              'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_centered'},
              'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_centered'},
              'd_phy': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_centered'},
              'nd_phy': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_centered'},
              #'euph_z': {'lon': 'glamf', 'lat': 'gphif','time': 'time_centered'},
              #'tpp': {'lon': 'glamf', 'lat': 'gphif','time': 'time_centered'}, # 'depth': 'depthw',
              'cons_temperature': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_centered'},
              'abs_salinity': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_centered'}}

initialgrid_mask = dirread+'ORCA0083-N06_20070105d05U.nc'
mask = xr.open_dataset(initialgrid_mask, decode_times=False)
Lat, Lon, Depth = mask.variables['nav_lat'], mask.variables['nav_lon'], mask.variables['depthu']
latvals = Lat[:]; lonvals = Lon[:] # extract lat/lon values to numpy arrays

iy_min, ix_min = getclosest_ij(latvals, lonvals, minlat, minlon)
iy_max, ix_max = getclosest_ij(latvals, lonvals, maxlat, maxlon)
indices = {'lon': range(ix_min, ix_max), 'lat': range(iy_min, iy_max)}  # 'depth': range(0, 2000)

fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=True,indices=indices)

lons = fieldset.U.lon
lats = fieldset.U.lat
depths = fieldset.U.depth

with open('/home/dlobelle/Kooi_data/data_input/profiles.pickle', 'rb') as f:
    depth,T_z,S_z,rho_z,upsilon_z,mu_z = pickle.load(f)

v_lon = np.array([minlon,maxlon]) #,[minlon,maxlon]])
v_lat = np.array([minlat,maxlat]) #,[minlat,maxlat]])

kv_or = np.transpose(np.tile(np.array(upsilon_z),(len(v_lon),len(v_lat),1)), (2,0,1)) # kinematic viscosity
sv_or = np.transpose(np.tile(np.array(mu_z),(len(v_lon),len(v_lat),1)), (2,0,1)) # dynamic viscosity of seawater    

KV = Field('KV',kv_or,lon=v_lon,lat=v_lat,depth = depths, mesh='spherical')#,transpose="True") #,fieldtype='U')
SV = Field('SV',sv_or,lon=v_lon,lat=v_lat,depth = depths, mesh='spherical')#,transpose="True") #,fieldtype='U')
fieldset.add_field(KV)
fieldset.add_field(SV)

# ------------- Using average diatom or non-diatom PP instead of TPP3 -----------

pp_orig = xr.open_dataset(ppfiles)
euph_z,nd_phy_ml,d_phy_ml = pp_orig.variables['MED_XZE'], pp_orig.variables['PRN'], pp_orig.variables['PRD']

z_nemo = Depth.data # depth levels of NEMO 
euph_z1 = euph_z[:,iy_min:iy_max,ix_min:ix_max].data #euph_z[:,iy_min+1,ix_min+1].data #50 # selecting euph layer depth where particle released
tot_phy_ml = nd_phy_ml[:,iy_min:iy_max,ix_min:ix_max].data + d_phy_ml[:,iy_min:iy_max,ix_min:ix_max].data 

z_all = np.transpose(np.tile(np.array(z_nemo),(len(lats),len(lats[0]),1)), (2,0,1))
id_ = z_all < euph_z1
nemo_euph = z_all[id_]

d = (len(z_all),len(z_all[0]),len(z_all[0][0]))
dz = np.zeros(d)

for z in range(len(dz)):
    if z == 0:
        dz[z] = (z_all[z+1]-z_all[z])
    elif z == 74:
        dz[z] = (z_all[z]-z_all[z-1])
    else:
        dz[z] = ((z_all[z]-z_all[z-1])/2)+((z_all[z+1]-z_all[z])/2) #(z_all[z+1]-z_all[z])

# print(dz.shape)
# print(tot_phy_ml.shape)
# print(id_.shape)
tpp_or = (tot_phy_ml*id_)/dz

print(tpp_or.shape,tpp_or[:,0,0])
# e1 = np.tile(np.array(euph_z1),(1,len(z_all),3,3))
# p1 = np.tile(np.array(tot_phy_ml),(1,len(z_all),3,3))
# i1 = np.transpose(np.tile(id_,(3,3,1,1)),(2,3,0,1))
# dz1 = np.transpose(np.tile(dz,(3,3,1,1)),(2,3,0,1))
# tpp_or= (p1*i1)/dz1


tpp = Field('tpp',tpp_or,lon=lons,lat=lats,depth = depths, mesh='spherical')#,fieldtype='U'
        
fieldset.add_field(tpp)


""" Defining the particle set """

pset = ParticleSet.from_list(fieldset=fieldset,       # the fields on which the particles are advected
                             pclass=plastic_particle, # the type of particles (JITParticle or ScipyParticle)
                             lon= lon_release, #-160.,  # a vector of release longitudes 
                             lat= lat_release, #36., 
                             time = [0],
                             depth = z_release) #[1.]

""" Kernal + Execution"""

kernels = pset.Kernel(AdvectionRK4_3D_vert) + pset.Kernel(polyTEOS10_bsq) + pset.Kernel(Profiles) + pset.Kernel(Kooi) #+ pset.Kernel(Sink) # pset.Kernel(AdvectionRK4_3D_vert) 

dirwrite = '/home/dlobelle/Kooi_data/data_output/tests/'
outfile = dirwrite + 'Kooi+NEMO_3DwWadv_rho'+str(int(rho_pl))+'_r'+ r_pl+'_'+str(simdays)+'days_'+str(secsdt)+'dtsecs_'+str(hrsoutdt)+'hrsoutdt'

pfile= ParticleFile(outfile, pset, outputdt=delta(hours = hrsoutdt)) #120

pset.execute(kernels, runtime=delta(days=simdays), dt=delta(seconds = secsdt), output_file=pfile, verbose_progress=True, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}) # minutes=0.1
pfile.close()

print('Execution finished')