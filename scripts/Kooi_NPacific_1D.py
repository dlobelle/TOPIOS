# created 19/12/19- North Pacific: Kooi et al. 2017 in 1D (depth)

from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4_3D, AdvectionRK4, ErrorCode, ParticleFile, Variable, Field, NestedField, VectorField, timer
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

#------ CHOOSE (Note: the same values must also be placed in the Kooi kernel: lines 53 and 54) -----
rho_pl = "920"                 # density of plastic (kg m-3): DEFAULT FOR FIG 1: 920 but full range is: 840, 920, 940, 1050, 1380 (last 2 are initially non-buoyant)
r_pl = "1e-04"                # radius of plastic (m): DEFAULT FOR FIG 1: 10-3 to 10-6 included but full range is: 10 mm to 0.1 um or 10-2 to 10-7

lon = np.array([-161,-159]) #lon release locations
lat = np.array([35,37]) #lat release locations
simdays =  150 #number of days running the sim
secsdt = 60 #timestep of sim

time0 = 0
secsoutdt = 60*60 # seconds in an hour (must be in hours due to algal pickle profiles being hours)
total_secs = secsoutdt*24.*simdays - secsoutdt # total time (in seconds) being run for the sim
dt_secs = total_secs/secsoutdt

'''Loading the Kooi theoretical profiles for physical seawater properties: not time-dependent. Generated in separate python file'''

with open('/home/dlobelle/Kooi_data/data_input/profiles.pickle', 'rb') as f:
    depth,T_z,S_z,rho_z,upsilon_z,mu_z = pickle.load(f)
depth = np.array(depth)

'''Loading the Kooi theoretical profiles for biological seawater properties: time-dependent. Generated in separate python file'''

with open('/home/dlobelle/Kooi_data/data_input/profiles_t.pickle', 'rb') as p:
    depth,time,A_A_t,mu_A_t = pickle.load(p)
    
time = np.linspace(time0,total_secs,dt_secs+1)
    
'''General functions and kernals'''

def Kooi(particle,fieldset,time):  
    #------ CHOOSE AGAIN-----
    rho_pl = 920.                 # density of plastic (kg m-3): DEFAULT FOR FIG 1: 920 but full range is: 840, 920, 940, 1050, 1380 (last 2 are initially non-buoyant)
    r_pl = 1e-04                  # radius of plastic (m): DEFAULT FOR FIG 1: 10-3 to 10-6 included but full range is: 10 mm to 0.1 um or 10-2 to 10-7
    
    z = particle.depth            # [m]
    t = particle.temp             # [oC]
    sw_visc = particle.sw_visc    # seawatar viscosity[kg m-1 s-1]
    aa = particle.aa              # ambient algal concentration[no m-3]
    mu_aa = particle.mu_aa/86400. # attached algal growth [s-1] 
    kin_visc = particle.kin_visc  # kinematic viscosity[m2 s-1]
    rho_sw = particle.rho_sw      # seawater density [kg m-3]
    a = particle.a                # number of attached algae[no. m-2]
    vs = particle.vs              # particle velocity [m s-1]

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
    
    #------ Diffusivity -----
    r_tot = r_pl + t_bf                               # total radius [m]
    rho_tot = (r_pl**3. * rho_pl + ((r_pl + t_bf)**3. - r_pl**3.)*rho_bf)/(r_pl + t_bf)**3. # total density [kg m-3]
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
    if delta_rho > 0: # sinks 
        vs = (g * kin_visc * w * delta_rho)**(1./3.)
    else: #rises 
        a_del_rho = delta_rho*-1.
        vs = -1.*(g * kin_visc * w * a_del_rho)**(1./3.)  # m s-1
    
    particle.vs_init = vs # initial particle velocity, before forcing a 0 m s-1 value when particle is above 0.6 m (in loop below)
    
    z0 = z + vs * particle.dt 
    if z0 <=0.6 or z0 >= 4000.: # NEMO's 'surface depth'
        vs = 0
        particle.depth = 0.6
    else:          
        particle.depth += vs * particle.dt 

    particle.vs = vs
    
def DeleteParticle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted') 
    #print(particle.lon, particle.lat, particle.depth)
    particle.delete()
    
# def Sink(particle, fieldset, time):
#     """Test to check that adding constant sinking speed works (to be replaced with Kooi equation later)"""
#     sp = 10./86400. #The sinkspeed m/day (CAN CHANGE THIS LATER- in Kooi et al. 2017 for particle of 0.1mm = 100 m d-1)
#     particle.depth += sp * particle.dt #(sp/(24*60*60)) * particle.dt # m/s : 1e-3

    
def Profiles(particle, fieldset, time):  
    particle.temp = fieldset.T[time, particle.depth,particle.lat,particle.lon]  
    particle.rho_sw = fieldset.D[time,particle.depth,particle.lat,particle.lon] 
    particle.kin_visc = fieldset.KV[time,particle.depth,particle.lat,particle.lon] 
    particle.sw_visc = fieldset.SV[time,particle.depth,particle.lat,particle.lon] 
    particle.aa = fieldset.AA[time,particle.depth,particle.lat,particle.lon]
    particle.mu_aa = fieldset.AAmu[time,particle.depth,particle.lat,particle.lon]
    
""" Defining the particle class """

class plastic_particle(JITParticle): 
    u = Variable('u', dtype=np.float32,to_write=False)
    v = Variable('v', dtype=np.float32,to_write=False)
    w = Variable('w',dtype=np.float32,to_write=True) 
    temp = Variable('temp',dtype=np.float32,to_write=True)
    rho_sw = Variable('rho_sw',dtype=np.float32,to_write=False)
    kin_visc = Variable('kin_visc',dtype=np.float32,to_write=False)
    sw_visc = Variable('sw_visc',dtype=np.float32,to_write=False)
    aa = Variable('aa',dtype=np.float32,to_write=True)
    mu_aa = Variable('mu_aa',dtype=np.float32,to_write=False)
    a = Variable('a',dtype=np.float32,to_write=True)
    vs = Variable('vs',dtype=np.float32,to_write=True)
    vs_init = Variable('vs_init',dtype=np.float32,to_write=True)
    rho_tot = Variable('rho_tot',dtype=np.float32,to_write=True)
    
    
""" Defining the fieldset"""

depth = np.array(depth)
S = np.transpose(np.tile(np.array(S_z),(len(lat),len(lon),len(time),1)), (2,3,0,1))*1000. # salinity (in Kooi equations/profiles, the salinity was in kg/kg so now converting to g/kg)
T = np.transpose(np.tile(np.array(T_z),(len(lat),len(lon),len(time),1)),(2,3,0,1)) # temperature
D = np.transpose(np.tile(np.array(rho_z),(len(lat),len(lon),len(time),1)), (2,3,0,1)) # density
KV = np.transpose(np.tile(np.array(upsilon_z),(len(lat),len(lon),len(time),1)), (2,3,0,1)) # kinematic viscosity
SV = np.transpose(np.tile(np.array(mu_z),(len(lat),len(lon),len(time),1)), (2,3,0,1)) # dynamic viscosity of seawater
AA = np.transpose(np.tile(np.array(A_A_t),(len(lat),len(lon),simdays,1)), (2,3,0,1)) # ambient algae 
AAmu = np.transpose(np.tile(np.array(mu_A_t),(len(lat),len(lon),simdays,1)), (2,3,0,1)) # ambient algae growth
U = np.zeros(shape=(len(time),len(S_z),len(lat),len(lon))) # this is just a filler since the particle set must have a U component (eastward velocity)
V = np.zeros(shape=(len(time),len(S_z),len(lat),len(lon))) # this is just a filler since the particle set must have a V component (northward velocity)

data = {'U': U,
        'V': V,
        'T': T,
        'D': D,
        'KV': KV,
        'SV': SV,
        'AA': AA,
        'AAmu': AAmu}

dimensions = {'U': {'time': time,'depth': depth, 'lat': lat, 'lon': lon},
             'V': {'time': time,'depth': depth, 'lat': lat, 'lon': lon},
             'T': {'time': time,'depth': depth, 'lat': lat, 'lon': lon},
             'D': {'time': time,'depth': depth, 'lat': lat, 'lon': lon},
             'KV': {'time': time,'depth': depth, 'lat': lat, 'lon': lon},
             'SV': {'time': time,'depth': depth, 'lat': lat, 'lon': lon},
             'AA':{'time': time, 'depth': depth, 'lat': lat, 'lon': lon},
             'AAmu':{'time': time, 'depth': depth, 'lat': lat, 'lon': lon}}

fieldset = FieldSet.from_data(data, dimensions, allow_time_extrapolation = True) #transpose=True, 

pset = ParticleSet.from_list(fieldset=fieldset, # the fields on which the particles are advected
                             pclass=plastic_particle, # the type of particles 
                             lon=-160., # a vector of release longitudes 
                             lat=36., 
                             time = [0],
                             depth = [0.6])

""" Kernal + Execution"""

kernels = pset.Kernel(AdvectionRK4) +  pset.Kernel(Profiles)  + pset.Kernel(Kooi) 

dirwrite = '/home/dlobelle/Kooi_data/data_output/1D_results/'
outfile = dirwrite+'Kooionly_1D_'+str(round(simdays,2))+'d_rho'+rho_pl+'_rpl'+r_pl+'_'+str(secsdt)+'dtsecs_'+str(round(secsoutdt/3600.,2))+'hrsoutdt.nc' 

pfile= ParticleFile(outfile, pset, outputdt=delta(seconds = secsoutdt))

pset.execute(kernels, runtime=delta(days=simdays), dt=delta(seconds = secsdt), output_file=pfile, verbose_progress=True, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
pfile.close()

print('Execution finished')