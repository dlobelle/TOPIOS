from parcels import FieldSet, Field, NestedField, VectorField, ParticleFile, ParticleSet, JITParticle
from parcels import Variable, AdvectionRK4, ErrorCode, AdvectionRK4_3D, plotTrajectoriesFile, random
import numpy as np
from glob import glob
import time as timelib
from datetime import timedelta 
import xarray as xr
from datetime import datetime
import math
from os import path
from random import gauss
from parcels.kernel import Kernel
from parcels.compiler import GNUCompiler
from parcels.particlefile import ParticleFile
from parcels.tools.loggers import logger
from parcels.grid import GridCode
from parcels.field import NestedField, SummedField
import progressbar
import collections
#import warnings
#%matplotlib inline
from operator import attrgetter
from scipy.stats import norm

#warnings.filterwarnings("ignore")

#Indices Limited Earth
minlat = 30
maxlat = 50
minlon = -10
maxlon = 55


#Parameters Program
rads = math.pi / 180


class ParticleVariables(JITParticle):
    """Particle velocities in degrees"""
    u = Variable('u', dtype=np.float32,to_write=False)
    v = Variable('v', dtype=np.float32,to_write=False)
    w = Variable('w', dtype=np.float32,to_write=False)
    age = Variable('age', dtype=np.float32,initial=0,to_write=False)
    windstress = Variable('windstress', dtype=np.float32)
    density = Variable('density', dtype=np.float32)

def WindStress(particle, fieldset, time): 
    stress = fieldset.TAU[time,0.,particle.lat,particle.lon] #TAU is stress in Pa taken from T from NEMO dataset                        
    particle.windstress = stress

def DeleteParticle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    particle.delete()
    
def getclosest_ij(lats,lons,latpt,lonpt):     
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_sq = (lats-latpt)**2 + (lons-lonpt)**2                 # find squared distance of every point on grid
    minindex_flattened = dist_sq.argmin()                       # 1D index of minimum dist_sq element
    return np.unravel_index(minindex_flattened, lats.shape)     # Get 2D index for latvals and lonvals arrays from 1D index

    
def Out(particle, fieldset, time):
    (uaux, vaux, waux) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon] #
    particle.u = uaux
    particle.v = vaux 
    particle.w = waux
    if (particle.u == 0 and particle.v == 0 and particle.w == 0):
        particle.delete()
        

def BrownianDiffusion(particle, fieldset, time):
    #GeneralParameters
    pi = math.acos(-1.0)
    rads = math.pi / 180
    degrees = 180 / math.pi
    rearth = 6371000
    h=particle.dt
    #Parameters Diffusion
    Dlon=7.25 #LATITUDE  m2/s Eddy diffusivity 0.0002055*l^1.15, l=12000m, 12km (Okubo's formula)
    Dlat=7.25 #m²/s
    Dv= 0.00001 #m2/s
    
    vlon=math.sqrt(2*Dlon*h) 
    vlat=math.sqrt(2*Dlat*h)  
    vxDIF=degrees*(vlon/(rearth*cos(rads*particle.lat)))
    vyDIF=degrees*(vlat/rearth)
    vzDIF=math.sqrt(2*Dv*h)
    
    particle.lon += random.uniform(-1., 1.)*vxDIF
    particle.lat += random.uniform(-1., 1.)*vyDIF
    particle.depth += random.uniform(-1., 1.)*vzDIF

        

def MaxeyRiley3(particle, fieldset, time):
    """Kernel for Inertial RK4 particles advection"""
    
    #Parameters to choose
    a=0.00005   
    beta=particle.density
    #General parameters
    Omega = 7.2921e-5
    g=9.80665  #m/s2
    viscosity=0.00000115  #m2/s
    stokes_time=(a*a)/(3*beta*viscosity)  #s
    settling_velocity=(1-beta)*g*stokes_time  #m/s
    
    (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
    w1=w1+settling_velocity
    lon1 = particle.lon + u1*.5*particle.dt
    lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    
    (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]
    w2=w2+settling_velocity
    lon2 = particle.lon + u2*.5*particle.dt
    lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    
    (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]
    w3=w3+settling_velocity
    lon3 = particle.lon + u3*particle.dt
    lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]
    w4=w4+settling_velocity
    
    
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt
    


def MaxeyRiley2(particle, fieldset, time):
    """Kernel for Inertial RK4 particles advection"""
    
    #Parameters to choose
    a=0.00005    
    beta=particle.density
    #General parameters
    Omega = 7.2921e-5
    g=9.80665  #m/s2
    viscosity=0.00000115  #m2/s
    stokes_time=(a*a)/(3*beta*viscosity)  #s
    settling_velocity=(1-beta)*g*stokes_time  #m/s
    pi = acos(-1.0)
    rads = math.pi / 180
    #Parameter MaterialDerivative
    Ddegree=0.001
    Ddepth=0.2
    Dlat=Ddegree                               
    Dlon=Ddegree 
    Dt=particle.dt


    #1
    lon0=particle.lon
    lat0=particle.lat
    dep0=particle.depth
    (u1f, v1f, w1f) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
               
    dudx1=(fieldset.U[time, dep0, lat0, lon0+Dlon]-fieldset.U[time, dep0, lat0, lon0-Dlon])/(2*Dlon)
    dudy1=(fieldset.U[time, dep0, lat0+Dlat, lon0]-fieldset.U[time, dep0, lat0-Dlat, lon0])/(2*Dlat)
    dudz1=(fieldset.U[time, dep0-Ddepth, lat0, lon0]-fieldset.U[time, dep0+Ddepth, lat0, lon0])/(2*Ddepth)
    dudt1=(fieldset.U[time+Dt, dep0, lat0, lon0]-fieldset.U[time, dep0, lat0, lon0])/(Dt)
    
    dvdx1=(fieldset.V[time, dep0, lat0, lon0+Dlon]-fieldset.V[time, dep0, lat0, lon0-Dlon])/(2*Dlon)
    dvdy1=(fieldset.V[time, dep0, lat0+Dlat, lon0]-fieldset.V[time, dep0, lat0-Dlat, lon0])/(2*Dlat)
    dvdz1=(fieldset.V[time, dep0-Ddepth, lat0, lon0]-fieldset.V[time, dep0+Ddepth, lat0, lon0])/(2*Ddepth)
    dvdt1=(fieldset.V[time+Dt, dep0, lat0, lon0]-fieldset.V[time, dep0, lat0, lon0])/(Dt)
    
    dwdx1=(fieldset.W[time, dep0, lat0, lon0+Dlon]-fieldset.W[time, dep0, lat0, lon0-Dlon])/(2*Dlon)
    dwdy1=(fieldset.W[time, dep0, lat0+Dlat, lon0]-fieldset.W[time, dep0, lat0-Dlat, lon0])/(2*Dlat)
    dwdz1=(fieldset.W[time, dep0-Ddepth, lat0, lon0]-fieldset.W[time, dep0+Ddepth, lat0, lon0])/(2*Ddepth)
    dwdt1=(fieldset.W[time+Dt, dep0, lat0, lon0]-fieldset.W[time, dep0, lat0, lon0])/(Dt)
   
    MatDevu1=dudt1+u1f*dudx1+v1f*dudy1+w1f*dudz1
    MatDevv1=dvdt1+u1f*dvdx1+v1f*dvdy1+w1f*dvdz1
    MatDevw1=dwdt1+u1f*dwdx1+v1f*dwdy1+w1f*dwdz1
    
                       
    u1=u1f+(stokes_time*(beta-1)*(MatDevu1-(v1f*sin(rads*lat0)-w1f*math.cos(rads*lat0)))) 
    v1=v1f+(stokes_time*(beta-1)*(MatDevv1-(-u1f*sin(rads*lat0)))) 
    w1=w1f+settling_velocity+(stokes_time*(beta-1)*(MatDevw1-(u1f*math.cos(rads*lat0))))
    lon1 = particle.lon + u1*.5*particle.dt
    lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    
                       
    #2
    (u2f, v2f, w2f) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]
                       
                       
    dudx2=(fieldset.U[time, dep1, lat1, lon1+Dlon]-fieldset.U[time, dep1, lat1, lon1-Dlon])/(2*Dlon)
    dudy2=(fieldset.U[time, dep1, lat1+Dlat, lon1]-fieldset.U[time, dep1, lat1-Dlat, lon1])/(2*Dlat)
    dudz2=(fieldset.U[time, dep1-Ddepth, lat1, lon1]-fieldset.U[time, dep1+Ddepth, lat1, lon1])/(2*Ddepth)
    dudt2=(fieldset.U[time+Dt, dep1, lat1, lon1]-fieldset.U[time, dep1, lat1, lon1])/(Dt)
    
    dvdx2=(fieldset.V[time, dep1, lat1, lon1+Dlon]-fieldset.V[time, dep1, lat1, lon1-Dlon])/(2*Dlon)
    dvdy2=(fieldset.V[time, dep1, lat1+Dlat, lon1]-fieldset.V[time, dep1, lat1-Dlat, lon1])/(2*Dlat)
    dvdz2=(fieldset.V[time, dep1-Ddepth, lat1, lon1]-fieldset.V[time, dep1+Ddepth, lat1, lon1])/(2*Ddepth)
    dvdt2=(fieldset.V[time+Dt, dep1, lat1, lon1]-fieldset.V[time, dep1, lat1, lon1])/(Dt)
    
    dwdx2=(fieldset.W[time, dep1, lat1, lon1+Dlon]-fieldset.W[time, dep1, lat1, lon1-Dlon])/(2*Dlon)
    dwdy2=(fieldset.W[time, dep1, lat1+Dlat, lon1]-fieldset.W[time, dep1, lat1-Dlat, lon1])/(2*Dlat)
    dwdz2=(fieldset.W[time, dep1-Ddepth, lat1, lon1]-fieldset.W[time, dep1+Ddepth, lat1, lon1])/(2*Ddepth)
    dwdt2=(fieldset.W[time+Dt, dep1, lat1, lon1]-fieldset.W[time, dep1, lat1, lon1])/(Dt)
    
    MatDevu2=dudt2+u2f*dudx2+v2f*dudy2+w2f*dudz2
    MatDevv2=dvdt2+u2f*dvdx2+v2f*dvdy2+w2f*dvdz2
    MatDevw2=dwdt2+u2f*dwdx2+v2f*dwdy2+w2f*dwdz2
                       
    u2=u2f+(stokes_time*(beta-1)*(MatDevu2-(v2f*sin(rads*lat1)-w2f*math.cos(rads*lat1)))) 
    v2=v2f+(stokes_time*(beta-1)*(MatDevv2-(-u2f*sin(rads*lat1)))) 
    w2=w2f+settling_velocity+(stokes_time*(beta-1)*(MatDevw2-(u2f*math.cos(rads*lat1))))                     
                       
    lon2 = particle.lon + u2*.5*particle.dt
    lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    
                       
    #3                 
    (u3f, v3f, w3f) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]
    
    
    dudx3=(fieldset.U[time, dep2, lat2, lon2+Dlon]-fieldset.U[time, dep2, lat2, lon2-Dlon])/(2*Dlon)
    dudy3=(fieldset.U[time, dep2, lat2+Dlat, lon2]-fieldset.U[time, dep2, lat2-Dlat, lon2])/(2*Dlat)
    dudz3=(fieldset.U[time, dep2-Ddepth, lat2, lon2]-fieldset.U[time, dep2+Ddepth, lat2, lon2])/(2*Ddepth)
    dudt3=(fieldset.U[time+Dt, dep2, lat2, lon2]-fieldset.U[time, dep2, lat2, lon2])/(Dt)
    
    dvdx3=(fieldset.V[time, dep2, lat2, lon2+Dlon]-fieldset.V[time, dep2, lat2, lon2-Dlon])/(2*Dlon)
    dvdy3=(fieldset.V[time, dep2, lat2+Dlat, lon2]-fieldset.V[time, dep2, lat2-Dlat, lon2])/(2*Dlat)
    dvdz3=(fieldset.V[time, dep2-Ddepth, lat2, lon2]-fieldset.V[time, dep2+Ddepth, lat2, lon2])/(2*Ddepth)
    dvdt3=(fieldset.V[time+Dt, dep2, lat2, lon2]-fieldset.V[time, dep2, lat2, lon2])/(Dt)
    
    dwdx3=(fieldset.W[time, dep2, lat2, lon2+Dlon]-fieldset.W[time, dep2, lat2, lon2-Dlon])/(2*Dlon)
    dwdy3=(fieldset.W[time, dep2, lat2+Dlat, lon2]-fieldset.W[time, dep2, lat2-Dlat, lon2])/(2*Dlat)
    dwdz3=(fieldset.W[time, dep2-Ddepth, lat2, lon2]-fieldset.W[time, dep2+Ddepth, lat2, lon2])/(2*Ddepth)
    dwdt3=(fieldset.W[time+Dt, dep2, lat2, lon2]-fieldset.W[time, dep2, lat2, lon2])/(Dt)
    
    MatDevu3=dudt3+u3f*dudx3+v3f*dudy3+w3f*dudz3
    MatDevv3=dvdt3+u3f*dvdx3+v3f*dvdy3+w3f*dvdz3
    MatDevw3=dwdt3+u3f*dwdx3+v3f*dwdy3+w3f*dwdz3
                       
    u3=u3f+(stokes_time*(beta-1)*(MatDevu3-(v3f*sin(rads*lat2)-w3f*math.cos(rads*lat2)))) 
    v3=v3f+(stokes_time*(beta-1)*(MatDevv3-(-u3f*sin(rads*lat2)))) 
    w3=w3f+settling_velocity+(stokes_time*(beta-1)*(MatDevw3-(u3f*math.cos(rads*lat2))))                                     

    lon3 = particle.lon + u3*particle.dt
    lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    
                       
                       
    #4
    (u4f, v4f, w4f) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3] 
        
    dudx4=(fieldset.U[time, dep3, lat3, lon3+Dlon]-fieldset.U[time, dep3, lat3, lon3-Dlon])/(2*Dlon)
    dudy4=(fieldset.U[time, dep3, lat3+Dlat, lon3]-fieldset.U[time, dep3, lat3-Dlat, lon3])/(2*Dlat)
    dudz4=(fieldset.U[time, dep3-Ddepth, lat3, lon3]-fieldset.U[time, dep3+Ddepth, lat3, lon3])/(2*Ddepth)
    dudt4=(fieldset.U[time+Dt, dep3, lat3, lon3]-fieldset.U[time, dep3, lat3, lon3])/(Dt)
    
    dvdx4=(fieldset.V[time, dep3, lat3, lon3+Dlon]-fieldset.V[time, dep3, lat3, lon3-Dlon])/(2*Dlon)
    dvdy4=(fieldset.V[time, dep3, lat3+Dlat, lon3]-fieldset.V[time, dep3, lat3-Dlat, lon3])/(2*Dlat)
    dvdz4=(fieldset.V[time, dep3-Ddepth, lat3, lon3]-fieldset.V[time, dep3+Ddepth, lat3, lon3])/(2*Ddepth)
    dvdt4=(fieldset.V[time+Dt, dep3, lat3, lon3]-fieldset.V[time, dep3, lat3, lon3])/(Dt)
    
    dwdx4=(fieldset.W[time, dep3, lat3, lon3+Dlon]-fieldset.W[time, dep3, lat3, lon3-Dlon])/(2*Dlon)
    dwdy4=(fieldset.W[time, dep3, lat3+Dlat, lon3]-fieldset.W[time, dep3, lat3-Dlat, lon3])/(2*Dlat)
    dwdz4=(fieldset.W[time, dep3-Ddepth, lat3, lon3]-fieldset.W[time, dep3+Ddepth, lat3, lon3])/(2*Ddepth)
    dwdt4=(fieldset.W[time+Dt, dep3, lat3, lon3]-fieldset.W[time, dep3, lat3, lon3])/(Dt)
    
    MatDevu4=dudt4+u4f*dudx4+v4f*dudy4+w4f*dudz4
    MatDevv4=dvdt4+u4f*dvdx4+v4f*dvdy4+w4f*dvdz4
    MatDevw4=dwdt4+u4f*dwdx4+v4f*dwdy4+w4f*dwdz4
                       
    u4=u4f+(stokes_time*(beta-1)*(MatDevu4-(v4f*sin(rads*lat3)-w4f*math.cos(rads*lat3)))) 
    v4=v4f+(stokes_time*(beta-1)*(MatDevv4-(-u4f*sin(rads*lat3)))) 
    w4=w4f+settling_velocity+(stokes_time*(beta-1)*(MatDevw4-(u4f*math.cos(rads*lat3))))  
    
    
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt
    

res='0083'
data_dir = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA%s-N006/' % res
ufiles = sorted(glob(data_dir+'means/ORCA%s-N06_200?????d05U.nc' % res))
vfiles = sorted(glob(data_dir+'means/ORCA%s-N06_200?????d05V.nc' % res))
wfiles = sorted(glob(data_dir+'means/ORCA%s-N06_200?????d05W.nc' % res))
taufiles = sorted(glob(data_dir+'means/ORCA%s-N06_200?????d05T.nc' % res))
mesh_mask = data_dir + 'domain/coordinates.nc'

filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles},
             'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles},
             'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': wfiles},
             'TAU': {'lon': mesh_mask, 'lat': mesh_mask, 'data': taufiles}}
                       
variables = {'U': 'uo',
             'V': 'vo',
             'W': 'wo',
             'TAU': 'taum'}
                       
dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
              'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
              'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
              'TAU': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}}




"""Find lon/lat indices for fieldset"""


initialgrid_mask = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/means/ORCA0083-N06_20000105d05U.nc'
mask = xr.open_dataset(initialgrid_mask, decode_times=False)
Lat, Lon = mask.variables['nav_lat'], mask.variables['nav_lon']

# extract lat/lon values (in degrees) to numpy arrays
latvals = Lat[:]; lonvals = Lon[:] 

iy_min, ix_min = getclosest_ij(latvals, lonvals, minlat, minlon)
iy_max, ix_max = getclosest_ij(latvals, lonvals, maxlat, maxlon)

#print('Index for latitude %s\xb0 = %s' % (minlat, iy_min))
#print('Index for latitude %s\xb0 = %s' % (maxlat, iy_max))
#print('Index for longitude %s\xb0 = %s' % (minlon, ix_min))
#print('Index for longitude %s\xb0 = %s' % (maxlon, ix_max))

indices = {'lon': range(ix_min, ix_max), 'lat': range(iy_min, iy_max)}  # 'depth': range(0, 2000)

fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=False, indices=indices)


data_random = np.loadtxt('random_p00125', dtype=np.str)
data = np.loadtxt('particles00125', dtype=np.str)
lon = data[:,0] 
lat = data[:,1] 
density = data_random[:]
initial_depth = 1
depths = initial_depth * np.ones(len(lon))
times = [datetime(year=2000,month=1,day=8)]*len(lon) 

simdays=100
pset = ParticleSet(fieldset=fieldset, pclass=ParticleVariables, lon=lon, lat=lat, depth=depths, time=times, density= density)

kernels = pset.Kernel(WindStress) + pset.Kernel(MaxeyRiley2) + pset.Kernel(BrownianDiffusion) + pset.Kernel(Out)

outdir = '/scratch/rebeca/output_S1/'
outfile = outdir+ 'sim1b.nc'


#Trajectory computation
pset.execute(kernels, runtime=timedelta(days=simdays), dt=timedelta(minutes=0.1), 
             output_file=pset.ParticleFile(name=outfile, outputdt=timedelta(hours=120)), 
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})


