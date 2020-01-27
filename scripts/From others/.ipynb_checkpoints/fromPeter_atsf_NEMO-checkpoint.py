# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:31:22 2017

@author: nooteboom
"""

from parcels import (FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D,
                     ErrorCode, ParticleFile, Variable)
from datetime import timedelta as delta
from datetime import  datetime
import numpy as np
import math
from glob import glob
import sys

dirread_pal = '/projects/0/palaeo-parcels/NEMOdata/'
dirread_top = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/'
dirread_top_bgc = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA_BGC/ORCA0083-N006/'

sp = 11. #The sinkspeed m/day
dd = 10. #The dwelling depth

dirwrite = '/projects/0/palaeo-parcels/NEMOres/atsf/particlefiles/global_sp%d_dd%d/'%(int(sp),int(dd))

posidx = int(sys.argv[1]) #ID of the file to define latitude and longitude ranges

latsz = np.load(dirread_pal + 'releaselocations/18cores/lats_id%d_dd%d.npy'%(posidx,int(dd)))
lonsz = np.load(dirread_pal + 'releaselocations/18cores/lons_id%d_dd%d.npy'%(posidx,int(dd)))

dep = dd * np.ones(latsz.shape)

times = np.array([datetime(2009, 12, 25) - delta(days=x) for x in range(0,int(365*8+1+30*10),9)])
time = np.empty(shape=(0));lons = np.empty(shape=(0));lats = np.empty(shape=(0));
for i in range(len(times)):
    lons = np.append(lons,lonsz)
    lats = np.append(lats, latsz)
    time = np.append(time, np.full(len(lonsz),times[i])) 
#%%
def set_nemo_fieldset(ufiles, vfiles, wfiles, tfiles, pfiles, dfiles, ifiles, bfile, mesh_mask='/projects/0/palaeo-parcels/NEMOdata/domain/coordinates.nc'):#dirread_top+'domain/coordinates.nc'):#
    filenames = { 'U': {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [ufiles[0]],
                        'data':ufiles},
                'V' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [ufiles[0]],
                        'data':vfiles},
                'W' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [wfiles[0]],
                        'data':wfiles},  
                'S' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [tfiles[0]],
                        'data':tfiles},   
                'T' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [tfiles[0]],
                        'data':tfiles}   ,   
 #                'O2':{'lon': mesh_mask,
 #                       'lat': mesh_mask,
 #                       'depth': [pfiles[0]],
 #                       'data':tfiles},
                 'NO3':{'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [pfiles[0]],
                        'data':pfiles},                 
                 'PP':{'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [dfiles[0]],
                        'data':dfiles}, 
                 'ICE':{'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [ifiles[0]],
                        'data':ifiles},                         
                'B' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [bfile[0]],
                        'data':bfile}#,      
                }
    if mesh_mask:
        filenames['mesh_mask'] = mesh_mask
    variables = {'U': 'uo',
                 'V': 'vo',
                 'W': 'wo',
                 'T': 'sst',
                 'S': 'sss',
#                 'O2': 'OXY',
                 'NO3': 'DIN',                 
                 'PP': 'TPP3',                
                 'ICE': 'sit',
                 'B':'Bathymetry'}

    dimensions = {'U':{'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},#
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},#
                    'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},#
                    'T': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                    'S': {'lon': 'glamf', 'lat': 'gphif',  'time': 'time_counter'},
#                    'O2': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'deptht', 'time': 'time_counter'},
                    'NO3': {'lon': 'glamf', 'lat': 'gphif',  'time': 'time_counter'},
                    'PP': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                    'ICE': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                    'B': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}}
                    
    latsmin = np.arange(-76,82,9)
    latsmax = np.arange(-67,90,9)
    #Chaning the lon and lat you must also do this within the kernels    
    minlat = latsmin[posidx];maxlat = latsmax[posidx]      

    latrange = 30
        
    if(latsmin[posidx]<40):
        latminind = max(0,(minlat-latrange+77)*20)
        latmaxind = min(3059,(maxlat+latrange+77)*20)    
    else: #Above 40 degrees North the grid curves
        latminind = max(0,(minlat-latrange+77)*20)
        latmaxind = 3059    
    latind = range(latminind, latmaxind)
    indices = {'lat': latind} 

    if mesh_mask:
        fieldset = FieldSet.from_nemo(filenames, variables, dimensions, indices = indices, allow_time_extrapolation=False)
        fieldset.U.vmax = 10
        fieldset.V.vmax = 10
        fieldset.W.vmax = 10        
        return fieldset
    else:
        filenames.pop('B')
        variables.pop('B')
        dimensions.pop('B') 
        fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, indices = indices, allow_time_extrapolation=False)
        fieldset.U.vmax = 10
        fieldset.V.vmax = 10
        fieldset.W.vmax = 10   
        return fieldset
        

def periodicBC(particle, fieldSet, time, dt):
    if particle.lon > 180:
        particle.lon -= 360        
    if particle.lon < -180:
        particle.lon += 360   
        
#Sink Kernel if only atsf is saved:
def Sink(particle, fieldset, time, dt):
    if(particle.depth>fieldset.dwellingdepth):
        particle.depth = particle.depth + fieldset.sinkspeed * dt
    elif(particle.depth<=fieldset.dwellingdepth and particle.depth>1):
        particle.depth = fieldset.surface
        particle.temp = fieldset.T[time+dt, particle.lon, particle.lat, fieldset.surface]
        particle.salin = fieldset.S[time+dt, particle.lon, particle.lat, fieldset.surface]
        particle.PP = fieldset.PP[time+dt, particle.lon, particle.lat, fieldset.surface]
        particle.NO3 = fieldset.NO3[time+dt, particle.lon, particle.lat, fieldset.surface]
        particle.ICE = fieldset.ICE[time+dt, particle.lon, particle.lat, fieldset.surface]   
        particle.delete()        

def SampleSurf(particle, fieldset, time, dt):
    particle.temp = fieldset.T[time+dt, particle.lon, particle.lat, fieldset.surface]
    particle.salin = fieldset.S[time+dt, particle.lon, particle.lat, fieldset.surface]               

def Age(particle, fieldset, time, dt):
    particle.age = particle.age + math.fabs(dt)  

def DeleteParticle(particle, fieldset, time, dt):
    particle.delete()

def initials(particle, fieldset, time, dt):
    if particle.age==0.:
        particle.depth = fieldset.B[time+dt, particle.lon, particle.lat, particle.depth]
        if(particle.depth  > 5800.):
            particle.age = (particle.depth - 5799.)*fieldset.sinkspeed
            particle.depth = 5799.        
        particle.lon0 = particle.lon
        particle.lat0 = particle.lat
        particle.depth0 = particle.depth
        
def FirstParticle(particle, fieldset, time, dt):
    particle.lon = particle.lon0
    particle.lat = particle.lat0
    particle.depth = fieldset.dwellingdepth   

def run_corefootprintparticles(dirwrite,outfile,lonss,latss,dep):
    ufiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200?????d05U.nc'))
    vfiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200?????d05V.nc'))
    wfiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200?????d05W.nc'))    
    tfiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200?????d05T.nc'))
    pfiles = sorted(glob(dirread_top_bgc + 'means/ORCA0083-N06_200?????d05P.nc'))    
    dfiles = sorted(glob(dirread_top_bgc + 'means/ORCA0083-N06_200?????d05D.nc'))    
    ifiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200?????d05I.nc'))     
    bfile = dirread_top+'domain/bathymetry_ORCA12_V3.3.nc'

    fieldset = set_nemo_fieldset(ufiles, vfiles, wfiles, tfiles, pfiles, dfiles, ifiles, bfile, dirread_pal + 'domain/coordinates.nc')    
    fieldset.B.allow_time_extrapolation = True
       
    fieldset.add_constant('dwellingdepth', np.float(dd))
    fieldset.add_constant('sinkspeed', sp/86400.)
    fieldset.add_constant('maxage', 300000.*86400)
    fieldset.add_constant('surface', 2.5)

    class DinoParticle(JITParticle):
        temp = Variable('temp', dtype=np.float32, initial=np.nan)
        age = Variable('age', dtype=np.float32, initial=0.)
        salin = Variable('salin', dtype=np.float32, initial=np.nan)
        lon0 = Variable('lon0', dtype=np.float32, initial=0.)
        lat0 = Variable('lat0', dtype=np.float32, initial=0.)
        depth0 = Variable('depth0',dtype=np.float32, initial=0.) 
        PP = Variable('PP',dtype=np.float32, initial=np.nan)
#        O2 = Variable('O2',dtype=np.float32, initial=np.nan)
        NO3 = Variable('NO3',dtype=np.float32, initial=np.nan)
        ICE = Variable('ICE',dtype=np.float32, initial=np.nan)
        
    pset = ParticleSet.from_list(fieldset=fieldset, pclass=DinoParticle, lon=lonss.tolist(), lat=latss.tolist(), 
                       time = time)

    # I use 'write on delete' here! So no information along the trajectory is written in the output
    pfile = ParticleFile(dirwrite + outfile, pset, write_ondelete=True)

    kernels = pset.Kernel(initials) + Sink  + pset.Kernel(AdvectionRK4_3D) + Age + periodicBC  

    pset.execute(kernels, runtime=delta(days=2170), dt=delta(minutes=-5), output_file=pfile, verbose_progress=False, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

    print('Execution finished')

outfile = "grid_id"+str(posidx)+'_dd'+str(int(dd)) +'_sp'+str(int(sp))+"_res"+str(res)
run_corefootprintparticles(dirwrite,outfile,lons,lats,dep)

