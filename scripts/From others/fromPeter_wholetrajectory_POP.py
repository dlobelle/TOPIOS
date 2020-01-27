# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:31:22 2017

@author: nooteboom
"""

from parcels import (FieldSet,ParticleSet, JITParticle, AdvectionRK4_3D,
                     ErrorCode, ParticleFile, Variable, ScipyParticle)
from datetime import timedelta as delta
from datetime import  datetime
import numpy as np
import math
from glob import glob
import sys

dirread_POP = '/projects/0/samoc/pop/tx0.1/output/run_henk_mixedbc_extravars_nooteboom/'
mesh = '/projects/0/palaeo-parcels/POP/POPdata/mesh_0.1degree/'

def snapshotfunction(days):
    #print the year + month + day
    snapshots = np.array(['1234567']*len(days))
    months = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    for i in range(len(days)):
        day = days[i] + 366
        year = 320+day/365
        day = day%365 + 1
        bo = True
        if(day<=months[0]): month = 1; bo = False; 
        for m in range(1,len(months)):     
            if(day<=np.sum(months[:m+1]) and bo): day -= np.sum(months[:m]); month = m+1; bo = False;
        s = str(year) + format(month, '02d') + format(day, '02d')
        snapshots[i] = s
    return snapshots 

sp = 6. #The sinkspeed m/day
dd = 10. #The dwelling depth
res = 1 #resolution in degrees
tempres = int(sys.argv[1])#9  # the temporal resolution in days


dirwrite = '/projects/0/palaeo-parcels/POP/POPres/0.1degree/wholetraj/'

lonsz = np.array([200.-360.])#np.array([312.1-360])
latsz = np.array([-18.])#np.array([-37.15])

if(not lonsz.size):
    sys.exit("Only land in the run with this idx")

dep = dd * np.ones(latsz.shape)

times = np.array([datetime(1975, 1, 3) - delta(days=x) for x in range(0,365*5,3)])
time = np.empty(shape=(0));lons = np.empty(shape=(0));lats = np.empty(shape=(0));
for i in range(len(times)):
    lons = np.append(lons,lonsz)
    lats = np.append(lats, latsz)
    time = np.append(time, np.full(len(lonsz),times[i])) 
#%%
def set_fieldset(snapshots, hormesh, sfile):
    ufiles = [dirread_POP+'tavg/'+'t.t0.1_42l_nccs01.0' + s + '.nc' for s in snapshots]
    env_files = [dirread_POP+'movie/'+'m.t0.1_42l_nccs01.0' + s + '.nc' for s in snapshots]
    bfile = hormesh
    filenames = { 'U': {'lon': hormesh,
                        'lat': hormesh,
                        'depth': sfile,
                        'data':ufiles},
                'V' : {'lon': hormesh,
                        'lat': hormesh,
                        'depth': sfile,
                        'data':ufiles},
                'W' : {'lon': hormesh,
                        'lat': hormesh,
                        'depth': sfile,
                        'data':ufiles},  
                'S' : {'lon': hormesh,
                        'lat': hormesh,
 #                       'depth': sfile,
                        'data':env_files},   
                'T' : {'lon': hormesh,
                        'lat': hormesh,
#                        'depth': sfile,
                        'data':env_files},
                'B' : {'lon': hormesh,
                        'lat': hormesh,
                        'depth': sfile,
                        'data':hormesh}#,      
                }

    variables = {'U': 'UVEL',
                 'V': 'VVEL',
                 'W': 'WVEL',
                 'T': 'TEMP_5m',
                 'S': 'SALT_5m',
                 'B':'BOT_DEP'}

    dimensions = {'U':{'lon': 'U_LON_2D', 'lat': 'U_LAT_2D', 'depth': 'BOTTOM_GRIDCELL','time': 'time'},#
                  'V': {'lon': 'U_LON_2D', 'lat': 'U_LAT_2D', 'depth': 'BOTTOM_GRIDCELL','time': 'time'},#
                    'W': {'lon': 'U_LON_2D', 'lat': 'U_LAT_2D', 'depth': 'BOTTOM_GRIDCELL','time': 'time'},#
                    'T': {'lon': 'U_LON_2D', 'lat': 'U_LAT_2D', 'time':'time'},
                    'S': {'lon': 'U_LON_2D', 'lat': 'U_LAT_2D', 'time':'time'},
                    'B': {'lon': 'U_LON_2D', 'lat': 'U_LAT_2D'} }

#    indices = {'lon': range(50,850),'lat': range(0, 1000) }# for location near Uruguay
    # 
    indices = {'lon': range(2400,3500),'lat': range(650, 1300) }
    fieldset = FieldSet.from_pop(filenames, variables, dimensions, indices=indices, allow_time_extrapolation=False)

    print 'latitude and longitude ranges from the indices:' 
    print 'longitude: ',fieldset.U.lon[0], fieldset.U.lon[-1]
    print 'latitude: ',fieldset.U.lat[0], fieldset.U.lat[-1]
    print 'lon, lat:  ',lonsz[0],latsz[0]
    print 'depth at bottom: ', fieldset.B[0,0,latsz[0], lonsz[0]]

    fieldset.U.vmax = 10    # set max of flow to 10 m/s
    fieldset.V.vmax = 10
    fieldset.W.vmax = 10
    fieldset.T.vmin = -5
    return fieldset
        

def periodicBC(particle, fieldSet, time):
    if particle.lon > 360:
        particle.lon -= 360        
    if particle.lon < 0:
        particle.lon += 360   
        
#Sink Kernel if only atsf is saved:
def Sink(particle, fieldset, time):
    if(particle.depth>fieldset.dwellingdepth):
        particle.depth = particle.depth + fieldset.sinkspeed * particle.dt
        #print fieldset.T[time+particle.dt, particle.lon, particle.lat, fieldset.surface]
    elif(particle.depth<=fieldset.dwellingdepth and particle.depth>1):
        particle.depth = fieldset.surface
        particle.temp = fieldset.T[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.salin = fieldset.S[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.delete()        

def SampleSurf(particle, fieldset, time):
    particle.temp = fieldset.T[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
    particle.salin = fieldset.S[time+particle.dt, fieldset.surface, particle.lat, particle.lon]               

def Age(particle, fieldset, time):
    particle.age = particle.age + math.fabs(particle.dt)  

def DeleteParticle(particle, fieldset, time):
    particle.delete()

def initials(particle, fieldset, time):
    if particle.age==0.:
        particle.depth = fieldset.B[time, particle.depth, particle.lat, particle.lon] - 1
        if(particle.depth  > 5370.):
            particle.age = (particle.depth - 5370.)*fieldset.sinkspeed
            particle.depth = 5370.        
        particle.lon0 = particle.lon
        particle.lat0 = particle.lat
        particle.depth0 = particle.depth
        
def FirstParticle(particle, fieldset, time):
    particle.lon = particle.lon0
    particle.lat = particle.lat0
    particle.depth = fieldset.dwellingdepth   

def run_corefootprintparticles(dirwrite,outfile,lonss,latss,dep):
    snapshots = snapshotfunction(range(0,365*5+8,tempres))  #max is 365*5+7
    hormesh = mesh + 'grid_coordinates_pop_tx0.1.nc'
    Sdepth = mesh + 'bottom_cell.nc'
    fieldset = set_fieldset(snapshots, hormesh, Sdepth)    
    fieldset.B.allow_time_extrapolation = True
       
    fieldset.add_constant('dwellingdepth', np.float(dd))
    fieldset.add_constant('sinkspeed', sp/86400.)
    fieldset.add_constant('maxage', 300000.*86400)
    fieldset.add_constant('surface', 5.00622)

    class DinoParticle(JITParticle):#JITParticle):
        temp = Variable('temp', dtype=np.float32, initial=np.nan)
        age = Variable('age', dtype=np.float32, initial=0.)
        salin = Variable('salin', dtype=np.float32, initial=np.nan)
        lon0 = Variable('lon0', dtype=np.float32, initial=0.)
        lat0 = Variable('lat0', dtype=np.float32, initial=0.)
        depth0 = Variable('depth0',dtype=np.float32, initial=0.) 

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=DinoParticle, lon=lonss.tolist(), lat=latss.tolist(), 
                       time = time)


    pfile = ParticleFile(dirwrite + outfile, pset, outputdt=delta(days=3))

    kernels = pset.Kernel(initials) + Sink  + pset.Kernel(AdvectionRK4_3D) + Age #+ periodicBC  

    pset.execute(kernels, runtime=delta(days=365*5), dt=delta(minutes=-5), 
        output_file=pfile, verbose_progress=True, 
        recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

    print 'Execution finished'

outfile = "oneloc"+'_lat'+str(latsz[0])+'_lon'+str(lonsz[0])+'_dd'+str(int(dd)) +'_sp'+str(int(sp)) +'_tempres'+str(tempres)
run_corefootprintparticles(dirwrite,outfile,lons,lats,dep)

