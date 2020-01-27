"""
TITLE
-------------------------------------------------------------------------

d.wichmann@uu.nl

##########################################################################

Code for the computation of 3D passive particle trajectories

"""

import numpy as np
from parcels import FieldSet, ParticleSet, JITParticle, ErrorCode, AdvectionRK4_3D
from argparse import ArgumentParser
from datetime import timedelta
from datetime import datetime
from glob import glob

datadir = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/'
outputdir = '/home/wichmann/example_delphine/'

def get_nemo():
    ufiles = sorted(glob(datadir+'means/ORCA0083-N06_2001????d05U.nc'))
    vfiles = sorted(glob(datadir+'means/ORCA0083-N06_2001????d05V.nc'))
    mesh_mask = datadir + 'domain/coordinates.nc'

    wfiles = sorted(glob(datadir+'means/ORCA0083-N06_2001????d05W.nc'))
    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles},
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles},
                 'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': wfiles}}
    variables = {'U': 'uo',
                 'V': 'vo',
                 'W': 'wo'}
    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}}

    fieldset = FieldSet.from_nemo(filenames, variables, dimensions)
    
    return fieldset

def DeleteParticle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    particle.delete()

def periodicBC(particle, fieldset, time):
    """
    Kernel for periodic values in longitude
    """
    if particle.lon < 0.:
        particle.lon += 360.
    elif particle.lon > 360.:
        particle.lon -= 360.

def surface_particle(particle, fieldset, time):
    if particle.depth < .5:
        particle.depth = .5
    
def p_advect(outname):
    """
    Main function for execution
        - outname: name of the output file. Note that all important parameters are also in the file name.
        - pos: Execution is manually parallelized over different initial position grids. These are indexed.
        - y, m, d: year, month an day of the simulation start
        - simdays: number of days to simulate
        - particledepth: for fixed-depth simulations. Index of nemo depth grid
    """
    
    print('-------------------------')
    print('Start run... Parameters: ')
    print('-------------------------')
    
    #Load initial particle positions (grids) from external files
    lons = [18.]
    lats = [36.]
    depths = [20.]
    
    times = [datetime(2001, 1, 5)]*len(lons)
    print('Number of particles: ', len(lons))
    
    fieldset = get_nemo()

    outfile = outputdir + 'example_delphine' + outname

    fieldset.U.vmax = 10
    fieldset.V.vmax = 10
    fieldset.W.vmax = 10

    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=lons, lat=lats, time=times, depth=depths)

    kernels= pset.Kernel(AdvectionRK4_3D) + pset.Kernel(periodicBC) + pset.Kernel(surface_particle)

    pset.execute(kernels, runtime=timedelta(days=30), dt=timedelta(minutes=10), 
                 output_file=pset.ParticleFile(name=outfile, outputdt=timedelta(days=1)),
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, verbose_progress=True)


if __name__=="__main__":
    p = ArgumentParser(description="""Global advection of different particles""")
    p.add_argument('-name', '--name', default='noname',help='name of output file')
    args = p.parse_args()
    p_advect(outname=args.name)
