import os

loc = 'north_global'
r_pl = '1e-07'
rho_pl = '920'

for year in ['2001']: #['2001','2002','2003','2004','2005', '2006','2007', '2008', '2009','2010']:
    for mon in ['12']: #, '03', '06', '12']: 
        if mon=='12':
            s = 'DJF'
        elif mon=='03':
            s = 'MAM'
        elif mon=='06':
            s = 'JJA'
        elif mon=='09':
            s = 'SON'

        if loc=='north_global':
            l='N'
        elif loc=='south_global':
            l='S'
        elif loc=='eq_global':
            l='E'

        with open('jobscript', 'w') as f:
            f.write("#!/bin/bash \n")
            f.write("#SBATCH -t 1-00:00:00 \n")
            f.write("#SBATCH -n 1 \n")
            f.write("#SBATCH --job-name %s%s%s\n" %(l, year, s))
            f.write("#SBATCH --output %s%s%s\n" %(l, year, s))
            f.write("#SBATCH --mail-type=begin \n")
            f.write("#SBATCH --mail-type=end \n")     
            f.write("#SBATCH --mail-type=fail \n")       
            f.write("#SBATCH --mail-user=d.m.a.lobelle@uu.nl \n")
            f.write("echo 'Initiating global run %s  %s %s %s %s...'\n" % (loc, s, year, r_pl, rho_pl))
            f.write("srun python global_Kooi+NEMO_3D.py -loc='%s' -mon='%s' -yr='%s' -r_pl='%s' -rho_pl='%s'\n" % (loc, mon, year, r_pl, rho_pl))
            f.write("echo 'Finished computation.'\n")
            f.close()

        os.system('sbatch jobscript')

