import os

for year in ['2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']:
    for mon in ['03', '06', '09', '12']:
        if mon=='12':
            s = 'DJF'
        elif mon=='03':
            s = 'MAM'
        elif mon=='06':
            s = 'JJA'
        elif mon=='09':
            s = 'SON'

        with open('jobscript', 'w') as f:
            f.write("#!/bin/bash \n")
            f.write("#SBATCH -t 5-00:00:00 \n")
            f.write("#SBATCH -p fat \n")
            f.write("#SBATCH -N 1 --ntasks-per-node=8\n")
            f.write("#SBATCH --job-name %s%s\n" %(year, s))
            f.write("#SBATCH --mail-type=begin \n")
            f.write("#SBATCH --mail-type=end \n")     
            f.write("#SBATCH --mail-type=fail \n")       
            f.write("#SBATCH --mail-user=d.m.a.lobelle@uu.nl \n")
            f.write("echo 'Initiating global run %s %s...'\n" % (s, year))
            f.write("srun python global_Kooi+NEMO_3D.py -mon='%s' -yr='%s'\n" % (mon, year))
            f.write("echo 'Finished computation.'\n")
            f.close()

        os.system('sbatch jobscript')

