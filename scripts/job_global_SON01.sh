#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH -p fat             
#SBATCH -n 4        
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-type=fail        
#SBATCH --mail-user=d.m.a.lobelle@uu.nl

echo 'Initiating global run SON 2001...'

srun python global_Kooi+NEMO_3D.py -mon='09' -yr='2001'


echo 'Finished computation.'
