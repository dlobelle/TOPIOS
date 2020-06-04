#!/bin/bash
#SBATCH -t 1-00:00:00             
#SBATCH -n 1        
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-type=fail        
#SBATCH --mail-user=d.m.a.lobelle@uu.nl

echo 'Initiating global run JJA 2001...'

python global_Kooi+NEMO_3D.py -mon='06' -yr='2001'


echo 'Finished computation.'
