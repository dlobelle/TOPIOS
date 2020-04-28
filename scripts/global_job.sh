#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -n 40 

echo 'Initiating global run ...'

python global_Kooi+NEMO_3D.py

echo 'Finished computation.'
