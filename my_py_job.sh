#!/bin/bash
#SBATCH -J my_python_program
#SBATCH --mem=30G
#SBATCH -t 8:30:00

module load miniconda
conda init bash
conda activate py3_env
python -u MH_test.py
