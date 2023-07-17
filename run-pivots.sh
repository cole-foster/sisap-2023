#!/bin/bash
#SBATCH -J HNSW-PIVOTS
#SBATCH -o slurms/slurm-%j.out 
#SBATCH --account=carney-bkimia-condo
#SBATCH -p batch
### SBATCH -p bigmem
#SBATCH -C intel|skylake
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem=128G
#SBATCH --time=24:00:00

module load anaconda/3-5.2.0
source activate ~/scratch/condas/faiss/

cd /users/cfoste18/data/cfoste18/SISAP2023/indexing-challenge/submission/sisap23-hnsw/

export PYTHONUNBUFFERED=TRUE

echo python3 search/pivot-selection.py --size 10M --radius1 0.7 --radius2 0.45 --threads 32 --save 1 --k 10000
python3 search/pivot-selection.py --size 10M --radius1 0.7 --radius2 0.45 --threads 32 --save 1 --k 10000

source deactivate


