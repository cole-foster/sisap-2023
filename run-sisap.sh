#!/bin/bash
#SBATCH -J S-HSP-10M
#SBATCH -o slurms/slurm-%j.out 
#SBATCH --account=carney-bkimia-condo
#SBATCH -p batch
### SBATCH -p bigmem
#SBATCH -C intel|skylake
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

module load anaconda/3-5.2.0
source activate ~/scratch/condas/faiss/

cd /users/cfoste18/data/cfoste18/SISAP2023/indexing-challenge/submission/sisap23-hnsw/

export PYTHONUNBUFFERED=TRUE

echo "Test: Closest Pivot in Layer"
echo python3 search/sisap-script.py --size 10M --M 20 --EF 400 --r1 0.65 --r2 0.45 -N 40
python3 search/sisap-script.py --size 10M --M 20 --EF 400 --r1 0.65 --r2 0.45 -N 40

source deactivate
