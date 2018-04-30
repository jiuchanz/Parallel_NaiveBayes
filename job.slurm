#!/bin/bash 
#SBATCH --ntasks=32 
#SBATCH --time=01:00:00
#SCATCH --mem-per-cpu=8GB
#SBATCH --output=job.output
#SBATCH --error=job.error
#SBATCH --mail-user=jiuchanz@usc.edu
#SBATCH --mail-type=END

cd ~/ee451/Parallel_NaiveBayes/src_omp
echo -e "omp version running time in seconds:\n"
echo 1 thread:
./NaiveBayes360 1
echo 2 threads:
./NaiveBayes360	2
echo 4 threads:
./NaiveBayes360	4
echo 8 threads:
./NaiveBayes360	8
echo 16 threads:
./NaiveBayes360	16
echo 32 threads:
./NaiveBayes360 32

cd ~/ee451/Parallel_NaiveBayes/src_pthread
echo -e "\n\npthread version running time in seconds:\n"
echo 1 thread:
./NaiveBayes360 1
echo 2 threads:
./NaiveBayes360	2
echo 4 threads:
./NaiveBayes360	4
echo 8 threads:
./NaiveBayes360	8
echo 16 threads:
./NaiveBayes360	16
echo 32 threads:
./NaiveBayes360 32

source /usr/usc/openmpi/default/setup.sh
cd ~/ee451/Parallel_NaiveBayes/src_mpi
echo -e "\n\nmpi version running time in seconds:\n"
echo 1 threads:
mpirun -np 1 NaiveBayes360
echo 2 threads:
mpirun -np 2 NaiveBayes360
echo 4 threads:
mpirun -np 4 NaiveBayes360
echo 8 threads:
mpirun -np 8 NaiveBayes360
echo 16 threads:
mpirun -np 16 NaiveBayes360
echo 32 threads:
mpirun -np 32 NaiveBayes360











