# Parallel_NaiveBayes
EE451 Final Project
Project members: Jiuchang Zhang, Jingwen Gong

We are optimizing naive bayesian learning by applying different parallel techniques including Pthread, MPI, OpenMP. We will then analyze the results. We used naive bayesian learning to classify handwritten digit from 0 to 9. We used the data set from MNIST database.

To run any job on HPC first do the following to allocate enough resources
  salloc --ntasks=32 --time=1:00:00 --mem-per-cpu=2GB
  
To run Pthread and Omp version on HPC, go to their src file folder
 1. cmake .
 2. make
 3. ./NaiveBayes360 #ofThreads
 
To run MPI on HPC, go to its src file foler
 1. cmake .
 2. make 
 3. mpirun -np #ofThreads NaiveBayes360
 
Alternatively, you can copy the slurm file in this project and push to the job queue, it already contains everything it need to execute for MPI, Pthread, and OMP for all different number of threads and ouput the results to the job.output
