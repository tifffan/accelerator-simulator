#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=milano
#SBATCH --job-name=xopt-impact
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=100:30:00
#SBATCH --nodes=2

export PMIX_MCA_psec=^munge

/usr/lib64/openmpi/bin/mpirun python -m mpi4py.futures -m xopt.mpi.run xopt2_Low_Fidelity_Gaussian_Rotate_Constant_Settings_5.yaml
