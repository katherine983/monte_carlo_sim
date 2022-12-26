#!/bin/bash
#SBATCH --partition=kamiak	# Partition/Queue to use
#SBATCH --job-name=Order2Alph4nSim100	# Job name
#SBATCH --output='outputs/Order2Alph4nSim100_%j.out'	# Output file (stdout)
#SBATCH --error='errors/Order2Alph4nSim100_%j.err'	# Error file (stderr)

#SBATCH --mail-type=ALL	# Email notification: BEGIN,END,FAIL,ALL
#SBATCH --mail-user=katherine.graham@wsu.edu	# Email address for notifications
#SBATCH --array=0-3:1		# Number of jobs, in steps of 1

#SBATCH --nodes=1		# Number of nodes (min-max)
#SBATCH --ntasks-per-node=1	# Number of tasks per node (max)
#SBATCH --cpus-per-task=10	# Number of cores per task (threads)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Starting job array $SLURM_ARRAY_TASK_ID for $SLURM_ARRAY_JOB_ID running on nodes $SLURM_JOB_NODELIST"
export myscratch="$(mkworkspace)"
echo "Storing simulation output in $myscratch"
module load miniconda3		# Load software module from Kamiak repository
source activate mcenv
srun -l python main.py 1000 --nobs 500 1000 2500 5000 -d markov -o $myscratch --jobarray $SLURM_ARRAY_TASK_ID --markovname Order2Alph4
source deactivate
echo "Completed job on node $HOSTNAME"

#==== END OF FILE
