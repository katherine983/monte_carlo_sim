#!/bin/bash
#SBATCH --partition=cook	# Partition/Queue to use
#SBATCH --job-name=testSim	# Job name
#SBATCH --output='outputs/testSim_%j.out'	# Output file (stdout)
#SBATCH --error='errors/testSim_%j.err'	# Error file (stderr)
#SBATCH --time=1-00:00:00	# Wall clock time limit Days-HH:MM:SS
#SBATCH --mail-type=ALL	# Email notification: BEGIN,END,FAIL,ALL
#SBATCH --mail-user=katherine.graham@wsu.edu	# Email address for notifications
#SBATCH --array=0-5:1		# Number of jobs, in steps of 1

#SBATCH --nodes=1		# Number of nodes (min-max)
#SBATCH --ntasks-per-node=1	# Number of tasks per node (max)
#SBATCH --cpus-per-task=10	# Number of cores per task (threads)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Starting job array $SLURM_ARRAY_TASK_ID for $SLURM_ARRAY_JOB_ID running on nodes $SLURM_JOB_NODELIST"
export myscratch="/scratch/user/katherine.graham/20220822_195206"
echo "Output saved in: $myscratch"
module load miniconda3		# Load software module from Kamiak repository
source activate mcenv
srun -l python main.py 5 --nobs 200 1000 5000 7500 10000 20000 -d uniform -o $myscratch --jobarray $SLURM_ARRAY_TASK_ID
source deactivate
echo "Completed job on node $HOSTNAME"

#==== END OF FILE
