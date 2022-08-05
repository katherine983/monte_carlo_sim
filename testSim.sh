#!/bin/bash
#SBATCH --partition=kamiak	# Partition/Queue to use
#SBATCH --job-name=testSim	# Job name
#SBATCH --output=testSim_%j.out	# Output file (stdout)
#SBATCH --error=testSim_%j.err	# Error file (stderr)
#SBATCH --time=1-00:00:00	# Wall clock time limit Days-HH:MM:SS
##SBATCH --mail-type=ALL	# Email notification: BEGIN,END,FAIL,ALL
##SBATCH --mail-user=katherine.graham@wsu.edu	# Email address for notifications

#SBATCH --nodes=1		# Number of nodes (min-max)
#SBATCH --ntasks-per-node=1	# Number of tasks per node (max)
#SBATCH --ntasks=1		# Number of tasks (processes)
#SBATCH --cpus-per-task=2	# Number of cores per task (threads)


echo "I am job $SLURM_JOBID running on nodes $SLURM_JOB_NODELIST"
export myscratch="$(mkworkspace -n $SLURM_JOBID)"
echo $myscratch
module load miniconda3		# Load software module from Kamiak repository
conda activate mcenv
srun -l python main.py 2 --nobs 200	-d uniform -o $myscratch # Each task runs this program (total 1 times)
				# Each srun is a job step, and spawns -ntasks
conda deactivate
echo "Completed job on node $HOSTNAME"

#==== END OF FILE
