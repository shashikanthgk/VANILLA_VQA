#!/bin/sh
#PBS -N VQA_WITH_ATTENTION
#PBS -P majorproject
#PBS -m bea
#PBS -M vrsandeep.181it151@nitk.edu.in
#PBS –l select=1:mem=64G:ncpus=6:ngpus=1
#PBS -l walltime=480:00:00
#PBS -j oe
#$PBS_O_WORKDIR="/home/181it151/MP/VANILLA_VQA"
echo "==============================="
echo $PBS_JOBID
#cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
#job
singularity exec --nv /home/181it151/local_sty_dkr_dlgpu2 python3 train.py --num_epochs 30 --batch_size 128 --model 'VWSA' --save_step 10

