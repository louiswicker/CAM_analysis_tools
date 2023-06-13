#!/bin/bash

#SBATCH --partition=batch
#SBATCH -J squall
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH -t 00:30:00
#SBATCH --output=/scratch/wicker/MPAS/ideal/squall/squall_run_%j.log

#set -eux

time1=$(date '+%s')
echo "Job Started: $(date). Job Id:  $SLURM_JOBID"
echo " "

ulimit -s unlimited

# Declare an array of string with type - no commas
#declare -a CapeArray=("C1000" "C1500" "C2000" "C2500" "C3000" "C3500" )
#declare -a CapeArray=("C2000" "C3500" )
declare -a CapeArray=("C1500" "C2500" "C3000" )
declare -a ShearArray=("06" "12" "18")

prefix="squall_3km"

# Iterate the string array using for loop
for shear in ${ShearArray[@]}; do

  for cape in ${CapeArray[@]}; do

    cd /scratch/wicker/MPAS/ideal/squall

    export EXP=${prefix}_${cape}_${shear}

    cp convert_mpas.py $EXP/.

    cd $EXP

    python convert_mpas.py

  done

done

exit

#if [[ $? -eq 0 ]]; then
#    touch done.static
#else
#    touch error.static
#fi

time2=$(date '+%s')

let diff=time2-time1
let hour=diff/3600
let diff=diff%3600
let min=diff/60
let sec=diff%60

echo -n "Job   Ended: $(date). "
printf 'Job run time:  %02d:%02d:%02d' $hour $min $sec
echo " "
