set -x
work_path=$(dirname $0)
filename=$(basename $work_path)
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p $1 -n $2 --ntasks-per-node=${GPUS_PER_NODE} --cpus-per-task=12 \
--gres=gpu:${GPUS_PER_NODE} --quotatype=spot -x SH-IDC1-10-142-5-136 \
python -u tools/train.py $3 \
    --run-dir $4 \
    --auto-resume \
    --launcher slurm \
