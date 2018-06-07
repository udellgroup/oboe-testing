DATA_DIR=$1
MAX_PROCS=$2
NUMERIC_INDICES=0

export OMP_NUM_THREADS=1
ls ${DATA_DIR}*.csv | xargs -i --max-procs=${MAX_PROCS} bash -c \
"python lowrank-automl-restricted.py {} ${NUMERIC_INDICES}"
