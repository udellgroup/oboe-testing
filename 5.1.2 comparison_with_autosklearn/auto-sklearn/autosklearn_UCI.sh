time=`date +%Y%m%d%H%M`
mkdir -p "autosklearn_"${time}

DATA_DIR=<directory_to_preprocessed_datasets>
SAVE_DIR="autosklearn_"${time}


git log -n 1 --pretty=format:"%H" >> ${SAVE_DIR}/README.md

ls ${DATA_DIR}/*.csv | xargs -i --max-procs=1 bash -c \
"python autosklearn_testing.py {} ${SAVE_DIR} ${WHOLE_FILENAME} 1 &>> ${SAVE_DIR}/log.txt"

