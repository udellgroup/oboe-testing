time=`date +%Y%m%d%H%M`
mkdir -p "autosklearn_"${time}

DATA_DIR=<directory_to_preprocessed_datasets>
SAVE_DIR="autosklearn_"${time}

git log -n 1 --pretty=format:"%H" >> ${SAVE_DIR}/README.md

export OPENBLAS_NUM_THREADS=1
for i in 3 8 11 12 14 16 18 20 22 23 28 30 31 36 37 39 40 41 43 44 46 48 50 53 54 59 60 61 181 182 183 187 285 300 307 311 312 313 316 329 333 334 335 336 337 338 373 375 377 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 400 401 446 450 458 463 464 469 475 512 679 694 715 717 718 720; do python autosklearn_testing.py ${DATA_DIR}/dataset_${i}_features_and_labels.csv ${SAVE_DIR} 0 &>> ${SAVE_DIR}/log.txt; done
