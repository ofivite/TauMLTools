#!/usr/bin/env bash

set -x
#for file in $(ls output/training_preparation/testing/*.root) ; do
#   ./run.sh TrainingTupleProducer --input $file \
#            --output output/tuples-v2-t2/testing/$(basename $file) \
#            &> output/tuples-v2-t2/testing/$(basename $file).log &
#done

for file in $(ls output/tuples-v2-t2/testing/*.root) ; do
    f_name_ext=${file##*/}
    f_name=${f_name_ext%.*}
   python ./TauML/Analysis/python/root_to_hdf.py --input $file \
            --output output/tuples-v2-t2/testing/$f_name.h5 --trees taus,inner_cells,outer_cells \
            &> output/tuples-v2-t2/testing/${f_name}_hdf.log &
done


wait
