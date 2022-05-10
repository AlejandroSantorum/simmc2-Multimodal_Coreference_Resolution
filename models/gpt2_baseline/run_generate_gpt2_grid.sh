#!/bin/bash

if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"

qsub -l h=!stg-gpu5\&!stg-gpu6\&!stg-gpu7\&!stg-gpu8\&!stg-gpu9,virtual_gpu_free=4000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh gpt2_dst/scripts/run_generation.py \
        --model_type=gpt2 \
        --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/model/ \
        --num_return_sequences=1 \
        --length=100 \
        --stop_token='<EOS>' \
        --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output="${PATH_DIR}"/gpt2_dst/results/simmc2_dials_dstc10_devtest_predicted.txt
