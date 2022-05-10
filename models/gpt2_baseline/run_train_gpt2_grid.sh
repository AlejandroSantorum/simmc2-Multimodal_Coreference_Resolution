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
    ${SCRIPTS_PATH}run_on_grid.sh gpt2_dst/scripts/run_language_modeling.py \
        --output_dir="${PATH_DIR}"/gpt2_dst/save/model \
        --model_type=gpt2 \
        --model_name_or_path=gpt2 \
        --line_by_line \
        --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/simmc2_special_tokens.json \
        --do_train \
        --train_data_file="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_train_target.txt \
        --do_eval --eval_all_checkpoints \
        --eval_data_file="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_dev_target.txt \
        --num_train_epochs=2 \
        --overwrite_output_dir \
        --per_gpu_train_batch_size=4 \
        --per_gpu_eval_batch_size=4
