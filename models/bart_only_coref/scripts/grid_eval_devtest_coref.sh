
SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"

qsub -l h=stg-gpu25,virtual_gpu_free=8000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/new_datasets/devtest_only_fashion_predict.txt \
        --path_output=../results/devtest_only_fashion/predictions_devtest_fashion.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_vanilla_no_heads_checking/checkpoint-57000