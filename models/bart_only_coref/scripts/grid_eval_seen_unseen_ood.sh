
SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"

qsub -l h=stg-gpu25,virtual_gpu_free=8000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/new_datasets/in_domain_held_out_predict.txt \
        --path_output=../results/in_domain_held_out/predictions_in_domain_held_out_48epochs.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/train_seen_unseen_OOD_seed3_48/checkpoint-87000