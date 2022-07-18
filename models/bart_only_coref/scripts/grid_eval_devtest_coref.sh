
SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"

qsub -l h=stg-gpu25,virtual_gpu_free=8000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output=../results/n_target_objs_head/predictions_num_targets_input_non_visual_attrs.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/num_targets_non_visual_attrs/checkpoint-21000 \
        --input_attrs True \
        --num_objs_head True