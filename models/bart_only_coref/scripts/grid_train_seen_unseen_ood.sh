
SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"

qsub -l h=stg-gpu25,virtual_gpu_free=12000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_train_bart_coref.py \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --item2id=./item2id.json \
        --train_input_file=../data_object_special/new_datasets/train_seen_unseen_OOD_predict.txt \
        --train_target_file=../data_object_special/new_datasets/train_seen_unseen_OOD_target.txt  \
        --eval_input_file=../data_object_special/new_datasets/in_domain_predict.txt \
        --eval_target_file=../data_object_special/new_datasets/in_domain_target.txt \
        --output_dir=../coref_models/train_seen_unseen_OOD_seed3_48 \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/train_seen_unseen_OOD_seed3_48/report.txt \
        --num_train_epochs=12  \
        --eval_steps=3000  \
        --warmup_steps=8000 \
        --seed 3