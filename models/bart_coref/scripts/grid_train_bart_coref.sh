
SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"

qsub -l h=stg-gpu25,virtual_gpu_free=12000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_train_bart_coref.py \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --item2id=./item2id.json \
        --train_input_file=../data_object_special/simmc2_dials_dstc10_train_predict.txt \
        --train_target_file=../data_object_special/simmc2_dials_dstc10_train_target.txt  \
        --eval_input_file=../data_object_special/simmc2_dials_dstc10_dev_predict.txt \
        --eval_target_file=../data_object_special/simmc2_dials_dstc10_dev_target.txt \
        --output_dir=../coref_models/model_coref \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/model_coref/report.txt \
        --num_train_epochs=8  \
        --eval_steps=3000  \
        --warmup_steps=8000 \
