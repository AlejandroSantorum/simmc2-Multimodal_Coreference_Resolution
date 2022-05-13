
SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"

qsub -l h=!stg-gpu5\&!stg-gpu6\&!stg-gpu7\&!stg-gpu8\&!stg-gpu9,virtual_gpu_free=4000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_bart_multi_task.py \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --item2id=./item2id.json \
        --train_input_file=../data_object_special/simmc2_dials_dstc10_train_predict.txt \
        --train_target_file=../data_object_special/simmc2_dials_dstc10_train_target.txt  \
        --disambiguation_file=../data_object_special/simmc2_dials_dstc10_train_disambiguation_label.txt \
        --response_file=../data_object_special/simmc2_dials_dstc10_train_response.txt \
        --eval_input_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --eval_target_file=../data_object_special/simmc2_dials_dstc10_devtest_target.txt \
        --output_dir=../multi_task/model \
        --train_batch_size=2 \
        --output_eval_file=../multi_task/model/report.txt \
        --num_train_epochs=10  \
        --eval_steps=3000  \
        --warmup_steps=10000 \
