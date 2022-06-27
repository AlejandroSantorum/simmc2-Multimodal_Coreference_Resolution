SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"

cd ../../

qsub -l h=stg-gpu15,virtual_gpu_free=12000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh train.py \
        --NAME UNITER_basic_all_objmen_noIDs_special_woman49 \
        --obj_id False \
        --vis_feats_clip True \
        --vis_feats_rcnn True \
        --pos True \
        --scene_seg True \
        --obj_embs_bert True \
        --obj_embs_sbert True \
        --kb_id_bert False \
        --kb_id_sbert False \
        --obj_men True \
        --SPLIT special_woman_store_train

cd ./sh/grid_train/
