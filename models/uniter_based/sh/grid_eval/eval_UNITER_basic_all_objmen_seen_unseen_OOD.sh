SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"

cd ../../

EVAL_NAME="eval_UNITER_basic_all_objmen_seen_unseen_OOD"
CHECKPOINT_NAME="UNITER_basic_all_objmen_seen_unseen_OOD"

qsub -l h=stg-gpu20,virtual_gpu_free=8000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh infer_eval.py \
        --NAME $EVAL_NAME \
        --CHECKPOINT $CHECKPOINT_NAME \
        --obj_id True \
        --vis_feats_clip True \
        --vis_feats_rcnn True \
        --pos True \
        --scene_seg True \
        --obj_embs_bert True \
        --obj_embs_sbert True \
        --kb_id_bert False \
        --kb_id_sbert False \
        --obj_men True \
        --SPLIT in_domain

qsub -l h=stg-gpu20,virtual_gpu_free=8000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh infer_eval.py \
        --NAME $EVAL_NAME \
        --CHECKPOINT $CHECKPOINT_NAME \
        --obj_id True \
        --vis_feats_clip True \
        --vis_feats_rcnn True \
        --pos True \
        --scene_seg True \
        --obj_embs_bert True \
        --obj_embs_sbert True \
        --kb_id_bert False \
        --kb_id_sbert False \
        --obj_men True \
        --SPLIT in_domain_held_out

qsub -l h=stg-gpu20,virtual_gpu_free=8000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh infer_eval.py \
        --NAME $EVAL_NAME \
        --CHECKPOINT $CHECKPOINT_NAME \
        --obj_id True \
        --vis_feats_clip True \
        --vis_feats_rcnn True \
        --pos True \
        --scene_seg True \
        --obj_embs_bert True \
        --obj_embs_sbert True \
        --kb_id_bert False \
        --kb_id_sbert False \
        --obj_men True \
        --SPLIT out_of_domain

cd ./sh/grid_eval/