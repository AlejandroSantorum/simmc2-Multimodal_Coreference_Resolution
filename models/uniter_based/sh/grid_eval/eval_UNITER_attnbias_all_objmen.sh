SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"

cd ../../

qsub -l h=stg-gpu15,virtual_gpu_free=8000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh infer_eval.py \
        --NAME eval_UNITER_attnbias_all_objmen \
        --CHECKPOINT UNITER_attnbias_all_objmen \
        --obj_id True \
        --vis_feats_clip True \
        --vis_feats_rcnn True \
        --pos True \
        --scene_seg True \
        --obj_embs_bert True \
        --obj_embs_sbert True \
        --kb_id_bert False \
        --kb_id_sbert False \
        --attn_bias True \
        --obj_men True

qsub -l h=stg-gpu15,virtual_gpu_free=8000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh infer_eval.py \
        --NAME eval_UNITER_attnbias_all_objmen \
        --CHECKPOINT UNITER_attnbias_all_objmen \
        --obj_id True \
        --vis_feats_clip True \
        --vis_feats_rcnn True \
        --pos True \
        --scene_seg True \
        --obj_embs_bert True \
        --obj_embs_sbert True \
        --kb_id_bert False \
        --kb_id_sbert False \
        --attn_bias True \
        --obj_men True \
        --SPLIT dev

cd ./sh/grid_eval/