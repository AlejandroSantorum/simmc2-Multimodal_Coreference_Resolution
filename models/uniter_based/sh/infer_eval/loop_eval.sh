
SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"
a=$(ls ../../result/checkpoint/UNITER_basic_all_objmen_noIDs_numTargetObjs_025)

for file in $a
do
echo "Copying..."
cp ../../result/checkpoint/UNITER_basic_all_objmen_noIDs_numTargetObjs_025/${file} ../../trained

cd ../../

qsub -l h=stg-gpu25,virtual_gpu_free=8000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh infer_eval.py \
        --NAME eval_${file} \
        --CHECKPOINT $file \
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
        --num_target_objs_head True

cd ./sh/grid_eval/
sleep 50s
done

echo "Done"