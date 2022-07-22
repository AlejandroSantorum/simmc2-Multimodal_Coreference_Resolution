# UNITER-based model focused on MM Coreference Resolution

## Overview
The UNITER-based system here is a modified version of the one proposed in [*"UNITER-Based Situated Coreference Resolution with Rich Multimodal Input"*](https://arxiv.org/abs/2112.03521) by the New York University of Shanghai for the [DSTC10](https://sites.google.com/dstc.community/dstc10/home) competition. The code here is a modified version of the one publicly available in [Yichen Huang's repository](https://github.com/i-need-sleep/MMCoref_Cleaned).

Several improvements are investigated and they are explained further below.

## **Dataset**
The original SIMMC2.0 dataset is hosted in [Meta's GitHub Repository](https://github.com/facebookresearch/simmc2) with [Git LFS](https://git-lfs.github.com/). However, the data used in this model is preprocessed before even training. Please, download the data from [data.zip](https://drive.google.com/file/d/1Cn3GCth4RfuF5zBLfeH9zTLW7GIJ9dd4/view?usp=sharing) **and** from [processed.zip](https://drive.google.com/file/d/1FtZ7EopmZ0WX8I0WPz8yLVsI61LDG6H_/view?usp=sharing) and place it in the corresponding folder. The structure of the folders should be something similar as:
```
|-- data
|   |-- images                                      # scene images
|   |   |-- cloth_store_1_1_1.png
|   |   |-- cloth_store_1_1_2.png
|   |   `-- ...
|   |      
|   |-- jsons                                       # bbox and scene jsons
|   |   |-- cloth_store_1_1_1_bbox.json
|   |   |-- cloth_store_1_1_1_scene.json
|   |   `-- ...
|   |
|   |-- simmc2_dials_dstc10_dev.json                # dialogue data (dev)
|   |-- simmc2_dials_dstc10_devtest.json            # dialogue data (devtest)
|   `-- simmc2_dials_dstc10_train.json              # dialogue data (train)
|
`-- processed
    |-- dev.json
    |-- devtest.json
    |-- train.json
    |-- img_features.pt
    |-- KB_train.pt
    |-- KB_devtest.pt
    |-- ...
    |-- KB_emb.pt
    |-- KB_dict.json
    |-- ...
    |-- simmc2_scenes_dev.txt
    |-- simmc2_scenes_devtest.txt
    |-- simmc2_scenes_train.txt 
    `-- ...
```

**NOTE**: Some of the scene images are corrupted and therefore ignored. We do not make use of images in this model other than getting image size.
```
./data/images/cloth_store_1416238_woman_4_8.png
./data/images/cloth_store_1416238_woman_19_0.png
./data/images/cloth_store_1416238_woman_20_6.png
```


## **Preprocessed data**
The dataset is preprocessed to be fed into the UNITER-based model. The folder [`processed`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based/processed) contains the preprocessed files and they should be downloaded from [processed.zip](https://drive.google.com/file/d/1FtZ7EopmZ0WX8I0WPz8yLVsI61LDG6H_/view?usp=sharing), or from the [original repository](https://github.com/i-need-sleep/MMCoref_Cleaned/tree/main/processed).


## **Training**
The folders [`sh/train`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based/sh/train) and [`sh/grid_train`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based/sh/grid_train) contain scripts to train the UNITER-based model in your computer or in a GPU grid, respectively. 

As an example, we can train an UNITER-based model in our **own computer** by executing
```bash
    python -u train.py \
        --NAME UNITER_basic_all_objmen \
        --obj_id True \
        --vis_feats_clip True \
        --vis_feats_rcnn True \
        --pos True \
        --scene_seg True \
        --obj_embs_bert True \
        --obj_embs_sbert True \
        --kb_id_bert False \
        --kb_id_sbert False \
        --obj_men True
```
or in a **GPU grid** by:
```bash
    SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"

    cd ../../

    qsub -l h=stg-gpu15,virtual_gpu_free=12000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
        -e /rmt/dialogue2/interns/alejandro/logs \
        -o /rmt/dialogue2/interns/alejandro/logs \
        ${SCRIPTS_PATH}run_on_grid.sh train.py \
            --NAME UNITER_basic_all_objmen \
            --obj_id True \
            --vis_feats_clip True \
            --vis_feats_rcnn True \
            --pos True \
            --scene_seg True \
            --obj_embs_bert True \
            --obj_embs_sbert True \
            --kb_id_bert False \
            --kb_id_sbert False \
            --obj_men True

    cd ./sh/grid_train/
```

The trained model (checkpoint) can be found at TODO folder.

These are just some examples. For more detailed information visit the folder [`sh`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based/sh).


## **Evaluation**
The folders [`sh/infer_eval`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based/sh/infer_eval) and [`sh/grid_eval`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based/sh/grid_eval) contain scripts to train the UNITER-based model in your computer or in a GPU grid, respectively.

**NOTE**: Before evaluating a model, the fine-tuned checkpoint should be moved/copied from TODO folder to TODO folder.

As an example, we can evaluate an UNITER-based model in our **own computer** by executing
```bash
    python -u infer_eval.py \
        --NAME eval_UNITER_basic_all_objmen \
        --CHECKPOINT UNITER_basic_all_objmen \
        --obj_id True \
        --vis_feats_clip True \
        --vis_feats_rcnn True \
        --pos True \
        --scene_seg True \
        --obj_embs_bert True \
        --obj_embs_sbert True \
        --kb_id_bert False \
        --kb_id_sbert False \
        --obj_men True\
        --SPLIT devtest
```
or in a **GPU grid** by:
```bash
    SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"

    cd ../../

    qsub -l h=stg-gpu25,virtual_gpu_free=8000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
        -e /rmt/dialogue2/interns/alejandro/logs \
        -o /rmt/dialogue2/interns/alejandro/logs \
        ${SCRIPTS_PATH}run_on_grid.sh infer_eval.py \
            --NAME eval_UNITER_basic_all_objmen \
            --CHECKPOINT UNITER_basic_all_objmen \
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
            --SPLIT devtest

    cd ./sh/grid_eval/
```
These are just some examples. For more detailed information visit the folder [`sh`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based/sh).


## **Report on MM Coreference Resolution**
After generating the predictions, the report containing the **Object F1 score**, **precision** and **recall** is printed in the terminal (or the specified stdout, like for example a log file). Also, the F1 score can be obtained using the script provided in [`sh/get_report`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based/sh/get_report) folder by executing:

```bash
cd sh/get_report

python get_report.py \
    --MODEL_EVAL_FILE ../../output/logit/eval_UNITER_basic_both_objid_pos_sceneseg_bothBERT_devtest.json
```
This is an example. Please change the path for the predictions file using the argument `--MODEL_EVAL_FILE`.