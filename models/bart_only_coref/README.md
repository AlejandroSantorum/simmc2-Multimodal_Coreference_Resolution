# BART-based model focused on MM Coreference Resolution

## Overview
The BART-based system here is a modified version of the one proposed in [*"Tackling Situated Multi-Modal Task-Oriented Dialogs with a Single Transformer Model"*](https://openreview.net/forum?id=NajekV9uBas) by the KAIST-AIPR Laboratory for the [DSTC10](https://sites.google.com/dstc.community/dstc10/home) competition. The version included here just uses the task-specific head focused on MM Coreference Resolution.

Several improvements are investigated and they are explained further below.


## **Dataset**
The dataset is hosted in [Meta's GitHub Repository](https://github.com/facebookresearch/simmc2) with [Git LFS](https://git-lfs.github.com/).

Run the script `rearrange.sh` to rearrange the `data` folder in the following format.

```
|-- images                                                # scene images
|   |-- cloth_store_1_1_1.png
|   |-- cloth_store_1_1_2.png
|   `-- ...
|-- jsons                                                 # bbox and scene jsons
|   |-- cloth_store_1_1_1_bbox.json
|   |-- cloth_store_1_1_1_scene.json
|   `-- ...
|-- fashion_prefab_metadata_all.json                      # metadata (fashion)
|-- furniture_prefab_metadata_all.json                    # metadata (furniture)
|-- simmc2_dials_dstc10_dev.json                          # dialogue data (dev)
|-- simmc2_dials_dstc10_devtest.json                      # dialogue data (devtest)
|-- simmc2_dials_dstc10_train.json                        # dialogue data (train)
|-- simmc2_dials_dstc10_dev_retrieval_candidate.json      # retrieval data (dev)
`-- simmc2_dials_dstc10_devtest_retrieval_candidate.json  # retrieval data (devtest)
```

**NOTE**: Some of the scene images are corrupted and therefore ignored. We do not make use of images in this model other than getting image size.
```
./data/images/cloth_store_1416238_woman_4_8.png
./data/images/cloth_store_1416238_woman_19_0.png
./data/images/cloth_store_1416238_woman_20_6.png
```


## **Preprocessed data**
The dataset is preprocessed to be fed into the BART-based model. The folder [`data_object_special`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/data_object_special) contains the majority of the preprocessed files and they can be directly downloaded cloning this repository. However, some special data files for some experiments are too large. They can be generated using the suitable scripts detailed in [`processing_data`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/processing_data) folder, or simple by downloading the full version of this folder: [data_object_special.zip](https://drive.google.com/file/d/1LsnwUyt_ZG-e1OS-Hiud7ERvU8jpD4KA/view?usp=sharing).


### **Data Preprocessing**
For our model input, preprocess the datasets to reformat the data. 

Make sure to download simmc2-data into `./data` before launching the code .

1. Move into `scripts` (`cd scripts`), run the following command.
```bash
python convert.py \
  --input_path_json=<YOUR INPUT PATH JSON> \
  --output_path_predict=<YOUR OUTPATH PREDICT> \
  --output_path_target=<YOUR OUTPATH TARGET> \
  --object_special_token_item2id=item2id.json \
  --scene_json_folder=../data/jsons \
  --image_folder ../data/images \
```

For example, for devtest dataset:
```bash
python convert.py \
  --input_path_json=../data/simmc2_dials_dstc10_devtest.json \
  --output_path_predict=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
  --output_path_target=../data_object_special/simmc2_dials_dstc10_devtest_target.txt \
  --object_special_token_item2id=item2id.json \
  --scene_json_folder=../data/jsons  \
  --image_folder=../data/images 
```
For teststd dataset without target(label) file,
```bash
python convert.py \
  --input_path_json=../data/simmc2_dials_dstc10_teststd_public.json \
  --output_path_predict=../teststd_data/teststd_predict.txt \
  --object_special_token_item2id=item2id.json \
  --scene_json_folder=../data/jsons  \
  --image_folder=../data/images \
  --with_target=0
```

## **Training**
Make sure to download simmc2-data into `./data` before training and preprocessing it into `data_object_special`.

Move into `scripts`: `cd scripts`. 

To train a model in **your own computer** just run:
```bash
bash run_train_bart_coref.sh
```
or 
```bash
python run_train_bart_coref.py \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --item2id=./item2id.json \
        --train_input_file=../data_object_special/simmc2_dials_dstc10_train_predict.txt \
        --train_target_file=../data_object_special/simmc2_dials_dstc10_train_target.txt  \
        --eval_input_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --eval_target_file=../data_object_special/simmc2_dials_dstc10_devtest_target.txt \
        --output_dir=../coref_models/model_only_coref \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/model_only_coref/report.txt \
        --num_train_epochs=8  \
        --eval_steps=3000  \
        --warmup_steps=8000
```

To train a model in **a GPU grid** run:
```bash
bash grid_train_bart_coref.sh
```
or
```bash
qsub -l h=stg-gpu25,virtual_gpu_free=12000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_train_bart_coref.py \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --item2id=./item2id.json \
        --train_input_file=../data_object_special/simmc2_dials_dstc10_train_predict.txt \
        --train_target_file=../data_object_special/simmc2_dials_dstc10_train_target.txt  \
        --eval_input_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --eval_target_file=../data_object_special/simmc2_dials_dstc10_devtest_target.txt \
        --output_dir=../coref_models/model_only_coref \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/model_only_coref/report.txt \
        --num_train_epochs=8  \
        --eval_steps=3000  \
        --warmup_steps=8000
```
where `SCRIPTS_PATH` is the path of the script to run a executable in the GPU cluster. As an example, in my case it would be `SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"`.

These are just some examples. For more detailed information visit the folder [`scripts`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/scripts).

## **Evaluation**
Move into `scripts`: `cd scripts`. 

To evaluate a model (i.e. get the predicted referred object IDs) in your own device, just run:
```bash
bash run_eval_bart_coref.sh
```
or
```bash
python run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output=../results/devtest/predictions_model_only_coref.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_only_coref/checkpoint-38000
```

To evaluate a fine-tuned model in a GPU grid execute:
```bash
bash grid_eval_devtest_coref.sh
```
or
```bash
qsub -l h=stg-gpu25,virtual_gpu_free=8000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output=../results/devtest/predictions_model_only_coref.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_only_coref/checkpoint-38000
```
where `SCRIPTS_PATH` is the path of the script to run a executable in the GPU cluster. As an example, in my case it would be `SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"`.

These are just some examples. For more detailed information visit the folder [`scripts`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/scripts).

## **Report on MM Coreference Resolution**
After generating the predictions, the report containing the **Object F1 score**, **precision** and **recall** can be obtained executing:
```bash
bash get_report_devtest_only_coref.sh
```
or
```bash
python ../evaluation_tools/evaluate_only_coref.py \
    --input_path_target ../data_object_special/simmc2_dials_dstc10_devtest_target.txt \
    --input_path_predicted ../results/devtest/predictions_model_only_coref.txt \
    --output_path_report ../results/devtest/report_model_only_coref.txt
```
The predictions and the final report can be found in the directory [`results`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/results). For more detailed information visit the folder [`scripts`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/scripts).

