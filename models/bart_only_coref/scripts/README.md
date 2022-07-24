# Scripts: BART-based model source code
This folder contains all the scripts and the source code to run all the experiments of this project.

## Index:
1. [GPU grid execution](#1-execution-on-a-gpu-cluster)
    1. [Training](#11-training)
    2. [Evaluating](#12-evaluating)
    3. [Getting the report](#13-getting-the-report)

2. [Local GPU execution](#2-local-gpu-execution)
    1. [Training](#21-training)
    2. [Evaluating](#22-evaluating)
    3. [Getting the report](#23-getting-the-report)

------------------------------------

## 1. Execution on a GPU cluster
In the following examples the variable `{SCRIPTS_PATH}` is the folder path that contains the script to run a python program in a GPU grid (e.g. `run_on_grid.sh`). In my case:
```bash
SCRIPTS_PATH="/rmt/dialogue2/interns/alejandro/scripts/"
```

### 1.1 Training

- Train a **standard** BART-based system with **only the Coreference Head** (check `grid_train_bart_coref.sh`):
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
        --num_train_epochs=10  \
        --eval_steps=3000  \
        --warmup_steps=8000
```

- Train a BART-based system using **non-visual attributes** as part of the input:
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
        --output_dir=../coref_models/model_only_coref_non_visual_attrs \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/model_only_coref_non_visual_attrs/report.txt \
        --num_train_epochs=10  \
        --eval_steps=3000  \
        --warmup_steps=8000 \
        --non_visual_attrs True
```
Note the new input argument `--non_visual_attrs` set to `True`.

- Train a BART-based system using **non-visual and visual attributes** as part of the input:
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
        --output_dir=../coref_models/model_only_coref_all_attrs \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/model_only_coref_all_attrs/report.txt \
        --num_train_epochs=10  \
        --eval_steps=3000  \
        --warmup_steps=8000 \
        --all_attrs True
```
Note the new input argument `--all_attrs` set to `True`.

- Train a standard 1-task head BART-based system on the `train_seen_unseen_OOD` set (~14.6K fashion examples) used for the **cross-domain experiments** (check `grid_train_seen_unseen_ood.sh`):
```bash
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
        --output_dir=../coref_models/model_cross_domain_exps \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/model_cross_domain_exps/report.txt \
        --num_train_epochs=10  \
        --eval_steps=3000  \
        --warmup_steps=8000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

- Train a standard 1-task head BART-based system on the `all_furniture` set (~12K furniture examples) used for a variation of the **cross-domain experiments**:
```bash
qsub -l h=stg-gpu25,virtual_gpu_free=12000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_train_bart_coref.py \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --item2id=./item2id.json \
        --train_input_file=../data_object_special/new_datasets/all_furniture_predict.txt \
        --train_target_file=../data_object_special/new_datasets/all_furniture_target.txt  \
        --eval_input_file=../data_object_special/new_datasets/in_domain_predict.txt \
        --eval_target_file=../data_object_special/new_datasets/in_domain_target.txt \
        --output_dir=../coref_models/model_trained_all_furniture \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/model_trained_all_furniture/report.txt \
        --num_train_epochs=10  \
        --eval_steps=3000  \
        --warmup_steps=8000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

It is *important to note* that since all the furniture examples are being used for training, there is no "furniture dev" set available, so here we use as `dev` set a group of 9K fashion examples (out-of-domain). Other option is to fix the number of epochs (`--num_train_epochs`) and use the same value for all the experiments, so they can be comparable too.

- Train a BART-based system on the original SIMMC2.0 training set using an **additional task head** that predicts the **number of referred objects** in the dialog (i.e. it predicts the *number of targets*). The prediction of this additional head is used at inference time to improve the predictions on coreference resolution.
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
        --output_dir=../coref_models/model_coref_num_targets_head \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/model_coref_num_targets_head/report.txt \
        --num_train_epochs=10  \
        --eval_steps=3000  \
        --warmup_steps=8000 \
        --num_objs_head True
```
Note the new input argument `--num_objs_head` set to `True`.

This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.


### 1.2 Evaluating

- Evaluate a **standard** BART-based system with **only the Coreference Head** (check `grid_eval_devtest_coref.sh`):
```bash
qsub -l h=stg-gpu25,virtual_gpu_free=8000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output=../results/devtest/predictions_only_coref.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_only_coref/checkpoint-38000
```
The **chosen checkpoint** (`--model_dir` argument) **can be changed** depending on the performance on the `dev` set or on the desired number of epochs.

- Evaluate a BART-based system using **non-visual attributes** as part of the input:
```bash
qsub -l h=stg-gpu25,virtual_gpu_free=12000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output=../results/devtest/predictions_only_coref_non_visual_attrs.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_only_coref_non_visual_attrs/checkpoint-38000 \
        --non_visual_attrs True
```
Note the new input argument `--non_visual_attrs` set to `True`.

- Evaluate a BART-based system using **non-visual and visual attributes** as part of the input:
```bash
qsub -l h=stg-gpu25,virtual_gpu_free=12000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output=../results/devtest/predictions_only_coref_all_attrs.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_only_coref_all_attrs/checkpoint-38000 \
        --all_attrs True
```
Note the new input argument `--all_attrs` set to `True`.

- Evaluate a standard 1-task head BART-based system trained on the `train_seen_unseen_OOD` set (~14.6K fashion examples) used for the **cross-domain experiments** (check `grid_eval_seen_unseen_ood.sh`). To evaluate it on the `in_domain` set (**in domain experiment**) run:
```bash
qsub -l h=stg-gpu25,virtual_gpu_free=12000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/new_datasets/in_domain_predict.txt \
        --path_output=../results/in_domain/predictions_cross_domain_exps.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_cross_domain_exps/checkpoint-38000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

- Evaluate a standard 1-task head BART-based system trained on the `train_seen_unseen_OOD` set (~14.6K fashion examples) used for the **cross-domain experiments**. To evaluate it on the `in_domain held-out` set (**in domain without scene overlapping experiment**) run:
```bash
qsub -l h=stg-gpu25,virtual_gpu_free=12000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/new_datasets/in_domain_held_out_predict.txt \
        --path_output=../results/in_domain_held_out/predictions_cross_domain_exps.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_cross_domain_exps/checkpoint-38000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

- Evaluate a standard 1-task head BART-based system trained on the `train_seen_unseen_OOD` set (~14.6K fashion examples) used for the **cross-domain experiments**. To evaluate it on the `out_of_domain` set (**out of domain experiment**) run:
```bash
qsub -l h=stg-gpu25,virtual_gpu_free=12000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/new_datasets/out_of_domain_predict.txt \
        --path_output=../results/out_of_domain/predictions_cross_domain_exps.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_cross_domain_exps/checkpoint-38000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

- Evaluate a standard 1-task head BART-based system trained on the `train_seen_unseen_OOD` set (~14.6K fashion examples) used for the **cross-domain experiments**. To evaluate it on the `devtest_only_furniture` set (**additional out of domain experiment**) run:
```bash
qsub -l h=stg-gpu25,virtual_gpu_free=12000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/new_datasets/devtest_only_furniture_predict.txt \
        --path_output=../results/devtest_only_furniture/predictions_cross_domain_exps.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_cross_domain_exps/checkpoint-38000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

- Evaluate a standard 1-task head BART-based system trained on the `all_furniture` set (~12K furniture examples) used for a variation of the **cross-domain experiments**. To evaluate it on the `devtest_only_fashion` set (**additional out of domain experiment**) run:
```bash
qsub -l h=stg-gpu25,virtual_gpu_free=12000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/new_datasets/devtest_only_fashion_predict.txt \
        --path_output=../results/devtest_only_fashion/predictions_cross_domain_exps.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_trained_all_furniture/checkpoint-38000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

- Evaluate a BART-based system on the original SIMMC2.0 `devtest` set using an **additional task head** that predicts the **number of referred objects** in the dialog (i.e. it predicts the *number of targets*). The prediction of this additional head is now used at inference time to improve the predictions on coreference resolution.
```bash
qsub -l h=stg-gpu25,virtual_gpu_free=8000M,gpu=1,gpu_queue=1,arch=*64*,test=*,centos=1,gpu_installed=1 \
    -e /rmt/dialogue2/interns/alejandro/logs \
    -o /rmt/dialogue2/interns/alejandro/logs \
    ${SCRIPTS_PATH}run_on_grid.sh run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output=../results/n_target_objs_head/predictions_coref_num_targets_head.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_coref_num_targets_head/checkpoint-38000 \
        --num_objs_head True
```
Note the new input argument `--num_objs_head` set to `True`.

This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.


### 1.3 Getting the report

To calculate the performance of any BART-based system just run (check `get_report_devtest_only_coref.sh`):
```bash
python ../evaluation_tools/evaluate_only_coref.py \
    --input_path_target <YOUR PATH TO TARGETS>\
    --input_path_predicted <YOUR PATH TO PREDICTIONS>  \
    --output_path_report <YOUR PATH TO STORE RESULTS>
```
In particular:
```bash
python ../evaluation_tools/evaluate_only_coref.py \
    --input_path_target ../data_object_special/simmc2_dials_dstc10_devtest_target.txt \
    --input_path_predicted ../results/devtest/predictions_only_coref.txt \
    --output_path_report ../results/devtest/report_only_coref.txt
```

In the particular case of models trained for the **cross-domain experiments** remember to specify the correct target file path. For example:
```bash
python ../evaluation_tools/evaluate_only_coref.py \
    --input_path_target ../data_object_special/new_datasets/in_domain_target.txt \
    --input_path_predicted ../results/in_domain/predictions_cross_domain_exps.txt \
    --output_path_report ../results/in_domain/report_cross_domain_exps.txt
```



-------------------------------------

## 2. Local GPU Execution
All the previous commands can also be run directly with python:

### 2.1 Training
- Train a **standard** BART-based system with **only the Coreference Head**:
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
        --num_train_epochs=10  \
        --eval_steps=3000  \
        --warmup_steps=8000
```

- Train a BART-based system using **non-visual attributes** as part of the input:
```bash
python run_train_bart_coref.py \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --item2id=./item2id.json \
        --train_input_file=../data_object_special/simmc2_dials_dstc10_train_predict.txt \
        --train_target_file=../data_object_special/simmc2_dials_dstc10_train_target.txt  \
        --eval_input_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --eval_target_file=../data_object_special/simmc2_dials_dstc10_devtest_target.txt \
        --output_dir=../coref_models/model_only_coref_non_visual_attrs \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/model_only_coref_non_visual_attrs/report.txt \
        --num_train_epochs=10  \
        --eval_steps=3000  \
        --warmup_steps=8000 \
        --non_visual_attrs True
```
Note the new input argument `--non_visual_attrs` set to `True`.

- Train a BART-based system using **non-visual and visual attributes** as part of the input:
```bash
python run_train_bart_coref.py \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --item2id=./item2id.json \
        --train_input_file=../data_object_special/simmc2_dials_dstc10_train_predict.txt \
        --train_target_file=../data_object_special/simmc2_dials_dstc10_train_target.txt  \
        --eval_input_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --eval_target_file=../data_object_special/simmc2_dials_dstc10_devtest_target.txt \
        --output_dir=../coref_models/model_only_coref_all_attrs \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/model_only_coref_all_attrs/report.txt \
        --num_train_epochs=10  \
        --eval_steps=3000  \
        --warmup_steps=8000 \
        --all_attrs True
```
Note the new input argument `--all_attrs` set to `True`.

- Train a standard 1-task head BART-based system on the `train_seen_unseen_OOD` set (~14.6K fashion examples) used for the **cross-domain experiments**:
```bash
python run_train_bart_coref.py \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --item2id=./item2id.json \
        --train_input_file=../data_object_special/new_datasets/train_seen_unseen_OOD_predict.txt \
        --train_target_file=../data_object_special/new_datasets/train_seen_unseen_OOD_target.txt  \
        --eval_input_file=../data_object_special/new_datasets/in_domain_predict.txt \
        --eval_target_file=../data_object_special/new_datasets/in_domain_target.txt \
        --output_dir=../coref_models/model_cross_domain_exps \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/model_cross_domain_exps/report.txt \
        --num_train_epochs=10  \
        --eval_steps=3000  \
        --warmup_steps=8000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

- Train a standard 1-task head BART-based system on the `all_furniture` set (~12K furniture examples) used for a variation of the **cross-domain experiments**:
```bash
python run_train_bart_coref.py \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --item2id=./item2id.json \
        --train_input_file=../data_object_special/new_datasets/all_furniture_predict.txt \
        --train_target_file=../data_object_special/new_datasets/all_furniture_target.txt  \
        --eval_input_file=../data_object_special/new_datasets/in_domain_predict.txt \
        --eval_target_file=../data_object_special/new_datasets/in_domain_target.txt \
        --output_dir=../coref_models/model_trained_all_furniture \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/model_trained_all_furniture/report.txt \
        --num_train_epochs=10  \
        --eval_steps=3000  \
        --warmup_steps=8000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

- Train a BART-based system on the original SIMMC2.0 training set using an **additional task head** that predicts the **number of referred objects** in the dialog (i.e. it predicts the *number of targets*). The prediction of this additional head is used at inference time to improve the predictions on coreference resolution.
```bash
python run_train_bart_coref.py \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --item2id=./item2id.json \
        --train_input_file=../data_object_special/simmc2_dials_dstc10_train_predict.txt \
        --train_target_file=../data_object_special/simmc2_dials_dstc10_train_target.txt  \
        --eval_input_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --eval_target_file=../data_object_special/simmc2_dials_dstc10_devtest_target.txt \
        --output_dir=../coref_models/model_coref_num_targets_head \
        --train_batch_size=8 \
        --output_eval_file=../coref_models/model_coref_num_targets_head/report.txt \
        --num_train_epochs=10  \
        --eval_steps=3000  \
        --warmup_steps=8000 \
        --num_objs_head True
```
Note the new input argument `--num_objs_head` set to `True`.

This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.


### 2.2 Evaluating
- Evaluate a **standard** BART-based system with **only the Coreference Head**:
```bash
python run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output=../results/devtest/predictions_only_coref.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_only_coref/checkpoint-38000
```
The **chosen checkpoint** (`--model_dir` argument) **can be changed** depending on the performance on the `dev` set or on the desired number of epochs.

- Evaluate a BART-based system using **non-visual attributes** as part of the input:
```bash
python run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output=../results/devtest/predictions_only_coref_non_visual_attrs.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_only_coref_non_visual_attrs/checkpoint-38000 \
        --non_visual_attrs True
```
Note the new input argument `--non_visual_attrs` set to `True`.

- Evaluate a BART-based system using **non-visual and visual attributes** as part of the input:
```bash
python run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output=../results/devtest/predictions_only_coref_all_attrs.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_only_coref_all_attrs/checkpoint-38000 \
        --all_attrs True
```
Note the new input argument `--all_attrs` set to `True`.

- Evaluate a standard 1-task head BART-based system trained on the `train_seen_unseen_OOD` set (~14.6K fashion examples) used for the **cross-domain experiments**. To evaluate it on the `in_domain` set (**in domain experiment**) run:
```bash
python run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/new_datasets/in_domain_predict.txt \
        --path_output=../results/in_domain/predictions_cross_domain_exps.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_cross_domain_exps/checkpoint-38000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

- Evaluate a standard 1-task head BART-based system trained on the `train_seen_unseen_OOD` set (~14.6K fashion examples) used for the **cross-domain experiments**. To evaluate it on the `in_domain held-out` set (**in domain without scene overlapping experiment**) run:
```bash
python run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/new_datasets/in_domain_held_out_predict.txt \
        --path_output=../results/in_domain_held_out/predictions_cross_domain_exps.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_cross_domain_exps/checkpoint-38000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

- Evaluate a standard 1-task head BART-based system trained on the `train_seen_unseen_OOD` set (~14.6K fashion examples) used for the **cross-domain experiments**. To evaluate it on the `out_of_domain` set (**out of domain experiment**) run:
```bash
python run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/new_datasets/out_of_domain_predict.txt \
        --path_output=../results/out_of_domain/predictions_cross_domain_exps.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_cross_domain_exps/checkpoint-38000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

- Evaluate a standard 1-task head BART-based system trained on the `train_seen_unseen_OOD` set (~14.6K fashion examples) used for the **cross-domain experiments**. To evaluate it on the `devtest_only_furniture` set (**additional out of domain experiment**) run:
```bash
python run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/new_datasets/devtest_only_furniture_predict.txt \
        --path_output=../results/devtest_only_furniture/predictions_cross_domain_exps.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_cross_domain_exps/checkpoint-38000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

- Evaluate a standard 1-task head BART-based system trained on the `all_furniture` set (~12K furniture examples) used for a variation of the **cross-domain experiments**. To evaluate it on the `devtest_only_fashion` set (**additional out of domain experiment**) run:
```bash
python run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/new_datasets/devtest_only_fashion_predict.txt \
        --path_output=../results/devtest_only_fashion/predictions_cross_domain_exps.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_trained_all_furniture/checkpoint-38000
```
This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.

- Evaluate a BART-based system on the original SIMMC2.0 `devtest` set using an **additional task head** that predicts the **number of referred objects** in the dialog (i.e. it predicts the *number of targets*). The prediction of this additional head is now used at inference time to improve the predictions on coreference resolution.
```bash
python run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output=../results/n_target_objs_head/predictions_coref_num_targets_head.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_coref_num_targets_head/checkpoint-38000 \
        --num_objs_head True
```
Note the new input argument `--num_objs_head` set to `True`.

This model can also be trained using **non-visual** or **non-visual+visual attributes** as part of the input. You just need to include the arguments `--non_visual_attrs=True` or `--all_attrs=True` respectively.


### 2.3 Getting the report
Identical to section [1.3](#13-getting-the-report).