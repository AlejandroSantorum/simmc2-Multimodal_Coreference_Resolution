# GPT-2 Baseline for MM Coreference Resolution

This directory contains the code and the scripts for running the baseline model provided by Facebook Research in [SIMMC2.0 mm_dst](https://github.com/facebookresearch/simmc2/tree/main/model/mm_dst).

### Training
To train the GPT-2 baseline execute:
```bash
    python -m gpt2_dst.scripts.run_language_modeling \
        --output_dir=./gpt2_dst/save/model \
        --model_type=gpt2 \
        --model_name_or_path=gpt2 \
        --line_by_line \
        --add_special_tokens=./gpt2_dst/data/simmc2_special_tokens.json \
        --do_train \
        --train_data_file=./gpt2_dst/data/simmc2_dials_dstc10_train_target.txt \
        --do_eval --eval_all_checkpoints \
        --eval_data_file=./gpt2_dst/data/simmc2_dials_dstc10_dev_target.txt \
        --num_train_epochs=2 \
        --overwrite_output_dir \
        --per_gpu_train_batch_size=4 \
        --per_gpu_eval_batch_size=4
```

### Evaluating
To evaluate the GPT-2 baseline execute:
```bash
    python -m gpt2_dst.scripts.run_generation \
        --model_type=gpt2 \
        --model_name_or_path=./gpt2_dst/save/model/ \
        --num_return_sequences=1 \
        --length=100 \
        --stop_token='<EOS>' \
        --prompts_from_file=./gpt2_dst/data/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output=./gpt2_dst/results/simmc2_dials_dstc10_devtest_predicted.txt
```

### Getting performance
To get the report after evaluating the GPT-2 baseline just execute:
```bash
    python -m gpt2_dst.scripts.evaluate \
        --input_path_target=./gpt2_dst/data/simmc2_dials_dstc10_devtest_target.txt \
        --input_path_predicted=./gpt2_dst/data/simmc2_dials_dstc10_devtest_predicted.txt \
        --output_path_report=./gpt2_dst/results/simmc2_dials_dstc10_devtest_report.json
```

The generated report can be found at [`results`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/gpt2_baseline/gpt2_dst/results) folder.