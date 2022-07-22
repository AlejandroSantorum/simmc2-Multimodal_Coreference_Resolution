python run_eval_bart_coref.py \
        --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
        --path_output=../results/devtest/predictions_model_only_coref.txt \
        --item2id=./item2id.json \
        --add_special_tokens=../data_object_special/simmc_special_tokens.json \
        --model_dir=../coref_models/model_only_coref/checkpoint-38000