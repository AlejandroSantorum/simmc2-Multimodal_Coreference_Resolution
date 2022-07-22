# evaluation tools for BART-based model
This folder contains the evaluation tools to assess BART-based model performance.

- `convert_baseline.py`: script for converting the main SIMMC datasets (.JSON format) into the line-by-line stringified format, and back.
- `error_analysis.py`: script to sample N random errors from a given file of predictions and illustrate those errors indicating the predicted object IDs, the target object IDs and the mentioned object IDs (multimodal context). As an example, the script can be executed as follows:
    ```bash
        python error_analysis.py \
            --model_name BART_only_coref \
            --predictions_file_path ../results/devtest/predictions_input_all_attrs_cp381.txt \
            --test_examples_file_path ../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
            --targets_file_path ../data_object_special/simmc2_dials_dstc10_devtest_target.txt \
            --test_scenes_file_path ../data_object_special/simmc2_scenes_devtest.txt \
            --n_errors 10
    ```
It is generated a PDF file illustrating these errors inside the folder [`pdf_error_analysis`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/evaluation_tools/pdf_error_analysis).
- `eval_mentioned_vs_new.py`: script to get the splitted report between mentioned object and new objects. It is indicated the F1 score, precision and recall within the previously mentioned objects and the new ones (not mentioned). As an example, the script can be executed as follows:
    ```bash
        python eval_mentioned_vs_new.py \
            --predictions_file_path ../results/devtest/predictions_input_all_attrs_cp381.txt \
            --test_data_path ../data_object_special/simmc2_dials_dstc10_devtest_target.txt
    ```
The splitted report is stored **by default** inside [`results`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/results) folder. The output path can be specified using the argument `--output_file_path`.
- `evaluate_dst.py`: util functions for evaluating the DST model predictions. The script includes a main function which takes the original JSON data file and the predicted model output file (in the same format), and outputs the report.
- `evaluate_only_coref.py`: adapted functions for evaluating the BART-based model that just uses the coreference head.
- `evaluate.py`: scripts for evaluating the GPT-2 DST model predictions. First, we parse the line-by-line stringified format into the structured DST output. We then run the main DST Evaluation script to get results.
- `pred_length_analysis.py`: script to get insights about the predictions of the given model. It counts the number of examples with *any wrong objects*, the number of examples *over-predicting* (predicting more referred objects than it should) and the number of examples *under-predicting*.As an example, the script can be executed as follows:
    ```bash
        python pred_length_analysis.py \
            --predictions_file_path ../results/devtest/predictions_input_all_attrs_cp381.txt \
            --targets_file_path ../data_object_special/simmc2_dials_dstc10_devtest_target.txt
    ```
The report is stored in the same folder as the predictions.