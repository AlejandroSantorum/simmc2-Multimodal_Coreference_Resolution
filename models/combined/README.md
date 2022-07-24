# Combined models
This module aims to **combine two models** that tackle coreference resolution in different ways so they can both benefit and increase the overall performance.

It was observed in the experiments that the BART-based model is better at recognizing non-mentioned objects, and the UNITER-based model better when the target objects were previously mentioned in the dialog. Therefore, the script `combine_mentioned_vs_new.py` will combine both models so each of them can target the objects accordingly to their strengths.

## Folder structure:
- [`model_outputs`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/combined/model_outputs): contains the files with the model predictions. In this case, there are stored the predictions of the best performing BART model and the best UNITER system.
- [`scripts`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/combined/scripts): contains the main **script to combine the predictions** (`combine_mentioned_vs_new.py`) and other useful source code to evaluate the combined predictions and get the F1-score.
- [`targets`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/combined/targets): folder containing the target object IDs. Obviously, the main script assumes that both models (BART and UNITER-based) were evaluated on the same dataset.
- [`utils`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/combined/utils): contains useful scripts to get the mentioned object IDs of the examples of the used test set (in this case, `devtest` set).

## Typical usage
1. Move to [`utils`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/combined/utils) folder (`cd utils`).
2. Execute `get_mentioned_ids.py` specifying the path of the used test set (BART format):
```bash
python get_mentioned_ids.py \
    --test_file_path simmc2_dials_dstc10_devtest_predict.txt \
    --store_path mentioned_ids.json
```
3. Make sure that both prediction files are placed in [`model_outputs`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/combined/model_outputs) folder.
4. Move to [`scripts`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/combined/scripts) folder (`cd ..` and `cd scripts` if you were in `utils` folder) and execute `combine_mentioned_vs_new.py` script. You can specify the predictions file path of both models, the targets file and the file containing the mentioned IDs per test example:
```bash
python combine_mentioned_vs_new.py \
    --model_for_new_objs_path ../model_outputs/predictions_BART_input_all_attrs_cp381.txt \
    --model_for_mentioned_objs_path ../model_outputs/predictions_UNITER_basic_all_objmen_noIDs_devtest.json \
    --targets_file_path ../targets/simmc2_dials_dstc10_devtest.json \
    --mentioned_ids_path ../utils/mentioned_ids.json
```
Typical output of previous command:
```bash
{
    'overall performance': {'F1 score': 0.8022403426406393,
                            'Precision': 0.7759719566602932,
                            'Recall': 0.8303495311167945},
    'performance on mentioned objects': {'F1 score': 0.8503699229956213,
                                         'Precision': 0.7990919409761634,
                                         'Recall': 0.9086802194256212},
    'performance on new objects': {'F1 score': 0.7444726350126857,
                                   'Precision': 0.7463662790697675,
                                   'Recall': 0.7425885755603759}
}
```
As we can see, the **overall performance of the combined model** is about **0.802 F1 score**.
