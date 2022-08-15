# data_object_special: preprocessed datasets for BART-based model
This folder contains the preprocessed datasets to be fed into the BART-based model and execute all the experiments of the project.


## Download the folder
This folder ([`data_object_special`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/data_object_special)) contains the majority of the preprocessed files and they can be directly downloaded cloning this repository. However, some special data files for some experiments are too large. They can be generated using the suitable scripts detailed in [`processing_data`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/processing_data) folder, or simple by downloading the full version of this folder: [data_object_special.zip](https://drive.google.com/file/d/1LsnwUyt_ZG-e1OS-Hiud7ERvU8jpD4KA/view?usp=sharing).


## File description

- `simmc2_special_tokens.json`: list of special tokens to fine-tune BART.
- `simmc2_dials_dstc10_dev_predict.txt`: preprocessed examples of the `dev` set.
- `simmc2_dials_dstc10_dev_target.txt`: preprocessed labels (targets) of the `dev` set.
- `simmc2_dials_dstc10_devtest_predict.txt`: preprocessed examples of the `devtest` set.
- `simmc2_dials_dstc10_devtest_target.txt`: preprocessed labels (targets) of the `devtest` set.
- `simmc2_dials_dstc10_train_predict.txt`: preprocessed examples of the `train` set.
- `simmc2_dials_dstc10_train_target.txt`: preprocessed labels (targets) of the `train` set.
- `simmc2_dials_dstc10_train_response.txt`: preprocessed examples to train response retrieval/generation tasks.
- `simmc2_dials_dstc10_train_disambiguation_label.txt`: preprocessed labels (targets) for the MM disambiguation task.
- `simmc2_dials_dstc10_devtest_retrieval.json`: preprocessed `devtest` examples for response retrieval task.
- `simmc2_dials_dstc10_devtest_inference_disambiguation.json`: preprocessed `devtest` examples for disambiguation task.
- `simmc2_dials_dstc10_dev_retrieval.json`: preprocessed `dev` examples for response retrieval task.
- `simmc2_dials_dstc10_dev_disambiguation_label.txt`: preprocessed labels for disambiguation task of the `dev` set.

