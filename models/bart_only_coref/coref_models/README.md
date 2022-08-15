# Pre-trained and fine-tuned models

The pre-trained and fine-tuned models can be directly downloaded using the following links.

All the fine-tuned models for the experiments of this project can be downloaded individually from this folder from Google Drive: [coref_models](https://drive.google.com/drive/folders/13ZNu1qCg8mfvroj1PsjVJIuzElhDoycH?usp=sharing).

The entire collection of models can be directly downloaded from: [coref_models.zip](https://drive.google.com/file/d/11LupifQrhIN1uSwqU946KhEBK9OEpENV/view?usp=sharing).

The downloaded parameters should be stored in [`coref_models`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/coref_models) folder. As said before, this folder with the fine-tuned models can also be directly downloaded from Google Drive: [coref_models.zip](https://drive.google.com/file/d/11LupifQrhIN1uSwqU946KhEBK9OEpENV/view?usp=sharing). The models are listed below:

- `model_no_attrs`: Plain BART-based model using only Coreference head.
- `model_4_common_attrs`: BART-based model that includes the object descriptions in the input using 4 attributes common for both fashion and furniture domains: 'color', 'type', 'brand' and 'price'.
- `model_non_visual_attrs`: BART-based model that includes the object descriptions in the input using non-visual attributes.
- `model_all_attrs`: BART-based model that includes the object descriptions in the input using both visual and non-visual attributes.
- `model_train_on_furniture`: BART-based model trained on all furniture dialogs, investigated on the cross-domain experiments.
- `train_on_furniture_non_visual_attrs`: BART-based model trained on all furniture dialogs that includes the object descriptions in the input non-visual attributes, investigated on the cross-domain experiments.
- `train_on_furniture_all_attrs`: BART-based model trained on all furniture dialogs that includes the object descriptions in the input using both visual and non-visual attributes, investigated on the cross-domain experiments.
- `train_seen_unseen_OOD`: BART-based model trained on a subset of the fashion dialogs, investigated on the cross-domain experiments.
- `train_OOD_non_visual_attrs`: BART-based model trained on a subset of the fashion dialogs that includes the object descriptions in the input non-visual attributes, investigated on the cross-domain experiments.
- `train_OOD_all_attrs`: BART-based model trained on a subset of the fashion dialogs that includes the object descriptions in the input using both visual and non-visual attributes, investigated on the cross-domain experiments.
- `num_objs_target_head`: BART-based model provided with the additional auxiliary task head that predicts the number of referred items in the last user utterance. The coreference predictions are modified accordingly the output of this head using heuristics.
- `num_targets_non_visual_attrs`: BART-based model provided with the additional auxiliary task head that includes the object descriptions in the input using non-visual attributes.
- `num_targets_all_attrs`: BART-based model provided with the additional auxiliary task head that includes the object descriptions in the input using both visual and non-visual attributes.