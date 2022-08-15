# Executables
The following directories contain the scripts to train and evaluate the different versions of the UNITER-based model.
- [`get_report`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based/sh/get_report): contains the script `get_report.py`, that computes the F1 score of a model given the predictions.
- [`grid_eval`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based/sh/grid_eval): contains the GPU-based evaluation scripts of the different versions of the UNITER-based model.
- [`grid_train`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based/sh/grid_train): contains the GPU-based training scripts of the different versions of the UNITER-based model.
- [`infer_eval`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based/sh/infer_eval): contains the CPU-based evaluation scripts of the different versions of the UNITER-based model.
- [`train`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based/sh/train): contains the CPU-based training scripts of the different versions of the UNITER-based model.

### Main UNITER-based models investigated in the original paper
Original paper: [*"UNITER-Based Situated Coreference Resolution with Rich Multimodal Input"*](https://arxiv.org/abs/2112.03521).
- `UNITER_basic_RCNN`: Plain UNITER + Faster RCNN for image embeddings
- `UNITER_basic_CLIP`: Plain UNITER + CLIP for image embeddings
- `UNITER_basic_both`: Plain UNITER + FasterRCNN+CLIP for image embeddings
- `UNITER_basic_both_objid`: UNITER + CLIP+FRCNN + Object IDs
- `UNITER_basic_both_objid_pos`: UNITER + CLIP+FRCNN + ObjIDs + coordinates
- `UNITER_basic_both_objid_pos_sceneseg`: UNITER + CLIP+FRCNN + ObjIDs + coordinates + active scene
- `UNITER_basic_both_objid_pos_sceneseg_BERT`: UNITER + CLIP+FRCNN + ObjIDs + coordinates + active scene + BERT for text embeddings
- `UNITER_basic_both_objid_pos_sceneseg_SBERT`: UNITER + CLIP+FRCNN + ObjIDs + coordinates + active scene + SBERT for text embeddings
- `UNITER_basic_both_objid_pos_sceneseg_bothBERT`: UNITER + CLIP+FRCNN + ObjIDs + coordinates + active scene + BERT+SBERT for text embeddings


### UNITER-based models investigated in this project
Note, `UNITER_basic_all_` means that the models is using UNITER + FasterRCNN+CLIP + ObjsIDs + coordinates + active scene + BERT. The different versions try different additional features:
- `UNITER_basic_all_objmen`: including previously mentioned objects
- `UNITER_basic_all_objmen_noIDs`: including previously mentioned objects but **removing object IDs**
- `UNITER_basic_all_objmen_just_fashion`: including previously mentioned objects and trained on the fashion examples of the `devtest` set
- `UNITER_basic_all_objmen_noIDs_just_fashion`: including previously mentioned objects but **removing object IDs** and trained on the fashion examples of the `devtest` set
- `UNITER_basic_all_objmen_seen_unseen_OOD`: including previously mentioned objects and trained on a subset of fashion
- `UNITER_basic_all_objmen_noIDs_seen_unseen_OOD`: including previously mentioned objects but **removing object IDs** and trained on a subset of fashion
- `UNITER_basic_all_objmen_train_on_all_furniture`: including previously mentioned objects and trained on a subset of furniture
- `UNITER_basic_all_objmen_noIDs_train_on_all_furniture`: including previously mentioned objects but **removing object IDs** and trained on a subset of furniture
- `UNITER_basic_all_objmen_numTargetObjs`: including previously mentioned objects and adding the auxiliary task head
- `UNITER_basic_all_objmen_noIDs_numTargetObjs`: including previously mentioned objects but **removing object IDs** and adding the auxiliary task head
- `UNITER_basic_all_objmen_numMentionedTargets`: including previously mentioned objects and adding the two auxiliary tasks head (number of mentioned + number of new targets)
- `UNITER_basic_all_objmen_noIDs_numMentionedTargets`: including previously mentioned objects but **removing object IDs** and adding the two auxiliary tasks head (number of mentioned + number of new targets)
- `UNITER_attnbias_all_objmen`: including previously mentioned objects and adding the attention bias mechanism to fine-tune UNITER layers

