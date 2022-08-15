# Pre-trained and fine-tuned models

All the parameters of the models investigated throughout this project can be downloaded from Google Drive: [trained_models](https://drive.google.com/drive/folders/1yF5eLE3E7tZMKKKnYUcaTYUDsunJtXKd?usp=sharing).
They should be placed in the current folder `trained`, since this is the directory where the models are searched by the evaluation scripts.

The main available checkpoints are:
Note, `UNITER_basic_all_` means that the models is using UNITER + FasterRCNN+CLIP + ObjsIDs + coordinates + active scene + BERT. The different versions try different additional features:
- `UNITER_basic_all_objmen.bin`: including previously mentioned objects
- `UNITER_basic_all_objmen_noIDs.bin`: including previously mentioned objects but **removing object IDs**
- `UNITER_basic_all_objmen_just_fashion.bin`: including previously mentioned objects and trained on the fashion examples of the `devtest` set
- `UNITER_basic_all_objmen_noIDs_just_fashion.bin`: including previously mentioned objects but **removing object IDs** and trained on the fashion examples of the `devtest` set
- `UNITER_basic_all_objmen_seen_unseen_OOD.bin`: including previously mentioned objects and trained on a subset of fashion
- `UNITER_basic_all_objmen_noIDs_seen_unseen_OOD.bin`: including previously mentioned objects but **removing object IDs** and trained on a subset of fashion
- `UNITER_basic_all_objmen_train_on_all_furniture.bin`: including previously mentioned objects and trained on a subset of furniture
- `UNITER_basic_all_objmen_noIDs_train_on_all_furniture.bin`: including previously mentioned objects but **removing object IDs** and trained on a subset of furniture
- `UNITER_basic_all_objmen_numTargetObjs.bin`: including previously mentioned objects and adding the auxiliary task head
- `UNITER_basic_all_objmen_noIDs_numTargetObjs.bin`: including previously mentioned objects but **removing object IDs** and adding the auxiliary task head
- `UNITER_basic_all_objmen_numMentionedTargets.bin`: including previously mentioned objects and adding the two auxiliary tasks head (number of mentioned + number of new targets)
- `UNITER_basic_all_objmen_noIDs_numMentionedTargets.bin`: including previously mentioned objects but **removing object IDs** and adding the two auxiliary tasks head (number of mentioned + number of new targets)
- `UNITER_attnbias_all_objmen.bin`: including previously mentioned objects and adding the attention bias mechanism to fine-tune UNITER layers