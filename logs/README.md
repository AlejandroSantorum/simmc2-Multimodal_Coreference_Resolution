# Logs of experiments
The following directories contain all the logs generated after executing the experiments of this project.

- [`gpt2`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/logs/gpt2): replication experiments of the GPT-2 model used as a baseline for the SIMMC2.0 dataset.
- [`kaist`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/logs/kaist): replication experiments of the BART-based model using all task-specfic heads.
- [`bart_coref`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/logs/bart_coref): experiments on the BART-based model removing all heads that are *not* related with coreference resolution.
- [`bart_only_coref`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/logs/bart_only_coref): experiments on the BART-based model removing all heads but the one performing coreference resolution.
- [`bart_only_coref_n_targets_head`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/logs/bart_only_coref_n_targets_head): experiments on the BART-based model that uses just coreference head, adding an additional head that tries to predict the number of referred objects (targets).
- [`uniter`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/logs/uniter):
experiments on the UNITER-based model.
- [`uniter_n_targets_head`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/logs/uniter_n_targets_head): experiments on the UNITER-based model, adding an additional head that tries to predict the number of referred objects (targets).