# tools for processing the data for BART-based model
This folder contains the tools to process the data for different BART-based model experiments.

- `attributes_processing.py`: script to generate all special tokens given all the objects attributes.
- `convert_line_to_json_for_mm_dst.py`: script to process datasets from line-by-line format to .JSON format.
- `convert_mm_dst_to_response.py`: script to parse DST results to response-task results.
- `get_random_samples_from_ref.py`: script to get the corresponding examples for the bART-based model given a reference dataset (e.g. [`random_devtest_samples.json`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/data_object_special/new_datasets/reference_sets)). This is mainly used to get the random subset of samples that originally was chosen from the [UNITER-based model](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based) preprocessed dataset. The same examples are chosen with the BART-based model so the results of both experiments can be compared across different models.
- `get_scenes_special_dataset.py`: script to get the list of scene names given a dataset. The i-th scene name would correspond to the scene where the i-th example of the given dataset is taking place. Executing
```bash
    python get_scenes_special_dataset.py
```
will create the files `simmc2_scenes_dev.txt`, `simmc2_scenes_devtest.txt` and `simmc2_scenes_train.txt` in [`data_object_special`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/data_object_special) folder.
- `get_seen_unseen_outdomain_sets.py`: script to get the **datasets for the cross-domain experiments**. They need the reference sets chosen for the [UNITER-based model](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/uniter_based) located at [`reference_sets`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/data_object_special/new_datasets/reference_sets) folder. Again, these datasets need to be the same for both models so the results are comparable.
- `mask_datasets.py`: script to generate the masked datasets located at [`masked_datasets`](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution/tree/main/models/bart_only_coref/data_object_special/masked_datasets).