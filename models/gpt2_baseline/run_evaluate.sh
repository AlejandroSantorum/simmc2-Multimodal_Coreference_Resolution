#!/bin/bash

PATH_DIR="."

# Evaluate (Example)
python -m utils.evaluate_dst \
    --input_path_target="${PATH_DATA_DIR}"/simmc2_dials_dstc10_devtest.json \
    --input_path_predicted="${PATH_DIR}"/simmc2_dials_dstc10_devtest_predicted.json \
    --output_path_report="${PATH_DIR}"/simmc2_dials_dstc10_report.json
