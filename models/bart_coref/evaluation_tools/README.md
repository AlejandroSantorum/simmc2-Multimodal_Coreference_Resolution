# Evaluation tools for BART-based model
This folder contains the evaluation tools to assess BART-based model performance.

- `convert_baseline.py`: script for converting the main SIMMC datasets (.JSON format) into the line-by-line stringified format, and back.
- `evaluate_dst.py`: util functions for evaluating the DST model predictions. The script includes a main function which takes the original JSON data file and the predicted model output file (in the same format), and outputs the report.
- `evaluate_only_coref.py`: adapted functions for evaluating the BART-based model that just uses the coreference head.
- `evaluate.py`: scripts for evaluating the GPT-2 DST model predictions. First, we parse the line-by-line stringified format into the structured DST output. We then run the main DST Evaluation script to get results.