#!/usr/bin/env python3

import argparse
import json
from convert_baseline import parse_flattened_results_from_file
from evaluate_dst import evaluate_from_flat_list
import pprint


def parse_for_only_coref(input_path_predicted):
    results = []
    with open(input_path_predicted, "r") as f_in:
        for line in f_in:
            parsed = parse_line_only_coref(line)
            results.append(parsed)
    return results


def parse_line_only_coref(line):
    d = {
            "act": [],
            "slots": [],
            "request_slots": [],
            "objects": [],
        }

    splits = line.split('<EOB>')
    splits = splits[0].split("[  ] ()")
    splits = splits[1].split("< ")
    splits = splits[1].split(" >")
    splits = splits[0].split(", ")

    for item in splits:
        try:
            d['objects'].append(int(item))
        except:
            pass

    return [d]

if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path_target", help="path for target, line-separated format (.txt)"
    )
    parser.add_argument(
        "--input_path_predicted",
        help="path for model prediction output, line-separated format (.txt)",
    )
    parser.add_argument(
        "--output_path_report", help="path for saving evaluation summary (.json)"
    )

    args = parser.parse_args()
    input_path_target = args.input_path_target
    input_path_predicted = args.input_path_predicted
    output_path_report = args.output_path_report

    # Convert the data from the GPT-2 friendly format to JSON
    list_target = parse_flattened_results_from_file(input_path_target)
    list_predicted = parse_for_only_coref(input_path_predicted)

    # Evaluate
    report = evaluate_from_flat_list(list_target, list_predicted, only_coref=True)
    pprint.pprint(report)
    # Save report
    with open(output_path_report, "w") as f_out:
        json.dump(report, f_out, indent=4)
