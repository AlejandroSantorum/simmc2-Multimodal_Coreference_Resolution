import json
import pprint
from evaluate_only_coref import parse_for_only_coref
from convert_baseline import parse_flattened_results_from_file


def main():
    predictions_file_path = "../results/devtest/predictions_input_all_attrs_cp381.txt"
    model_name = predictions_file_path[predictions_file_path.find("/predictions_")+len("/predictions_"):predictions_file_path.find(".txt")]
    target_file_path = "../data_object_special/simmc2_dials_dstc10_devtest_target.txt"
    output_file_path = "../results/devtest/span_report_"+model_name+".txt"

    # Convert the data from the GPT-2 friendly format to JSON
    list_target = parse_flattened_results_from_file(target_file_path)
    list_predicted = parse_for_only_coref(predictions_file_path)
    assert len(list_predicted) == len(list_target)

    n_errors = 0
    underpred = 0
    overpred = 0
    gt_no_coref = 0
    gt_coref = 0
    for i in range(len(list_predicted)):
        pred = list_predicted[i][0]['objects']
        target = list_target[i][0]['objects']
        
        # The model is under-predicting
        if len(pred) < len(target):
            underpred += 1
        # The model is over-predicting
        elif len(pred) > len(target):
            overpred += 1
        # Would the "empty coref" head help? How many "zero targets but several predictions"?
        if len(pred) > 0 and len(target)==0:
            gt_no_coref += 1
        # Would the complementary of "empty coref" head help? How many "zero predictions but with targets"?
        elif len(pred)==0 and len(target)>0:
            gt_coref += 1
        
        # Total number of wrong examples (not wrong objects, but examples with some errors)
        if (len(pred) != len(target)) or (len(pred) != len(set(pred).intersection(target))):
            n_errors += 1
        
    report = {}
    report['Total no. examples with any wrong objects'] = n_errors
    report['No. wrong examples over-predicting'] = overpred
    report['No. wrong examples under-predicting'] = underpred
    report['No. wrong examples with zero targets but several predictions'] = gt_no_coref
    report['No. wrong examples with several targets but zero predictions'] = gt_coref

    print("Report of model:", model_name)
    pprint.pprint(report)
    with open(output_file_path, 'w') as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    main()