import json
import pprint


def parse_objects_from_simmc2_format(data):
    objects_list = []
    for dial in data['dialogue_data']:
        for turn in dial['dialogue']:
            try:
                if turn['disambiguation_label'] == 1:
                    continue
            except:
                pass
            objs = turn['transcript_annotated']['act_attributes']['objects']
            objects_list.append(objs)
    return objects_list



def main():
    predictions_file_path = "../output/eval_UNITER_basic_all_objmen_devtest.json"
    model_name = predictions_file_path[predictions_file_path.find("/eval_")+len("/eval_"):predictions_file_path.find(".json")]
    target_file_path = "../data/simmc2_dials_dstc10_devtest.json"
    output_file_path = "./span_reports/span_report_"+model_name+".txt"

    # Reading predictions file and parsing it
    with open(predictions_file_path, 'r') as f:
        pred_data = json.load(f)
    list_predicted = parse_objects_from_simmc2_format(pred_data)
    
    # Reading target file and parsing it
    with open(target_file_path, 'r') as f:
        target_data = json.load(f)
    list_target = parse_objects_from_simmc2_format(target_data)

    assert len(list_predicted) == len(list_target)

    n_errors = 0
    underpred = 0
    overpred = 0
    gt_no_coref = 0
    gt_coref = 0
    for i in range(len(list_predicted)):
        pred = list_predicted[i]
        target = list_target[i]
        
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