import json
import pprint
from convert_baseline import parse_flattened_results_from_file
from evaluate_only_coref import parse_line_only_coref

def rec_prec_f1(n_correct, n_true, n_pred):
    rec = n_correct / n_true if n_true != 0 else 0
    prec = n_correct / n_pred if n_pred != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

    return rec, prec, f1


def extract_mentioned_ids(example):
    mentioned_ids = set()

    start = example.find('<SOM>')
    while start != -1:
        end = example.find('<EOM>', start)
        som_part = example[start+len('<SOM>'):end]

        id_start = som_part.find('<')
        while id_start != -1:
            id_end = som_part.find('>', id_start)
            id = int(som_part[id_start+1:id_end])
            mentioned_ids.add(id)
            id_start = som_part.find('<', id_end)

        start = example.find('<SOM>', end)

    return list(mentioned_ids)


def main():
    test_data_path = "../data_object_special/simmc2_dials_dstc10_devtest_target.txt"
    predictions_file_path = "../results/n_target_objs_head/predictions_num_objs_target.txt"

    results_path = predictions_file_path[:predictions_file_path.find('/predictions_')]
    model_name = predictions_file_path[predictions_file_path.find("/predictions_")+len("/predictions_"):predictions_file_path.find(".txt")]
    output_file_path = results_path + "/split_f1_report_"+model_name+".txt"

    with open(test_data_path, 'r') as f:
        test_data = f.readlines()
    
    with open(predictions_file_path, 'r') as f:
        predictions = f.readlines()
    
    list_target = parse_flattened_results_from_file(test_data_path)
    
    assert len(test_data) == len(predictions) == len(list_target)

    # mentioned objects subset
    n_pred_men = 0
    n_true_men = 0
    n_correct_men = 0
    # new objects subset
    n_pred_new = 0
    n_true_new = 0
    n_correct_new = 0

    for line_idx in range(len(test_data)):
        test_data_line = test_data[line_idx]
        pred_line = predictions[line_idx]
        target_ids = list_target[line_idx][0]['objects']
        pred_ids = parse_line_only_coref(pred_line)[0]['objects']

        mentioned_ids = extract_mentioned_ids(test_data_line)

        pred_mentioned_ids = [id for id in pred_ids if id in mentioned_ids]
        pred_new_ids = [id for id in pred_ids if id not in mentioned_ids]
        real_mentioned_ids = [id for id in target_ids if id in mentioned_ids]
        real_new_ids = [id for id in target_ids if id not in mentioned_ids]

        n_pred_men += len(set(pred_mentioned_ids))
        n_pred_new += len(set(pred_new_ids))

        n_true_men += len(set(real_mentioned_ids))
        n_true_new += len(set(real_new_ids))

        n_correct_men += len(set(pred_mentioned_ids).intersection(set(real_mentioned_ids)))
        n_correct_new += len(set(pred_new_ids).intersection(set(real_new_ids)))
    
    rec_men, prec_men, f1_men = rec_prec_f1(n_correct_men, n_true_men, n_pred_men)
    rec_new, prec_new, f1_new = rec_prec_f1(n_correct_new, n_true_new, n_pred_new)
    rec, prec, f1 = rec_prec_f1(n_correct_men+n_correct_new, n_true_men+n_true_new, n_pred_men+n_pred_new)

    report = {'performance on mentioned objects':
                {
                    'F1 score': f1_men,
                    'Precision': prec_men,
                    'Recall': rec_men
                },
              'performance on new objects':
                {
                    'F1 score': f1_new,
                    'Precision': prec_new,
                    'Recall': rec_new
                },
              'overall performance':
                {
                    'F1 score': f1,
                    'Precision': prec,
                    'Recall': rec
                }}

    print("REPORT splitting mentioned and new objects of:", model_name)
    pprint.pprint(report)
    with open(output_file_path, 'w') as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    main()