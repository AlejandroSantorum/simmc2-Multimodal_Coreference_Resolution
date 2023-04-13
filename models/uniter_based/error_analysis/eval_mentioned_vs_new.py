import argparse
import json
import pprint
import numpy as np


def rec_prec_f1(n_correct, n_true, n_pred):
    rec = n_correct / n_true if n_true != 0 else 0
    prec = n_correct / n_pred if n_pred != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

    return rec, prec, f1


def d_f1(n_correct, n_true, n_pred):
    # 1/r + 1/p = 2/F1
    # dr / r^2 + dp / p^2 = 2dF1 /F1^2
    # dF1 = 1/2 F1^2 (dr/r^2 + dp/p^2)
    dr = b_stderr(n_true, n_correct)
    dp = b_stderr(n_pred, n_correct)

    r = n_correct / n_true
    p = n_correct / n_pred
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0

    d_f1 = 0.5 * f1 ** 2 * (dr / r ** 2 + dp / p ** 2)
    return d_f1


def b_stderr(n_total, n_pos):
    return np.std(b_arr(n_total, n_pos)) / np.sqrt(n_total)


def b_arr(n_total, n_pos):
    out = np.zeros(int(n_total))
    out[: int(n_pos)] = 1.0
    return out


def main(args):
    test_data_path = args.test_data_path
    predictions_file_path = args.predictions_file_path
    if args.output_file_path != '':
        output_file_path = args.output_file_path
    else:
        results_path = predictions_file_path[:predictions_file_path.find('/eval_')]
        model_name = predictions_file_path[predictions_file_path.find("/eval_")+len("/eval_"):predictions_file_path.find(".json")]
        output_file_path = results_path + "/split_f1_report_"+model_name+".txt"

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    with open(predictions_file_path, 'r') as f:
        eval_data = json.load(f)

    # mentioned objects subset
    n_pred_men = 0
    n_true_men = 0
    n_correct_men = 0
    # new objects subset
    n_pred_new = 0
    n_true_new = 0
    n_correct_new = 0

    cont = 0
    for dial in eval_data['dialogue_data']:
        for turn in dial['dialogue']:
            try:
                if turn['disambiguation_label'] == 1:
                    continue
            except:
                pass

            mentioned_ids = [id for i,id in enumerate(test_data[cont]['candidate_ids']) if test_data[cont]['candidate_mentioned'][i]]
            cont += 1

            pred_mentioned_ids = [id for id in turn['transcript_annotated']['act_attributes']['objects'] if id in mentioned_ids]
            pred_new_ids = [id for id in turn['transcript_annotated']['act_attributes']['objects'] if id not in mentioned_ids]
            real_mentioned_ids = [id for id in turn['transcript_annotated']['act_attributes']['objects_real'] if id in mentioned_ids]
            real_new_ids = [id for id in turn['transcript_annotated']['act_attributes']['objects_real'] if id not in mentioned_ids]
    
            n_pred_men += len(set(pred_mentioned_ids))
            n_pred_new += len(set(pred_new_ids))

            n_true_men += len(set(real_mentioned_ids))
            n_true_new += len(set(real_new_ids))

            n_correct_men += len(set(real_mentioned_ids).intersection(set(pred_mentioned_ids)))
            n_correct_new += len(set(real_new_ids).intersection(set(pred_new_ids)))
    
    rec_men, prec_men, f1_men = rec_prec_f1(n_correct_men, n_true_men, n_pred_men)
    rec_new, prec_new, f1_new = rec_prec_f1(n_correct_new, n_true_new, n_pred_new)
    rec, prec, f1 = rec_prec_f1(n_correct_men+n_correct_new, n_true_men+n_true_new, n_pred_men+n_pred_new)

    stderr_men = d_f1(n_correct_men, n_true_men, n_pred_men)
    stderr_new = d_f1(n_correct_new, n_true_new, n_pred_new)
    stderr = d_f1(n_correct_men+n_correct_new, n_true_men+n_true_new, n_pred_men+n_pred_new)

    report = {'performance on mentioned objects':
                {
                    'F1 score': f1_men,
                    'Precision': prec_men,
                    'Recall': rec_men,
                    'Std. error': stderr_men
                },
              'performance on new objects':
                {
                    'F1 score': f1_new,
                    'Precision': prec_new,
                    'Recall': rec_new,
                    'Std. error': stderr_new
                },
              'overall performance':
                {
                    'F1 score': f1,
                    'Precision': prec,
                    'Recall': rec,
                    'Std. error': stderr
                }
            }

    print("REPORT splitting mentioned and new objects of:", model_name)
    pprint.pprint(report)
    with open(output_file_path, 'w') as f:
        json.dump(report, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # path of the test set data
    parser.add_argument('--test_data_path', default='../processed/devtest.json')
    # path of the model predictions file
    parser.add_argument('--predictions_file_path', default='../output/eval_UNITER_basic_all_objmen_noIDs_devtest.json')
    # path to store the analysis results: by default their are placed in the 'output' folder
    parser.add_argument('--output_file_path', default='')
    
    args = parser.parse_args()
    main(args)