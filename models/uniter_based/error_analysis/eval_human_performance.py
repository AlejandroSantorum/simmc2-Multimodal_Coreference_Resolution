import json
import argparse
import pprint

def rec_prec_f1(n_correct, n_true, n_pred):
    rec = n_correct / n_true if n_true != 0 else 0
    prec = n_correct / n_pred if n_pred != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

    return rec, prec, f1


def main(args):
    predictions_path = args.predictions_path
    subset_data_path = args.subset_data_path

    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    with open(subset_data_path, 'r') as f:
        test_data = json.load(f)
    
    assert len(test_data) == len(predictions)

    # mentioned objects subset
    n_pred_men = 0
    n_true_men = 0
    n_correct_men = 0
    # new objects subset
    n_pred_new = 0
    n_true_new = 0
    n_correct_new = 0
    for i in range(len(predictions)):
        mentioned_ids = [id for j,id in enumerate(test_data[i]['candidate_ids']) if test_data[i]['candidate_mentioned'][j]]
        reference = [test_data[i]['candidate_ids'][idx] for idx in test_data[i]['reference_idx']]

        pred_mentioned_ids = [id for id in predictions[i] if id in mentioned_ids]
        pred_new_ids = [id for id in predictions[i] if id not in mentioned_ids]
        real_mentioned_ids = [id for id in reference if id in mentioned_ids]
        real_new_ids = [id for id in reference if id not in mentioned_ids]

        n_pred_men += len(set(pred_mentioned_ids))
        n_pred_new += len(set(pred_new_ids))

        n_true_men += len(set(real_mentioned_ids))
        n_true_new += len(set(real_new_ids))

        n_correct_men += len(set(real_mentioned_ids).intersection(set(pred_mentioned_ids)))
        n_correct_new += len(set(real_new_ids).intersection(set(pred_new_ids)))
    
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

    pprint.pprint(report)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--predictions_path', default="./estimate_human_performance/estim_human_perform_asantorum.json")
    parser.add_argument('--subset_data_path', default="../processed/random_test_subset/random_devtest_samples.json")

    args = parser.parse_args()
    main(args)