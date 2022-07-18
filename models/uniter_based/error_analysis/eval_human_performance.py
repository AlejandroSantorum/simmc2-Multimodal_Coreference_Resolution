import json


def rec_prec_f1(n_correct, n_true, n_pred):
    rec = n_correct / n_true if n_true != 0 else 0
    prec = n_correct / n_pred if n_pred != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

    return rec, prec, f1


def main():
    predictions_path = "./pdf_error_analysis/estim_human_perform_asantorum.json"
    subset_data_path = "../processed/random_test_subset/random_devtest_samples.json"

    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    with open(subset_data_path, 'r') as f:
        test_data = json.load(f)
    
    assert len(test_data) == len(predictions)
    
    n_pred = 0
    n_true = 0
    n_correct = 0
    for i in range(len(predictions)):
        n_pred += len(predictions[i])
        n_true += len(test_data[i]['reference_idx'])

        reference = [test_data[i]['candidate_ids'][idx] for idx in test_data[i]['reference_idx']]
        n_correct += len(set(reference).intersection(predictions[i]))

    rec, prec, f1 = rec_prec_f1(n_correct, n_true, n_pred)
    print("F1 score:", f1)
    print("Precision:", prec)
    print("Recall:", rec)



if __name__ == "__main__":
    main()