import json, pprint, argparse
from utils import parse_line_only_coref, rec_prec_f1


def open_file(file_path):
    extension = file_path[file_path.rfind('.'):]

    predictions = []
    if extension == '.txt':
        with open(file_path, 'r') as f:
            data = f.readlines()
        for line in data:
            predictions.append(parse_line_only_coref(line)[0]['objects'])
    else:
        with open(file_path, 'r') as f:
            data = json.load(f)
        for dial in data['dialogue_data']:
            for turn in dial['dialogue']:
                predictions.append(turn['transcript_annotated']['act_attributes']['objects'])
    return predictions


def open_uniter_file(uniter_file_path):
    predictions = []
    with open(uniter_file_path, 'r') as f:
        data = json.load(f)
    for dial in data['dialogue_data']:
        for turn in dial['dialogue']:
            predictions.append(turn['transcript_annotated']['act_attributes']['objects'])

    return predictions


def open_bart_file(bart_file_path):
    predictions = []
    with open(bart_file_path, 'r') as f:
        data = f.readlines()
    for line in data:
        predictions.append(parse_line_only_coref(line)[0]['objects'])
    return predictions




def main(args):
    model_for_new_objs_path = args.model_for_new_objs_path
    model_for_mentioned_objs_path = args.model_for_mentioned_objs_path
    target_file_path = args.targets_file_path
    mentioned_filepath = args.mentioned_ids_path

    men_predictions = open_uniter_file(model_for_mentioned_objs_path)
    new_predictions = open_bart_file(model_for_new_objs_path)
    with open(mentioned_filepath, 'r') as f:
        mentioned_list = json.load(f)
    assert len(men_predictions) == len(new_predictions) == len(mentioned_list)

    with open(target_file_path, 'r') as f:
        test_data = json.load(f)

    # mentioned objects subset
    n_pred_men = 0
    n_true_men = 0
    n_correct_men = 0
    # new objects subset
    n_pred_new = 0
    n_true_new = 0
    n_correct_new = 0

    cont = 0
    for dial in test_data['dialogue_data']:
        for turn in dial['dialogue']:
            try:
                if turn['disambiguation_label']:
                    cont += 1
                    continue
            except: pass

            targets = turn['transcript_annotated']['act_attributes']['objects']
            men_model_preds = men_predictions[cont]
            new_model_preds = new_predictions[cont]
            mentioned_ids = mentioned_list[cont]

            pred_men_ids = [id for id in men_model_preds if id in mentioned_ids]
            pred_new_ids = [id for id in new_model_preds if id not in mentioned_ids]

            true_men_ids = [id for id in targets if id in mentioned_ids]
            true_new_ids = [id for id in targets if id not in mentioned_ids]

            n_pred_men += len(set(pred_men_ids))
            n_pred_new += len(set(pred_new_ids))

            n_true_men += len(set(true_men_ids))
            n_true_new += len(set(true_new_ids))

            n_correct_men += len(set(true_men_ids).intersection(set(pred_men_ids)))
            n_correct_new += len(set(true_new_ids).intersection(set(pred_new_ids)))

            cont += 1

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # path of the BART model predictions file (will focus on non-mentioned objects)
    parser.add_argument('--model_for_new_objs_path', default='../model_outputs/predictions_BART_input_all_attrs_cp381.txt')
    # path of the UNITER model predictions file (will focus on mentioned objects)
    parser.add_argument('--model_for_mentioned_objs_path', default='../model_outputs/predictions_UNITER_basic_all_objmen_noIDs_devtest.json')
    # path of the file with the targets
    parser.add_argument('--targets_file_path', default='../targets/simmc2_dials_dstc10_devtest.json')
    # path file with the mentioned object IDs
    parser.add_argument('--mentioned_ids_path', default='../utils/mentioned_ids.json')

    args = parser.parse_args()
    main(args)