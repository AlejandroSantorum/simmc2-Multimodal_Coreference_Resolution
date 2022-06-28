import torch
import json

STORE_BASE_PATH = "../data_object_special/new_datasets/"
REF_SETS_BASE_PATH = STORE_BASE_PATH+"reference_sets"


def _get_last_user_turn(line):
    idx_usr = line.rfind('User :')
    idx_soo = line.find(' <SOO>')
    if idx_soo != -1:
        return line[idx_usr+len('User : '):idx_soo]
    return line[idx_usr+len('User : '):]


def get_out_domain_set(train_pred_processed_path, train_target_processed_path):
    out_domain_predict = []
    out_domain_target = []

    with open(train_target_processed_path, 'r') as f:
        train_target_data = f.readlines()
    
    with open(train_pred_processed_path, 'r') as f:
        for line_idx, line in enumerate(f.readlines()):
            if len(line) > 1:
                idx = line.find("<@")
                if line[idx+2] == '2': # furniture example
                    out_domain_predict.append(line)
                    out_domain_target.append(train_target_data[line_idx])
    
    return out_domain_predict, out_domain_target


def get_all_fashion_examples(predict_path_list, target_path_list):
    all_predict_ex = []
    all_target_ex = []

    assert len(predict_path_list) == len(target_path_list)

    for i in range(len(predict_path_list)):
        with open(target_path_list[i], 'r') as f:
            target_data = f.readlines()
        
        with open(predict_path_list[i], 'r') as f:
            for line_idx, line in enumerate(f.readlines()):
                if len(line) > 1:
                    idx = line.find("<@")
                    if line[idx+2] == '1': # fashion example
                        all_predict_ex.append(line)
                        all_target_ex.append(target_data[line_idx])

    return all_predict_ex, all_target_ex


def get_examples_given_ref(ref_path, all_fash_pred_examples, all_fash_target_examples):
    with open(ref_path, 'r') as f:
        ref_data = json.load(f)
    
    pred_examples = []
    target_examples = []

    for line in ref_data:
        last_user_turn = _get_last_user_turn(line['dial'])
        for idx, example in enumerate(all_fash_pred_examples):
            if last_user_turn == _get_last_user_turn(example):
                pred_examples.append(example)
                target_examples.append(all_fash_target_examples[idx])
    
    return pred_examples, target_examples


def _store_data(data, path):
    with open(path, 'w') as f:
        for line in data:
            f.write(line)


def main():
    TRAIN_PRED_PROCESSED_PATH = "../data_object_special/simmc2_dials_dstc10_train_predict.txt"
    TRAIN_TARGET_PROCESSED_PATH = "../data_object_special/simmc2_dials_dstc10_train_target.txt"

    DEV_PRED_PROCESSED_PATH = "../data_object_special/simmc2_dials_dstc10_dev_predict.txt"
    DEV_TARGET_PROCESSED_PATH = "../data_object_special/simmc2_dials_dstc10_dev_target.txt"

    DEVTEST_PRED_PROCESSED_PATH = "../data_object_special/simmc2_dials_dstc10_devtest_predict.txt"
    DEVTEST_TARGET_PROCESSED_PATH = "../data_object_special/simmc2_dials_dstc10_devtest_target.txt"

    OUT_DOMAIN_REF_PATH = REF_SETS_BASE_PATH + "/out_of_domain_test.json"
    IN_DOMAIN_HELD_OUT_REF_PATH = REF_SETS_BASE_PATH + "/in_domain_held_out_test.json"
    IN_DOMAIN_REF_PATH = REF_SETS_BASE_PATH + "/in_domain_test.json"
    TRAIN_REF_PATH = REF_SETS_BASE_PATH + "/seen_unseen_OOD_train.json"

    OUT_DOMAIN_PRED_STORE_PATH = STORE_BASE_PATH + "/out_of_domain_predict.txt"
    OUT_DOMAIN_TARGET_STORE_PATH = STORE_BASE_PATH + "/out_of_domain_target.txt"

    IN_DOMAIN_HELD_OUT_PRED_STORE_PATH = STORE_BASE_PATH + "/in_domain_held_out_predict.txt"
    IN_DOMAIN_HELD_OUT_TARGET_STORE_PATH = STORE_BASE_PATH + "/in_domain_held_out_target.txt"

    IN_DOMAIN_PRED_STORE_PATH = STORE_BASE_PATH + "/in_domain_predict.txt"
    IN_DOMAIN_TARGET_STORE_PATH = STORE_BASE_PATH + "/in_domain_target.txt"

    TRAIN_PRED_STORE_PATH = STORE_BASE_PATH + "/train_seen_unseen_OOD_predict.txt"
    TRAIN_TARGET_STORE_PATH = STORE_BASE_PATH + "/train_seen_unseen_OOD_target.txt"

    # Get all fashion examples from the three original sets: train, dev and devtest
    pred_path_list = [TRAIN_PRED_PROCESSED_PATH, DEV_PRED_PROCESSED_PATH, DEVTEST_PRED_PROCESSED_PATH]
    target_path_list = [TRAIN_TARGET_PROCESSED_PATH, DEV_TARGET_PROCESSED_PATH, DEVTEST_TARGET_PROCESSED_PATH]
    all_fash_pred_ex, all_fash_target_ex = get_all_fashion_examples(pred_path_list, target_path_list)

    print("Getting Out-Of-Domain sets...")
    out_domain_predict, out_domain_target = get_out_domain_set(TRAIN_PRED_PROCESSED_PATH, TRAIN_TARGET_PROCESSED_PATH)
    print("\tStoring...")
    _store_data(out_domain_predict, OUT_DOMAIN_PRED_STORE_PATH)
    _store_data(out_domain_target, OUT_DOMAIN_TARGET_STORE_PATH)

    print("Getting In-Domain-Held-Out sets...")
    in_dom_held_out_predict, in_dom_held_out_target = get_examples_given_ref(IN_DOMAIN_HELD_OUT_REF_PATH, all_fash_pred_ex, all_fash_target_ex)
    print("\tStoring...")
    _store_data(in_dom_held_out_predict, IN_DOMAIN_HELD_OUT_PRED_STORE_PATH)
    _store_data(in_dom_held_out_target, IN_DOMAIN_HELD_OUT_TARGET_STORE_PATH)

    print("Getting In-Domain sets...")
    in_dom_predict, in_dom_target = get_examples_given_ref(IN_DOMAIN_REF_PATH, all_fash_pred_ex, all_fash_target_ex)
    print("\tStoring...")
    _store_data(in_dom_predict, IN_DOMAIN_PRED_STORE_PATH)
    _store_data(in_dom_target, IN_DOMAIN_TARGET_STORE_PATH)

    print("Getting Training sets...")
    train_predict, train_target = get_examples_given_ref(TRAIN_REF_PATH, all_fash_pred_ex, all_fash_target_ex)
    print("\tStoring...")
    _store_data(train_predict, TRAIN_PRED_STORE_PATH)
    _store_data(train_target, TRAIN_TARGET_STORE_PATH)



if __name__ == "__main__":
    main()