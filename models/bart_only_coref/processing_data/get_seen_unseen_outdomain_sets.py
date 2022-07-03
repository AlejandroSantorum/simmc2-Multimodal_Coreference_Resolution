import torch
import json
from tqdm import tqdm

STORE_BASE_PATH = "../data_object_special/new_datasets/"
REF_SETS_BASE_PATH = STORE_BASE_PATH+"reference_sets"


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

DEVTEST_FURN_PRED_STORE_PATH = STORE_BASE_PATH + "/devtest_only_furniture_predict.txt"
DEVTEST_FURN_TARGET_STORE_PATH = STORE_BASE_PATH + "/devtest_only_furniture_target.txt"

DEVTEST_FASH_PRED_STORE_PATH = STORE_BASE_PATH + "/devtest_only_fashion_predict.txt"
DEVTEST_FASH_TARGET_STORE_PATH = STORE_BASE_PATH + "/devtest_only_fashion_target.txt"

ALL_FURN_PRED_STORE_PATH = STORE_BASE_PATH + "/all_furniture_predict.txt"
ALL_FURN_TARGET_STORE_PATH = STORE_BASE_PATH + "/all_furniture_target.txt"


def get_out_domain_set(train_pred_processed_path, train_target_processed_path, keyword='furniture'):
    out_domain_predict = []
    out_domain_target = []

    if keyword == 'furniture':
        look_for_str = '2'
    elif keyword == 'fashion':
        look_for_str = '1'
    else:
        print("Error: keyword parameter must be fashion or furniture")
        exit()

    with open(train_target_processed_path, 'r') as f:
        train_target_data = f.readlines()
    
    with open(train_pred_processed_path, 'r') as f:
        for line_idx, line in enumerate(f.readlines()):
            if len(line) > 1:
                idx = line.find("<@")
                if line[idx+2] == look_for_str: # keyword is usally furniture
                    out_domain_predict.append(line)
                    out_domain_target.append(train_target_data[line_idx])
    
    return out_domain_predict, out_domain_target


def get_all_domain_examples(predict_path_list, target_path_list, domain='fashion'):
    all_predict_ex = []
    all_target_ex = []

    assert len(predict_path_list) == len(target_path_list)

    if domain == 'fashion':
        look_for_str = '1'
    elif domain == 'furniture':
        look_for_str = '2'
    else:
        print("Error: domain parameter must be fashion or furniture")
        exit()

    for i in range(len(predict_path_list)):
        with open(target_path_list[i], 'r') as f:
            target_data = f.readlines()
        
        with open(predict_path_list[i], 'r') as f:
            for line_idx, line in enumerate(f.readlines()):
                if len(line) > 1:
                    idx = line.find("<@")
                    if line[idx+2] == look_for_str: # look_for_str is usually '1' (fashion example)
                        all_predict_ex.append(line)
                        all_target_ex.append(target_data[line_idx])

    return all_predict_ex, all_target_ex


def _get_last_sys_turn(line):
    check = line.find("<@")
    idx_sys = line.rfind('System :')
    if check == -1: # UNITER MODEL
        idx_som = line.rfind(' System mentions :')
        if idx_som == -1 or idx_som < idx_sys:
            idx_som = line.rfind(' User :')

    else: # BART MODEL
        idx_som = line.rfind(' <SOM>')
    
    if idx_sys != -1 and idx_som != -1:
        return line[idx_sys+len('System : '):idx_som]
    if idx_sys != -1:
        return line[idx_sys+len('System : '):]
    return -1


def _get_last_user_turn(line):
    idx_usr = line.rfind('User :')
    idx_soo = line.find(' <SOO>')
    if idx_soo != -1:
        return line[idx_usr+len('User : '):idx_soo]
    return line[idx_usr+len('User : '):]


def _get_line_object_ids(line):
    line_ids = []
    pos = 0
    idx = line.find("<@", pos)
    while idx != -1:
        # get absolute object ID
        abs_id = line[idx+3:idx+6]
        line_ids.append(int(abs_id)+1)
        # update pos and idx
        pos = idx+4
        idx = line.find("<@", pos)
    return line_ids


def get_examples_given_ref(ref_path, all_fash_pred_examples, all_fash_target_examples):
    with open(ref_path, 'r') as f:
        ref_data = json.load(f)
    
    pred_examples = []
    target_examples = []

    for line in tqdm(ref_data):
        check = False
        aux_list = []
        last_user_turn = _get_last_user_turn(line['dial'])
        last_sys_turn = _get_last_sys_turn(line['dial'])
        KB_ids = line['KB_ids']
        temp_pred = []
        temp_target = []
        for idx, example in enumerate(all_fash_pred_examples):
            if last_user_turn == _get_last_user_turn(example) and last_sys_turn == _get_last_sys_turn(example):
                aux_list.append((example, idx))
                if set(sorted(_get_line_object_ids(example))) <= set(sorted(KB_ids)):
                    check = True
                    temp_pred.append(example)
                    temp_target.append(all_fash_target_examples[idx])
        if not check and len(aux_list) == 1: # Object sets didn't match, but we can keep the only one possible match
            pred_examples.append(aux_list[0][0])
            target_examples.append(all_fash_target_examples[aux_list[0][1]])
            continue
        if len(temp_pred) > 1:
            # if 'bad image' in last_user_turn: continue
            continue
        if not check:
            print("Error: no matchings")
            break
        pred_examples += temp_pred
        target_examples += temp_target
    
    return pred_examples, target_examples


def _store_data(data, path):
    with open(path, 'w') as f:
        for line in data:
            f.write(line)


def main():
    # Get all fashion examples from the three original sets: train, dev and devtest
    pred_path_list = [TRAIN_PRED_PROCESSED_PATH, DEV_PRED_PROCESSED_PATH, DEVTEST_PRED_PROCESSED_PATH]
    target_path_list = [TRAIN_TARGET_PROCESSED_PATH, DEV_TARGET_PROCESSED_PATH, DEVTEST_TARGET_PROCESSED_PATH]
    all_fash_pred_ex, all_fash_target_ex = get_all_domain_examples(pred_path_list, target_path_list, domain='fashion')

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

    print("Getting furniture subsets of devtest...")
    devtest_furn_predict, devtest_furn_target = get_out_domain_set(DEVTEST_PRED_PROCESSED_PATH, DEVTEST_TARGET_PROCESSED_PATH, keyword='furniture')
    print("\tStoring...")
    _store_data(devtest_furn_predict, DEVTEST_FURN_PRED_STORE_PATH)
    _store_data(devtest_furn_target, DEVTEST_FURN_TARGET_STORE_PATH)

    print("Getting fashion subsets of devtest...")
    devtest_fash_predict, devtest_fash_target = get_out_domain_set(DEVTEST_PRED_PROCESSED_PATH, DEVTEST_TARGET_PROCESSED_PATH, keyword='fashion')
    print("\tStoring...")
    _store_data(devtest_fash_predict, DEVTEST_FASH_PRED_STORE_PATH)
    _store_data(devtest_fash_target, DEVTEST_FASH_TARGET_STORE_PATH)

    print("Getting all furniture examples from the three original sets: train, dev and devtest...")
    pred_path_list = [TRAIN_PRED_PROCESSED_PATH, DEV_PRED_PROCESSED_PATH, DEVTEST_PRED_PROCESSED_PATH]
    target_path_list = [TRAIN_TARGET_PROCESSED_PATH, DEV_TARGET_PROCESSED_PATH, DEVTEST_TARGET_PROCESSED_PATH]
    all_furn_pred_ex, all_furn_target_ex = get_all_domain_examples(pred_path_list, target_path_list, domain='furniture')
    print("\tStoring...")
    _store_data(all_furn_pred_ex, ALL_FURN_PRED_STORE_PATH)
    _store_data(all_furn_target_ex, ALL_FURN_TARGET_STORE_PATH)


if __name__ == "__main__":
    main()