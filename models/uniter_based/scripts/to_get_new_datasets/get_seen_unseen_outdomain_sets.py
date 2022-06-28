import json
import random
from params import TRAIN_DSTC10_DATAPATH, DEV_DSTC10_DATAPATH, DEVTEST_DSTC10_DATAPATH
from params import TRAIN_PROCESSED_DATAPATH, DEV_PROCESSED_DATAPATH, DEVTEST_PROCESSED_DATAPATH



def get_out_domain_sets(train_processed_path, train_dstc10_path):
    # Getting examples of Out-Of-Domain test set
    with open(train_processed_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_set = []
    for line in data:
        if line['domain'] == 'furniture':
            test_set.append(line)
    
    # Getting examples of Out-Of-Domain target set
    with open(train_dstc10_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    target_set = []
    for dial in data['dialogue_data']:
        if dial['domain'] == 'furniture':
            target_set.append(dial)
    
    return test_set, target_set
    


def get_all_fashion_examples(processed_sets_path_list, dstc10_sets_path_list):
    all_processed_fashion_examples = []
    all_target_fashion_examples = []

    for set_path in processed_sets_path_list:
        with open(set_path, 'r') as f:
            data = json.load(f)
        for line in data:
            if line['domain'] == 'fashion':
                all_processed_fashion_examples.append(line)
    
    for set_path in dstc10_sets_path_list:
        with open(set_path, 'r') as f:
            data = json.load(f)
        for dial in data['dialogue_data']:
            if dial['domain'] == 'fashion':
                all_target_fashion_examples.append(dial)
    
    return all_processed_fashion_examples, all_target_fashion_examples


        
def extract_woman49_examples(all_processed_fashion_examples, all_target_fashion_examples):
    woman49_processed_examples = []
    remaining_processed_examples = []
    woman49_target_examples = []
    remaining_target_examples = []

    for line in all_processed_fashion_examples:
        if 'cloth_store_1498649_woman' in line['scenes'][0]:
            woman49_processed_examples.append(line)
        else:
            remaining_processed_examples.append(line)
    
    for dial in all_target_fashion_examples:
        if 'cloth_store_1498649_woman' in dial['scene_ids']['0']:
            woman49_target_examples.append(dial)
        else:
            remaining_target_examples.append(dial)
    
    return woman49_processed_examples, remaining_processed_examples, woman49_target_examples, remaining_target_examples



def _get_remaining_dials_indices(remaining_target):
    indices = []
    for dial in remaining_target:
        indices.append(dial['dialogue_idx'])
    return indices


def extract_in_domain_examples(remaining_processed, remaining_target, size):
    in_domain_test = []
    in_domain_target = []

    remaining_dials_indices = _get_remaining_dials_indices(remaining_target)
    used_dials_ids = []

    while len(in_domain_test) < size:
        dial_idx = random.sample(remaining_dials_indices,1)[0]
        remaining_dials_indices.remove(dial_idx)

        for line in remaining_processed:
            if line['dial_idx'] == dial_idx:
                in_domain_test.append(line)
        
        for dial in remaining_target:
            if dial['dialogue_idx'] == dial_idx:
                in_domain_target.append(dial)
        
        used_dials_ids.append(dial_idx)
    
    train_set = []
    for line in remaining_processed:
        if line['dial_idx'] not in used_dials_ids:
            train_set.append(line)
    
    return in_domain_test, in_domain_target, train_set



def main():
    out_domain_test_save_path = '../../processed/new_datasets/out_of_domain_test.json'
    out_domain_target_save_path = '../../processed/new_datasets/out_of_domain_target.json'

    in_domain_held_out_test_save_path = "../../processed/new_datasets/in_domain_held_out_test.json"
    in_domain_held_out_target_save_path = "../../processed/new_datasets/in_domain_held_out_target.json"

    in_domain_test_save_path = "../../processed/new_datasets/in_domain_test.json"
    in_domain_target_save_path = "../../processed/new_datasets/in_domain_target.json"

    train_set_save_path = "../../processed/new_datasets/seen_unseen_OOD_train.json"

    ###
    print("Building test and target sets for OUT-OF-DOMAIN experiment...")
    out_domain_set, out_domain_target = get_out_domain_sets(TRAIN_PROCESSED_DATAPATH, TRAIN_DSTC10_DATAPATH)
    # Storing Out-of-Domain test set
    with open(out_domain_test_save_path, 'w') as f:
        json.dump(out_domain_set, f)
    # Storing Out-of-Domain target set
    out_domain_target_dict = {'dialogue_data': out_domain_target}
    with open(out_domain_target_save_path, 'w') as f:
        json.dump(out_domain_target_dict, f)

    print("Getting all fashion examples in original train, dev and devtest sets...")
    processed_sets_path_list = [TRAIN_PROCESSED_DATAPATH, DEV_PROCESSED_DATAPATH, DEVTEST_PROCESSED_DATAPATH]
    dstc10_sets_path_list = [TRAIN_DSTC10_DATAPATH, DEV_DSTC10_DATAPATH, DEVTEST_DSTC10_DATAPATH]
    all_processed_fashion_examples, all_target_fashion_examples = get_all_fashion_examples(processed_sets_path_list, dstc10_sets_path_list)

    ###
    print("Building test and target sets for IN-DOMAIN-HELD-OUT experiment...")
    woman49_processed, remaining_processed, woman49_target, remaining_target = extract_woman49_examples(all_processed_fashion_examples, all_target_fashion_examples)
    # Storing In-Domain-Held-Out test set
    with open(in_domain_held_out_test_save_path, 'w') as f:
        json.dump(woman49_processed, f)
    # Storing In-Domain-Held-Out target set
    woman49_target_dict = {'dialogue_data': woman49_target}
    with open(in_domain_held_out_target_save_path, 'w') as f:
        json.dump(woman49_target_dict, f)

    ###
    print("Building test and target sets for IN-DOMAIN experiment...")
    size = 9000
    in_domain_test, in_domain_target, train = extract_in_domain_examples(remaining_processed, remaining_target, size)
    # Storing In-Domain test set
    with open(in_domain_test_save_path, 'w') as f:
        json.dump(in_domain_test, f)
    # Storing In-Domain target set
    in_domain_target_dict = {'dialogue_data': in_domain_target}
    with open(in_domain_target_save_path, 'w') as f:
        json.dump(in_domain_target_dict, f)
    
    ###
    # Storing training set (remaining examples)
    with open(train_set_save_path, 'w') as f:
        json.dump(train, f)


if __name__ == '__main__':
    main()