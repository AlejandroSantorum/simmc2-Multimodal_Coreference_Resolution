import json
import re

STORE_BASE_PATH = "../data_object_special/masked_datasets/"


TRAIN_PRED_PROCESSED_PATH = "../data_object_special/simmc2_dials_dstc10_train_predict.txt"
TRAIN_TARGET_PROCESSED_PATH = "../data_object_special/simmc2_dials_dstc10_train_target.txt"

DEV_PRED_PROCESSED_PATH = "../data_object_special/simmc2_dials_dstc10_dev_predict.txt"
DEV_TARGET_PROCESSED_PATH = "../data_object_special/simmc2_dials_dstc10_dev_target.txt"

DEVTEST_PRED_PROCESSED_PATH = "../data_object_special/simmc2_dials_dstc10_devtest_predict.txt"
DEVTEST_TARGET_PROCESSED_PATH = "../data_object_special/simmc2_dials_dstc10_devtest_target.txt"

DEVTEST_PRED_ONLY_FURN_PATH = "../data_object_special/new_datasets/devtest_only_furniture_predict.txt"
DEVTEST_TARGET_ONLY_FURN_PATH = "../data_object_special/new_datasets/devtest_only_furniture_target.txt"


def read_dataset(datapath):
    with open(datapath, 'r') as f:
        data = f.readlines()
    return data


def _store_data(data, path):
    with open(path, 'w') as f:
        for line in data:
            f.write(line)


def mask_dialogue(dataset, dataset_type='predict'):
    # dataset type can be 'predict' or 'target'
    if dataset_type == "predict":
        keyword = " <SOO>"
    elif dataset_type == "target":
        keyword = " => Belief State"
    else:
        print("Error: dataset_type parameter must be \'predict\' or \'target\'")
        exit()

    masked_dataset = []

    for line in dataset:
        soo_idx = line.find(keyword)
        dial_part = line[:soo_idx].strip()
        #print(dial_part)
        dial_split = dial_part.split(' ')
        for i,word in enumerate(dial_split):
            if (word not in ["User", "System", ':', '']) and ('<' not in word):
                dial_split[i] = ' '

        #print(' '.join(dial_split))
        #exit()
        masked_line = (' '.join(dial_split)).strip() + line[soo_idx:]
        masked_dataset.append(masked_line)

    return masked_dataset


def _build_masked_som(length):
    s = '<SOM> '
    for i in range(length-1):
        s += '<@0000>, '
    return s+'<@0000> '

def mask_mentioned_objs(dataset):
    masked_dataset = []

    for line in dataset:
        som_idx = line.find('<SOM>')
        while som_idx != -1:
            eom_idx = line.find('<EOM>', som_idx)
            n_men_objs = len(re.findall('<', line[som_idx:eom_idx]))-1 # -1 because of '<SOM>'
            masked_part = _build_masked_som(n_men_objs)
            line = line[:som_idx] + masked_part + line[eom_idx:]
            som_idx = line.find('<SOM>', som_idx+1)

        masked_dataset.append(line)
    return masked_dataset


def mask_global_ids(dataset):
    masked_dataset = []

    for line in dataset:
        start_idx = line.find('<@')
        while start_idx != -1:
            end_idx = line.find('>', start_idx)
            line = line[:start_idx] + '<@0000>' + line[end_idx:]
            start_idx = line.find('<@', start_idx+1)

        masked_dataset.append(line)
    return masked_dataset


def mask_bounding_boxes(dataset):
    masked_dataset = []

    for line in dataset:
        start_idx = line.find('[(')
        while start_idx != -1:
            end_idx = line.find(')]', start_idx)
            line = line[:start_idx] + '[(0.0,0.0,0.0,0.0,0.0,0.0' + line[end_idx:]
            start_idx = line.find('[(', start_idx+1)

        masked_dataset.append(line)
    return masked_dataset


def mask_canonical_ids(dataset):
    masked_dataset = []

    for line in dataset:
        start_idx = line.find('<OBJ>')
        while start_idx != -1:
            end_idx = line.find('[(', start_idx)
            line = line[:start_idx] + '<OBJ><199>' + line[end_idx:]
            start_idx = line.find('<OBJ>', start_idx+1)
        
        start_idx = line.find('<PREVIOBJ>')
        while start_idx != -1:
            end_idx = line.find('[(', start_idx)
            line = line[:start_idx] + '<PREVIOBJ><199>' + line[end_idx:]
            start_idx = line.find('<PREVIOBJ>', start_idx+1)

        masked_dataset.append(line)
    return masked_dataset 


def main():
    # train_pred = read_dataset(TRAIN_PRED_PROCESSED_PATH)
    # train_target= read_dataset(TRAIN_TARGET_PROCESSED_PATH)
    # dev_pred = read_dataset(DEV_PRED_PROCESSED_PATH)
    # dev_target= read_dataset(DEV_TARGET_PROCESSED_PATH)
    devtest_pred = read_dataset(DEVTEST_PRED_PROCESSED_PATH)
    devtest_target= read_dataset(DEVTEST_TARGET_PROCESSED_PATH)
    devtest_only_furn_pred = read_dataset(DEVTEST_PRED_ONLY_FURN_PATH)
    devtest_only_furn_target = read_dataset(DEVTEST_TARGET_ONLY_FURN_PATH)

    # Standard devset
    DEVTEST_PRED_MASK_DIAL_STORE_PATH = STORE_BASE_PATH + "devtest_masked_dials_predict.txt"
    DEVTEST_TARGET_MASK_DIAL_STORE_PATH = STORE_BASE_PATH + "devtest_masked_dials_target.txt"
    DEVTEST_PRED_MASK_OBJMEN_STORE_PATH = STORE_BASE_PATH + "devtest_masked_objmen_predict.txt"
    DEVTEST_TARGET_MASK_OBJMEN_STORE_PATH = STORE_BASE_PATH + "devtest_masked_objmen_target.txt"
    DEVTEST_PRED_MASK_GLOBALIDS_STORE_PATH = STORE_BASE_PATH + "devtest_masked_globalIDs_predict.txt"
    DEVTEST_TARGET_MASK_GLOBALIDS_STORE_PATH = STORE_BASE_PATH + "devtest_masked_globalIDs_target.txt"
    DEVTEST_PRED_MASK_BOXES_STORE_PATH = STORE_BASE_PATH + "devtest_masked_bboxes_predict.txt"
    DEVTEST_TARGET_MASK_BOXES_STORE_PATH = STORE_BASE_PATH + "devtest_masked_bboxes_target.txt"
    DEVTEST_PRED_MASK_CANONICAL_STORE_PATH = STORE_BASE_PATH + "devtest_masked_canonicalIDs_predict.txt"
    DEVTEST_TARGET_MASK_CANONICAL_STORE_PATH = STORE_BASE_PATH + "devtest_masked_canonicalIDs_target.txt"

    print("Masking dialogues of devtest sets...")
    devtest_pred_masked_dial = mask_dialogue(devtest_pred, dataset_type='predict')
    devtest_target_masked_dial = mask_dialogue(devtest_target, dataset_type='target')
    _store_data(devtest_pred_masked_dial, DEVTEST_PRED_MASK_DIAL_STORE_PATH)
    _store_data(devtest_target_masked_dial, DEVTEST_TARGET_MASK_DIAL_STORE_PATH)

    print("Masking prev. mentioned objects of devtest sets...")
    devtest_pred_masked_objmen = mask_mentioned_objs(devtest_pred)
    devtest_target_masked_objmen = mask_mentioned_objs(devtest_target)
    _store_data(devtest_pred_masked_objmen, DEVTEST_PRED_MASK_OBJMEN_STORE_PATH)
    _store_data(devtest_target_masked_objmen, DEVTEST_TARGET_MASK_OBJMEN_STORE_PATH)

    print("Masking global IDs of devtest sets...")
    devtest_pred_masked_globalids = mask_global_ids(devtest_pred)
    devtest_target_masked_globalids = mask_global_ids(devtest_target)
    _store_data(devtest_pred_masked_globalids, DEVTEST_PRED_MASK_GLOBALIDS_STORE_PATH)
    _store_data(devtest_target_masked_globalids, DEVTEST_TARGET_MASK_GLOBALIDS_STORE_PATH)

    print("Masking bounding boxes of devtest sets...")
    devtest_pred_masked_boxes = mask_bounding_boxes(devtest_pred)
    devtest_target_masked_boxes = mask_bounding_boxes(devtest_target)
    _store_data(devtest_pred_masked_boxes, DEVTEST_PRED_MASK_BOXES_STORE_PATH)
    _store_data(devtest_target_masked_boxes, DEVTEST_TARGET_MASK_BOXES_STORE_PATH)

    print("Masking canonical IDs of devtest sets...")
    devtest_pred_masked_canonicalIDs = mask_canonical_ids(devtest_pred)
    devtest_target_masked_canonicalIDs = mask_canonical_ids(devtest_target)
    _store_data(devtest_pred_masked_canonicalIDs, DEVTEST_PRED_MASK_CANONICAL_STORE_PATH)
    _store_data(devtest_target_masked_canonicalIDs, DEVTEST_TARGET_MASK_CANONICAL_STORE_PATH)

    # Devtest only furniture
    DEVTEST_FURN_PRED_MASK_DIAL_STORE_PATH = STORE_BASE_PATH + "devtest_only_furn_masked_dials_predict.txt"
    DEVTEST_FURN_TARGET_MASK_DIAL_STORE_PATH = STORE_BASE_PATH + "devtest_only_furn_masked_dials_target.txt"
    DEVTEST_FURN_PRED_MASK_OBJMEN_STORE_PATH = STORE_BASE_PATH + "devtest_only_furn_masked_objmen_predict.txt"
    DEVTEST_FURN_TARGET_MASK_OBJMEN_STORE_PATH = STORE_BASE_PATH + "devtest_only_furn_masked_objmen_target.txt"
    DEVTEST_FURN_PRED_MASK_GLOBALIDS_STORE_PATH = STORE_BASE_PATH + "devtest_only_furn_masked_globalIDs_predict.txt"
    DEVTEST_FURN_TARGET_MASK_GLOBALIDS_STORE_PATH = STORE_BASE_PATH + "devtest_only_furn_masked_globalIDs_target.txt"
    DEVTEST_FURN_PRED_MASK_BOXES_STORE_PATH = STORE_BASE_PATH + "devtest_only_furn_masked_bboxes_predict.txt"
    DEVTEST_FURN_TARGET_MASK_BOXES_STORE_PATH = STORE_BASE_PATH + "devtest_only_furn_masked_bboxes_target.txt"
    DEVTEST_FURN_PRED_MASK_CANONICAL_STORE_PATH = STORE_BASE_PATH + "devtest_only_furn_masked_canonicalIDs_predict.txt"
    DEVTEST_FURN_TARGET_MASK_CANONICAL_STORE_PATH = STORE_BASE_PATH + "devtest_only_furn_masked_canonicalIDs_target.txt"

    print("Masking dialogues of furniture of devtest sets...")
    devtest_only_furnpred_masked_dial = mask_dialogue(devtest_only_furn_pred, dataset_type='predict')
    devtest_only_furntarget_masked_dial = mask_dialogue(devtest_only_furn_target, dataset_type='target')
    _store_data(devtest_only_furnpred_masked_dial, DEVTEST_FURN_PRED_MASK_DIAL_STORE_PATH)
    _store_data(devtest_only_furntarget_masked_dial, DEVTEST_FURN_TARGET_MASK_DIAL_STORE_PATH)

    print("Masking prev. mentioned objects of furniture of devtest sets...")
    devtest_only_furnpred_masked_objmen = mask_mentioned_objs(devtest_only_furn_pred)
    devtest_only_furntarget_masked_objmen = mask_mentioned_objs(devtest_only_furn_target)
    _store_data(devtest_only_furnpred_masked_objmen, DEVTEST_FURN_PRED_MASK_OBJMEN_STORE_PATH)
    _store_data(devtest_only_furntarget_masked_objmen, DEVTEST_FURN_TARGET_MASK_OBJMEN_STORE_PATH)

    print("Masking global IDs of furniture of devtest sets...")
    devtest_only_furnpred_masked_globalids = mask_global_ids(devtest_only_furn_pred)
    devtest_only_furntarget_masked_globalids = mask_global_ids(devtest_only_furn_target)
    _store_data(devtest_only_furnpred_masked_globalids, DEVTEST_FURN_PRED_MASK_GLOBALIDS_STORE_PATH)
    _store_data(devtest_only_furntarget_masked_globalids, DEVTEST_FURN_TARGET_MASK_GLOBALIDS_STORE_PATH)

    print("Masking bounding boxes of furniture of devtest sets...")
    devtest_only_furnpred_masked_boxes = mask_bounding_boxes(devtest_only_furn_pred)
    devtest_only_furntarget_masked_boxes = mask_bounding_boxes(devtest_only_furn_target)
    _store_data(devtest_only_furnpred_masked_boxes, DEVTEST_FURN_PRED_MASK_BOXES_STORE_PATH)
    _store_data(devtest_only_furntarget_masked_boxes, DEVTEST_FURN_TARGET_MASK_BOXES_STORE_PATH)

    print("Masking canonical IDs of furniture of devtest sets...")
    devtest_only_furnpred_masked_canonical = mask_canonical_ids(devtest_only_furn_pred)
    devtest_only_furntarget_masked_canonical = mask_canonical_ids(devtest_only_furn_target)
    _store_data(devtest_only_furnpred_masked_canonical, DEVTEST_FURN_PRED_MASK_CANONICAL_STORE_PATH)
    _store_data(devtest_only_furntarget_masked_canonical, DEVTEST_FURN_TARGET_MASK_CANONICAL_STORE_PATH)

    return


if __name__ == "__main__":
    main()