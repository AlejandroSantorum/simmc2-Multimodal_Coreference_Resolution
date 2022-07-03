import json
from params import TRAIN_PROCESSED_DATAPATH, DEVTEST_PROCESSED_DATAPATH


def collect_examples(dataset_path, train_list, test_list):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for line in data:
        if 'cloth_store_1498649_woman' in line['scenes'][0]:
            test_list.append(line)
        else:
            train_list.append(line)
    
    return train_list, test_list


def main():
    train_store_path = "../../processed/new_datasets/woman49_store_train.json"
    test_store_path = "../../processed/new_datasets/woman49_store_test.json"

    train_list = []
    test_list = []
    train_list, test_list = collect_examples(TRAIN_PROCESSED_DATAPATH, train_list, test_list)
    train_list, test_list = collect_examples(DEVTEST_PROCESSED_DATAPATH, train_list, test_list)

    with open(train_store_path, 'w') as f:
        json.dump(train_list, f)

    with open(test_store_path, 'w') as f:
        json.dump(test_list, f)


if __name__ == '__main__':
    main()