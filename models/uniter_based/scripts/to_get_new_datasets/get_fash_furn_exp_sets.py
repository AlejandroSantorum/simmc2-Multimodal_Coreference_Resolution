import json
from params import TRAIN_PROCESSED_DATAPATH, DEVTEST_PROCESSED_DATAPATH


def collect_examples(dataset_path, fash_train_list, furn_test_list):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for line in data:
        if line['domain'] == 'fashion':
            fash_train_list.append(line)
        elif line['domain'] == 'furniture':
            furn_test_list.append(line)
        else:
            print("Error: undefined domain ->", line['domain'])
            exit()
    
    return fash_train_list, furn_test_list


def main():
    fash_train_store_path = "../../processed/new_datasets/train_on_just_fashion.json"
    furn_test_store_path = "../../processed/new_datasets/test_on_just_furniture.json"

    fash_train_list = []
    furn_test_list = []
    fash_train_list, furn_test_list = collect_examples(TRAIN_PROCESSED_DATAPATH, fash_train_list, furn_test_list)
    fash_train_list, furn_test_list = collect_examples(DEVTEST_PROCESSED_DATAPATH, fash_train_list, furn_test_list)

    with open(fash_train_store_path, 'w') as f:
        json.dump(fash_train_list, f)

    with open(furn_test_store_path, 'w') as f:
        json.dump(furn_test_list, f)


if __name__ == '__main__':
    main()