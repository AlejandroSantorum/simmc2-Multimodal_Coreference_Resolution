import json
from params import TRAIN_DSTC10_DATAPATH, DEVTEST_DSTC10_DATAPATH


def collect_furniture_dials(dataset_path, furn_dials_list):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    for dial in data['dialogue_data']:
        if dial['domain'] == 'furniture':
            furn_dials_list.append(dial)
    
    return furn_dials_list


def main():
    store_path = "../../processed/new_datasets/test_furniture_target_dials.json"

    furn_dials_list = []
    furn_dials_list = collect_furniture_dials(TRAIN_DSTC10_DATAPATH, furn_dials_list)
    furn_dials_list = collect_furniture_dials(DEVTEST_DSTC10_DATAPATH, furn_dials_list)

    furn_dials_dict = {'dialogue_data': furn_dials_list}
    with open(store_path, 'w') as f:
        json.dump(furn_dials_dict, f)


if __name__ == "__main__":
    main()