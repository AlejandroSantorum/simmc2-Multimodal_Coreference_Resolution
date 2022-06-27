import json


def collect_furniture_dials(dataset_path, furn_dials_list):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    for dial in data['dialogue_data']:
        if dial['domain'] == 'furniture':
            furn_dials_list.append(dial)
    
    return furn_dials_list


def main():
    train_data_path = "../data/simmc2_dials_dstc10_train.json"
    devtest_data_path = "../data/simmc2_dials_dstc10_devtest.json"
    store_path = "../processed/test_furniture_target_dials.json"

    furn_dials_list = []
    furn_dials_list = collect_furniture_dials(train_data_path, furn_dials_list)
    furn_dials_list = collect_furniture_dials(devtest_data_path, furn_dials_list)

    furn_dials_dict = {'dialogue_data': furn_dials_list}
    with open(store_path, 'w') as f:
        json.dump(furn_dials_dict, f)


if __name__ == "__main__":
    main()