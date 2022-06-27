import json


def collect_special_woman_store_dials(dataset_path, dials_list):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    for dial in data['dialogue_data']:
        if 'cloth_store_1498649_woman' in dial['scene_ids']['0']:
            dials_list.append(dial)
    
    return dials_list



def main():
    train_data_path = "../data/simmc2_dials_dstc10_train.json"
    devtest_data_path = "../data/simmc2_dials_dstc10_devtest.json"
    store_path = "../processed/special_woman_store_target.json"

    dials_list = []
    dials_list = collect_special_woman_store_dials(train_data_path, dials_list)
    dials_list = collect_special_woman_store_dials(devtest_data_path, dials_list)

    furn_dials_dict = {'dialogue_data': dials_list}
    with open(store_path, 'w') as f:
        json.dump(furn_dials_dict, f)


if __name__ == "__main__":
    main()