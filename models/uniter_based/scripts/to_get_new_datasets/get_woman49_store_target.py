import json
from params import TRAIN_DSTC10_DATAPATH, DEVTEST_DSTC10_DATAPATH


def collect_special_woman_store_dials(dataset_path, dials_list):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    for dial in data['dialogue_data']:
        if 'cloth_store_1498649_woman' in dial['scene_ids']['0']:
            dials_list.append(dial)
    
    return dials_list



def main():
    store_path = "../../processed/new_datasets/special_woman_store_target.json"

    dials_list = []
    dials_list = collect_special_woman_store_dials(TRAIN_DSTC10_DATAPATH, dials_list)
    dials_list = collect_special_woman_store_dials(DEVTEST_DSTC10_DATAPATH, dials_list)

    furn_dials_dict = {'dialogue_data': dials_list}
    with open(store_path, 'w') as f:
        json.dump(furn_dials_dict, f)


if __name__ == "__main__":
    main()