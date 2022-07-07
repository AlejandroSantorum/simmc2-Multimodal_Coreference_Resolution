import json
from tqdm import tqdm



def get_last_user_turns(filepath):
    last_user_turns_list = []

    with open(filepath, 'r') as f:
        data = json.load(f)
    
    for line in data:
        dial = line['dial']
        idx_usr = dial.rfind('User :')
        last_user_turn = dial[idx_usr+len('User : '):]
        last_user_turns_list.append(last_user_turn)
   
    return last_user_turns_list



def _get_scene_given_turn_idx(scene_ids, idx):
    prev_key = list(scene_ids.keys())[0]
    for key in scene_ids.keys():
        if int(key) <= idx:
            prev_key = key
    return scene_ids[prev_key]


def find_corresponding_scene(last_user_turn, raw_data, lut_dict):
    n_found = 0
    for dialogue in raw_data['dialogue_data']:
        for idx, turn in enumerate(dialogue['dialogue']):
            if turn['transcript'] == last_user_turn:
                n_found += 1
                if last_user_turn in lut_dict:
                    if n_found <= lut_dict[last_user_turn]:
                        continue
                    else:
                        lut_dict[last_user_turn] += 1
                else:
                    lut_dict[last_user_turn] = 1

                return _get_scene_given_turn_idx(dialogue['scene_ids'], idx), lut_dict



def get_scenes_files(processed_path, raw_path, store_scenes_path):
    with open(raw_path, 'r') as f:
        raw_data = json.load(f)
    
    last_user_turns = get_last_user_turns(processed_path)

    print(len(last_user_turns))

    scenes = []
    lut_dict = {}
    for lut in tqdm(last_user_turns):
        scene_name, lut_dict = find_corresponding_scene(lut, raw_data, lut_dict)
        scenes.append(scene_name)
    
    with open(store_scenes_path, 'w') as f:
        json.dump(scenes, f)


def main():
    TRAIN_PROCESSED_PATH = "../processed/train.json"
    TRAIN_RAW_PATH = "../data/simmc2_dials_dstc10_train.json"
    TRAIN_STORE_PATH = "../processed/simmc2_scenes_train.txt"

    DEV_PROCESSED_PATH = "../processed/dev.json"
    DEV_RAW_PATH = "../data/simmc2_dials_dstc10_dev.json"
    DEV_STORE_PATH = "../processed/simmc2_scenes_dev.txt"

    DEVTEST_PROCESSED_PATH = "../processed/devtest.json"
    DEVTEST_RAW_PATH = "../data/simmc2_dials_dstc10_devtest.json"
    DEVTEST_STORE_PATH = "../processed/simmc2_scenes_devtest.txt"

    print("Getting train scenes...")
    get_scenes_files(TRAIN_PROCESSED_PATH, TRAIN_RAW_PATH, TRAIN_STORE_PATH)
    print("Getting dev scenes...")
    get_scenes_files(DEV_PROCESSED_PATH, DEV_RAW_PATH, DEV_STORE_PATH)
    print("Getting devtest scenes...")
    get_scenes_files(DEVTEST_PROCESSED_PATH, DEVTEST_RAW_PATH, DEVTEST_STORE_PATH)
    print("Done")



if __name__ == '__main__':
    main()
