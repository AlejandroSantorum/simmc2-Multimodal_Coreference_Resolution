import json
import random


def main(n_random_examples=100):
    BAG_TEST_SET_PATH = "../processed/devtest.json"
    BAG_SCENES_SET_PATH = "../processed/simmc2_scenes_devtest.txt"

    RANDOM_SET_STORE_PATH = "../processed/random_test_subset/random_devtest_samples.json"
    SCENES_SET_STORE_PATH = "../processed/random_test_subset/random_devtest_scenes.json"

    with open(BAG_TEST_SET_PATH, 'r') as f:
        bag_data = json.load(f)
    
    with open(BAG_SCENES_SET_PATH, 'r') as f:
        bag_scenes = json.load(f)
    
    assert len(bag_data) == len(bag_scenes)

    chosen_examples_ids = []   # line indices of randomly chosen examples
    random_examples = []       # list gathering random examples
    random_scenes = []         # list gathering scenes corresponding to random examples
    while len(random_examples) < n_random_examples:
        idx = random.randint(0, len(bag_data)-1)
        while idx in chosen_examples_ids:
            idx = random.randint(0, len(bag_data)-1)
        
        random_examples.append(bag_data[idx])
        random_scenes.append(bag_scenes[idx])
        chosen_examples_ids.append(idx)
    
    with open(RANDOM_SET_STORE_PATH, 'w') as f:
        json.dump(random_examples, f)
    
    with open(SCENES_SET_STORE_PATH, 'w') as f:
        json.dump(random_scenes, f)

    # Checking everything went potentially as desired
    with open(SCENES_SET_STORE_PATH, 'r') as f:
        checking = json.load(f)
        assert len(checking) == n_random_examples



if __name__ == "__main__":
    random.seed(12)
    main(n_random_examples=100)