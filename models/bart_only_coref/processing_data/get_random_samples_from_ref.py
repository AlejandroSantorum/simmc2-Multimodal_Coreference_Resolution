from get_seen_unseen_outdomain_sets import get_examples_given_ref


def _store_data(data, path):
    with open(path, 'w') as f:
        for line in data:
            f.write(line)


def main():
    RANDOM_SAMPLES_REF_PATH = "../data_object_special/new_datasets/reference_sets/random_devtest_samples.json"

    BAG_PRED_TEST_SET_PATH = "../data_object_special/simmc2_dials_dstc10_devtest_predict.txt"
    BAG_TARGET_TEST_SET_PATH = "../data_object_special/simmc2_dials_dstc10_devtest_target.txt"

    PRED_STORE_PATH = "../data_object_special/random_test_subset/random_devtest_samples_predict.json"
    TARGET_STORE_PATH = "../data_object_special/random_test_subset/random_devtest_samples_target.json"

    with open(BAG_PRED_TEST_SET_PATH, 'r') as f:
        pred_data = f.readlines()
    
    with open(BAG_TARGET_TEST_SET_PATH, 'r') as f:
        target_data = f.readlines()
    
    assert len(pred_data) == len(target_data)

    pred, target = get_examples_given_ref(RANDOM_SAMPLES_REF_PATH, pred_data, target_data)
    assert len(pred) == len(target)

    print("Number of retrieved examples:", len(pred))

    _store_data(pred, PRED_STORE_PATH)
    _store_data(target, TARGET_STORE_PATH)



if __name__ == "__main__":
    main()