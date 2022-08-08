import json
import numpy as np


def _order_data(data):
    ret = []
    for l in data:
        ret.append(sorted(l))
    return ret


def _get_unique(ann1, ann2):
    unique = []
    for ann in ann1:
        if ann not in unique:
            unique.append(ann)
    for ann in ann2:
        if ann not in unique:
            unique.append(ann)
    return unique


def cohen_kappa(ann1, ann2):
    assert len(ann1) == len(ann2)

    count = 0
    for an1, an2 in zip(ann1, ann2):
        if sorted(an1) == sorted(an2):
            count += 1
    # observed agreement (Po)
    Po = count / len(ann1)

    uniq = _get_unique(ann1, ann2)
    Pe = 0 # expected agreement (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        Pe += count

    return round((Po - Pe) / (1 - Pe), 4)


def main():
    ann1_path = "./estimate_human_performance/estim_human_perform_asantorum.json"
    ann2_path = "./estimate_human_performance/estim_human_perform_svetla.json"
    ann3_path = "./estimate_human_performance/estim_human_perform_simon.json"

    with open(ann1_path, 'r') as f:
        ann1_data = json.load(f)
    with open(ann2_path, 'r') as f:
        ann2_data = json.load(f)
    with open(ann3_path, 'r') as f:
        ann3_data = json.load(f)

    ann1_data = _order_data(ann1_data)
    ann2_data = _order_data(ann2_data)
    ann3_data = _order_data(ann3_data)

    iaa_1_2 = cohen_kappa(ann1_data, ann2_data)
    iaa_1_3 = cohen_kappa(ann1_data, ann3_data)
    iaa_2_3 = cohen_kappa(ann2_data, ann3_data)

    print("Cohen's kappa between annotator 1 and 2:", iaa_1_2)
    print("Cohen's kappa between annotator 1 and 3:", iaa_1_3)
    print("Cohen's kappa between annotator 2 and 3:", iaa_2_3)
    print("Average Cohen's kappa:", np.mean([iaa_1_2, iaa_1_3, iaa_2_3]))



if __name__ == "__main__":
    main()