####################################################################
#
#   Author: Alejandro Santorum Varela (@AlejandroSantorum)
#           alejandro.santorum@gmail.com
#   Date: May 22, 2022
#   Project: Multimodal Coreference Resolution
#            MPhil MLMI dissertation at Cambridge University
#
#   Description: Given a file with the generated evaluations, this
#       script calculates the F1 score.
#
####################################################################

import json
import argparse
from sklearn.metrics import f1_score


def get_f1_score(args):
    file_path = args.MODEL_EVAL_FILE

    with open(file_path, 'r') as data_file:
        data = json.load(data_file)
    
    predict_list = []
    target_list = []

    for dial_idx in data:
        for turn_idx in data[dial_idx]:
            for object_head in data[dial_idx][turn_idx]:
                prediction = int(object_head[0] > 0)
                predict_list.append(prediction)
                target_list.append(object_head[1])
    
    f1 = f1_score(target_list, predict_list)

    print("\tFile path:", file_path)
    print("\tObject MM Coref F1-Score:", f1)

    return f1
    
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--MODEL_EVAL_FILE', required=True, type=str)

    args = parser.parse_args()

    get_f1_score(args)
