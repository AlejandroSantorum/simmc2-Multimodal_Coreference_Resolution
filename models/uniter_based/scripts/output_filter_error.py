import json
import copy
import glob
import tqdm

def filter_output(path):
    data = json.load(open(path, "r"))

    out = {'dialogue_data': [], 'domain': data['domain'], 'split': data['split']}

    for dial_idx, dial in enumerate(data['dialogue_data']):
        dial_out = copy.deepcopy(dial)
        dial_out['dialogue'] = []
        error = []
        for round_idx, round in enumerate(dial['dialogue']):
            try:
                if round['disambiguation_label'] == 1:
                    continue
            except:
                pass

            objects = round['transcript_annotated']['act_attributes']['objects']
            objects_real = round['transcript_annotated']['act_attributes']['objects_real']
            round['objects'] = objects
            round['objects_real'] = objects_real
            dial_out['dialogue'].append(round)

            if set(objects) != set(objects_real) or len(objects) != len(objects_real):
                error.append(round_idx)
        if len(error) > 0:
            dial_out['error_round'] = error
            out['dialogue_data'].append(dial_out)
    
    name = path.split('/')[-1]
    with open(f'../output/filtered/{name}', 'w', encoding='utf-8') as f:
        json.dump(out, f)

if __name__ == '__main__':
    paths = glob.glob('../output/*.json')
    for path in tqdm.tqdm(paths):
        filter_output(path.replace('\\', '/'))