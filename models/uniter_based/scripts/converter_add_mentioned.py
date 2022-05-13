import json
from PIL import Image
import tqdm

def fix_preprocessed(split):
    processed_path = f"../processed/{split}.json"
    DATA_ROOT = f'../data'
    
    with open(processed_path, 'r') as f:
        processed = json.load(f, encoding='utf-8')

    for idx, line in enumerate(tqdm.tqdm(processed)):
        dial = line['dial']
        processed[idx]['candidate_mentioned'] = []

        # Add mentioned flags (1 for mentioned before, 0 for not)
        for object_id in line['candidate_ids']:
            if f' {str(object_id)} ' in dial:
                processed[idx]['candidate_mentioned'].append(1)
            else:
                processed[idx]['candidate_mentioned'].append(0)
    
    with open(processed_path, 'w', encoding='utf-8') as out_file:
        json.dump(processed, out_file)

if __name__ == '__main__':
    fix_preprocessed('dev')
    fix_preprocessed('devtest')
    fix_preprocessed('train')
    fix_preprocessed('teststd')