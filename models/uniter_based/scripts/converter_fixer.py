import json
from PIL import Image
import tqdm

def fix_preprocessed(split):
    processed_path = f"../processed/{split}.json"
    DATA_ROOT = f'../data'
    
    with open(processed_path, 'r') as f:
        processed = json.load(f, encoding='utf-8')

    for idx, line in enumerate(tqdm.tqdm(processed)):
        scene_paths = line['scenes']

        # Added bboxes for scnenes (entire image)
        additional_bboxes = []
        for scene_path in scene_paths:
            scene_path_corrected = scene_path.replace('m_','')
            w, h = Image.open(f'{DATA_ROOT}/all_images/{scene_path_corrected}.png').size
            additional_bboxes.append([0,0,h,w])

        if len(additional_bboxes) == 1:
            additional_bboxes.append([0,0,0,0])

        processed[idx]['candidate_bbox'] += additional_bboxes
    
    with open(processed_path, 'w', encoding='utf-8') as out_file:
        json.dump(processed, out_file)

if __name__ == '__main__':
    fix_preprocessed('dev')
    fix_preprocessed('devtest')
    fix_preprocessed('train')
    fix_preprocessed('teststd')