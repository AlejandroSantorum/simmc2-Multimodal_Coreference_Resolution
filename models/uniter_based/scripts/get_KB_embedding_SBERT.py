import json
import torch
from sentence_transformers import SentenceTransformer

PROCESSED_ROOT = '../processed'

def get_KB_embedding(split):
    # Set up
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = SentenceTransformer('../pretrained/sentence-transformers_paraphrase-xlm-r-multilingual-v1').to(device)

    # Load data
    with open(f'{PROCESSED_ROOT}/{split}.json', 'r') as f:
        data = json.load(f)

    out = {}
    for line in data:
        model.eval()
        with torch.no_grad():
            objects = line['objects']
            object_str = json.dumps(objects)
            if object_str in out.keys():
                continue
            emb = torch.tensor(model.encode(objects))
            out[object_str] = emb.to('cpu')
            print(emb.shape)


    torch.save(out, f'KB_SBERT_{split}.pt')

if __name__ == "__main__":
    print('working...')
    get_KB_embedding('dev')
    get_KB_embedding('devtest')
    get_KB_embedding('train')
    get_KB_embedding('teststd')
    print('...done')