import json
import torch
from transformers import AutoConfig, AutoModel, BertTokenizer

PROCESSED_ROOT = '../processed'

def get_KB_embedding(split):
    # Set up
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    configuration = AutoConfig.from_pretrained('../pretrained/bert-large-uncased')
    model = AutoModel.from_pretrained('../pretrained/bert-large-uncased').to(device)
    tokenizer = BertTokenizer.from_pretrained('../pretrained/bert-large-uncased')
    print(configuration)

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

            if len(objects) < 60:
                tokenized = tokenizer(objects, padding='longest')
                tokens, attn_mask = tokenized['input_ids'], tokenized['attention_mask']
                tokens = torch.tensor(tokens).to(device)
                attn_mask = torch.tensor(attn_mask).to(device)

                emb = model(tokens, attn_mask)['last_hidden_state'][:,0,:]
                out[object_str] = emb.to('cpu')
            
            else:
                print(len(objects))
                batch1 = objects[:50]
                batch2 = objects[50:]

                tokenized = tokenizer(batch1, padding='longest')
                tokens, attn_mask = tokenized['input_ids'], tokenized['attention_mask']
                tokens = torch.tensor(tokens).to(device)
                attn_mask = torch.tensor(attn_mask).to(device)
                emb1 = model(tokens, attn_mask)['last_hidden_state'][:,0,:]

                tokenized = tokenizer(batch2, padding='longest')
                tokens, attn_mask = tokenized['input_ids'], tokenized['attention_mask']
                tokens = torch.tensor(tokens).to(device)
                attn_mask = torch.tensor(attn_mask).to(device)
                emb2 = model(tokens, attn_mask)['last_hidden_state'][:,0,:]
                
                emb = torch.cat((emb1, emb2), 0)
                out[object_str] = emb.to('cpu')
                print(emb.shape)
            # if len(out) >= 3:
            #     break

    torch.save(out, f'KB_{split}.pt')

if __name__ == "__main__":
    print('working...')
    # get_KB_embedding('dev')
    # get_KB_embedding('devtest')
    # get_KB_embedding('train')
    get_KB_embedding('teststd')
    # print(torch.load('KB_dev.pt'))