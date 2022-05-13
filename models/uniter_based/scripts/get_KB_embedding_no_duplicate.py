import json
import torch
from transformers import AutoConfig, AutoModel, BertTokenizer

PROCESSED_ROOT = '../processed'

def get_KB_embedding_learnable():
    # Set up
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    configuration = AutoConfig.from_pretrained('../pretrained/bert-large-uncased')
    model = AutoModel.from_pretrained('../pretrained/bert-large-uncased').to(device)
    tokenizer = BertTokenizer.from_pretrained('../pretrained/bert-large-uncased')
    print(configuration)

    # Load data
    with open(f'{PROCESSED_ROOT}/KB_dict.json', 'r') as f:
        data = json.load(f)

    out = {}
    for key, val in data.items():
        model.eval()
        with torch.no_grad():
            try:
                key = int(key)
            except:
                continue

            obj_string = [val['string']]
            tokenized = tokenizer(obj_string, padding='longest')
            tokens, attn_mask = tokenized['input_ids'], tokenized['attention_mask']
            tokens = torch.tensor(tokens).to(device)
            attn_mask = torch.tensor(attn_mask).to(device)

            emb = model(tokens, attn_mask)['last_hidden_state'][:,0,:]
            out[key] = emb.to('cpu')
            # if len(out) >= 3:
            #     break

    torch.save(out, f'{PROCESSED_ROOT}/KB_emb.pt')

if __name__ == "__main__":
    print('working...')
    get_KB_embedding_learnable()
    print('...aaaaand done')