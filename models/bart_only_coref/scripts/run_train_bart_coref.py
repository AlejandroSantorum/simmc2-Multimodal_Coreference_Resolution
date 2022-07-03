import re
import os
import ast
import copy
import json
import random
import argparse
import logging

import numpy as np
import torch

from torch import nn
from torch.optim import AdamW as torch_AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    BartTokenizerFast, 
    BartForConditionalGeneration, 
    get_linear_schedule_with_warmup
)
from transformers.tokenization_utils import PreTrainedTokenizer


from utils import api, util
from utils.metadata import (FASHION_SIZES, FASHION_AVAILABLE_SIZES, FASHION_BRAND, FASHION_COLOR, 
FASHION_PATTERN, FASHION_SLEEVE_LENGTH, FASHION_ASSET_TYPE, FASHION_TYPE, 
FASHION_PRICE, FASHION_CUSTOMER_REVIEW, FURNITURE_BRAND, FURNITURE_COLOR, 
FURNITURE_MATERIALS, FURNITURE_TYPE, FURNITURE_PRICE, FURNITURE_CUSTOMER_RATING)

fashion_meta_attrs = {
    'size': FASHION_SIZES,
    'available_sizes': FASHION_AVAILABLE_SIZES,
    'brand': FASHION_BRAND,
    'color': FASHION_COLOR,
    'pattern': FASHION_PATTERN,
    'sleeve_length': FASHION_SLEEVE_LENGTH,
    'asset_type': FASHION_ASSET_TYPE,
    'type': FASHION_TYPE,
    'price': FASHION_PRICE,
    'customer_review': FASHION_CUSTOMER_REVIEW,
    }
furniture_meta_attrs = {
    'brand': FURNITURE_BRAND,
    'color': FURNITURE_COLOR,
    'materials': FURNITURE_MATERIALS,
    'type': FURNITURE_TYPE,
    'price': FURNITURE_PRICE,
    'customer_review': FURNITURE_CUSTOMER_RATING  # key is "review"!!
}
available_sizes2st = {
    'XS': '<A>',
    'S': '<B>',
    'M': '<C>',
    'L': '<D>',
    'XL': '<E>',
    'XXL': '<F>' 
}

train_api = api.PromptAPI(dial_split='train')
CELoss = nn.CrossEntropyLoss()
BCELoss = nn.BCEWithLogitsLoss()

logger = logging.getLogger(__name__)

NUM_FASHION_ITEMS = 288
NUM_FURNITURE_ITEMS = 57
FASHION_SPECIAL_TOKENS = [f"<@1{i:03}>" for i in range(NUM_FASHION_ITEMS)]
FURNITURE_SPECIAL_TOKENS = [f"<@2{i:03}>" for i in range(NUM_FURNITURE_ITEMS)]

MAX_NUM_OBJ_IN_SCENE = 200
OBJECT_INDICES = [f"<{i}>" for i in range(MAX_NUM_OBJ_IN_SCENE)]

START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_OF_OBJ_TOKEN = "<SOO>"
END_OF_OBJ_TOKEN = "<EOO>"

###################################################################################
FASHION_METAFILE="../data/fashion_prefab_metadata_all.json"
FURNITURE_METAFILE="../data/furniture_prefab_metadata_all.json"

with open(FASHION_METAFILE) as f:
    fash_meta = json.load(f)

with open(FURNITURE_METAFILE) as f:
    fur_meta = json.load(f)

name2id_fash = dict()
for id, name in enumerate(fash_meta):
    name2id_fash[name] = id

name2id_fur = dict()
for id, name in enumerate(fur_meta):
    name2id_fur[name] = id

id2name_fash = dict()
for id, name in enumerate(fash_meta):
    id2name_fash[id] = name

id2name_fur = dict()
for id, name in enumerate(fur_meta):
    id2name_fur[id] = name

def get_line_object_ids(line):
    line_ids = []
    pos = 0
    idx = line.find("<@", pos)
    while idx != -1:
        # get absolute object ID
        abs_id = line[idx+2:idx+6]
        line_ids.append(abs_id)
        # update pos and idx
        pos = idx+4
        idx = line.find("<@", pos)
    return line_ids

def insert_attributes(line):
    pos = 0
    idx = line.find("<@", pos)
    while idx != -1:
        # get absolute object ID
        abs_id = line[idx+2:idx+6]
        # get object type
        meta = abs_id[0]
        abs_id = int(abs_id[1:])
        obj_brand = fash_meta[id2name_fash[abs_id]]['brand'] if meta=='1' else fur_meta[id2name_fur[abs_id]]['brand']
        obj_price = fash_meta[id2name_fash[abs_id]]['price'] if meta=='1' else fur_meta[id2name_fur[abs_id]]['price']
        # append object type to line
        line = line[:idx+7] + obj_brand + str(obj_price) + line[idx+7:]
        # update pos and idx
        pos = idx+4
        idx = line.find("<@", pos)
    return line


ATTR_NAME_LIST = ['color', 'type', 'brand', 'price']
def get_attribute_embeddings(line_ids, tokenizer, model, device):
    line_object_embeddings = []
    for abs_id in line_ids:
        # get object type
        meta = abs_id[0]
        abs_id = int(abs_id[1:])
        object_attrs = [str(fash_meta[id2name_fash[abs_id]][attr_name]) if meta=='1'
                            else str(fur_meta[id2name_fur[abs_id]][attr_name])
                            for attr_name in ATTR_NAME_LIST]
        # get embedding
        object_int_tokens = [torch.tensor(get_input_id(tokenizer, attr)).to(device) for attr in object_attrs]
        object_embeddings = [torch.sum(model.model.encoder.embed_tokens(obj_tok), dim=0) # summing over columns handling multiple integer tokens
                                for obj_tok in object_int_tokens]
        line_object_embeddings.append(object_embeddings)
    return line_object_embeddings

####################################################################################

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_input_id(tokenizer, tokens):
    return tokenizer(tokens).input_ids[1:-1]

def id_converter(tokenizer):
    id2index = {get_input_id(tokenizer, index)[0]: index for index in OBJECT_INDICES}
    id2fashion_st = {get_input_id(tokenizer, st)[0]: st for st in FASHION_SPECIAL_TOKENS}
    id2furniture_st = {get_input_id(tokenizer, st)[0]: st for st in FURNITURE_SPECIAL_TOKENS}
    return id2index, id2fashion_st, id2furniture_st

class LineByLineDataset(Dataset):
    def __init__(self, input_file, target_file, tokenizer: PreTrainedTokenizer, all_objects_meta, evaluation=False):

        print(f"Data file : {input_file}")
        self.evaluation = evaluation
        
        # Other tasks
        lines = []
        self.boxes = []
        self.obj_ids_per_line = []
        vocab2id = tokenizer.get_vocab()
        id2vocab = {v: k for k, v in vocab2id.items()}
        SOM_id = vocab2id[START_OF_MULTIMODAL_CONTEXTS]
        EOM_id = vocab2id[END_OF_MULTIMODAL_CONTEXTS]

        # extract input sequence to BART, and bbox info to be embedded
        with open(input_file, encoding="utf-8") as f:
            cont = 0
            for line in f.read().splitlines():
                if (len(line) > 0 and not line.isspace()):
                    # [[0.2, 0.3, 0.1, 0.2, 0.43, 3.4], [0.2, 0.1, 0.1, 0.2, 0.43, 3.7], ...]
                    line_boxes = [ast.literal_eval(position.replace('(', '').replace(')', '')) for position in re.findall(r"\[\([^\)]+\)\]", line)]
                    self.boxes.append(line_boxes)
                    line = re.sub(r"\[\([^\)]*\)\]", "", line)
                    line_ids = get_line_object_ids(line)
                    self.obj_ids_per_line.append(line_ids)
                    lines.append("<DISAM> "+line)
        encode_text = tokenizer(lines, add_special_tokens=True)
        self.examples = encode_text.input_ids
        self.examples_attention_mask = encode_text.attention_mask
        # extract generation target
        targets = []
        with open(target_file, encoding="utf-8") as f:
            target_lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        
        assert len(target_lines) == len(self.examples)
        
        corefs = []  # [ [corefobj1(index), corefobj2], [corefobj1], [...], ...]
        for line in target_lines:
            dst_start = line.index('Belief State : ')
            dst_end = line.index('<EOB>')
            dst = line[dst_start:dst_end]
            coref_referred = [obj_index for obj_index in re.findall(r"<[^<^>^ ]+>", dst)]
            corefs.append(coref_referred)
            # if 'availableSizes =' in dst:                
            #     available_sizes = [ast.literal_eval(availSize.split('=')[1].strip()) for availSize in re.findall(r"availableSizes = \[.*\]", dst)][0]
            #     available_sizes = [available_sizes2st[size] for size in available_sizes]
            line_split = line.split('Belief State : ')
            after_belief_state = line_split[1]
            after_belief_state = re.sub(r"<((<[0-9]+>)|,| )*>", "", after_belief_state)
            # if 'availableSizes =' in after_belief_state:
            #     after_belief_state = re.sub(r"availableSizes = \[.*\]", str(available_sizes), after_belief_state)
            # targets.append('=====' + after_belief_state)
            targets.append(after_belief_state)
        self.generation = targets

        self.misc = []  # [ [ {pos, coref_label, is_fashion}, ... ], ...]
        id2index, id2fashion_st, id2furniture_st = id_converter(tokenizer)
        for idx, tokenized_line in enumerate(self.examples):
            tl = tokenized_line

            EOM_indices = [i for i, tokenized_id in enumerate(tl) if tokenized_id ==EOM_id]
            if EOM_indices:
                EOM_last_idx = EOM_indices[-1]
            else:
                EOM_last_idx = -1

            is_fashion = True
            for token_id in tl:
                if token_id in id2fashion_st:
                    break
                if token_id in id2furniture_st:
                    is_fashion = False
                    break

            line_labels = []
            if is_fashion:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index
                        temp = dict()
                        pos = i; item_index = id2index[token_id]; fashion_st = id2fashion_st[tl[i+1]]
                        temp['is_fashion'] = True
                        temp['pos'] = pos
                        temp['coref_label'] = 1 if item_index in corefs[idx] else 0
                        line_labels.append(temp)
            else:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index
                        temp = dict()
                        pos = i; item_index = id2index[token_id]; furniture_st = id2furniture_st[tl[i+1]]
                        temp['is_fashion'] = False
                        temp['pos'] = pos
                        temp['coref_label'] = 1 if item_index in corefs[idx] else 0
                        line_labels.append(temp)
            self.misc.append(line_labels)
        print("Done Load Main File....")
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
            self.generation[i], self.boxes[i], self.misc[i], self.obj_ids_per_line[i]


def get_dataset(args, tokenizer, all_objects_meta, train=True):
    if train:
        dataset = LineByLineDataset(args.train_input_file, args.train_target_file, tokenizer, all_objects_meta)
    else:
        dataset = LineByLineDataset(args.eval_input_file, args.eval_target_file, tokenizer, all_objects_meta, evaluation=True)

    # Unknown issues have been reported around not being able to handle incomplete batches (e.g. w/ older CUDA 9.2)
    # Below is a workaround in case you encounter this issue.
    # Alternatively, --nocuda could avoid this issue too.
    # Comment out the following if you do not encounuter this issue or if you are not using any GPU.
    n = len(dataset) % args.train_batch_size if train else len(dataset) % args.eval_batch_size
    if n != 0:
        print(f"Truncating from {len(dataset.examples)} examples to {len(dataset.examples[:-n])}")
        dataset.examples = dataset.examples[:-n]
        dataset.generation = dataset.generation[:-n]
        dataset.boxes = dataset.boxes[:-n]
        dataset.misc = dataset.misc[:-n]
        dataset.obj_ids_per_line = dataset.obj_ids_per_line[:-n]
    return dataset


class BoxEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super(BoxEmbedding, self).__init__()
        self.box_linear = nn.Linear(6, hidden_dim)  
        self.box_layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, box_feat):
        transformed_box = self.box_layer_norm(self.box_linear(box_feat))
        return transformed_box

class CorefEncoderHead(nn.Module):
    def __init__(self, hidden_dim):
        super(CorefEncoderHead, self).__init__()
        self.aggregator = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.coref_linear = nn.Linear(2*hidden_dim, 2)

    def forward(self, concat_vector):
        ''' concat_vector: concat of obj_index_vector and st_vector '''
        aggregated = self.aggregator(concat_vector)
        coref = self.coref_linear(aggregated)
        return coref


def train_embedding_clip_way(args, model, tokenizer, all_objects_meta, num_iter=50):
    emb = model.model.encoder.embed_tokens
    emb.weight.requires_grad = True
    emb_weight_clone = emb.weight.detach().clone()
    emb_opt = torch_AdamW(emb.parameters())

    # fashion_attr_dict: {'sleeve_length': {'sleeveless':[(tok1, ctr1), (tok2, ctr2)], 'long':[(tok3,ctr3)] ...},
    #                'color':{'red':[(tok1, ctr1)], ...}, 
    #                ...}
    fashion_attr_dict = dict()
    fashion_meta_attrs_copied = copy.deepcopy(fashion_meta_attrs)
    for attr_name, attr_values in fashion_meta_attrs_copied.items():
        fashion_attr_dict[attr_name] = dict()
        if '' in attr_values:
            attr_values.remove('')
            attr_values.append('none')
        attr_values = list(attr_values)
        attr_values.sort()
        accum_token_counter = 0
        for attr_value in attr_values:
            fashion_attr_dict[attr_name][attr_value] = [
                (x, accum_token_counter + i) for i, x in enumerate(get_input_id(tokenizer, attr_value))
                ]
            accum_token_counter += len(get_input_id(tokenizer, attr_value))
    # print(fashion_attr_dict)
    # furniture_attr_dict: same as fashion_attr_dict
    furniture_attr_dict = dict()
    for attr_name, attr_values in furniture_meta_attrs.items():
        furniture_attr_dict[attr_name] = dict()
        attr_values = list(attr_values)
        attr_values.sort()
        accum_token_counter = 0
        for attr_value in attr_values:
            if not attr_value:  # skip empty string
                continue
            furniture_attr_dict[attr_name][attr_value] = [
                (x, accum_token_counter + i) for i, x in enumerate(get_input_id(tokenizer, attr_value))
                ]
            accum_token_counter += len(get_input_id(tokenizer, attr_value))
    # print(fashion_attr_dict)
    # fashion_item_label: {
    #    '<@1001>': {'sleeve_length': [pos1, pos2], 'color': [pos1], ...}
    # }
    fashion_item_label = {fashion_st: dict() for fashion_st in FASHION_SPECIAL_TOKENS}
    for attr_name, token_dict in fashion_attr_dict.items():
        for fashion_st in FASHION_SPECIAL_TOKENS:
            if attr_name == 'available_sizes':
                item_meta = all_objects_meta[fashion_st]
                sizes = item_meta[attr_name]
                sizes.sort()
                # sizes = ['^'+size for size in attr_value.split('^') if size]
                fashion_item_label[fashion_st][attr_name] = [token_dict[size][0][1] for size in sizes] 
            else:
                item_meta = all_objects_meta[fashion_st]
                attr_value = item_meta[attr_name] if item_meta[attr_name] != '' else 'none'  # for sleeve_length ''
                fashion_item_label[fashion_st][attr_name] = [idx for tok, idx in token_dict[attr_value]]

    furniture_item_label = {furniture_st: dict() for furniture_st in FURNITURE_SPECIAL_TOKENS}
    for attr_name, token_dict in furniture_attr_dict.items():
        for furniture_st in FURNITURE_SPECIAL_TOKENS:
            item_meta = all_objects_meta[furniture_st]
            attr_value = item_meta[attr_name]
            furniture_item_label[furniture_st][attr_name] = [idx for tok, idx in token_dict[attr_value]]
    
    # fashion_CELoss_label: {attr_name: [gt1, gt2, gt3, ...], ...} 
    fashion_CELoss_label = dict()
    for attr_name in fashion_attr_dict.keys():
        gt_list = []
        for item in FASHION_SPECIAL_TOKENS:
            gt_list.extend(fashion_item_label[item][attr_name])
        fashion_CELoss_label[attr_name] = torch.tensor(gt_list).to(args.device)
    furniture_CELoss_label = dict()
    for attr_name in furniture_attr_dict.keys():
        gt_list = []
        for item in FURNITURE_SPECIAL_TOKENS:
            gt_list.extend(furniture_item_label[item][attr_name])
        furniture_CELoss_label[attr_name] = torch.tensor(gt_list).to(args.device)
    # print(fashion_CELoss_label)
    
    fashion_attr_embed_matrix = dict()
    for attr_name, tok_dict in fashion_attr_dict.items():
        fashion_attr_embed_matrix[attr_name] = torch.stack([emb_weight_clone[t[0]] for tl in tok_dict.values() for t in tl]).to(args.device)
    furniture_attr_embed_matrix = dict()
    for attr_name, tok_dict in furniture_attr_dict.items():
        furniture_attr_embed_matrix[attr_name] = torch.stack([emb_weight_clone[t[0]] for tl in tok_dict.values() for t in tl]).to(args.device)

    # print(furniture_attr_embed_matrix)

    for i in range(num_iter):
        for j, attr_name in enumerate(fashion_attr_dict.keys()):
            st_indices = []
            for fashion_st in FASHION_SPECIAL_TOKENS:
                st_repeat = len(fashion_item_label[fashion_st][attr_name])
                st_indices.extend(get_input_id(tokenizer, fashion_st) * st_repeat)

            # logits: (num_possibly_duplicated_items, num_concatenated_tokens)
            logits = emb(torch.tensor(st_indices).to(args.device)) @ fashion_attr_embed_matrix[attr_name].t()
            if j == 0:
                fashion_emb_loss = CELoss(logits, fashion_CELoss_label[attr_name])
            else: 
                fashion_emb_loss += CELoss(logits, fashion_CELoss_label[attr_name])
        for j, attr_name in enumerate(furniture_attr_dict.keys()):
            st_indices = []
            for furniture_st in FURNITURE_SPECIAL_TOKENS:
                st_repeat = len(furniture_item_label[furniture_st][attr_name])
                st_indices.extend(get_input_id(tokenizer, furniture_st) * st_repeat)
            # logits: (num_possibly_duplicated_items, num_concatenated_tokens)
            logits = emb(torch.tensor(st_indices).to(args.device)) @ furniture_attr_embed_matrix[attr_name].t()
            if j == 0:
                furniture_emb_loss = CELoss(logits, furniture_CELoss_label[attr_name])
            else:
                furniture_emb_loss += CELoss(logits, furniture_CELoss_label[attr_name])

        (fashion_emb_loss + furniture_emb_loss).backward()
        emb_opt.step()
        emb.zero_grad()



def train(args, model, tokenizer, box_embedding, coref_enc_head, all_objects_meta):

    def collate_bart(examples):
        enc_input = list(map(lambda x: x[0], examples))
        enc_attention_mask = list(map(lambda x: x[1], examples))
        decoder_input = list(map(lambda x: x[2], examples))
        boxes = list(map(lambda x: x[3], examples))  
        misc = list(map(lambda x: x[4], examples))
        obj_ids_per_line = list(map(lambda x: x[5], examples))
        if tokenizer._pad_token is None:
            enc_input_pad = pad_sequence(enc_input, batch_first=True)
        else:
            enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
        enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0)
        decoder_input_pad = tokenizer(decoder_input, padding="longest", truncation=True, return_tensors="pt")

        return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                boxes, misc, obj_ids_per_line
    
    train_dataset = get_dataset(args, tokenizer, all_objects_meta, train=True)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_bart)
    t_total = len(train_dataloader) * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": coref_enc_head.parameters()
        },
        {
            "params": box_embedding.parameters()
        }

    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if (args.model_dir and os.path.isfile(os.path.join(args.model_dir, "optimizer.pt")) and os.path.isfile(os.path.join(args.model_dir, "scheduler.pt"))):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_dir, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    for net in [model, box_embedding, coref_enc_head]:
        net.train()

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    if args.model_dir and os.path.exists(args.model_dir):
        try:
            checkpoint_suffix = args.model_dir.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // len(train_dataloader)
            steps_trained_in_current_epoch = global_step % len(train_dataloader)
            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info(
                "  Will skip the first %d steps in the first epoch",
                steps_trained_in_current_epoch,
            )
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss = 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training

            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            enc_input = batch[0].to(args.device)
            enc_attention_mask = batch[1].to(args.device)
            decoder_input = batch[2].to(args.device)
            decoder_attention_mask = batch[3].to(args.device)
            boxes = batch[4] # batch, num_obj_per_line, 6
            misc = batch[5]  # batch, num_obj_per_line, dict
            obj_ids_per_line = batch[6] # batch, num_obj_per_line, num_attrs

            # follow `class BartEncoder`. shape of (batch, seqlen, d_model)
            inputs_embeds = model.model.encoder.embed_tokens(enc_input) * model.model.encoder.embed_scale

            if args.train_batch_size != len(misc):
                print(f"Something's Wrong!! batch_size:{batch_size}, len(misc):{len(misc)}")
            batch_size = len(misc)

            for b_idx in range(batch_size):  # in a batch
                box_embedded = box_embedding(torch.tensor(boxes[b_idx]).to(args.device))  # (num_obj_per_line, d_model)
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    inputs_embeds[b_idx][pos] += box_embedded[obj_idx]
                
                line_embeddings = get_attribute_embeddings(obj_ids_per_line[b_idx], tokenizer, model, args.device)
                for idx, abs_id_embs in enumerate(line_embeddings):
                    pos = misc[b_idx][idx]['pos']
                    for embs in abs_id_embs:
                        inputs_embeds[b_idx][pos] += torch.reshape(embs, (-1,))


            model_outputs = model(
                inputs_embeds=inputs_embeds, 
                attention_mask=enc_attention_mask, 
                decoder_input_ids=decoder_input[:, :-1],
                decoder_attention_mask=decoder_attention_mask[:, :-1],
                labels=decoder_input[:, 1:].contiguous()
                )
            
            model_loss = model_outputs.loss
            enc_last_state = model_outputs.encoder_last_hidden_state  # (bs, seqlen, d_model)

            misc_loss = 0
            for b_idx in range(batch_size):  # in a batch
                #is_fashion = misc[b_idx][0]['is_fashion']
                coref_label = [misc[b_idx][obj_idx]['coref_label'] for obj_idx in range(len(misc[b_idx]))]  # (num_obj)  0 or 1
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    # hidden_concat: (num_obj, 2*model)
                    if obj_idx == 0:
                        hidden_concat = torch.reshape(enc_last_state[b_idx][pos:pos+2], (1,-1))
                    else:
                        hidden_concat = torch.cat([hidden_concat, torch.reshape(enc_last_state[b_idx][pos:pos+2], (1,-1))], dim=0)

                coref = coref_enc_head(hidden_concat)  # (num_obj, num_logits)
                loss_per_line = 10 * CELoss(coref, torch.tensor(coref_label, dtype=torch.long).to(args.device))

                misc_loss += loss_per_line
            misc_loss /= batch_size

            (0.3*misc_loss).backward()

            tr_loss += model_loss.item()
            parameters_to_clip = [p for p in model.parameters() if p.grad is not None] + \
                                 [p for p in box_embedding.parameters() if p.grad is not None] + \
                                 [p for p in coref_enc_head.parameters() if p.grad is not None]
            
            torch.nn.utils.clip_grad_norm_(parameters_to_clip, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            box_embedding.zero_grad()
            coref_enc_head.zero_grad()
            global_step += 1
            
            if global_step % args.embedding_train_steps == 0:
                train_embedding_clip_way(args, model, tokenizer, all_objects_meta, args.embedding_train_epochs_ongoing)

            if (global_step % args.eval_steps == 0) and (global_step > 15000):
                results = evaluate(args, model, tokenizer, box_embedding, coref_enc_head, all_objects_meta)
                for net in [model, box_embedding, coref_enc_head]:
                    net.train()

            if args.save_steps > 0 and (global_step % args.save_steps == 0) and (global_step > 15000):
                print('checkpoint saving!!')
                checkpoint_prefix = "checkpoint"
                output_dir = os.path.join(
                    args.output_dir, "{}-{}".format(checkpoint_prefix, global_step)
                )
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)
                torch.save({'box_embedding_dict': box_embedding.state_dict(),
                            'coref_enc_head': coref_enc_head.state_dict()},
                            os.path.join(output_dir, 'aux_nets.pt'))

    return global_step, tr_loss/global_step

def evaluate(args, model, tokenizer, box_embedding, coref_enc_head, all_objects_meta):
    
    def collate_eval_bart(examples):
        enc_input = list(map(lambda x: x[0], examples))
        enc_attention_mask = list(map(lambda x: x[1], examples))
        decoder_input = list(map(lambda x: x[2], examples))
        boxes = list(map(lambda x: x[3], examples))  
        misc = list(map(lambda x: x[4], examples))
        obj_ids_per_line = list(map(lambda x: x[5], examples))
        if tokenizer._pad_token is None:
            enc_input_pad = pad_sequence(enc_input, batch_first=True)
        else:
            enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
        enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0)
        decoder_input_pad = tokenizer(decoder_input, padding="longest", truncation=True, return_tensors="pt")

        return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                boxes, misc, obj_ids_per_line
    
    def add_dicts(d1, d2):
        return {k: d1[k] + d2[k] for k in d1}

    def rec_prec_f1(n_correct, n_true, n_pred):
        rec = n_correct / n_true if n_true != 0 else 0
        prec = n_correct / n_pred if n_pred != 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0
        return rec, prec, f1

    eval_dataset = get_dataset(args, tokenizer, all_objects_meta, train=False)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_eval_bart)
    
    # Eval!
    for net in [model, box_embedding, coref_enc_head]:
        net.eval()
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    eval_loss = 0.0
    nb_eval_steps = 0

    report_template = {'fashion_coref': 0, 'furniture_coref': 0}
    total_report = copy.deepcopy(report_template)

    n_pred_objects = 0
    n_true_objects = 0
    n_correct_objects = 0

    num_fashions = 0
    num_furnitures = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        enc_input = batch[0].to(args.device)
        enc_attention_mask = batch[1].to(args.device)
        decoder_input = batch[2].to(args.device)
        decoder_attention_mask = batch[3].to(args.device)
        boxes = batch[4] # batch, num_obj_per_line, 6
        misc = batch[5]  # batch, num_obj_per_line, dict
        obj_ids_per_line = batch[6] # batch, num_obj_per_line, num_attrs

        with torch.no_grad():
            inputs_embeds = model.model.encoder.embed_tokens(enc_input) * model.model.encoder.embed_scale
            batch_size = len(misc)
            for b_idx in range(batch_size):  # in a batch
                box_embedded = box_embedding(torch.tensor(boxes[b_idx]).to(args.device))  # (num_obj_per_line, d_model)
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    inputs_embeds[b_idx][pos] += box_embedded[obj_idx]
                
                line_embeddings = get_attribute_embeddings(obj_ids_per_line[b_idx], tokenizer, model, args.device)
                for idx, abs_id_embs in enumerate(line_embeddings):
                    pos = misc[b_idx][idx]['pos']
                    for embs in abs_id_embs:
                        inputs_embeds[b_idx][pos] += torch.reshape(embs, (-1,))

            model_outputs = model(
                inputs_embeds=inputs_embeds, 
                attention_mask=enc_attention_mask, 
                decoder_input_ids=decoder_input[:, :-1],
                decoder_attention_mask=decoder_attention_mask[:, :-1],
                labels=decoder_input[:, 1:].contiguous()
                )

        model_loss = model_outputs.loss
        enc_last_state = model_outputs.encoder_last_hidden_state  # (bs, seqlen, d_model)

        batch_report = copy.deepcopy(report_template)
        
        for b_idx in range(batch_size):  # in a batch
            
            for obj_idx in range(len(misc[b_idx])):
                pos = misc[b_idx][obj_idx]['pos']
                # hidden_concat: (num_obj, 2*model)
                if obj_idx == 0:
                    hidden_concat = torch.reshape(enc_last_state[b_idx][pos:pos+2], (1,-1))
                else:
                    hidden_concat = torch.cat([hidden_concat, torch.reshape(enc_last_state[b_idx][pos:pos+2], (1,-1))], dim=0)
            
            is_fashion = misc[b_idx][0]['is_fashion']
            coref_label = torch.tensor([misc[b_idx][obj_idx]['coref_label'] for obj_idx in range(len(misc[b_idx]))]).to(args.device)  # (num_obj)  0 or 1
            n_true_objects += coref_label.sum().item()
            coref = coref_enc_head(hidden_concat)  # (num_obj, num_logits)
            if is_fashion:
                num_fashions += 1
                n_pred_objects += coref.argmax(dim=1).sum().item()
                n_correct_objects += torch.logical_and(coref.argmax(dim=1), coref_label).int().sum().item()
                batch_report['fashion_coref'] += torch.all(coref.argmax(dim=1) == coref_label, dim=0).float()  # 1. or 0.
            else:
                num_furnitures += 1
                n_pred_objects += coref.argmax(dim=1).sum().item()
                n_correct_objects += torch.logical_and(coref.argmax(dim=1), coref_label).int().sum().item()
                batch_report['furniture_coref'] += torch.all(coref.argmax(dim=1) == coref_label, dim=0).float()  # 1. or 0.
      
        total_report = add_dicts(total_report, batch_report)
        nb_eval_steps += 1

    for k, v in total_report.items():
        if ('fashion' in k) and num_fashions:
            total_report[k] = v/num_fashions
        if ('furniture' in k) and num_furnitures:
            total_report[k] = v/num_furnitures

    print('total coref result:', n_correct_objects, n_true_objects, n_pred_objects)
    coref_rec, coref_prec, coref_f1 = rec_prec_f1(n_correct_objects, n_true_objects, n_pred_objects)
    total_report['coref_info'] = f'rec: {coref_rec}, prec: {coref_prec}, f1: {coref_f1}'

    if args.output_eval_file:
        os.makedirs(args.output_eval_file.rsplit('/', 1)[0], exist_ok=True)
        with open(args.output_eval_file, 'a') as writer:
            for key in total_report.keys():
                writer.write("%s = %s\n\n" % (key, str(total_report[key])))

    print('EVALUATION:', total_report)
    return total_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_batch_size',
        default=4,
        type=int,
    )
    parser.add_argument(
        '--eval_batch_size',
        default=4,
        type=int,
    )
    parser.add_argument(
        '--num_train_epochs',
        default=3,
        type=int,
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--warmup_steps", default=1000, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--eval_steps", default=1000, type=int
    )
    parser.add_argument(
        "--embedding_train_steps", default=200, type=int
    )
    parser.add_argument(
        "--save_steps", default=1000, type=int
    )
    parser.add_argument(
        "--add_special_tokens",
        default=None,
        required=True,
        type=str,
        help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.",
    )
    parser.add_argument(
        "--item2id",
        required=True,
        type=str,
        help='item2id filepath'
    )
    parser.add_argument(
        "--train_input_file",
        required=True,
        type=str,
        help='preprocessed input file path'
    )
    parser.add_argument(
        "--train_target_file",
        required=True,
        type=str,
        help='preprocessed target file path'
    )
    parser.add_argument(
        "--eval_input_file",
        required=True,
        type=str,
        help='preprocessed input file path'
    )
    parser.add_argument(
        "--eval_target_file",
        required=True,
        type=str,
        help='preprocessed target file path'
    )
    parser.add_argument(
        "--embedding_train_epochs_start",
        type=int,
        default=400
    )
    parser.add_argument(
        "--embedding_train_epochs_ongoing",
        type=int,
        default=100
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help='model dir which contains model, optimizer, scheduler weight files'
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str
    )
    parser.add_argument(
        "--output_eval_file",
        type=str,
        default=""
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    set_seed(args)

    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')

    if args.add_special_tokens:
        if not os.path.exists(args.add_special_tokens):
            raise ValueError("Additional special tokens file {args.add_special_tokens} not found}")
        with open(args.add_special_tokens, "rb") as handle:
                special_tokens_dict = json.load(handle)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} tokens")
    
    # Define Model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    if args.add_special_tokens:
        model.resize_token_embeddings(len(tokenizer))
        model.vocab_size = len(tokenizer)
    model.config.decoder_start_token_id = 0
    model.to(args.device)

    box_embedding = BoxEmbedding(model.config.d_model).to(args.device)
    coref_enc_head = CorefEncoderHead(model.config.d_model).to(args.device)
    
    with open(args.item2id, 'r') as f:
        item2id = json.load(f)
    
    fashion_meta = train_api.fashion_meta
    furniture_meta = train_api.furniture_meta
    all_objects_meta = dict()
    for meta in fashion_meta:
        object_special_id = item2id[meta.name]
        object_meta = {'asset_type': meta.asset_type, 'customer_review': str(meta.customer_review),
        'available_sizes': [available_sizes2st[size] for size in meta.available_sizes], 
        'color': meta.color, 'pattern': meta.pattern, 'brand': meta.brand, 
        'sleeve_length': meta.sleeve_length, 'type': meta.type, 'price': str(meta.price), 'size': meta.size
        }
        all_objects_meta[object_special_id] = object_meta
    for meta in furniture_meta:
        object_special_id = item2id[meta.name]
        object_meta = {'brand': meta.brand, 'color': meta.color, 'customer_review': str(meta.customer_review),
        'materials': meta.materials, 'price': meta.price, 'type': meta.type}
        all_objects_meta[object_special_id] = object_meta

    train_embedding_clip_way(args, model, tokenizer, all_objects_meta, args.embedding_train_epochs_start)
    
    global_step, train_loss = train(args, model, tokenizer, box_embedding, coref_enc_head, all_objects_meta)


if __name__ == '__main__':
    main()