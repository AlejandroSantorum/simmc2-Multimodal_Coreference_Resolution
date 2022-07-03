import os
import re
import ast
import copy
import json
import argparse
import logging

from tqdm import tqdm

import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BartForConditionalGeneration, BartTokenizerFast

from run_train_bart_coref import BoxEmbedding, CorefEncoderHead

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
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

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_input_id(tokenizer, tokens):
    return tokenizer(tokens).input_ids[1:-1]


###################################################################################
TRAIN_SCENES_FILE="../data_object_special/simmc2_scenes_train.txt"
DEV_SCENES_FILE="../data_object_special/simmc2_scenes_dev.txt"
DEVTEST_SCENES_FILE="../data_object_special/simmc2_scenes_devtest.txt"

SCENE_EMBEDDING_SIZE = 512
CORRUPTED_SCENE_IMGS = ['cloth_store_1416238_woman_4_8', 'm_cloth_store_1416238_woman_20_6', 'cloth_store_1416238_woman_20_6', 'cloth_store_1416238_woman_19_0']


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

def id_converter(tokenizer):
    id2index = {get_input_id(tokenizer, index)[0]: index for index in OBJECT_INDICES}
    id2fashion_st = {get_input_id(tokenizer, st)[0]: st for st in FASHION_SPECIAL_TOKENS}
    id2furniture_st = {get_input_id(tokenizer, st)[0]: st for st in FURNITURE_SPECIAL_TOKENS}
    return id2index, id2fashion_st, id2furniture_st

def correct_action(text, correction_dict):
    for k, v in correction_dict.items():
        text = text.replace(k, v)
    return text

def correct_available_sizes(text):
    SIZES =['<A>', '<B>', '<C>', '<D>', '<E>', '<F>']
    try:
        if 'availableSizes =' in text:
            available_sizes_str_list = [(m.start(0), m.end(0)) for m in re.finditer(r"availableSizes =", text)]
            if not available_sizes_str_list:  # empty available_sizes_str_list: in case of (availableSizes)
                return text
            availableSizes_idx = available_sizes_str_list[0][1]
            start_bracket_idx = -1
            end_bracket_idx = -1
            for i in range(70):
                if text[availableSizes_idx+i] == '[':
                    start_bracket_idx = availableSizes_idx+i
                if text[availableSizes_idx+i] == ']':
                    end_bracket_idx = availableSizes_idx+i
                if start_bracket_idx != -1 and end_bracket_idx != -1:
                    break
            assert start_bracket_idx != -1 and end_bracket_idx != -1, f"ERROR AT def correct_available_sizes!!\n{text}"
            list_str = text[start_bracket_idx:end_bracket_idx].replace("'", "")
            new_list = []
            for size in SIZES:
                if size in list_str:
                    new_list.append(size)
            new = ", ".join(new_list)
            return text[:start_bracket_idx] + '['+new + text[end_bracket_idx:]
        else:
            return text
    except:
        print('text:', text)

def remove_bos_eos_startequal(text):
    text = text.split("</s>")[0].replace('<s>', '')
    return text

def replace_special_chars(text):
    def rep(match_re_obj):
        return match_re_obj.group(0).replace('<','').replace('>','')
    available_sizes_st_list = [('<A>', "'XS'"), ('<B>', "'S'"), ('<C>', "'M'"), ('<D>', "'L'"), ('<E>', "'XL'"), ('<F>', "'XXL'")]
    for size_tuple in available_sizes_st_list:
        text = text.replace(size_tuple[0], size_tuple[1])
    text = re.sub("<[0-9]+>", rep, text)
    return text

def insert_coref(text, coref_chars: list):
    """ coref_chars: [<11>, <44>, ...] """
    try:
        coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]
    except:
        text = text[:text.rfind("<pad>")+5] + " [  ] ()  <EOB>" + text[text.rfind("<pad>")+5:]
        coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]
        
    coref_list = [int(coref.replace('<', '').replace('>', '')) for coref in coref_chars]
    coref_str = str(coref_list).replace('[', '< ').replace(']',' >') if coref_list else '<  >'
    return text[:coref_pos_start+1] + ' ' + coref_str + ' <EOB>' + text[coref_pos_end:]

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

class GenerationDataset(Dataset):
    def __init__(self, prompts_from_file, tokenizer):

        lines = []
        self.original_lines = []
        self.boxes = []
        self.obj_ids_per_line = []

        # WARNING!!! WE ARE ASSUMING THAT SCENE EMBEDDINGS ARE ONLY USED WITH STANDARD PREPROCESSED DATASETS:
        #       e.g. data_object_special/simmc2_dials_dstc10_devtest_predict.txt
        self.scene_names = json.load(open(DEVTEST_SCENES_FILE, 'r'))

        vocab2id = tokenizer.get_vocab()
        id2vocab = {v: k for k, v in vocab2id.items()}
        SOM_id = vocab2id[START_OF_MULTIMODAL_CONTEXTS]
        EOM_id = vocab2id[END_OF_MULTIMODAL_CONTEXTS]

        # extract input sequence to BART, and bbox info to be embedded
        with open(prompts_from_file, encoding="utf-8") as f:
            for line in f.read().splitlines():
                if (len(line) > 0 and not line.isspace()):
                    # [[0.2, 0.3, 0.1, 0.2, 0.4, 0.8], [0.2, 0.1, 0.1, 0.2, 0.3, 0.1], ...]
                    line_boxes = [ast.literal_eval(position.replace('(', '').replace(')', '')) for position in re.findall(r"\[\([^\)]+\)\]", line)]
                    self.boxes.append(line_boxes)
                    line = re.sub(r"\[\([^\)]*\)\]", "", line)
                    line_ids = get_line_object_ids(line)
                    self.obj_ids_per_line.append(line_ids)
                    original_line = copy.deepcopy(line)
                    original_line = re.sub(r" <SOO.*EOO>", "", original_line)
                    lines.append(line)
                    self.original_lines.append(original_line)
        encode_text = tokenizer(lines, add_special_tokens=True)
        self.examples = encode_text.input_ids
        self.examples_attention_mask = encode_text.attention_mask

        self.misc = []  # [ [ {pos, is_fashion}, ... ], ...]
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
                        pos = i; item_index = id2index[token_id]
                        temp['canonical_id'] = int(item_index[1:-1])
                        temp['is_fashion'] = True
                        temp['pos'] = pos
                        
                        line_labels.append(temp)
            else:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index
                        temp = dict()
                        pos = i; item_index = id2index[token_id]
                        temp['canonical_id'] = int(item_index[1:-1])
                        temp['is_fashion'] = False
                        temp['pos'] = pos
                        line_labels.append(temp)
            self.misc.append(line_labels)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.examples_attention_mask[i], dtype=torch.long), self.original_lines[i], self.boxes[i], self.misc[i], self.obj_ids_per_line[i], self.scene_names[i]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        help='model dir which contains model, optimizer, scheduler weight files'
    )
    parser.add_argument(
        "--prompts_from_file",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        '--item2id',
        type=str,
        required=True
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=36
    )
    parser.add_argument(
        "--add_special_tokens",
        default=None,
        required=True,
        type=str,
        help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.",
    )
    parser.add_argument("--length", type=int, default=150)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of samples to generate.",
    )
    parser.add_argument(
        "--correct_act",
        type=str,
        default=None,
        help="correct wrongly generated action with correct_act dictionary",
    )
    parser.add_argument(
        "--path_output",
        type=str,
        required=True,
        help="Path to output predictions in a line separated text file.",
    )
    ### New:
    parser.add_argument(
        "--input_attrs",
        default=False
    )
    parser.add_argument(
        "--scene_embeddings",
        default=False
    )
    parser.add_argument(
        "--obj_img_embeddings",
        default=False
    )

    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.eval_input_file = args.prompts_from_file
    set_seed(args)

    if args.prompts_from_file and not os.path.exists(args.prompts_from_file):
        raise Exception(f"prompt file '{args.prompts_from_file}' not found")
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    with open(args.add_special_tokens, "rb") as handle:
            special_tokens_dict = json.load(handle)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_toks} tokens")

    model = BartForConditionalGeneration.from_pretrained(args.model_dir)
    model.config.decoder_start_token_id = 0
    model.to(args.device)

    checkpoint = torch.load(os.path.join(args.model_dir, 'aux_nets.pt'), map_location=torch.device(args.device))
    box_embedding = BoxEmbedding(model.config.d_model).to(args.device)
    if args.scene_embeddings or args.obj_img_embeddings:
        args.scene_embeddings_dict = torch.load("../data_object_special/img_features.pt", map_location=args.device)
        coref_enc_head = CorefEncoderHead(model.config.d_model, scene_embed_size=SCENE_EMBEDDING_SIZE).to(args.device)
    else:
        coref_enc_head = CorefEncoderHead(model.config.d_model).to(args.device)

    box_embedding.load_state_dict(checkpoint['box_embedding_dict'])
    coref_enc_head.load_state_dict(checkpoint['coref_enc_head'])
    
    args.length = adjust_length_to_model(
        args.length, max_sequence_length=model.config.max_position_embeddings
    )
    #logger.info(args)

    def collate_bart(examples):
        enc_input = list(map(lambda x: x[0], examples))
        enc_attention_mask = list(map(lambda x: x[1], examples))
        original_lines = list(map(lambda x: x[2], examples))
        boxes = list(map(lambda x: x[3], examples))
        misc = list(map(lambda x: x[4], examples))
        obj_ids_per_line = list(map(lambda x: x[5], examples))
        scene_names = list(map(lambda x: x[6], examples))
        if tokenizer._pad_token is None:
            enc_input_pad = pad_sequence(enc_input, batch_first=True)
        else:
            enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
        enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0)
        return enc_input_pad, enc_attention_pad, original_lines, boxes, misc, obj_ids_per_line, scene_names
    
    with open(args.item2id, 'r') as f:
        item2id = json.load(f)
    
    decode_dataset = GenerationDataset(args.prompts_from_file, tokenizer)
    decode_sampler = SequentialSampler(decode_dataset)
    decode_dataloader = DataLoader(
        decode_dataset,
        sampler=decode_sampler,
        batch_size=args.batch_size,
        collate_fn=collate_bart
    )

    tokenizer_id2token = {v: k for k, v in tokenizer.get_vocab().items()}
    results = []
    results_coref_replaced = []
    n_prompts = len(decode_dataset)
    for i, batch in enumerate(tqdm(decode_dataloader, desc='Decoding')):  # should be 1-batchsized batch
        
        enc_input = batch[0].to(args.device)
        enc_input_attention = batch[1].to(args.device)
        original_lines = batch[2]
        boxes = batch[3] # batch, num_obj_per_line, 6
        misc = batch[4]  # batch, num_obj_per_line, dict
        obj_ids_per_line = batch[5] # batch, num_obj_per_line, num_attrs
        scene_names = batch[6] # batch, SCENE_EMBEDDING_SIZE
        batch_size = len(misc)

        with torch.no_grad():
            inputs_embeds = model.model.encoder.embed_tokens(enc_input) * model.model.encoder.embed_scale
            for b_idx in range(batch_size):  # in a batch
                box_embedded = box_embedding(torch.tensor(boxes[b_idx]).to(args.device))  # (num_obj_per_line, d_model)
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    inputs_embeds[b_idx][pos] += box_embedded[obj_idx]
                
                if args.input_attrs:
                    line_embeddings = get_attribute_embeddings(obj_ids_per_line[b_idx], tokenizer, model, args.device)
                    for idx, abs_id_embs in enumerate(line_embeddings):
                        pos = misc[b_idx][idx]['pos']
                        for embs in abs_id_embs:
                            inputs_embeds[b_idx][pos] += torch.reshape(embs, (-1,))

            encoder_outputs = model.model.encoder(inputs_embeds=inputs_embeds, attention_mask=enc_input_attention, return_dict=True)  # check this line

        coref_obj_list = []
        for b_idx in range(batch_size):
            coref_obj_each_batch = []
            for obj_idx in range(len(misc[b_idx])):
                pos = misc[b_idx][obj_idx]['pos']
                # hidden_concat: (num_obj, 2*model)
                if obj_idx == 0:
                    hidden_concat = torch.reshape(encoder_outputs.last_hidden_state[b_idx][pos:pos+2], (1,-1))
                else:
                    hidden_concat = torch.cat([hidden_concat, torch.reshape(encoder_outputs.last_hidden_state[b_idx][pos:pos+2], (1,-1))], dim=0)
            
            ## OBJECT IMAGE EMBEDDING STUFF!
            if args.obj_img_embeddings and not args.scene_embeddings:
                obj_img_embeds = torch.zeros(hidden_concat.shape[0], SCENE_EMBEDDING_SIZE).to(args.device)
                if scene_names[b_idx] not in CORRUPTED_SCENE_IMGS:
                    for obj_idx in range(len(misc[b_idx])):
                        canonical_id = misc[b_idx][obj_idx]['canonical_id']
                        scene_name = scene_names[b_idx]+'_scene'
                        if canonical_id in args.scene_embeddings_dict[scene_name]: # <OBJ>
                            obj_img_embeds_aux = args.scene_embeddings_dict[scene_name][canonical_id]
                        else: # <PREVOBJ>
                            obj_img_embeds_aux = torch.zeros(1, SCENE_EMBEDDING_SIZE).to(args.device)
                        if obj_idx == 0:
                            obj_img_embeds = obj_img_embeds_aux
                        else:
                            obj_img_embeds = torch.cat([obj_img_embeds, obj_img_embeds_aux], dim=0)
                hidden_concat = torch.cat([hidden_concat, obj_img_embeds], dim=1)

            ## SCENE EMBEDDING STUFF!
            if args.scene_embeddings and not args.obj_img_embeddings:
                scene_embedding = torch.zeros(hidden_concat.shape[0], SCENE_EMBEDDING_SIZE).to(args.device)
                if scene_names[b_idx] not in CORRUPTED_SCENE_IMGS:
                    scene_name = scene_names[b_idx]+'_scene'
                    scene_embedding = args.scene_embeddings_dict[scene_name]['scene']
                    scene_embedding = scene_embedding.repeat(hidden_concat.shape[0], 1)
                hidden_concat = torch.cat([hidden_concat, scene_embedding], dim=1)
            
            objs_pos = [misc[b_idx][obj_idx]['pos'] for obj_idx in range(len(misc[b_idx]))]
            obj_indices = [tokenizer_id2token[enc_input[b_idx][pos].item()] for pos in objs_pos]  # ex) [<11>, <41>, ...]

            coref = coref_enc_head(hidden_concat)

            coref_predict = coref.argmax(dim=1).tolist()  # (num_objs)
            for i, coref_signal in enumerate(coref_predict):
                if coref_signal:
                    coref_obj_each_batch.append(obj_indices[i])
            coref_obj_list.append(coref_obj_each_batch)

        for j in range(len(coref_obj_list)):
            total_sequence_coref_replaced = insert_coref("", coref_obj_list[j])
            results_coref_replaced.append(total_sequence_coref_replaced)

    with open(args.path_output, "w") as f_out:
        f_out.write("\n".join(results_coref_replaced))
    
    return 

if __name__ == "__main__":
    main()
