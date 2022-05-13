import os
import json
import torch

from Transformers_VQA.dataset_final import make_final_loader
from Transformers_VQA.modified_uniter import Modified_Uniter
from Transformers_VQA.modified_uniter_KBid import Modified_Uniter_KBid
from Transformers_VQA.modified_uniter_sceneseg import Modified_Uniter_sceneseg

def inference(checkpoint, model_name, test_set = 'devtest'):
    BATCH_SIZE = 1
    
    torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make loaders
    loader = make_final_loader(test_set, BATCH_SIZE, rcnn=False, test=True)

    # Load model
    if model_name == 'base':
        model = Modified_Uniter().to(device)
    elif model_name == 'KBid':
        model = Modified_Uniter_KBid().to(device)
    elif model_name == 'sceneseg':
        model = Modified_Uniter_sceneseg().to(device)
    model.load_state_dict(torch.load(f'./trained/{checkpoint}.bin', map_location=device)['model_state_dict'], strict=False)
    
    mask_stepper = torch.ones(1, 12, 512, 512).to(device)
    for i in range(12):
        mask_stepper[0, i, :, :] *= i+1

    # Infer
    out = {}
    out_obj_logit = {}
    n_hit = 0
    n_pred_pos = 0
    n_real_pos = 0
    logit_out = torch.tensor([[]]).to(device)
    truth_out = torch.tensor([[]]).to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            txt_seg_ids = batch['txt_seg_ids'].to(device)
            vis_feats = batch['vis_feats'].to(device)
            obj_embs = batch['obj_embs'].to(device)
            KB_ids = batch['KB_ids'].to(device)
            obj_ids = batch['obj_ids'].to(device)
            pos_x = batch['pos_x'].to(device)
            pos_y = batch['pos_y'].to(device)
            pos_z = batch['pos_z'].to(device)
            bboxes = batch['bboxes'].to(device)
            vis_seg = batch['vis_seg'].to(device)
            extended_attention_mask = batch['extended_attention_mask'].to(device)
            output_mask = batch['output_mask'].to(device)
            reference = batch['reference'].to(device)
            scene_seg = batch['scene_segs'].to(device)
            dial_idx = batch['dial_idx']
            round_idx = batch['round_idx']


            if model_name == 'base':
                pred = model(input_ids , txt_seg_ids, vis_feats, obj_embs, obj_ids, pos_x, pos_y, pos_z, bboxes, vis_seg, extended_attention_mask)
            elif model_name == 'KBid':
                pred = model(input_ids , txt_seg_ids, vis_feats, KB_ids, obj_ids, pos_x, pos_y, pos_z, bboxes, vis_seg, extended_attention_mask)
            elif model_name == 'sceneseg':
                pred = model(input_ids , txt_seg_ids, vis_feats, obj_embs, obj_ids, pos_x, pos_y, pos_z, bboxes, vis_seg, extended_attention_mask, scene_seg)
            
            pred = pred.reshape(1,-1)
            pred = pred[output_mask==1].reshape(-1,1)
            
            truth = reference.float().reshape(-1,1)

            pred_bin = pred > 0
            truth_bin = truth > 0.5
            
            n_hit += torch.sum(pred_bin*truth_bin == 1).detach().item()
            n_pred_pos += torch.sum((pred.reshape(1,-1) > 0) ).to('cpu').item()
            n_real_pos += torch.sum((reference.reshape(1,-1) > 0.1)).to('cpu').item()

            logit_out = torch.cat((logit_out, pred.reshape(1,-1)), axis=1)
            truth_out = torch.cat((truth_out, reference.reshape(1,-1)), axis=1)

            obj_ids -= 1 # -1 for padding. 0 is a valid index
            line_out = []
            dial_idx = dial_idx[0]
            round_idx = round_idx[0]

            for idx, prediction in enumerate(pred):
                if prediction > 0:
                    line_out.append(obj_ids[0][idx].item())
                try:
                    out_obj_logit[dial_idx][round_idx][obj_ids[0][idx].item()] = prediction.item()
                except:
                    try:
                        out_obj_logit[dial_idx][round_idx] = {obj_ids[0][idx].item(): prediction.item()}
                    except:
                        out_obj_logit[dial_idx] = {round_idx: {obj_ids[0][idx].item(): prediction.item()}}
            
            try:
                out[dial_idx][round_idx] = line_out
            except:
                out[dial_idx] = {round_idx: line_out}

    try:
        with open(f'../data/simmc2_dials_dstc10_{test_set}.json', 'r') as data_file:
            data = json.load(data_file)
    except:
        with open(f'../data/simmc2_dials_dstc10_{test_set}_public.json', 'r') as data_file:
            data = json.load(data_file)

    for dial in data['dialogue_data']:
        dial_mentions = []
        dial_idx = dial['dialogue_idx']
        for round_idx, round in enumerate(dial['dialogue']):
            try:
                round['transcript_annotated']['act_attributes']['objects'] = out[dial_idx][round_idx]
                for obj_idx in out[dial_idx][round_idx]:
                    if obj_idx not in dial_mentions:
                        dial_mentions.append(obj_idx)
            except:
                try:
                    round['transcript_annotated']['act_attributes']['objects'] = []
                except:
                    round['transcript_annotated'] = {'act_attributes': {"objects": []}}
        dial['mentioned_object_ids'] = dial_mentions

    # uncomment this to output coref predictions for each model
    # with open(f'./inference/{checkpoint}_{test_set}.json', 'w', encoding='utf-8') as out_file:
    #     json.dump(data, out_file)
    
    with open(f'./inference/{checkpoint}_{test_set}_obj_logits.json', 'w', encoding='utf-8') as out_file:
        json.dump(out_obj_logit, out_file)
    print(test_set)
    print(n_hit)
    print(n_pred_pos)
    print(n_real_pos)

    torch.save({'logit': logit_out, 'truth': truth_out}, f'./inference/{checkpoint}_{test_set}_logit_truth.pt')
    return

if __name__ == '__main__':
    inference('base', 'base', test_set='dev')
    inference('base', 'base', test_set='devtest')
    inference('base', 'base', test_set='teststd')
    inference('KBid', 'KBid', test_set='dev')
    inference('KBid', 'KBid', test_set='devtest')
    inference('KBid', 'KBid', test_set='teststd')
    inference('sceneseg', 'sceneseg', test_set='dev')
    inference('sceneseg', 'sceneseg', test_set='devtest')
    inference('sceneseg', 'sceneseg', test_set='teststd')