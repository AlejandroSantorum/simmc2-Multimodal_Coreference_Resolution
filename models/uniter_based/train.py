import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup

from utils.focalloss import FocalLoss
from utils.dataset import make_loader

from model.modified_uniter import Modified_Uniter
from model.modified_lxmert import Modified_Lxmert
from model.model_utils import make_KBid_emb_init


def train(args):
    # Constant setup
    NAME = args.NAME
    MODEL = args.MODEL
    SPLIT = args.SPLIT
    BATCH_SIZE = args.BATCH_SIZE
    BATCH_SIZE_DEV = args.BATCH_SIZE_DEV
    LR = args.LR
    N_EPOCH = args.N_EPOCH
    GAMMA = args.GAMMA
    ALPHA = args.ALPHA
    SCHEDULER = args.scheduler
    print(f'{NAME} batch_size={BATCH_SIZE}, Adam_lr={LR}, FocalAlpha={ALPHA}, GAMMA={GAMMA}, scheduler={SCHEDULER}\n')

    arg_dict = vars(args)
    for model_flag in ['obj_id', 'vis_feats_clip', 'vis_feats_rcnn', 'pos', 'scene_seg', 'obj_embs_bert', 'obj_embs_sbert', 'kb_id_bert', 'kb_id_sbert', 'attn_bias','graph_attn','obj_men','pred_men','more_roi']:
        print(model_flag)
        if arg_dict[model_flag] == 'True':
            arg_dict[model_flag] = True
            print('True')
        else:
            arg_dict[model_flag] = False
            print('False')

    if arg_dict['pred_men']:
        print(f'Auxilliary focal loss: alpha {args.men_alpha}, gamma {args.men_gamma}, weight {args.men_weight}')

    torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make loaders
    train_loader = make_loader(SPLIT, BATCH_SIZE, more_roi=arg_dict['more_roi'], add_visual_attrs=arg_dict['visual_attrs'])
    if SPLIT == 'seen_unseen_OOD_train':
        dev_loader = make_loader('in_domain', BATCH_SIZE_DEV, more_roi=arg_dict['more_roi'], add_visual_attrs=arg_dict['visual_attrs'])
    else:
        dev_loader = make_loader('dev', BATCH_SIZE_DEV, more_roi=arg_dict['more_roi'], add_visual_attrs=arg_dict['visual_attrs'])

    # Setup Tensorboard
    writer = SummaryWriter(log_dir='./result/runs' ,comment=f'{NAME} batch_size={BATCH_SIZE}, Adam_lr={LR}, FocalAlpha={ALPHA}, GAMMA={GAMMA}')

    # Eval for F1
    def eval(model):
        model.eval()
        with torch.no_grad():
            total_hit, total_pred_positive, total_truth_positive, total_loss, total_pred = 0, 0, 0, [], 0
            n_target_objs_acc_list = []
            n_mentioned_targets_acc_list = []
            n_new_targets_acc_list = []
            for idx, batch in enumerate(dev_loader):
                input_ids = batch['input_ids'].to(device)
                txt_seg_ids = batch['txt_seg_ids'].to(device)
                vis_feats_clip = batch['vis_feats'].to(device)
                vis_feats_rcnn = batch['vis_feats_rcnn'].to(device)
                kb_embs_bert = batch['obj_embs'].to(device)
                kb_embs_sbert = batch['obj_embs_SBERT'].to(device)
                obj_ids = batch['obj_ids'].to(device)
                pos_x = batch['pos_x'].to(device)
                pos_y = batch['pos_y'].to(device)
                pos_z = batch['pos_z'].to(device)
                bboxes = batch['bboxes'].to(device)
                vis_seg = batch['vis_seg'].to(device)
                extended_attention_mask = batch['extended_attention_mask'].to(device)
                output_mask = batch['output_mask'].to(device)
                reference = batch['reference'].to(device)
                num_gt_targets = batch['num_gt_targets'].to(device)
                num_mentioned_targets = batch['num_mentioned_targets'].to(device)
                num_new_targets = batch['num_new_targets'].to(device)
                scene_seg = batch['scene_segs'].to(device)
                kb_id = batch['KB_ids'].to(device)
                rel_mask_left = batch['rel_mask_left'].to(device).long()
                rel_mask_right = batch['rel_mask_right'].to(device).long()
                rel_mask_up = batch['rel_mask_up'].to(device).long()
                rel_mask_down = batch['rel_mask_down'].to(device).long()
                input_ids_uncased = batch['input_ids_uncased'].to(device)
                input_mask_uncased = batch['input_mask_uncased'].to(device).long()
                segment_ids_uncased = batch['segment_ids_uncased'].to(device)
                bboxes_lxmert = batch['bboxes_lxmert'].to(device).float()
                vis_mask = batch['vis_mask'].to(device).long()
                obj_men = batch['obj_men'].to(device).long()

                if MODEL == "UNITER":
                    if arg_dict['pred_men']:
                        pred, pred_men = model(input_ids, txt_seg_ids, vis_seg, bboxes, extended_attention_mask, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down)
                    elif arg_dict['num_target_objs_head']:
                        pred, num_target_objs_logits = model(input_ids, txt_seg_ids, vis_seg, bboxes, extended_attention_mask, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down, obj_men=obj_men)
                    elif arg_dict['mentioned_and_new_head']:
                        pred, num_mentioned_targets_logits, num_new_targets_logits = model(input_ids, txt_seg_ids, vis_seg, bboxes, extended_attention_mask, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down, obj_men=obj_men)
                    else:
                        pred = model(input_ids, txt_seg_ids, vis_seg, bboxes, extended_attention_mask, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down, obj_men=obj_men)
                elif MODEL == 'LXMERT':
                    pred = model(input_ids_uncased, input_mask_uncased, segment_ids_uncased, bboxes_lxmert, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, vis_mask, obj_men=obj_men)
                
                # shape of pred before reshaping: (batch, 512, 1)
                pred = pred.reshape(1,-1)
                # shape of pred after reshaping: (1, batch*512)
                pred = pred[output_mask==1].reshape(-1,1)
                # shape of pred after masking + reshaping: (n_scene_objs, 1)
                truth = reference.float().reshape(-1,1)
                loss = criterion(pred, truth).detach()

                pred_bin = pred > 0
                truth_bin = truth > 0.5
                
                hit = torch.sum(pred_bin*truth_bin == 1).detach()
                pred_positive = torch.sum(pred > 0).detach()
                truth_positive = torch.sum(truth > 0.5).detach()

                if arg_dict['num_target_objs_head']:
                    pred_n_target_objs = torch.argmax(num_target_objs_logits, dim=1)
                    n_target_objs_accuracy = torch.sum((pred_n_target_objs == num_gt_targets))/len(num_gt_targets)
                    n_target_objs_acc_list.append(n_target_objs_accuracy.detach().item())
                
                if arg_dict['mentioned_and_new_head']:
                    pred_num_mentioned_targets = torch.argmax(num_mentioned_targets_logits, dim=1)
                    pred_num_new_targets = torch.argmax(num_new_targets_logits, dim=1)
                    mentioned_targets_accuracy = torch.sum((pred_num_mentioned_targets == num_mentioned_targets))/len(num_mentioned_targets)
                    new_targets_accuracy = torch.sum((pred_num_new_targets == num_new_targets))/len(num_new_targets)
                    n_mentioned_targets_acc_list.append(mentioned_targets_accuracy.detach().item())
                    n_new_targets_acc_list.append(new_targets_accuracy.detach().item())

                total_loss.append(float(loss))
                total_hit += int(hit)
                total_pred_positive += int(pred_positive)
                total_truth_positive += int(truth_positive)
                total_pred += int(pred.shape[0])
            print('#pred positives',total_pred_positive)
            print('#groundtruth positives',total_truth_positive)
            print('#total pred', total_pred)
            print('#hit', total_hit)
            if arg_dict['num_target_objs_head']:
                print('Num target obj head accuracy:', torch.tensor(n_target_objs_acc_list).mean().item())
            if arg_dict['mentioned_and_new_head']:
                print('Num mentioned targets head accuracy:', torch.tensor(n_mentioned_targets_acc_list).mean().item())
                print('Num new targets head accuracy:', torch.tensor(n_new_targets_acc_list).mean().item())
            total_loss = sum(total_loss)/len(total_loss)
            if (total_pred_positive == 0):
                total_pred_positive = 1e10
            prec = total_hit / total_pred_positive
            recall = total_hit / total_truth_positive
            try:
                f1 = 2/(1/prec + 1/recall)
            except:
                f1 = 0
            print('f1', f1)
        return total_loss, prec, recall, f1

    # Training setup
    if MODEL == 'UNITER':
        model = Modified_Uniter(arg_dict['obj_id'], arg_dict['vis_feats_clip'], arg_dict['vis_feats_rcnn'], arg_dict['pos'], arg_dict['scene_seg'], arg_dict['obj_embs_bert'], arg_dict['obj_embs_sbert'], arg_dict['kb_id_bert'], arg_dict['kb_id_sbert'], arg_dict['attn_bias'], arg_dict['graph_attn'], arg_dict['obj_men'], arg_dict['pred_men'], n_target_objs_head=arg_dict['num_target_objs_head'], mentioned_and_new_head=arg_dict['mentioned_and_new_head']).to(device)
    elif MODEL == 'LXMERT':
        model = Modified_Lxmert(arg_dict['obj_id'], arg_dict['vis_feats_clip'], arg_dict['vis_feats_rcnn'], arg_dict['pos'], arg_dict['scene_seg'], arg_dict['obj_embs_bert'], arg_dict['obj_embs_sbert'], arg_dict['kb_id_bert'], arg_dict['kb_id_sbert'], arg_dict['attn_bias'], arg_dict['graph_attn'], arg_dict['obj_men']).to(device)

    if arg_dict['kb_id_bert']:
        # Overwrite the initial KB id embedding with BERT output
        # This is not used in the final implementation: 
        # Using trainable embeddings for prefab entries allows the model to 'cheat' by memorising the prefab feature of each prefab item,
        # which is not good for generalization.
        # SBERT version not implemented
        state_dict = model.state_dict()
        emb_mat = state_dict['kb_id_enc_bert.weight']
        state_dict['kb_id_enc_bert.weight'] = make_KBid_emb_init(emb_mat)
        model.load_state_dict(state_dict)

    criterion = FocalLoss(gamma=GAMMA, alpha=ALPHA)
    criterion_men = FocalLoss(gamma=args.men_gamma, alpha=args.men_alpha)
    if arg_dict['num_target_objs_head']:
        criterion_n_target_objs = FocalLoss(gamma=args.men_gamma, alpha=args.men_alpha)
    if arg_dict['mentioned_and_new_head']:
        criterion_n_mentioned_targets = FocalLoss(gamma=args.men_gamma, alpha=args.men_alpha)
        criterion_n_new_targets = FocalLoss(gamma=args.men_gamma, alpha=args.men_alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if SCHEDULER == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=len(train_loader)*N_EPOCH)

    scaler = GradScaler()

    # Train
    n_iter = 0
    n_prev_iter = 0
    running_loss = 0
    best_f1 = 0
    for epoch in range(N_EPOCH):
        for batch_idx, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            txt_seg_ids = batch['txt_seg_ids'].to(device)
            vis_feats_clip = batch['vis_feats'].to(device)
            vis_feats_rcnn = batch['vis_feats_rcnn'].to(device)
            kb_embs_bert = batch['obj_embs'].to(device)
            kb_embs_sbert = batch['obj_embs_SBERT'].to(device)
            obj_ids = batch['obj_ids'].to(device)
            pos_x = batch['pos_x'].to(device)
            pos_y = batch['pos_y'].to(device)
            pos_z = batch['pos_z'].to(device)
            bboxes = batch['bboxes'].to(device)
            vis_seg = batch['vis_seg'].to(device)
            extended_attention_mask = batch['extended_attention_mask'].to(device)
            output_mask = batch['output_mask'].to(device)
            reference = batch['reference'].to(device)
            num_gt_targets = batch['num_gt_targets'].to(device)
            num_mentioned_targets = batch['num_mentioned_targets'].to(device)
            num_new_targets = batch['num_new_targets'].to(device)
            scene_seg = batch['scene_segs'].to(device)
            kb_id = batch['KB_ids'].to(device)
            rel_mask_left = batch['rel_mask_left'].to(device).long()
            rel_mask_right = batch['rel_mask_right'].to(device).long()
            rel_mask_up = batch['rel_mask_up'].to(device).long()
            rel_mask_down = batch['rel_mask_down'].to(device).long()
            input_ids_uncased = batch['input_ids_uncased'].to(device)
            input_mask_uncased = batch['input_mask_uncased'].to(device).long()
            segment_ids_uncased = batch['segment_ids_uncased'].to(device)
            bboxes_lxmert = batch['bboxes_lxmert'].to(device).float()
            vis_mask = batch['vis_mask'].to(device).long()
            obj_men = batch['obj_men'].to(device).long()

            truth = reference.float().reshape(-1,1)

            if MODEL == "UNITER":
                if arg_dict['pred_men']:
                    pred, pred_men = model(input_ids, txt_seg_ids, vis_seg, bboxes, extended_attention_mask, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down)
                elif arg_dict['num_target_objs_head']:
                    pred, num_target_objs_logits = model(input_ids, txt_seg_ids, vis_seg, bboxes, extended_attention_mask, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down, obj_men=obj_men)
                elif arg_dict['mentioned_and_new_head']:
                    pred, num_mentioned_targets_logits, num_new_targets_logits = model(input_ids, txt_seg_ids, vis_seg, bboxes, extended_attention_mask, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down, obj_men=obj_men)
                else:
                    pred = model(input_ids, txt_seg_ids, vis_seg, bboxes, extended_attention_mask, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down, obj_men=obj_men)
            elif MODEL == 'LXMERT':
                pred = model(input_ids_uncased, input_mask_uncased, segment_ids_uncased, bboxes_lxmert, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, vis_mask, obj_men=obj_men)


            # shape of pred before reshaping: (batch, 512, 1)
            pred = pred.reshape(1,-1)
            # shape of pred after reshaping: (1, batch*512)
            pred = pred[output_mask==1].reshape(-1,1)
            # shape of pred after masking + reshaping: (n_scene_objs_batch, 1)
            loss = criterion(pred, truth)

            if arg_dict['num_target_objs_head']:
                gt_targets_array = torch.zeros(num_gt_targets.shape[0], 4).to(device)
                for i in range(len(num_gt_targets)):
                    idx = num_gt_targets[i] if num_gt_targets[i] < 3 else 3
                    gt_targets_array[i][idx] = 1

                loss_num_objs_head = criterion_n_target_objs(num_target_objs_logits, gt_targets_array)
                # Right now we are re-using pred_men weightss
                loss = args.men_weight * loss_num_objs_head + (1 - args.men_weight) * loss
            
            if arg_dict['mentioned_and_new_head']:
                mentioned_targets_array = torch.zeros(num_mentioned_targets.shape[0], 4).to(device)
                new_targets_array = torch.zeros(num_new_targets.shape[0], 4).to(device)
                for i in range(len(num_mentioned_targets)):
                    idx = num_mentioned_targets[i] if num_mentioned_targets[i] < 3 else 3
                    mentioned_targets_array[i][idx] = 1
                for i in range(len(num_new_targets)):
                    idx = num_new_targets[i] if num_new_targets[i] < 3 else 3
                    new_targets_array[i][idx] = 1
                
                loss_mentioned_targets_head = criterion_n_mentioned_targets(num_mentioned_targets_logits, mentioned_targets_array)
                loss_new_targets_head = criterion_n_new_targets(num_new_targets_logits, new_targets_array)
                loss = args.men_weight/2 * loss_mentioned_targets_head + args.men_weight/2 * loss_new_targets_head + (1 - args.men_weight) * loss

            if arg_dict['pred_men']:
                pred_men = pred_men.reshape(1,-1)
                pred_men = pred_men[output_mask==1].reshape(-1,1)
                
                zeros = torch.zeros(obj_men.shape[0], 512-obj_men.shape[1]).to(device)
                obj_men = torch.cat((zeros, obj_men), axis=1)

                truth_men = obj_men.reshape(1,-1)[output_mask==1].reshape(-1,1)
                loss_men = criterion_men(pred_men, truth_men)
                loss = args.men_weight * loss_men + (1 - args.men_weight) * loss

            loss.backward()
            optimizer.step()

            if SCHEDULER:
                scheduler.step()

            n_iter += 1
            writer.add_scalar('Loss/train_batch', loss, n_iter)
            running_loss += loss.detach()

            if batch_idx % 250 == 0:
                print(epoch, batch_idx)
                print(running_loss/(n_iter-n_prev_iter))
                loss, prec, recall, f1 = eval(model)
                writer.add_scalar('Loss/train_avg', running_loss/(n_iter-n_prev_iter), n_iter)
                n_prev_iter = n_iter
                running_loss = 0
                writer.add_scalar('Loss/dev', loss, n_iter)
                writer.add_scalar('Precision/dev', prec, n_iter)
                writer.add_scalar('Recall/dev', recall, n_iter)
                writer.add_scalar('F1/dev', f1, n_iter)

                try:
                    os.makedirs(f'./result/checkpoint/{NAME}')
                except:
                    pass

                if f1 > best_f1:
                    best_f1 = f1
                    torch.save({
                        'epoch': epoch,
                        'step': n_iter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'dev_loss': loss,
                        }, f'./result/checkpoint/{NAME}/batchsize{BATCH_SIZE}_lr{LR}_FocalALPHA{ALPHA}_GAMMA{GAMMA}_{epoch}_{batch_idx}_{loss}_{f1}.bin')
    print('DONE !!!')
    print(f'Best F1: {best_f1}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--NAME', default='unnamed')
    parser.add_argument('--MODEL', default='UNITER')
    parser.add_argument('--BATCH_SIZE', default=16, type=int)
    parser.add_argument('--BATCH_SIZE_DEV', default=8, type=int)
    parser.add_argument('--LR', default=5e-6, type=float)
    parser.add_argument('--N_EPOCH', default=30, type=int)
    parser.add_argument('--GAMMA', default=2, type=int)
    parser.add_argument('--ALPHA', default=5, type=int)
    parser.add_argument('--scheduler', default=False)
    
    parser.add_argument('--obj_id', default=False)
    parser.add_argument('--vis_feats_clip', default=False)
    parser.add_argument('--vis_feats_rcnn', default=False)
    parser.add_argument('--pos', default=False)
    parser.add_argument('--scene_seg', default=False)
    parser.add_argument('--obj_embs_bert', default=False)
    parser.add_argument('--obj_embs_sbert', default=False)
    parser.add_argument('--kb_id_bert', default=False)
    parser.add_argument('--kb_id_sbert', default=False) # Not implemented. This line should always be False
    parser.add_argument('--obj_men', default=False)

    parser.add_argument('--pred_men', default=False) # Auxilliary prediction 
    parser.add_argument('--men_gamma', default=2, type=int)
    parser.add_argument('--men_alpha', default=5, type=int)
    parser.add_argument('--men_weight', default=0.3, type=float)

    parser.add_argument('--attn_bias', default=False)
    parser.add_argument('--graph_attn', default=False)
    
    parser.add_argument('--more_roi', default=False)

    parser.add_argument('--SPLIT', default='train')
    parser.add_argument('--visual_attrs', default=False) # not fully implemented!
    parser.add_argument('--num_target_objs_head', default=False) # Auxiliary task to predict the number of referred objects
    parser.add_argument('--mentioned_and_new_head', default=False) # Auxiliary task to predict the number of previously mentioned and new referred objs
    args = parser.parse_args()

    train(args)