import json
import torch
from torch.utils.data import Dataset, DataLoader
from model.Transformers_VQA_master.src.tokenization import BertTokenizer

PROCESSED_ROOT = './processed'

class UNITER_on_CLIP_BERT_Dataset(Dataset):
    def __init__(self, split, more_roi, max_n_obj=200):
        self.file_path = f'{PROCESSED_ROOT}/{split}.json'
        self.KB_emb_path = f'{PROCESSED_ROOT}/KB_{split}.pt'
        self.KB_emb_SBERT_path = f'{PROCESSED_ROOT}/KB_SBERT_{split}.pt'
        self.vis_feat_path = f'{PROCESSED_ROOT}/img_features.pt'
        self.vis_feat_rcnn_path = f'{PROCESSED_ROOT}/img_features_rcnn.json'
        self.KB_dict_path = f'{PROCESSED_ROOT}/KB_dict.json'
        self.roi_path = f'{PROCESSED_ROOT}/more_img_roi.pt'
        self.max_n_obj = max_n_obj
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.more_roi = more_roi

        self.tokenizer = BertTokenizer.from_pretrained('./pretrained/bert-base-cased/bert-base-cased-vocab.txt')
        self.tokenizer_uncased = BertTokenizer.from_pretrained('./pretrained/bert-base-uncased/bert-base-uncased-vocab.txt')

        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        with open(self.KB_dict_path, 'r', encoding='utf-8') as f:
            self.KB_dict = json.load(f)

        with open(self.vis_feat_rcnn_path, 'r', encoding='utf-8') as f:
            self.vis_feat_dict_rcnn = json.load(f)

        self.KB_emb_dict = torch.load(self.KB_emb_path, map_location=self.device) # KB_emb_dict['obj_string']
        self.KB_emb_dict_SBERT = torch.load(self.KB_emb_SBERT_path, map_location=self.device)
        self.vis_feat_dict = torch.load(self.vis_feat_path, map_location=self.device) # vis_feat_dict[scene_name][local_idx or 'Scene']
        self.roi_dict = torch.load(self.roi_path, map_location=self.device)

    def __len__(self):
        return len(self.data)
    
    def _uniterBoxes(self, boxes):#uniter requires a 7-dimensiom beside the regular 4-d bbox. From the Transformer VQA repo
        new_boxes = torch.zeros(boxes.shape[0],7)
        new_boxes = torch.zeros(boxes.shape[0],7)
        new_boxes[:,1] = boxes[:,0]
        new_boxes[:,0] = boxes[:,1]
        new_boxes[:,3] = boxes[:,2]
        new_boxes[:,2] = boxes[:,3]
        new_boxes[:,4] = new_boxes[:,3]-new_boxes[:,1] #w
        new_boxes[:,5] = new_boxes[:,2]-new_boxes[:,0] #h
        new_boxes[:,6] = new_boxes[:,4]*new_boxes[:,5] #area
        return new_boxes  

    def _make_relationship_mask(self, obj_rels, obj_ids):
        out = []
        for rel in ['left', 'right', 'up', 'down']:
            rel_dict = obj_rels[rel]
            mask = torch.zeros(self.max_n_obj, self.max_n_obj)
            for idx, obj_id in enumerate(obj_ids):
                if str(obj_id) in rel_dict.keys():
                    for val in rel_dict[str(obj_id)]:
                        mask[idx][obj_ids.index(val)] = 1
            mask = torch.nn.ZeroPad2d((512 - self.max_n_obj, 0, 0, 512 - self.max_n_obj))(mask) # zero pad to (512, 512)
            mask = mask.unsqueeze(0)
            out.append(mask)
        return out

    def __getitem__(self, index):
        line = self.data[index]
        dial, objects, reference, obj_ids, obj_pos, obj_bbox, scenes, KB_ids, scene_segs, obj_rels, obj_mens = line['dial'], line['objects'], line['reference_mask'], line['candidate_ids'], line['candidate_pos'], line['candidate_bbox'], line['scenes'], line['KB_ids'], line['scene_seg'], line['candidate_relations'], line['candidate_mentioned']
        
        # Retrive add extra ROI features (#roi, 1024)
        if self.more_roi:
            roi_boxes, roi_feats = get_extra_roi_feats(scenes, obj_bbox, self.roi_dict)
            roi_boxes = roi_boxes.to(self.device)
            roi_feats = roi_feats.to(self.device)

        # relationship mask (512, 512) * 4
        rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down = self._make_relationship_mask(obj_rels, obj_ids)

        # obj_embs
        object_string = json.dumps(objects)
        obj_embs = self.KB_emb_dict[object_string] # (#object, 1024)
        obj_embs = torch.nn.ZeroPad2d((0,0,0, self.max_n_obj - obj_embs.shape[0]))(obj_embs) # zero pad to (max_n_obj, 1024)

        obj_embs_SBERT = self.KB_emb_dict_SBERT[object_string] # (#object, 768)
        obj_embs_SBERT = torch.nn.ZeroPad2d((0,0,0, self.max_n_obj - obj_embs_SBERT.shape[0]))(obj_embs_SBERT) # zero pad to (max_n_obj, 768)

        # scene_segs (1, #obj + 2)
        scene_segs = torch.tensor([scene_segs])
        active_scene_h = obj_bbox[-1][2] + 1e-7
        active_scene_w = obj_bbox[-1][3] + 1e-7
        inactive_scene_h = obj_bbox[-2][2] + 1e-7
        inactive_scene_w = obj_bbox[-2][3] + 1e-7
        if scene_segs[0,-2] == 1:
            inactive_scene_h = obj_bbox[-1][2] + 1e-7
            inactive_scene_w = obj_bbox[-1][3] + 1e-7
            active_scene_h = obj_bbox[-2][2] + 1e-7
            active_scene_w = obj_bbox[-2][3] + 1e-7
        
        scene_segs = torch.nn.ZeroPad2d((0, self.max_n_obj - scene_segs.shape[1],0,0))(scene_segs) # zero pad to (1, max_n_obj)

        # KB_ids (1, #obj)
        KB_ids = torch.tensor([KB_ids])
        KB_ids = torch.nn.ZeroPad2d((0, self.max_n_obj - KB_ids.shape[1],0,0))(KB_ids) # zero pad to (1, max_n_obj)

        # Merge vis feats of all scenes
        vis_feat_scenes = {}
        scene_feats = []
        for scene in scenes:
            vis_feat_scene = self.vis_feat_dict[scene+'_scene']
            for key, val in vis_feat_scene.items():
                vis_feat_scenes[key] = val
            scene_feats.append(vis_feat_scene['scene'])

        vis_feat_scenes_rcnn = {}
        scene_feats_rcnn = []
        for scene in scenes:
            vis_feat_scene_rcnn = self.vis_feat_dict_rcnn[scene]
            for key, val in vis_feat_scene_rcnn.items():
                vis_feat_scenes_rcnn[key] = torch.tensor([val])
            scene_feats_rcnn.append(vis_feat_scenes_rcnn['scene'])
        
        # Make the vis feat tensor (CLIP) (#obj + #scene, 512+1024)
        for _, obj_id in enumerate(obj_ids):
            if _ == 0:
                vis_feats = vis_feat_scenes[obj_id]
            else:
                vis_feats = torch.cat((vis_feats, vis_feat_scenes[obj_id]), axis=0)
        for scene_feat in scene_feats:
            vis_feats = torch.cat((vis_feats, scene_feat), axis=0) 

        # Vis feats using BUTD (Fast rcnn)
        for _, obj_id in enumerate(obj_ids):
            if _ == 0:
                vis_feats_rcnn = vis_feat_scenes_rcnn[str(obj_id)]
            else:
                vis_feats_rcnn = torch.cat((vis_feats_rcnn, vis_feat_scenes_rcnn[str(obj_id)]), axis=0)
        for scene_feat_rcnn in scene_feats_rcnn:
            vis_feats_rcnn = torch.cat((vis_feats_rcnn, scene_feat_rcnn), axis=0)

        vis_feats_rcnn = torch.cat((vis_feats_rcnn.to(self.device), roi_feats.to(self.device)), axis=0)

        vis_seg = torch.ones_like(vis_feats_rcnn[:,0]).unsqueeze(0)
        vis_mask = torch.ones_like(vis_feats_rcnn[:,0]).unsqueeze(0)
        vis_feats = torch.nn.ZeroPad2d((0,0,0, self.max_n_obj - vis_feats.shape[0]))(vis_feats) # zero pad to (max_n_obj, 512)
        vis_feats_rcnn = torch.nn.ZeroPad2d((0,0,0, self.max_n_obj - vis_feats_rcnn.shape[0]))(vis_feats_rcnn)
        vis_seg = torch.nn.ZeroPad2d((0, self.max_n_obj - vis_seg.shape[1],0,0))(vis_seg).long() # zero pad to (1, max_n_obj)
        vis_mask = torch.nn.ZeroPad2d((0,self.max_n_obj - vis_mask.shape[1],0,0))(vis_mask)
            
        # Obj pos (1, #obj)
        pos_x = []
        pos_y = []
        pos_z = []
        for pos in obj_pos:
            pos_x.append(pos[0])
            pos_y.append(pos[1])
            pos_z.append(pos[2])
        pos_x = torch.nn.ZeroPad2d((0, self.max_n_obj - len(obj_ids), 0, 0))(torch.tensor(pos_x).reshape(1,-1)) # zero pad to (1, max_n_obj)
        pos_y = torch.nn.ZeroPad2d((0, self.max_n_obj - len(obj_ids), 0, 0))(torch.tensor(pos_y).reshape(1,-1))
        pos_z = torch.nn.ZeroPad2d((0, self.max_n_obj - len(obj_ids), 0, 0))(torch.tensor(pos_z).reshape(1,-1))

        # Bboxes (#obj, 7)
        for _, boxes in enumerate(obj_bbox):
            if _ == 0:
                bboxes = torch.tensor([boxes])
            else:
                bboxes = torch.cat((bboxes, torch.tensor([boxes])), axis=0)
        bboxes = torch.cat((bboxes.to(self.device), roi_boxes.to(self.device)), axis=0)
        bboxes = torch.nn.ZeroPad2d((0,0,0,self.max_n_obj-bboxes.shape[0]))(bboxes) # zero pad to (max_n_obj, 4) x,y,h,w
        new_bboxes = torch.zeros_like(bboxes)
        new_bboxes[:,0:2] = bboxes[:,0:2] 
        new_bboxes[:,2] = bboxes[:,0] + bboxes[:,3]
        new_bboxes[:,3] = bboxes[:,1] + bboxes[:,2] # x1, y1, x2, y2
        bboxes = self._uniterBoxes(new_bboxes) # convert to uniter's bbox representation (max_n_obj, 7)

        # keep a copy for lxmert. Divide over the image width/height
        bboxes_lxmert = new_bboxes.float()
        # get scene widths/heights
        scene_segs_expanded_w = scene_segs.squeeze(0).unsqueeze(1).repeat(1,4)
        scene_segs_expanded_w[:,1] = 3 
        scene_segs_expanded_w[:,3] = 3 
        scene_segs_expanded_h = scene_segs.squeeze(0).unsqueeze(1).repeat(1,4)
        scene_segs_expanded_h[:,0] = 3 
        scene_segs_expanded_h[:,2] = 3 
        # active
        bboxes_lxmert[scene_segs_expanded_h == 1] /= active_scene_h
        bboxes_lxmert[scene_segs_expanded_w == 1] /= active_scene_w
        bboxes_lxmert[scene_segs_expanded_h == 2] /= inactive_scene_h
        bboxes_lxmert[scene_segs_expanded_w == 2] /= inactive_scene_w

        # Obj_ids (1, #obj)
        obj_ids = torch.tensor([obj_ids])
        obj_ids += 1 # 0 is reserved for padding
        output_mask = torch.ones_like(obj_ids)
        obj_ids = torch.nn.ZeroPad2d((0, self.max_n_obj - obj_ids.shape[1], 0, 0))(obj_ids) # zero pad to (1, max_n_obj)
        output_mask = torch.nn.ZeroPad2d((512 - self.max_n_obj, self.max_n_obj - output_mask.shape[1], 0, 0))(output_mask) # zero pad to (1, 512)

        # Mentioned flag (1, #obj)
        obj_men = torch.tensor([obj_mens])
        obj_men = torch.nn.ZeroPad2d((0, self.max_n_obj - obj_men.shape[1], 0, 0))(obj_men)

        # Input ids (512-max_n_obj)
        max_n_ids = 512 - self.max_n_obj
        tokens_a = self.tokenizer.tokenize(dial.strip())
        # Brutally handle the over-sized text inputs
        if len(tokens_a) > 510-self.max_n_obj:
            # print(len(tokens_a))
            tokens_a = tokens_a[510-self.max_n_obj-len(tokens_a):]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        if len(tokens) > 512-self.max_n_obj:
            print(len(tokens))
            print('TOO LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOONG')
            raise 
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(input_ids)

        # Zero pad
        padding = [0] * (512-self.max_n_obj - len(input_ids))
        
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        input_ids = torch.tensor([input_ids])
        input_mask = torch.tensor([input_mask])
        txt_seg_ids = torch.tensor([segment_ids])

        attn_mask = torch.cat((input_mask.to(self.device), vis_mask.to(self.device)), axis=1)
        extended_attention_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # input_ids with uncased tokenizer (for lxmert)
        tokens_b = self.tokenizer_uncased.tokenize(dial.strip())
        if len(tokens_b) > 510-self.max_n_obj:
            tokens_b = tokens_b[510-self.max_n_obj-len(tokens_b):]

        tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
        input_ids_uncased = self.tokenizer_uncased.convert_tokens_to_ids(tokens_b)

        padding = [0] * (512-self.max_n_obj - len(input_ids_uncased))

        segment_ids_uncased = [0] * len(tokens_b)
        input_mask_uncased = [1] * len(input_ids_uncased)
        
        input_ids_uncased += padding
        input_mask_uncased += padding
        segment_ids_uncased += padding
        input_ids_uncased = torch.tensor([input_ids_uncased])
        input_mask_uncased = torch.tensor([input_mask_uncased])
        segment_ids_uncased = torch.tensor([segment_ids_uncased])

        reference = torch.tensor(reference)

        round_idx = line['round_idx']
        dial_idx = line['dial_idx']

        return input_ids, txt_seg_ids, input_ids_uncased, input_mask_uncased, segment_ids_uncased, vis_mask, vis_feats, vis_feats_rcnn, KB_ids, obj_ids, obj_men, pos_x, pos_y, pos_z, bboxes, bboxes_lxmert, vis_seg, extended_attention_mask, output_mask, reference, dial_idx, round_idx, obj_embs, scene_segs, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down, obj_embs_SBERT

def mr_collate(data):
    
    input_ids, txt_seg_ids, input_ids_uncased, input_mask_uncased, segment_ids_uncased, vis_mask, vis_feats, vis_feats_rcnn, KB_ids, obj_ids, obj_men, pos_x, pos_y, pos_z, bboxes, bboxes_lxmert, vis_seg, extended_attention_mask, output_mask, reference, dial_idx, round_idx, obj_embs, scene_segs, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down, obj_embs_SBERT = data[0]
    vis_feats = vis_feats.unsqueeze(0)
    vis_feats_rcnn = vis_feats_rcnn.unsqueeze(0)
    bboxes = bboxes.unsqueeze(0)
    bboxes_lxmert = bboxes_lxmert.unsqueeze(0)
    obj_embs = obj_embs.unsqueeze(0)
    obj_embs_SBERT = obj_embs_SBERT.unsqueeze(0)
    dial_idx = [dial_idx]
    round_idx = [round_idx]

    for idx, line in enumerate(data):
        if idx == 0:
            continue
        input_ids_l, txt_seg_ids_l, input_ids_uncased_l, input_mask_uncased_l, segment_ids_uncased_l, vis_mask_l, vis_feats_l, vis_feats_rcnn_l, KB_ids_l, obj_ids_l, obj_men_l, pos_x_l, pos_y_l, pos_z_l, bboxes_l, bboxes_lxmert_l, vis_seg_l, extended_attention_mask_l, output_mask_l, reference_l, dial_idx_l, round_idx_l, obj_embs_l, scene_segs_l, rel_mask_left_l, rel_mask_right_l, rel_mask_up_l, rel_mask_down_l, obj_embs_SBERT_l = line 
        vis_feats_l = vis_feats_l.unsqueeze(0)
        vis_feats_rcnn_l = vis_feats_rcnn_l.unsqueeze(0)
        bboxes_l = bboxes_l.unsqueeze(0)
        bboxes_lxmert_l = bboxes_lxmert_l.unsqueeze(0)
        obj_embs_l = obj_embs_l.unsqueeze(0)
        obj_embs_SBERT_l = obj_embs_SBERT_l.unsqueeze(0)
        
        input_ids = torch.cat((input_ids, input_ids_l), dim=0)
        txt_seg_ids = torch.cat((txt_seg_ids, txt_seg_ids_l), dim=0)
        input_ids_uncased = torch.cat((input_ids_uncased, input_ids_uncased_l), dim=0)
        input_mask_uncased = torch.cat((input_mask_uncased, input_mask_uncased_l), dim=0)
        segment_ids_uncased = torch.cat((segment_ids_uncased, segment_ids_uncased_l), dim=0)
        vis_mask = torch.cat((vis_mask, vis_mask_l), dim=0)
        vis_feats  = torch.cat((vis_feats, vis_feats_l), dim=0)
        vis_feats_rcnn  = torch.cat((vis_feats_rcnn, vis_feats_rcnn_l), dim=0)
        KB_ids = torch.cat((KB_ids, KB_ids_l), axis=0)
        obj_ids = torch.cat((obj_ids, obj_ids_l), dim=0)
        obj_men = torch.cat((obj_men, obj_men_l), dim=0)
        pos_x = torch.cat((pos_x, pos_x_l), dim=0)
        pos_y = torch.cat((pos_y, pos_y_l), dim=0)
        pos_z = torch.cat((pos_z, pos_z_l), dim=0)
        bboxes = torch.cat((bboxes, bboxes_l), dim=0)
        bboxes_lxmert = torch.cat((bboxes_lxmert, bboxes_lxmert_l), dim=0)
        vis_seg = torch.cat((vis_seg, vis_seg_l), dim=0)
        extended_attention_mask = torch.cat((extended_attention_mask, extended_attention_mask_l), dim=0)
        output_mask = torch.cat((output_mask, output_mask_l), dim=1)
        reference = torch.cat((reference,reference_l), dim=0)
        dial_idx.append(dial_idx_l)
        round_idx.append(round_idx_l)
        obj_embs = torch.cat((obj_embs, obj_embs_l), dim=0)
        obj_embs_SBERT = torch.cat((obj_embs_SBERT, obj_embs_SBERT_l), dim=0)
        scene_segs = torch.cat((scene_segs, scene_segs_l), dim=0)
        rel_mask_left = torch.cat((rel_mask_left, rel_mask_left_l), dim=0)
        rel_mask_right = torch.cat((rel_mask_right, rel_mask_right_l), dim=0)
        rel_mask_up = torch.cat((rel_mask_up, rel_mask_up_l), dim=0)
        rel_mask_down = torch.cat((rel_mask_down, rel_mask_down_l), dim=0)
        
    return {
        'input_ids': input_ids, 
        'txt_seg_ids': txt_seg_ids, 
        'input_ids_uncased': input_ids_uncased,
        'input_mask_uncased': input_mask_uncased,
        'segment_ids_uncased': segment_ids_uncased,
        'vis_mask': vis_mask,
        'vis_feats': vis_feats, 
        'vis_feats_rcnn': vis_feats_rcnn, 
        'KB_ids': KB_ids, 
        'obj_ids': obj_ids, 
        'obj_men': obj_men, 
        'pos_x': pos_x, 
        'pos_y': pos_y, 
        'pos_z': pos_z, 
        'bboxes': bboxes, 
        'bboxes_lxmert': bboxes_lxmert, 
        'vis_seg': vis_seg, 
        'extended_attention_mask': extended_attention_mask, 
        'output_mask': output_mask, 
        'reference': reference, 
        'dial_idx': dial_idx,       
        'round_idx': round_idx,
        'obj_embs': obj_embs,
        'obj_embs_SBERT': obj_embs_SBERT,
        'scene_segs': scene_segs, 
        'rel_mask_left': rel_mask_left,
        'rel_mask_right': rel_mask_right,
        'rel_mask_up': rel_mask_up,
        'rel_mask_down': rel_mask_down
    }
    

def get_extra_roi_feats(scene_paths, bboxes, roi_dict, IOU_thresh=0.8):
    for scene_path in scene_paths:
        if scene_path[:2] == 'm_':
            scene_path = scene_path[2:]
        entry = roi_dict["/kaggle/input/simmc-img/data/all_images/"+scene_path+".png"]
        roi_boxes = entry["instances"]
        features = entry["roi_features"]

        for i in range(roi_boxes.shape[0]):
            roi_box = roi_boxes[i,:]
            try:
                if len(bboxes) + bboxes_out.shape[0] > 198:
                    break
            except:
                pass

            # Rearrange into the SIMMC's format
            # x0 y0 x1 y1 to x0 y0 h w
            x0_r = roi_box[0]
            y0_r = roi_box[1]
            x1_r = roi_box[2]
            y1_r = roi_box[3]
            roi_box_new = [x0_r, y0_r, y1_r-y0_r, x1_r-x0_r]
            add = True
            
            # for obj_box in bboxes:
            #     # Compute ROI and filter out the ones high IOUs
            #     x0 = obj_box[0]
            #     y0 = obj_box[1]
            #     x1 = x0 + obj_box[3]
            #     y1 = y0 + obj_box[2]

            #     xA = max(x0, x0_r)
            #     yA = max(y0, y0_r)
            #     xB = min(x1, x1_r)
            #     yB = min(y1, y1_r)

            #     interArea = max(0, xB - xA + 1e-6) * max(0, yB - yA + 1e-6)
            #     # compute the area of both the prediction and ground-truth
            #     # rectangles
            #     boxAArea = (x1 - x0 + 1e-6) * (y1 - y0 + 1e-6)
            #     boxBArea = (x1_r - x0_r + 1e-6) * (y1_r - y0_r + 1e-6)
                
            #     iou = interArea / float(boxAArea + boxBArea - interArea)

            #     # Filter out ROIs with high IOU with obj bboxes
            #     if iou > IOU_thresh:
            #         add = False
            #         break
            if add:
                try:
                    bboxes_out = torch.cat((bboxes_out, torch.tensor(roi_box_new).reshape(1,-1)), axis=0)
                    feats_out = torch.cat((feats_out, features[i,:].reshape(1,-1)), axis=0)
                except:
                    bboxes_out = torch.tensor(roi_box_new).reshape(1,-1)
                    feats_out = torch.tensor(features[i,:]).reshape(1,-1)

    return bboxes_out, feats_out

        

def make_loader(split, batch_size, more_roi=True):
    dataset = UNITER_on_CLIP_BERT_Dataset(split, more_roi)
    loader = DataLoader(dataset, batch_size=batch_size ,shuffle=True, collate_fn=mr_collate)
    return loader

if __name__ == "__main__":
    pass