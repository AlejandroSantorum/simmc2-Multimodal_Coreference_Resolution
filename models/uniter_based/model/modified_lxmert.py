import torch
from model.Transformers_VQA_master.vqa_model import VQAModel
from model.model_utils import ObjPositionalEncoding

class Modified_Lxmert(torch.nn.Module):
    def __init__(self, obj_id, vis_feats_clip, vis_feats_rcnn, pos, scene_seg, obj_embs_bert, obj_embs_sbert, kb_id_bert, kb_id_sbert, attn_bias=False, graph_attn=False, obj_men=False):
        super(Modified_Lxmert, self).__init__()

        self.obj_id = obj_id
        self.vis_feats_clip = vis_feats_clip
        self.vis_feats_rcnn = vis_feats_rcnn
        self.pos = pos
        self.scene_seg = scene_seg
        self.obj_embs_bert = obj_embs_bert
        self.obj_embs_sbert = obj_embs_sbert
        self.kb_id_bert = kb_id_bert
        self.kb_id_sbert = kb_id_sbert
        self.obj_men = obj_men
        if not (obj_id or vis_feats_clip or vis_feats_rcnn or pos or scene_seg or obj_embs_bert or obj_embs_sbert or kb_id_bert or kb_id_sbert):
            print('Must have some feature for each object')
            raise

        self.attn_bias = attn_bias
        self.graph_attn = graph_attn

        # Load pretrained LXMERT
        self.lxmert = VQAModel(num_answers=69, model='lxmert', attn_bias=attn_bias, graph_attn=graph_attn)
        self.lxmert.encoder.load('./model/Transformers_VQA_master/models/pretrained/model_LXRT.pth')

        # Convert input (Ojb_idx: 512, visual(CLIP): 512, visual(BUTD): 2048, pos: 3*128, scene_seg: 128, kb_embs(BERT): 1024, kb_embs(SBERT): 768)
        emb_bag_in = 0
        if self.obj_id:
            self.obj_idx_emb = torch.nn.Embedding(200, 512, padding_idx=0)
            emb_bag_in += 512
        if self.vis_feats_clip:
            emb_bag_in += 512
        if self.vis_feats_rcnn:
            emb_bag_in += 2048
        if self.pos:
            self.obj_pos_enc = ObjPositionalEncoding()
            emb_bag_in += 3*128
        if self.scene_seg:
            self.scene_seg_emb = torch.nn.Embedding(3, 128, padding_idx=0)
            emb_bag_in += 128
        if self.obj_embs_bert:
            emb_bag_in += 1024
        if self.obj_embs_sbert:
            emb_bag_in += 768
        if self.kb_id_bert:
            self.kb_id_enc_bert = torch.nn.Embedding(500, 1024, padding_idx=0)
            emb_bag_in += 1024
        if self.kb_id_sbert:
            self.kb_id_enc_sbert = torch.nn.Embedding(500, 768, padding_idx=0)
            emb_bag_in += 768
        if self.obj_men:
            self.obj_men_emb = torch.nn.Embedding(2, 128, padding_idx=0)
            emb_bag_in += 128
        self.emb_bag = torch.nn.Linear(emb_bag_in, 2048)

        if self.graph_attn:
            self.rel_emb = torch.nn.Embedding(4, 2048)

        # Convert output
        self.clsHead = torch.nn.Linear(768, 1)
            
    def forward(self, input_ids_uncased, input_mask_uncased, segment_ids_uncased, bboxes_lxmert, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, vis_mask, rel_mask_left=None, rel_mask_right=None, rel_mask_up=None, rel_mask_down=None, obj_men=None):
        # combine object features
        
        def cat_with_obj_feats(obj_feats, new_feat):
            if obj_feats.shape == torch.Size([0,0]): 
                obj_feats = new_feat
            else:
                obj_feats = torch.cat((obj_feats, new_feat), axis=2)
            return obj_feats

        obj_feats = torch.zeros(0,0)

        if self.vis_feats_rcnn and not (self.obj_id or self.vis_feats_clip or self.pos or self.scene_seg or self.obj_embs_bert or self.obj_embs_sbert or self.kb_id_bert or self.kb_id_sbert or self.attn_bias or self.graph_attn):
            obj_feats = vis_feats_rcnn
        else:
            if self.obj_id:
                obj_id_embs = self.obj_idx_emb(obj_ids)
                obj_feats = cat_with_obj_feats(obj_feats, obj_id_embs)
            if self.vis_feats_clip:
                obj_feats = cat_with_obj_feats(obj_feats, vis_feats_clip)
            if self.vis_feats_rcnn:
                obj_feats = cat_with_obj_feats(obj_feats, vis_feats_rcnn)
            if self.pos:
                pos_x_emb = self.obj_pos_enc(pos_x)
                pos_y_emb = self.obj_pos_enc(pos_y)
                pos_z_emb = self.obj_pos_enc(pos_z)
                obj_feats = cat_with_obj_feats(obj_feats, pos_x_emb)
                obj_feats = cat_with_obj_feats(obj_feats, pos_y_emb)
                obj_feats = cat_with_obj_feats(obj_feats, pos_z_emb)
            if self.scene_seg:
                scene_seg_emb = self.scene_seg_emb(scene_seg)
                obj_feats = cat_with_obj_feats(obj_feats, scene_seg_emb)
            if self.obj_embs_bert:
                obj_feats = cat_with_obj_feats(obj_feats, kb_embs_bert)
            if self.obj_embs_sbert:
                obj_feats = cat_with_obj_feats(obj_feats, kb_embs_sbert)
            if self.kb_id_bert:
                kb_emb_bert = self.kb_id_enc_bert(kb_id)
                obj_feats = cat_with_obj_feats(obj_feats, kb_emb_bert)
            if self.kb_id_sbert:
                kb_emb_sbert = self.kb_id_enc_sbert(kb_id)
                obj_feats = cat_with_obj_feats(obj_feats, kb_emb_sbert)
            if self.obj_men:
                obj_mem_emb = self.obj_men_emb(obj_men)
                obj_feats = cat_with_obj_feats(obj_feats, obj_mem_emb)

            obj_feats = self.emb_bag(obj_feats.float())

        # Add relationship tokens for structure-aware self-attention
        # at the last 4 positions (alsways vacant)
        if self.graph_attn:
            obj_feats_obj, obj_feats_rel = obj_feats[:,:-4,:], torch.zeros_like(obj_feats[:,-4:,0])
            obj_feats_rel[:,0] = 0 
            obj_feats_rel[:,1] = 1 
            obj_feats_rel[:,2] = 2 
            obj_feats_rel[:,3] = 3
            obj_feats_rel = self.rel_emb(obj_feats_rel.long())
            obj_feats = torch.cat([obj_feats_obj, obj_feats_rel], axis=1)
            vis_mask[:,:,-4:] = 1

        seq_feats = self.lxmert.encoder.model(input_ids_uncased, token_type_ids=segment_ids_uncased, attention_mask=input_mask_uncased, visual_feats=(obj_feats, bboxes_lxmert), visual_attention_mask=vis_mask)
        seq_feats = torch.cat(seq_feats, axis=1)

        out = self.clsHead(seq_feats)
        return out

if __name__ == '__main__':
    pass