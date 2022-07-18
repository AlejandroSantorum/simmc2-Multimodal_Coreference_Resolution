import torch
from model.Transformers_VQA_master.vqa_model import VQAModel
from model.model_utils import ObjPositionalEncoding

class Modified_Uniter(torch.nn.Module):
    def __init__(self, obj_id, vis_feats_clip, vis_feats_rcnn, pos, scene_seg, obj_embs_bert, obj_embs_sbert, kb_id_bert, kb_id_sbert, attn_bias=False, graph_attn=False, obj_men=False, pred_men=False, n_target_objs_head=False):
        super(Modified_Uniter, self).__init__()

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
        self.pred_men = pred_men
        if not (obj_id or vis_feats_clip or vis_feats_rcnn or pos or scene_seg or obj_embs_bert or obj_embs_sbert or kb_id_bert or kb_id_sbert):
            print('Must have some feature for each object')
            raise
    
        self.attn_bias = attn_bias
        self.graph_attn = graph_attn

        self.n_target_objs_head = n_target_objs_head

        # Load pretrained UNITER
        self.uniter = VQAModel(num_answers=69, model='uniter', attn_bias=attn_bias, graph_attn=graph_attn)
        self.uniter.encoder.load('./model/Transformers_VQA_master/models/pretrained/uniter-base.pt')

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

        # Auxilliary task to predict whether an object is previously mentioned
        if self.pred_men:
            self.menHead = torch.nn.Linear(768, 1)
        
        # Auxiliary task to predict the number of targets
        if self.n_target_objs_head:
            self.numTargetObjsHead = torch.nn.Linear(768, 4) # 4 for multi-class classification: 0 targets, 1 target, 2 targets, 3+ targets
            
    def forward(self, input_ids, txt_seg_ids, vis_seg, bboxes, extended_attention_mask, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, rel_mask_left=None, rel_mask_right=None, rel_mask_up=None, rel_mask_down=None, obj_men=None):
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
            extended_attention_mask[:,:,:,-4:] = 0

        word_embeddings = self.uniter.encoder.model.uniter.embeddings(input_ids, txt_seg_ids)
        img_type_embeddings = self.uniter.encoder.model.uniter.embeddings.token_type_embeddings(vis_seg)
        img_embeddings = self.uniter.encoder.model.uniter.img_embeddings(obj_feats, bboxes, img_type_embeddings)
        embeddings = torch.cat([word_embeddings,img_embeddings],dim=1)

        lang_v_feats = self.uniter.encoder.model.uniter.encoder(hidden_states=embeddings, attention_mask=extended_attention_mask, rel_mask_left=rel_mask_left, rel_mask_right=rel_mask_right, rel_mask_up=rel_mask_up, rel_mask_down=rel_mask_down)
        # shape of lang_v_feats: (batch, 512, 768)
        out = self.clsHead(lang_v_feats)
        # shape of out: (batch, 512, 1)

        if self.pred_men:
            men_out = self.menHead(lang_v_feats)
            return out, men_out
        
        if self.n_target_objs_head:
            cls_token_embeddings = lang_v_feats[:,0,:] # (batch, 1, 768) = (batch, 768)
            head_output = self.numTargetObjsHead(cls_token_embeddings) # (batch, 4)
            return out, head_output

        return out

if __name__ == '__main__':
    pass