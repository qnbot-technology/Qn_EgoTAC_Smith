"""Fixed-prompt SAM3 wrappers. Imported only during conversion, never runtime."""
import torch


class EmptyGeometryEncoder(torch.nn.Module):
    """Specialize the original encoder for exactly zero point/box/mask prompts.

    Empty ROIAlign/grid_sample branches have no values to contribute. Keep the
    learned CLS token and every image-attention layer; do not substitute padded
    boxes or a cached image-independent prompt. Export must compare this branch
    with the native encoder before accepting it.
    """
    def __init__(self, original):
        super().__init__()
        if original.cls_embed is None:
            raise ValueError('Empty geometry export requires original CLS token')
        self.cls_embed=original.cls_embed
        self.d_model=original.d_model
        self.final_proj=original.final_proj
        if self.final_proj is not None:self.norm=original.norm
        self.encode=original.encode
        if self.encode is not None:self.encode_norm=original.encode_norm

    def forward(self, geo_prompt, img_feats, img_sizes, img_pos_embeds=None):
        if geo_prompt.point_embeddings.shape[0] or geo_prompt.box_embeddings.shape[0] or geo_prompt.mask_embeddings is not None:
            raise ValueError('Only the original empty geometric prompt is supported')
        memory=img_feats[-1]
        batch=memory.shape[1]
        embeds=self.cls_embed.weight.view(1,1,self.d_model).repeat(1,batch,1)
        mask=torch.zeros(batch,1,dtype=torch.bool,device=memory.device)
        if self.final_proj is not None:embeds=self.norm(self.final_proj(embeds))
        if self.encode is not None:
            pos=img_pos_embeds[-1] if img_pos_embeds is not None else torch.zeros_like(memory)
            for layer in self.encode:
                embeds=layer(tgt=embeds,memory=memory,tgt_key_padding_mask=mask,pos=pos)
            embeds=self.encode_norm(embeds)
        return embeds,mask


class ImageEncoder(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.neck = backbone.vision_backbone
        self.scalp = backbone.scalp

    def forward(self, image):
        fpn, pos, _, _ = self.neck(image)
        if self.scalp:
            fpn, pos = fpn[:-self.scalp], pos[:-self.scalp]
        return tuple(fpn) + tuple(pos)


class TextOnlyGrounding(torch.nn.Module):
    def __init__(self, model, text):
        super().__init__()
        # The image and text encoders are not traced into this second graph.
        model.backbone = None
        self.model = model
        self.register_buffer('language_features', text['language_features'].detach())
        self.register_buffer('language_mask', text['language_mask'].detach())
        from sam3.model.data_misc import FindStage
        self.find = FindStage(img_ids=torch.tensor([0]), text_ids=torch.tensor([0]),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None)

    def forward(self, fpn0, fpn1, fpn2, pos0, pos1, pos2):
        backbone = dict(backbone_fpn=[fpn0, fpn1, fpn2],
            vision_pos_enc=[pos0, pos1, pos2], vision_features=fpn2,
            language_features=self.language_features, language_mask=self.language_mask)
        out = self.model.forward_grounding(backbone_out=backbone, find_input=self.find,
            find_target=None, geometric_prompt=self.model._get_dummy_prompt())
        return tuple(out[k] for k in ('pred_logits','pred_boxes','pred_masks','presence_logit_dec'))
