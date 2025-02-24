# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    build_box_head,
    build_mask_head,
    select_foreground_proposals,
    ROI_HEADS_REGISTRY,
    ROIHeads,
    Res5ROIHeads,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.poolers import ROIPooler


class AttributePredictor(nn.Module):
    """
    Head for attribute prediction, including feature/score computation and
    loss computation.

    """
    def __init__(self, cfg, input_dim):
        super().__init__()

        # fmt: off
        self.num_objs          = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.obj_embed_dim     = cfg.MODEL.ROI_ATTRIBUTE_HEAD.OBJ_EMBED_DIM
        self.fc_dim            = cfg.MODEL.ROI_ATTRIBUTE_HEAD.FC_DIM
        self.num_attributes    = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_CLASSES
        self.max_attr_per_ins  = cfg.INPUT.MAX_ATTR_PER_INS
        self.loss_weight       = cfg.MODEL.ROI_ATTRIBUTE_HEAD.LOSS_WEIGHT
        # fmt: on

        # object class embedding, including the background class
        self.obj_embed = nn.Embedding(self.num_objs + 1, self.obj_embed_dim)
        input_dim += self.obj_embed_dim
        self.fc = nn.Sequential(
                nn.Linear(input_dim, self.fc_dim),
                nn.ReLU()
            )
        self.attr_score = nn.Linear(self.fc_dim, self.num_attributes)
        nn.init.normal_(self.attr_score.weight, std=0.01)
        nn.init.constant_(self.attr_score.bias, 0)

    def forward(self, x, obj_labels):
        attr_feat = torch.cat((x, self.obj_embed(obj_labels)), dim=1)
        return self.attr_score(self.fc(attr_feat))

    def loss(self, score, label):
        n = score.shape[0]
        score = score.unsqueeze(1)
        score = score.expand(n, self.max_attr_per_ins, self.num_attributes).contiguous()
        score = score.view(-1, self.num_attributes)
        inv_weights = (
            (label >= 0).sum(dim=1).repeat(self.max_attr_per_ins, 1).transpose(0, 1).flatten()
        )
        weights = inv_weights.float().reciprocal()
        weights[weights > 1] = 0.
        n_valid = len((label >= 0).sum(dim=1).nonzero())
        label = label.view(-1)
        attr_loss = F.cross_entropy(score, label, reduction="none", ignore_index=-1)
        attr_loss = (attr_loss * weights).view(n, -1).sum(dim=1)

        if n_valid > 0:
            attr_loss = attr_loss.sum() * self.loss_weight / n_valid
        else:
            attr_loss = attr_loss.sum() * 0.
        return {"loss_attr": attr_loss}


class AttributeROIHeads(ROIHeads):
    """
    An extension of ROIHeads to include attribute prediction.
    """
    def forward_attribute_loss(self, proposals, box_features):
        proposals, fg_selection_attributes = select_foreground_proposals(
            proposals, self.num_classes
        )
        attribute_features = box_features[torch.cat(fg_selection_attributes, dim=0)]
        obj_labels = torch.cat([p.gt_classes for p in proposals])
        attribute_labels = torch.cat([p.gt_attributes for p in proposals], dim=0)
        attribute_scores = self.attribute_predictor(attribute_features, obj_labels)
        return self.attribute_predictor.loss(attribute_scores, attribute_labels)


@ROI_HEADS_REGISTRY.register()
class AttributeRes5ROIHeads(AttributeROIHeads, Res5ROIHeads):
    """
    An extension of Res5ROIHeads to include attribute prediction.
    """
    def __init__(self, cfg, input_shape):
        super(Res5ROIHeads, self).__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        self.attribute_on = cfg.MODEL.ATTRIBUTE_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

        if self.attribute_on:
            self.attribute_predictor = AttributePredictor(cfg, out_channels)

    def forward(self, images, features, proposals, targets=None):
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])
        predictions = self.box_predictor(feature_pooled)

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            if self.attribute_on:
                losses.update(self.forward_attribute_loss(proposals, feature_pooled))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def get_conv5_features(self, features):
        features = [features[f] for f in self.in_features]
        return self.res5(features[0])


@ROI_HEADS_REGISTRY.register()
class AttributeStandardROIHeads(AttributeROIHeads, StandardROIHeads):
    """
    An extension of StandardROIHeads to include attribute prediction.
    """
    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        attribute_on: bool = False,
        attribute_predictor: Optional[nn.Module] = None,
        **kwargs
    ):
        super(StandardROIHeads, self).__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

        self.attribute_on = attribute_on
        if self.attribute_on:
            # self.attribute_predictor = AttributePredictor(cfg, self.box_head.output_shape.channels)
            self.attribute_predictor = attribute_predictor

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["attribute_on"] = cfg.MODEL.ATTRIBUTE_ON
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = StandardROIHeads._init_box_head(cfg, input_shape)
        if cfg.MODEL.ATTRIBUTE_ON:
            ret.update({"attribute_predictor": AttributePredictor(cfg, ret["box_head"].output_shape.channels)})
        return ret

    def _forward_box(self, features, proposals):
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            losses = self.box_predictor.losses(predictions, proposals)
            if self.attribute_on:
                losses.update(self.forward_attribute_loss(proposals, box_features))
                del box_features

            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def get_conv5_features(self, features):
        assert len(self.in_features) == 1

        features = [features[f] for f in self.in_features]
        return features[0]

