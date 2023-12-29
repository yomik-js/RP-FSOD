#这里的代码是 基类新类都按照距离进库，然后加入一个原型损失让类间距离增大，然后又生成了一个新特征


import torch
import logging
import numpy as np
from torch import nn
from typing import Dict
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils.events import get_event_storage
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .box_head import build_box_head
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, FastRCNNOutputLayers, FastRCNNOutputs
from .gcn import GCN
import pickle
import os
import torchvision.transforms as transforms
# from detectron2.modeling.roi_heads import SigmoidFocalLoss

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix
            )
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith(
                        "gt_"
                    ) and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets]
                        )
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4)
                    )
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item()
            )
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()

import fvcore.nn.weight_init as weight_init
@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        # print('pooler:', x.size())
        x = self.res5(x)
        # print('res5:', x.size())
        return x

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled
        )
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )

        self.cls_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )

        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

        self.cls_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        proposal_list = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(
            features, proposal_list
        )

        cls_features = self.cls_head(box_features)
        pred_class_logits, _ = self.cls_predictor(
            cls_features
        )

        box_features = self.box_head(box_features)
        _, pred_proposal_deltas = self.box_predictor(
            box_features
        )
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances



import copy
import torch.distributed as dist
from torch import distributions
import torch.nn.functional as F
from detectron2.layers import cat
import fvcore.nn.weight_init as weight_init

@ROI_HEADS_REGISTRY.register()
class CommonalityROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        # assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # self.box_head = build_box_head(
        #     cfg,
        #     ShapeSpec(
        #         channels=in_channels,
        #         height=pooler_resolution,
        #         width=pooler_resolution,
        #     ),
        # )
        
        # out_channels = 1024
        self.res5, out_channels = self._build_res5_block(cfg)
        self.fc_s = nn.Linear(out_channels, out_channels)
        self.fc_l = nn.Linear(out_channels, out_channels)
        
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

        self.fc_1 = nn.Linear(out_channels, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_3 = nn.Linear(1024, out_channels)
        
        # for layer in [self.fc_s, self.fc_l,self.fc_1,self.fc_2,self.fc_3]:
        #     weight_init.c2_xavier_fill(layer)
        for layer in [self.fc_s, self.fc_l]:
            weight_init.c2_xavier_fill(layer)
        
     
        self.cfg = cfg
        self.num = cfg.MODEL.ROI_HEADS.num
        self.memory = cfg.MODEL.ROI_HEADS.MEMORY
        self.augmentation = cfg.MODEL.ROI_HEADS.AUGMENTATION
        self.save = cfg.MODEL.ROI_HEADS.SAVE
        self.output_dir = cfg.OUTPUT_DIR
        # self.support_dict_base = dict()
        # self.base_mean1 = torch.zeros(20,2048)``
        # self.base_std1 = torch.zeros(20,2048)
        self.shot = cfg.MODEL.ROI_HEADS.Shot
        self.alpha = cfg.MODEL.ROI_HEADS.alpha
        self.beta = cfg.MODEL.ROI_HEADS.beta

        # create the queue
        self.queue_len = cfg.MODEL.ROI_HEADS.QUEUE_LEN
        self.dists = torch.zeros(self.num_classes,self.queue_len)
        if self.memory:
            self.register_buffer("queue_s", torch.zeros(self.num_classes, self.queue_len, out_channels))
            # self.register_buffer("queue_l", torch.zeros(self.num_classes, self.queue_len, out_channels))
            self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
            self.register_buffer("queue_full", torch.zeros(self.num_classes, dtype=torch.long))
        self.base_index = []
        self.novel_mean=torch.zeros(5,2048)
        self.novel_std=torch.zeros(5,2048)
        if self.num_classes ==15:
            for i in range(self.num_classes):
                self.base_index.append(i)
        if self.num_classes ==20 or self.num_classes ==5:
            # self.novel_index = [4,7,13,14,17]   #split3
            # self.novel_index = [0,6,8,9,15]   #split2
            self.novel_index = [1,2,3,11,16]   #split1
            self.base_pro = []
            for i in range(20):
                if i not in self.novel_index:
                    self.base_index.append(i)
        elif self.num_classes in [60, 80]:   #把新类的挑出来
            self.novel_index = [0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62]

        if self.num_classes ==20 or self.num_classes==5:
            # self.load_prototype_voc()
            self.base_mean,self.base_pro = self.load_base_mean()
            self.base_std = self.load_base_std()
        # self.base_pro = []
        # if self.num_classes ==20 or self.num_classes ==5:
        #     for i in range(self.num_classes):
        #         if i not in self.novel_index:
        #             self.base_pro.append(self.support_dict_base[i])
        #     self.base_pro = torch.stack(self.base_pro, dim=0)
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_s, gt_class):

        keys_s = keys_s[:self.queue_len]
        batch_size = keys_s.shape[0]
        ptr = int(self.queue_ptr[gt_class])
        if ptr + batch_size <= self.queue_len:
            self.queue_s[gt_class, ptr:ptr + batch_size] = keys_s  #该维度是20*2048*2048，更新一下 3是第一个维度存储类别，然后存了两个该类别的2048进去
        else:
            self.queue_s[gt_class, ptr:] = keys_s[:self.queue_len - ptr]
            self.queue_s[gt_class, :(ptr + batch_size) % self.queue_len] = keys_s[self.queue_len - ptr:]
        
    
        if ptr + batch_size >= self.queue_len:
            self.queue_full[gt_class] = 1
        ptr = (ptr + batch_size) % self.queue_len    #ptr代表的是该类别存了多少个 记了个数
        self.queue_ptr[gt_class] = ptr


    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys_s, gt_class):
        keys_s = keys_s[:self.queue_len]  #这里得到了要存储的特征，先截断一下
        batch_size = keys_s.shape[0]  #这个类别一共传进来多少个
        num = batch_size
        # if gt_class in self.novel_index:   #只更新了新类
        ptr = int(self.queue_ptr[gt_class])
        if self.queue_full[gt_class] != 1:
            if ptr + batch_size <= self.queue_len:
                self.queue_s[gt_class, ptr:ptr + batch_size] = keys_s 
                ptr = (ptr + batch_size) % self.queue_len    #ptr代表的是该类别存了多少个 记了个数
                self.queue_ptr[gt_class] = ptr #该维度是20*2048*2048，更新一下 3是第一个维度存储类别，然后存了两个该类别的2048进去
            else:
                self.queue_s[gt_class, ptr:] = keys_s[:self.queue_len - ptr]
                self.queue_full[gt_class] = 1
                num = (ptr + batch_size) % self.queue_len
                self.queue_full[gt_class] = 1

        if self.queue_full[gt_class] == 1:
            mean, std = self.calculate_distribution()
            gt_mean = mean[gt_class]
            for i in range(self.queue_len):
                dist = torch.norm(gt_mean - self.queue_s[gt_class,i], p=2)
                self.dists[gt_class,i] = dist
            dist, index = self.dists[gt_class,:].sort(descending=True)
            for i in range(num):
                dist_batch = torch.norm(gt_mean - keys_s[i], p=2)
                if dist_batch <= self.dists[gt_class,index[0]]:
                    self.queue_s[gt_class, index[0]] = keys_s[i] #
                    gt_mean, std = self.calculate_distribution(gt_class)
                    mean[gt_class,:] = gt_mean[0,:]
                    for j in range(self.queue_len):
                        dist = torch.norm(gt_mean - self.queue_s[gt_class,j], p=2)
                        self.dists[gt_class,j] = dist
                    dist, index = self.dists[gt_class,:].sort(descending=True)

    @torch.no_grad()
    def update_memory(self, features_s, gt_classes,score,index):

        #单卡训练时就将这些注释掉
        # features_s = concat_all_gather(features_s)
        # gt_classes = concat_all_gather(gt_classes)
        # score = concat_all_gather(score)
        
        fg_cases = (gt_classes >= 0) & (gt_classes < self.num_classes)
        features_fg_s = features_s[fg_cases]
        gt_classes_fg = gt_classes[fg_cases]
        score_1 = score[fg_cases]
        
        if len(gt_classes_fg) == 0:
            return
        uniq_c = torch.unique(gt_classes_fg)
            #这一次传进来的4个类别标签
        for c in uniq_c:
            c = int(c)
            c_index = torch.nonzero(
                gt_classes_fg == c, as_tuple=False
            ).squeeze(1)
            score_re = torch.nonzero(score_1>0.5)   #这里说明置信度的筛选
            if len(score_re)==0:
                return 
            list1=[]
            for i in c_index:   #判断类别预测是否与类别标签一致
                if i in score_re:
                    list1.append(i)
            list2=torch.tensor(list1,device=c_index.device,dtype=torch.long)
            _,b=score_1[list2].sort(descending=True)
            list3 = list2[b]

            if len(list2)==0:
                return
            features_c_s = features_fg_s[list3]
            storage = get_event_storage()
            # aa=self.num
            if self.num_classes ==15:
                if int(storage.iter) >= 900:
                    self._dequeue_and_enqueue1(features_c_s, c)
                else:
                    self._dequeue_and_enqueue(features_c_s, c)
            else:
                if int(storage.iter) >= 1000:
                    self._dequeue_and_enqueue1(features_c_s, c)
                else:
                    self._dequeue_and_enqueue(features_c_s, c)
            # if int(storage.iter) >= aa:
            #     if self.num_classes ==15:
            #         self._dequeue_and_enqueue1(features_c_s, c)
            #     else:
            #         if c in self.novel_index:
            #             self._dequeue_and_enqueue1(features_c_s, c)
            # else:
            #     if self.num_classes ==15:
            #         self._dequeue_and_enqueue(features_c_s, c)
            #     else:
            #         if c in self.novel_index:
            #             self._dequeue_and_enqueue(features_c_s, c)
            
            # self._dequeue_and_enqueue(features_c_s, c)

    @torch.no_grad()
    def predict_prototype(self, feature_pooled_s, gt_classes):
        
        prototypes_s = []
        kth=1
        prototypes_l = []
        if self.num_classes ==20:
            for i in range(self.num_classes):
                #原型校准
                if i in self.novel_index:
                    if self.queue_full[i] or self.queue_ptr[i] == 0:
                        p_n = self.queue_s[i].mean(dim=0)
                    else:
                        p_n = self.queue_s[i][:self.queue_ptr[i]].mean(dim=0)
                    dists = torch.norm(p_n[None,:] - self.base_pro, p=2, dim=1)
                    _, index = dists.sort()
                    aa = 2
                    dists = torch.pow(dists,aa)
                    w=1.0/dists.add(1)
                    mean_1=torch.zeros(2048,device=p_n.device)
                    w_l=0
                    for j in range(kth):
                        mean_1 = mean_1+w[index[j]]*self.base_pro[index[j]]
                        w_l = w_l+w[index[j]]*self.base_pro[index[j]]
                    p_nc = (p_n+mean_1)/(1+w_l)
                    prototypes_s.append(p_nc)
                else:
                    prototypes_s.append(self.base_mean[i].cuda())
        else:
            for i in range(self.num_classes):
                if self.queue_full[i] or self.queue_ptr[i] == 0:
                    prototypes_s.append(self.queue_s[i].mean(dim=0))
                else:
                    prototypes_s.append(self.queue_s[i][:self.queue_ptr[i]].mean(dim=0))
                    
                    
                    
        #不使用原型校准     
        # for i in range(self.num_classes):
        #     if self.queue_full[i] or self.queue_ptr[i] == 0:
        #         prototypes_s.append(self.queue_s[i].mean(dim=0))
        #         # prototypes_l.append(self.queue_l[i].mean(dim=0))
        #     else:
        #         prototypes_s.append(self.queue_s[i][:self.queue_ptr[i]].mean(dim=0))
        prototypes_s = torch.stack(prototypes_s, dim=0)
        return prototypes_s
    @torch.no_grad()
    def calculate_distribution(self,gt_class=None):
        mean, std = [], []
        if gt_class:
            c_mean = self.queue_s[gt_class].mean(dim=0)
            c_std = self.queue_s[gt_class].var(dim=0, unbiased=False)
            c_std = c_std * self.queue_len / (self.queue_len - 1)
            mean.append(c_mean) 
            std.append(c_std)
        else:
            if self.num_classes == 15:
                for c in self.base_index:
                    if self.queue_full[c]:
                        c_mean = self.queue_s[c].mean(dim=0)
                        c_std = self.queue_s[c].var(dim=0, unbiased=False)
                        c_std = c_std * self.queue_len / (self.queue_len - 1)
                    else:
                        c_mean = self.queue_s[c][:self.queue_ptr[c]].mean(dim=0)
                        c_std = self.queue_s[c][:self.queue_ptr[c]].var(dim=0, unbiased=False)
                        if self.queue_ptr[c] > 1:
                            c_std = c_std * self.queue_ptr[c] / (self.queue_ptr[c] - 1)
                    mean.append(c_mean)
                    std.append(c_std)
            else:
                for c in range(self.num_classes):
                    if c in self.novel_index:
                        if self.queue_full[c]:
                            c_mean = self.queue_s[c].mean(dim=0)
                            c_std = self.queue_s[c].var(dim=0, unbiased=False)
                            c_std = c_std * self.queue_len / (self.queue_len - 1)
                        else:
                            c_mean = self.queue_s[c][:self.queue_ptr[c]].mean(dim=0)
                            c_std = self.queue_s[c][:self.queue_ptr[c]].var(dim=0, unbiased=False)
                            if self.queue_ptr[c] > 1:
                                c_std = c_std * self.queue_ptr[c] / (self.queue_ptr[c] - 1)
                    else:
                        c_mean = self.base_mean[c].cuda()
                        c_std = self.base_std[c].cuda()
                    mean.append(c_mean)
                    std.append(c_std)
        mean = torch.stack(mean, dim=0)
        std = torch.stack(std, dim=0)
        return mean,std
    @torch.no_grad()
    def generate_features(self, gt_classes, base_mean, base_std):
        new_features = []
        new_classes = []
        uniq_c = torch.unique(gt_classes)
        kth = 2
        # num_samples = self.shot
        num_samples = 5
        for c in self.novel_index:
            if np.random.rand() < 0.7 or (c not in uniq_c):
                continue
            if c not in [2,11,14]:
                continue
            if self.queue_full[c]:
                c_mean = self.queue_s[c].mean(dim=0)
                c_std = self.queue_s[c].var(dim=0, unbiased=False)
                c_std = c_std * self.queue_len / (self.queue_len - 1)
            else:
                c_mean = self.queue_s[c][:self.queue_ptr[c]].mean(dim=0)
                c_std = self.queue_s[c][:self.queue_ptr[c]].var(dim=0, unbiased=False)
                if self.queue_ptr[c] > 1:
                    c_std = c_std * self.queue_ptr[c] / (self.queue_ptr[c] - 1)
            
            self.base_mean = self.base_mean.to(c_mean.device)
            self.base_std = self.base_std.to(c_mean.device)
            dists = torch.norm(c_mean[None,:] - self.base_mean, p=2, dim=1)
            for j in range(self.num_classes):
                if j in self.novel_index:
                    dists[j] = dists[j]+1000
            _, index = dists.sort()
            #mean = torch.cat([self.features_mean[c].unsqueeze(0), self.features_mean[index[:kth]]])
            w_l =0
            w=1.0/dists.add(1)
            #mean = torch.cat([self.features_mean[c].unsqueeze(0), self.features_mean[index[:kth]]])
            mean_1=torch.zeros(2048,device=dists.device)
            for i in range(kth):
                mean_1 = mean_1+w[index[i]]*self.base_mean[index[i]]
                w_l = w_l+w[index[i]]
            calibrated_mean = (c_mean+mean_1)/(1+w_l)
            # calibrated_mean = (c_mean+mean_1)/(1+(dists[index[:kth]]).sum(dim=0))
            # calibrated_std = self.base_std[index[:kth]].mean(dim=0)+0.1
            
            univariate_normal_dists = distributions.normal.Normal(
                calibrated_mean, scale=torch.sqrt(c_std))
            
            feaures_rsample = univariate_normal_dists.rsample(
                (num_samples,))
            classes_rsample = gt_classes.new_full((num_samples, ), c)
            
            new_features.append(feaures_rsample)
            new_classes.append(classes_rsample)
        if len(new_features) == 0:
            return [], []
        else:
            return torch.cat(new_features), torch.cat(new_classes)

    def _build_res5_block(self, cfg):    #推理时可以去掉该部分，加快速度！
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            #first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        # print('pooler:', x.size())
        x = self.res5(x) 
        # print('res5:', x.size())
        return x

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        features_list =  [features[f] for f in self.in_features]
        box_features = self._shared_roi_transform(
            features_list, proposal_boxes
        )
        # box_features = self.pooler(
        #     features_list, proposal_boxes
        # )
        
        
        # box_features = self._shared_roi_transform(
        #     [features[f] for f in self.in_features], proposal_boxes
        # )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        
        feature_pooled_s = F.relu(self.fc_s(feature_pooled))
        feature_pooled_l = F.relu(self.fc_l(feature_pooled))
        
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled_s, feature_pooled_l
        )

        score= pred_class_logits.softmax(dim=1)
        score, index = score.max(dim=1)
        
        feature_new = feature_pooled_s
        feature_new = F.relu(self.fc_3(F.relu(self.fc_2(F.relu(self.fc_1(feature_new))))))
        pred_class_logits_new, _ = self.box_predictor(
            feature_new, feature_new
                        )
        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            pred_class_logits_new
        )

        if self.training:          
            if self.memory:
                with torch.no_grad():
                    gt_classes = outputs.gt_classes

                    #单卡训练
                    pad_size = self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE \
                        * self.cfg.SOLVER.IMS_PER_BATCH

#                     #单卡训练
#                     pad_size = self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE \
#                         * self.cfg.SOLVER.IMS_PER_BATCH

#                     #多卡训练
#                     # pad_size = self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE \
#                     #     * self.cfg.SOLVER.IMS_PER_BATCH // torch.distributed.get_world_size()
#                     # pad_size *= 2

                    feature_pooled_pad_s = feature_pooled_s.new_full((
                        pad_size, feature_pooled_s.size(1)), -1)
                    feature_pooled_pad_s[: feature_pooled_s.size(0)] = feature_pooled_s
                    gt_classes_pad = gt_classes.new_full((pad_size,), -1)
                    gt_classes_pad[: gt_classes.size(0)] = gt_classes
                    score_pad = score.new_full((pad_size,), -1)
                    score_pad[: gt_classes.size(0)] = score
                    
                    self.update_memory(feature_pooled_pad_s.detach(), gt_classes_pad,score_pad,index)

            losses = outputs.losses() 
            del features

            storage = get_event_storage()
            if int(storage.iter) >= 100:

                if 1>0:
                    gt_classes = outputs.gt_classes
                    bg_class_ind = pred_class_logits.shape[1] - 1
                    true_cases = (gt_classes >= 0) & (gt_classes < bg_class_ind)
                    #这里在进行原型校准
                    self.prototypes_s= self.predict_prototype(feature_pooled_s, gt_classes)
                    dot_product_mat = torch.mm(self.prototypes_s, torch.transpose(self.prototypes_s, 0, 1))
                    len_vec = torch.unsqueeze(torch.sqrt(torch.sum(self.prototypes_s * self.prototypes_s, dim=1)), dim=0)
                    len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
                    cos_sim_mat = dot_product_mat / len_mat
                    cos_diag = torch.diag(cos_sim_mat)
                    a_diag = torch.diag_embed(cos_diag)
                    cos_sim_mat = cos_sim_mat-a_diag
                    cos_sim_mat = torch.sum(torch.sum(cos_sim_mat))
                    num = self.num_classes -1
                    loss_prototype = cos_sim_mat/(num*num)
                    losses.update({"loss_prototype": loss_prototype*self.alpha})   ##代表论文中的alpha


                    feature_new = feature_new[true_cases]
                    gt_classes_new = gt_classes[true_cases]
                    loss_cls_score_new = F.cross_entropy(
                                pred_class_logits_new, gt_classes,reduction="mean"
                        )
                    losses.update({"loss_cls_score_new": loss_cls_score_new})

                    gt_class_new = torch.zeros(feature_new.size()[0],self.prototypes_s.size()[0],device=feature_pooled_s.device)
                    for i in range(len(gt_classes_new)):
                        idx = gt_classes_new[i]
                        gt_class_new[i,idx]=1
                    weight = torch.mm(gt_class_new,self.prototypes_s)  #获得这个类的类别原型
                    differ = feature_new-weight
                    differ = torch.sum(torch.norm(differ,p=1,dim=1)/(differ.size()[0]*differ.size()[1]))
                    loss_new_feature = differ
                    losses.update({"loss_new_feature": loss_new_feature*self.beta})    ##代表论文中的beta







                if self.save:

                    base_mean,base_std = self.calculate_distribution()
                    output_file_mean = os.path.join(self.output_dir, 'base_mean.pickle')
                    with open(output_file_mean, 'wb') as handle:
                        pickle.dump(base_mean, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    output_file_std = os.path.join(self.output_dir, 'base_std.pickle')
                    with open(output_file_std, 'wb') as handle:
                        pickle.dump(base_std,handle, protocol=pickle.HIGHEST_PROTOCOL)

                if self.augmentation:
                    new_features,new_classes = self.generate_features(gt_classes,self.base_mean,self.base_std)
                    if len(new_features) == 0:
                        loss_cls_score_aug = feature_pooled_pad_s.new_full((1,), 0).mean()
                    else:
                        pred_class_logits_aug, _ = self.box_predictor(
                            new_features, new_features
                        )
                        loss_cls_score_aug = F.cross_entropy(
                                pred_class_logits_aug, new_classes, reduction="mean"
                        )

                    losses.update({"loss_cls_score_aug": loss_cls_score_aug * 0.1})

            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}
    # def load_prototype_voc(self):

    #     output_file = os.path.join(self.output_dir, 'base_class_prototype.pickle')
    #     if self.num_classes == 20:
    #         ###如果使用vscode的话需要对应修改一下
    #         output_dir = self.output_dir
    #         path_parts = output_dir.split('/')
    #         new_path_parts = [part if part != "defrcn_gfsod_r101_novel1" else "defrcn_det_r101_base1" for part in path_parts]
    #         new_path_parts = new_path_parts[:-2]
    #         new_path = os.path.join(*new_path_parts)
    #         output_file = os.path.join(new_path, 'base_class_prototype.pickle')
    #     if os.path.exists(output_file):
    #         with open(output_file, 'rb') as handle:
    #             prototype_base = pickle.load(handle)
    #         print("loading base class prototype from ", output_file)
    #         flag=0
    #         for cls_key, feature in prototype_base.items():
    #                 while cls_key+flag in self.novel_index:
    #                     flag = flag+1
    #                 self.support_dict_base[cls_key+flag] = feature.cuda()
    #         return


    def load_base_mean(self):
        output_file = os.path.join(self.output_dir, 'base_mean.pickle')
        if self.num_classes == 20:
            # output_dir = os.path.replace(self.output_dir, 'base_mean.pickle')
            print("num_classes=20")
            output_dir = self.output_dir
            path_parts = output_dir.split('/')
            new_path_parts = [part if part != "defrcn_gfsod_r101_novel1" else "defrcn_det_r101_base1" for part in path_parts]
            new_path_parts = new_path_parts[:-2]
            new_path = os.path.join(*new_path_parts)
            print(output_dir)
            output_file = os.path.join(new_path, 'base_mean.pickle')
            print(output_file)
        if os.path.exists(output_file):
            with open(output_file, 'rb') as handle:
                base_mean = pickle.load(handle)
            # base_mean1 = base_mean
            flag=0
            base_mean1 = torch.zeros(20,2048)
            for i in range(len(base_mean)):
                while i+flag in self.novel_index:
                    flag = flag+1
                base_mean1[i+flag] = base_mean[i].cuda()
        return base_mean1 ,base_mean
    def load_base_std(self):
        output_file = os.path.join(self.output_dir, 'base_std.pickle')
        if self.num_classes == 20:
            # output_dir = os.path.replace(self.output_dir, 'base_mean.pickle')
            output_dir = self.output_dir
            path_parts = output_dir.split('/')
            new_path_parts = [part if part != "defrcn_gfsod_r101_novel1" else "defrcn_det_r101_base1" for part in path_parts]
            new_path_parts = new_path_parts[:-2]
            new_path = os.path.join(*new_path_parts)
            output_file = os.path.join(new_path, 'base_std.pickle')
        if os.path.exists(output_file):
            with open(output_file, 'rb') as handle:
                base_std = pickle.load(handle)
            flag=0
            base_std1 = torch.zeros(20,2048)
            for i in range(len(base_std)):
                while i+flag in self.novel_index:
                    flag = flag+1
                base_std1[i+flag] = base_std[i].cuda()
        
        return base_std1
@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output




def sigmoid_focal_loss(
    inputs,
    targets,
    alpha: float = -1,
    gamma: float = 5,
    reduction: str = "none",
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
