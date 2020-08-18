# -*- encoding: utf-8 -*-

# @File    : fcos.py
# @Time    : 2020-07-22
# @Author  : zjh

r"""
"""

import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss, giou_loss
from torch import nn
from torch.nn import functional as F

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from ..anchor_generator import build_anchor_generator
from ..backbone import build_backbone
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..postprocessing import detector_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["FCOS"]


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


@META_ARCH_REGISTRY.register()
class FCOS(nn.Module):
    """
    Implement FCOS in :paper:`FCOS`.
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = cfg.MODEL.FCOS.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # Vis parameters
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT
        # fmt: on

        self.backbone = build_backbone(cfg)

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = FCOSHead(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)
        self.feature_shapes = feature_shapes

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.FCOS.IOU_THRESHOLDS,
            cfg.MODEL.FCOS.IOU_LABELS,
            allow_low_quality_matches=True,
        )

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]

        points = self.anchor_generator(features)
        pred_logits, pred_anchor_deltas, pred_centerness = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]
        pred_centerness = [permute_to_N_HWA_K(x, 1) for x in pred_centerness]

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes, gt_centerness = self.label_points(points, gt_instances)
            losses = self.losses(points, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes,
                                 pred_centerness, gt_centerness)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        points, pred_logits, pred_anchor_deltas, pred_centerness, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(points, pred_logits, pred_anchor_deltas, pred_centerness, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, points, pred_logits, gt_labels, pred_deltas, gt_boxes,
               pred_centerness, gt_centerness):
        """
        Args:
            points (list[Tensor]): a list of #feature level Points
            gt_labels, gt_boxes: see output of :meth:`FCOS.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)
        gt_boxes = torch.stack(gt_boxes)  # (N, R, 4)
        gt_centerness = torch.stack(gt_centerness)[:, :, None]  # (N, R, 1)

        pred_logits = cat(pred_logits, dim=1)  # (N, R, C)
        pred_deltas = cat(pred_deltas, dim=1)  # (N, R, 4)
        pred_centerness = cat(pred_centerness, dim=1)  # (N, R, 1)

        strides = [x.stride for x in self.feature_shapes]
        point_strides = [p.new_full([p.shape[0], 1], s) for p, s in zip(points, strides)]
        point_strides = cat(point_strides, dim=0)  # (R, 1)
        points = cat(points, dim=0)  # (R, 2)
        pred_boxes = self.transfer_boxes(points, pred_deltas, point_strides)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + \
                               (1 - self.loss_normalizer_momentum) * max(num_pos_anchors, 1)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
                           :, :-1
                           ]  # no loss for the last (background) class
        loss_cls = sigmoid_focal_loss_jit(
            pred_logits[valid_mask],
            gt_labels_target.to(pred_logits.dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )
        pred_boxes_pos = pred_boxes[pos_mask]
        gt_boxes_pos = gt_boxes[pos_mask]
        if not torch.isfinite(pred_boxes_pos).all() or not torch.isfinite(gt_boxes_pos).all():
            print(pred_boxes_pos)
            print(gt_boxes_pos)
        loss_box_reg = giou_loss(
            pred_boxes_pos,
            gt_boxes_pos,
            reduction="sum"
        )
        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_centerness[pos_mask],
            gt_centerness[pos_mask],
            reduction="sum")
        return {
            "loss_cls": loss_cls / self.loss_normalizer,
            "loss_box_reg": loss_box_reg / self.loss_normalizer,
            "loss_centerness": loss_centerness / self.loss_normalizer,
        }

    def transfer_boxes(self, points, deltas, strides=1):
        deltas = torch.exp(deltas) * strides
        boxes_tl = points - deltas[..., :2]
        boxes_br = points + deltas[..., 2:]
        return torch.cat([boxes_tl, boxes_br], dim=-1)

    @torch.no_grad()
    def label_points(self, points, gt_instances):
        """
        Args:
            points (list[Tensor]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps (sum(Hi * Wi * A)).
                Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.
            list[Tensor]:
                i-th element is a Rx4 tensor, where R is the total number of anchors across
                feature maps. The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as foreground.
        """
        szr = [0, 64, 128, 256, 512, 1 << 30]
        size_range = [torch.ones_like(p) * p.new_tensor([szr[i], szr[i + 1]])
                      for i, p in enumerate(points)]
        points = torch.cat(points, dim=0)
        size_range = torch.cat(size_range, dim=0)

        def sort_instances_by_area(instances):
            if len(instances) <= 1:
                return instances
            boxes = instances.gt_boxes.tensor.cpu().numpy()
            sizes = boxes[:, 2:] - boxes[:, :2]
            areas = sizes[:, 0] * sizes[:, 1]
            idc = np.argsort(areas)
            for field in instances.get_fields():
                instances.set(field, getattr(instances, field)[idc])
            return instances

        gt_labels = []
        gt_boxes = []
        gt_centerness = []
        for gt_per_image in gt_instances:
            if len(gt_per_image) > 0:
                # sort gt boxes by area
                gt_per_image = sort_instances_by_area(gt_per_image)

                # select valid predictions
                lt = points[:, None, :] - gt_per_image.gt_boxes.tensor[:, :2]
                rb = gt_per_image.gt_boxes.tensor[:, 2:] - points[:, None, :]
                ltrb = torch.cat([lt, rb], dim=-1)
                max_sz, _ = ltrb.max(dim=-1)
                min_sz, _ = ltrb.min(dim=-1)
                d = (max_sz >= size_range[:, None, 0]) & \
                    (max_sz <= size_range[:, None, 1]) & \
                    (min_sz > 0)

                matched_vals, matched_idxs = torch.max(d.to(dtype=torch.uint8), dim=-1)

                adx = torch.arange(points.shape[0])
                gt_delta_i = ltrb[adx, matched_idxs]
                gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]

                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_labels_i[matched_vals == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                pass

                min_lr, _ = gt_delta_i[:, [0, 2]].min(dim=-1)
                max_lr, _ = gt_delta_i[:, [0, 2]].max(dim=-1)
                min_tb, _ = gt_delta_i[:, [1, 3]].min(dim=-1)
                max_tb, _ = gt_delta_i[:, [1, 3]].max(dim=-1)
                gt_centerness_i = torch.sqrt((min_lr / max_lr) * (min_tb / max_tb))

            else:
                num = points.shape[0]
                gt_boxes_i = points.new_zeros([num, 4])
                gt_labels_i = points.new_zeros([num]) + self.num_classes
                gt_centerness_i = points.new_zeros([num])

            gt_labels.append(gt_labels_i)
            gt_boxes.append(gt_boxes_i)
            gt_centerness.append(gt_centerness_i)

        return gt_labels, gt_boxes, gt_centerness

    def inference(self, points, pred_logits, pred_anchor_deltas, pre_centerness, image_sizes):
        """
        Arguments:
            points (list[Tensor]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            centerness_per_image = [x[img_idx] for x in pre_centerness]
            results_per_image = self.inference_single_image(
                points, pred_logits_per_image, deltas_per_image, centerness_per_image, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, points, box_cls, box_delta, box_ctn, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            points (list[Tensor]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        strides = [x.stride for x in self.feature_shapes]
        for box_cls_i, box_reg_i, box_ctn_i, points_i, stride_i in zip(box_cls, box_delta, box_ctn, points, strides):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.sigmoid_()
            box_ctn_i = box_ctn_i.sigmoid_()
            box_cls_i = (box_cls_i * box_ctn_i).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            points_i = points_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.transfer_boxes(points_i, box_reg_i, stride_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class FCOSHead(nn.Module):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        num_convs = cfg.MODEL.FCOS.NUM_CONVS
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        num_anchors = 1

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred, self.centerness]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
            centerness (list[Tensor]): #lvl tensors, each has shape (N, 1, Hi, Wi).
                The tensor predicts the centerness of position belongs to object.
        """
        logits = []
        bbox_reg = []
        centerness = []
        for feature in features:
            feat = self.cls_subnet(feature)
            logits.append(self.cls_score(feat))
            centerness.append(self.centerness(feat))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg, centerness
