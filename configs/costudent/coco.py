#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii Inc. and its affiliates.

import imp
from PIL import Image
import torchvision.transforms as tfs
import copy
import numpy as np
import megfile
import torch
from cvpods.data.datasets import COCODataset
from cvpods.data.registry import DATASETS, PATH_ROUTES
from cvpods.data.detection_utils import (
    annotations_to_instances,
    check_image_size,
    filter_empty_instances,
    read_image
)

_UPDATE_DICT = {
    "coco_2017_train_full":
        ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_train_easy":
        ("coco/train2017", "coco/annotations/instances_train2017_easy.json"),
    "coco_2017_train_hard":
        ("coco/train2017", "coco/annotations/instances_train2017_hard.json"),
    "coco_2017_train_extreme":
        ("coco/train2017", "coco/annotations/instances_train2017_extreme.json"),
    "coco_missing_50p":
        ("coco/train2017", "coco_missing_50p.json"),
    "coco_voc_0712_trainval":
        ("coco/train2017", "coco/annotations/pascal_trainval0712_missing50p.json"),

    "coco_loose_redcup17711_train_og_only":
        ("loose/loose_17711/images", "loose/loose_17711/annotations/instances_train_og_only.json"),
    "coco_loose_redcup17711_train_cropped_only":
        ("loose/loose_17711/images", "loose/loose_17711/annotations/instances_train_cropped_only.json"),
    "coco_loose_redcup17648_train_og_only":
        ("loose/loose_17648/images", "loose/loose_17648/annotations/instances_train_og_only.json"),
    "coco_loose_redcup17648_train_cropped_only": 
        ("loose/loose_17648/images", "loose/loose_17648/annotations/instances_train_cropped_only.json"),
    "coco_loose_redcup17647_train_og_only":
        ("loose/loose_17647/images", "loose/loose_17647/annotations/instances_train_og_only.json"),
    "coco_loose_redcup17647_train_cropped_only":
        ("loose/loose_17647/images", "loose/loose_17647/annotations/instances_train_cropped_only.json"),
    
    "coco_squidle_redcup_train_sparse":
        ("squidle_coco/squidle_redcup_full_train_sparse/images", "squidle_coco/squidle_redcup_full_train_sparse/annotations/instances_train.json"),
    "coco_squidle_redcup_test":
        ("squidle_coco/squidle_redcup_test/test2023", "squidle_coco/squidle_redcup_test/annotations/instances_test.json"),    
    
    
    
    "coco_loose_urchin17714_train_og_only":
        ("loose/loose_17714/images", "loose/loose_17714/annotations/instances_train_og_only.json"),
    "coco_loose_urchin17714_train_cropped_only":
        ("loose/loose_17714/images", "loose/loose_17714/annotations/instances_train_cropped_only.json"),
    "coco_loose_urchin17631_train_og_only":
        ("loose/loose_17631/images", "loose/loose_17631/annotations/instances_train_og_only.json"),
    "coco_loose_urchin17631_train_cropped_only":
        ("loose/loose_17631/images", "loose/loose_17631/annotations/instances_train_cropped_only.json"),    
    "coco_loose_urchin17630_train_og_only":
        ("loose/loose_17630/images", "loose/loose_17630/annotations/instances_train_og_only.json"),
    "coco_loose_urchin17630_train_cropped_only":
        ("loose/loose_17630/images", "loose/loose_17630/annotations/instances_train_cropped_only.json"),
    
    "coco_squidle_urchin_train_sparse":
        ("squidle_coco/squidle_urchin_full_train_sparse/images", "squidle_coco/squidle_urchin_full_train_sparse/annotations/instances_train.json"),
    "coco_squidle_urchin_2011_test":
        ("squidle_coco/squidle_urchin_2011/test2023", "squidle_coco/squidle_urchin_2011/annotations/instances_test2023.json"),

    "coco_squidle_ob_test":
        ("squidle_coco/squidle_ob_sparse_point_test/test2023", "squidle_coco/squidle_ob_sparse_point_test/annotations/instances_test.json"),  
    "coco_squidle_ob_train":
        ("squidle_coco/squidle_ob_sparse_point_train/train2023", "squidle_coco/squidle_ob_sparse_point_train/annotations/instances_train.json"),      
    "coco_loose_ob17863_train_og_only":
        ("loose/loose_17863/images", "loose/loose_17863/annotations/instances_train_og_only.json"),
    "coco_loose_ob17863_train_cropped_only":
        ("loose/loose_17863/images", "loose/loose_17863/annotations/instances_train_cropped_only.json"),
    
        "coco_squidle_anemone_sparse_test":
        ("squidle_coco/squidle_anemone_sparse_point_test/test2023", "squidle_coco/squidle_anemone_sparse_point_test/annotations/instances_test.json"),  
        "coco_squidle_anemone_full_test":
        ("squidle_coco/squidle_anemone_full_test/test2023", "squidle_coco/squidle_anemone_full_test/annotations/instances_test.json"),  
    "coco_squidle_anemone_train_sparse":
        ("squidle_coco/squidle_anemone_sparse_point_train/train2023", "squidle_coco/squidle_anemone_sparse_point_train/annotations/instances_train.json"),      
    "coco_loose_anemone18051_train_og_only":
        ("loose/loose_18051/images", "loose/loose_18051/annotations/instances_train_og_only.json"),
    "coco_loose_anemone18051_train_cropped_only":
        ("loose/loose_18051/images", "loose/loose_18051/annotations/instances_train_cropped_only.json"),
    
    "coco_weddell_sea":
        ("WeddellSea/Results/images", "WeddellSea/Results/annotations/dataset_anemones.json"),  
}



PATH_ROUTES.get("COCO")["coco"].update(_UPDATE_DICT)
PATH_ROUTES.get("COCO")["dataset_type"] = "COCOMutiBranch"



multi_branch = ["Raw","Nor","Str"]
@DATASETS.register()
class COCOMutiBranch(COCODataset):
    
    def _apply_transforms(self, image, annotations=None, **kwargs):
        """
        Apply a list of :class:`TransformGen` on the input image, and
        returns the transformed image and a list of transforms.

        We cannot simply create and return all transforms without
        applying it to the image, because a subsequent transform may
        need the output of the previous one.

        Args:
            transform_gens (list): list of :class:`TransformGen` instance to
                be applied.
            img (ndarray): uint8 or floating point images with 1 or 3 channels.
            annotations (list): annotations
        Returns:
            ndarray: the transformed image
            TransformList: contain the transforms that's used.
        """
        if self.transforms is None:
            return image, annotations

        if isinstance(self.transforms, dict):
            dataset_dict = {}
            for key, tfms in self.transforms.items():
                img = copy.deepcopy(image)
                annos = copy.deepcopy(annotations)
                for tfm in tfms:
                    img, annos = tfm(img, annos, **kwargs)
                dataset_dict[key] = (img, annos)
            return dataset_dict, None
        elif isinstance(self.transforms, list):
            tfm = self.transforms[kwargs['index']]
            image, annotations = tfm(image, annotations, **kwargs)
            return image, annotations
        else:
            for tfm in self.transforms:
                image, annotations = tfm(image, annotations, **kwargs)

            return image, annotations


    def __getitem__(self, index):
        """Load data, apply transforms, converto to Instances.
        """

        dataset_dict = copy.deepcopy(self.dataset_dicts[index])

        # read image

        image = read_image(dataset_dict["file_name"], format=self.data_format)
        check_image_size(dataset_dict, image)

        if "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            annotations = [
                ann for ann in annotations if ann.get("iscrowd", 0) == 0]
        else:
            annotations = None

        annotations_copy = copy.deepcopy(annotations)

        if len(self.transforms) == 1:
            image_i, annotations_i = self._apply_transforms(image, annotations,index=0)
            image_shape = image_i.shape[:2]  # h, w

            instances = annotations_to_instances(
                annotations_i, image_shape, mask_format=self.mask_format
            )


            dataset_dict["instances"] = filter_empty_instances(instances)
            
            dataset_dict["image"] = torch.as_tensor(
                    np.ascontiguousarray(image_i.transpose(2, 0, 1)))

            return dataset_dict

        for i in range(3):
            annotations = copy.deepcopy(annotations_copy)
            image_i, annotations_i = self._apply_transforms(image, annotations,index=i)


            image_shape = image_i.shape[:2]  # h, w

            instances = annotations_to_instances(
                annotations_i, image_shape, mask_format=self.mask_format
            )

            # # Create a tight bounding box from masks, useful when image is cropped
            # if self.crop_gen and instances.has("gt_masks"):
            #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            dataset_dict["instances_"+multi_branch[i]] = filter_empty_instances(instances)
            if 'transform_matrix' in annotations_i[0]:
                dataset_dict["transform_matrix_"+multi_branch[i]] = annotations_i[0]['transform_matrix']
            # convert to Instance type
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            # h, w, c -> c, h, w
            dataset_dict["image_"+multi_branch[i]] = torch.as_tensor(
                    np.ascontiguousarray(image_i.transpose(2, 0, 1)))


        return dataset_dict