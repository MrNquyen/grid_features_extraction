#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Grid features extraction script.
"""
import argparse
import os
import torch
from tqdm import tqdm

import torch.nn as nn
import torchvision.transforms as transforms


from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from icecream import ic
from fvcore.common.file_io import PathManager
from utils import load_json, save_json, load_image

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.evaluation import inference_context
from detectron2.modeling import build_model

from grid_feats import (
    add_attribute_config,
    build_detection_test_loader_with_attributes,
)

# A simple mapper from object detection dataset to VQA dataset names
dataset_to_folder_mapper = {}
dataset_to_folder_mapper['coco_2014_train'] = 'train2014'
dataset_to_folder_mapper['coco_2014_val'] = 'val2014'
# One may need to change the Detectron2 code to support coco_2015_test
# insert "coco_2015_test": ("coco/test2015", "coco/annotations/image_info_test2015.json"),
# at: https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/builtin.py#L36
dataset_to_folder_mapper['coco_2015_test'] = 'test2015'

# Data class
class myDataset(Dataset):
    def __init__(self, data_save_dir, image_save_dir):
        super().__init__()
        data = setup_data(data_save_dir=data_save_dir)
        self.data = []
        self.transform = transform = transforms.Compose([
            transforms.PILToTensor()
        ])

        for data_dict in data.values():
            ic(None in data.values())
            for id, v in data_dict.items():
                img_path = os.path.join(image_save_dir, f'{id}.png')
                v['img_path'] = img_path
                v['img_id'] = id
                self.data.append(v)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['img_path']
        # image = torch.tensor(load_image(img_path))
        # image = ToTensor(load_image(img_path))
        image = load_image(img_path)
        image = self.transform(image)

        return {
            'image_id': item['img_id'],
            'image': image,
        }
    
    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    # img_paths = [item['img_path'] for item in batch]
    # image = [torch.tensor(load_image(img_path)) for img_path in img_paths]
    # return {
    #     'image': image,
    #     'image_id': 
    # }
    return batch

# Extract features section
def extract_grid_feature_argument_parser():
    # data_save_dir
    data_save_dir = "F:\\UNIVERSITY\\NCKH\\NCKH_V2\\code\\feature_extraction\\data"
    img_save_dir = "F:\\UNIVERSITY\\NCKH\\NCKH_V2\\baseline_final\\AoANet_Official\\AoANet\\data\\images"

    # Args Parsers
    parser = argparse.ArgumentParser(description="Grid feature extraction")
    parser.add_argument("--config-file", default="configs/X-101-grid.yaml", metavar="FILE", help="path to config file",
                        choices=["configs/R-50-grid.yaml", "configs/X-101-grid.yaml", "configs/X-152-grid.yaml"])
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2014_train",
                        choices=['coco_2014_train', 'coco_2014_val', 'coco_2015_test'])
    parser.add_argument("--data_save_dir", help="dataset save location", default=data_save_dir)
    parser.add_argument("--img_save_dir", help="images save location", default=img_save_dir)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()

def extract_grid_feature_on_dataset(model, data_loader, dump_folder):
    for idx, inputs in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            # image_id = inputs[0]['image_id']
            image_ids = [item['image_id'] for item in inputs]
            file_names = [f'{img_id}.pth' for img_id in image_ids]
            # compute features
            images = model.preprocess_image(inputs)
            features = model.backbone(images.tensor)
            outputs = model.roi_heads.get_conv5_features(features)
            outputs = outputs.cpu()
            if len(outputs) != len(file_names):
                raise Exception("Ouputs length and Number of id has different length")
            for id, file_name in tqdm(enumerate(file_names)):
                with PathManager.open(os.path.join(dump_folder, file_name), "wb") as f:
                    # save as CPU tensors
                    torch.save(outputs[id], f)

def do_feature_extraction(cfg, model, dataset_name):
    with inference_context(model):
        dump_folder = os.path.join(cfg.OUTPUT_DIR, "features")
        PathManager.mkdirs(dump_folder)
        # data_loader = build_detection_test_loader_with_attributes(cfg, dataset_name)
        dataset = myDataset(cfg.DATA_SAVE_DIR, cfg.IMG_SAVE_DIR)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            collate_fn=collate_fn
        )
        extract_grid_feature_on_dataset(model, data_loader, dump_folder)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    # Setup
    device = "cpu" if torch.cpu.is_available() else "cuda"
    
    # Get config
    cfg = get_cfg()
    add_attribute_config(cfg)
    ic('Finish add_attribute_config')
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # force the final residual block to have dilations 1
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.MODEL.DEVICE = device
    cfg.SOLVER.IMS_PER_BATCH = 1
    # setup path
    data_save_dir = "F:\\UNIVERSITY\\NCKH\\NCKH_V2\\code\\feature_extraction\\data"
    img_save_dir = "F:\\UNIVERSITY\\NCKH\\NCKH_V2\\baseline_final\\AoANet_Official\\AoANet\\data\\images"

    cfg.DATA_SAVE_DIR = data_save_dir
    cfg.IMG_SAVE_DIR = img_save_dir

    cfg.freeze()
    default_setup(cfg, args)
    ic('Finish default_setup')
    return cfg


def setup_data(data_save_dir, file_names=['train_segmented', 'dev_segmented', 'test_segmented']):
    data_paths = [
        os.path.join(data_save_dir, f'{file_name}.json') 
        for file_name in file_names
    ]
    data_dicts = [
        load_json(data_path) 
        for data_path in data_paths
    ]
    data = {
        'train': None,
        'val': None,
        'test': None,
    }

    for file_name, data_dict in zip(file_names, data_dicts):
        if 'train' in file_name:
            data['train'] = data_dict
        elif 'dev' in file_name:
            data['val'] = data_dict
        elif 'test' in file_name:
            data['test'] = data_dict
    return data


def main(args):
    # Config setup
    cfg = setup(args)

    # Setup model
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    dataset = setup_data(cfg.DATA_SAVE_DIR)
    do_feature_extraction(cfg, model, dataset)


if __name__ == "__main__":
    args = extract_grid_feature_argument_parser()
    print("Command Line Args:", args)
    main(args)
