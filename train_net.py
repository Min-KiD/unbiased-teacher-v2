#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data.datasets import register_coco_instances

# hacky way to register
from ubteacher.modeling import *
from ubteacher.engine import *
from ubteacher import add_ubteacher_config
import json
import numpy as np
import os

def generate_data_seed_file(dataset_size, percentage, seed, output_path):
    data_seeds = {}
    num_label = int(percentage / 100.0 * dataset_size)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.random.seed(seed)
    indices = np.random.choice(dataset_size, num_label, replace=False).tolist()
    data_seeds[str(seed)] = indices

    data_seed_dict = {str(percentage): data_seeds}

    with open(output_path, 'w') as f:
        json.dump(data_seed_dict, f, indent=4)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ("coco_train",)
    cfg.DATASETS.TEST = ("coco_val",)
    cfg.DATALOADER.RANDOM_DATA_SEED_PATH = './data_seed.json'
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    train_ann_file = "/kaggle/working/datasets/coco/annotations/instances_train2017.json"
    val_ann_file = "/kaggle/working/datasets/coco/annotations/instances_val2017.json"
    train_path = "/kaggle/working/datasets/coco/train2017"
    val_path = "/kaggle/working/datasets/coco/val2017"
    
    register_coco_instances("coco_train", {}, train_ann_file, train_path)
    register_coco_instances("coco_val", {}, val_ann_file, val_path)

    dataset_size = 27358  #
    percentage = 10
    seed = 21
    output_path = './data_seed.json'  # Specify your desired output path
    
    generate_data_seed_file(dataset_size, percentage, seed, output_path)

    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "ubteacher_rcnn":
        Trainer = UBRCNNTeacherTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
