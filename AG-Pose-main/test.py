import os
import sys
import argparse
import logging
import random

import torch
import gorilla

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from Net import Net
# from solver import test_func, get_logger
from nocs_dataset import TestDataset
from evaluation_utils import evaluate

import time
from tqdm import tqdm
import pickle as cPickle


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        default='/workspace/code/AG-Pose-main/config/REAL/camera_real.yaml',
                        help="path to config file")
    parser.add_argument("--test_epoch",
                        type=int,
                        default=30,
                        help="test epoch")
    parser.add_argument("--cat_id",
                        type=int,
                        default=-1,
                        help="category id, -1 for mean aps")
    parser.add_argument('--mask_label', action='store_true', default=False,
                        help='whether having mask labels of real data')
    parser.add_argument('--only_eval', action='store_true', default=False,
                        help='whether directly evaluating the results')
    args_cfg = parser.parse_args()
    return args_cfg

def init(epoch):
    args = get_parser()
    exp_name = args.config.split("/")[-1].split(".")[0]
    log_dir = os.path.join("log", exp_name)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    cfg.ckpt_dir = os.path.join(cfg.log_dir, 'ckpt')
    cfg.gpus = args.gpus
    cfg.test_epoch = epoch
    cfg.mask_label = args.mask_label
    cfg.only_eval = args.only_eval
    cfg.cat_id = args.cat_id

    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)
    logger = get_logger(level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir+"/test_epoch" + str(cfg.test_epoch)  + "_logger.log")

    return logger, cfg

def test_func(model, dataloder, save_path):
    model.eval()
    time_all = 0
    with tqdm(total=len(dataloder)) as t:
        for i, data in enumerate(dataloder):
            path = dataloder.dataset.result_pkl_list[i]

            inputs = {
                'rgb': data['rgb'][0].cuda(),
                'pts': data['pts'][0].cuda(),
                'choose': data['choose'][0].cuda(),
                'category_label': data['category_label'][0].cuda(),
            }
            
            start_time = time.time()
            end_points = model(inputs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_all += elapsed_time

            pred_translation = end_points['pred_translation']
            pred_size = end_points['pred_size']
            pred_scale = torch.norm(pred_size, dim=1, keepdim=True)
            pred_size = pred_size / pred_scale
            pred_rotation = end_points['pred_rotation']

            num_instance = pred_rotation.size(0)
            pred_RTs =torch.eye(4).unsqueeze(0).repeat(num_instance, 1, 1).float().to(pred_rotation.device)
            pred_RTs[:, :3, 3] = pred_translation
            pred_RTs[:, :3, :3] = pred_rotation * pred_scale.unsqueeze(2)
            pred_scales = pred_size

            # save
            result = {}

            result['gt_class_ids'] = data['gt_class_ids'][0].numpy()

            result['gt_bboxes'] = data['gt_bboxes'][0].numpy()
            result['gt_RTs'] = data['gt_RTs'][0].numpy()

            result['gt_scales'] = data['gt_scales'][0].numpy()
            result['gt_handle_visibility'] = data['gt_handle_visibility'][0].numpy()

            result['pred_class_ids'] = data['pred_class_ids'][0].numpy()
            result['pred_bboxes'] = data['pred_bboxes'][0].numpy()
            result['pred_scores'] = data['pred_scores'][0].numpy()

            result['pred_RTs'] = pred_RTs.detach().cpu().numpy()
            result['pred_scales'] = pred_scales.detach().cpu().numpy()

            with open(os.path.join(save_path, path.split('/')[-1]), 'wb') as f:
                cPickle.dump(result, f)

            t.set_description(
                "Test [{}/{}][{}]: ".format(i+1, len(dataloder), num_instance)
            )

            t.update(1)
    print(f"time_all: {time_all}")
    
def get_logger(level_print, level_save, path_file, name_logger = "logger"):
    # level: logging.INFO / logging.WARN
    logger = logging.getLogger(name_logger)
    logger.setLevel(level = logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # set file handler
    handler_file = logging.FileHandler(path_file)
    handler_file.setLevel(level_save)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    # set console holder
    handler_view = logging.StreamHandler()
    handler_view.setFormatter(formatter)
    handler_view.setLevel(level_print)
    logger.addHandler(handler_view)
    return logger

def main(epoch=30, model=None):
    logger, cfg = init(epoch)

    logger.warning("************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    if cfg.setting == 'supervised':
        save_path = os.path.join(cfg.log_dir, 'eval_epoch' + str(cfg.test_epoch))
        setting = 'supervised'
    else:
        if cfg.mask_label:
            save_path = os.path.join(cfg.log_dir, 'eval_withMaskLabel_epoch' + str(cfg.test_epoch))
            setting = 'unsupervised_withMask'
        else:
            save_path = os.path.join(cfg.log_dir, 'eval_woMaskLabel_epoch' + str(cfg.test_epoch))
            setting = 'unsupervised'

    if not cfg.only_eval:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # model
        if not model:
            logger.info("=> creating model ...")
            model = Net(cfg.pose_net)
            model = model.cuda()

        checkpoint = os.path.join(cfg.ckpt_dir, 'epoch_' + str(cfg.test_epoch) + '.pt')
        logger.info("=> loading checkpoint from path: {} ...".format(checkpoint))
        gorilla.solver.load_checkpoint(model=model, filename=checkpoint)
 
        # data loader
        dataset = TestDataset(cfg.test_dataset.img_size, cfg.test_dataset.sample_num, cfg.test_dataset.dataset_dir, cfg.setting, cfg.test_dataset.dataset_name)

        dataloder = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=8,
                shuffle=True,
                drop_last=False
            )
        test_func(model, dataloder, save_path)

    evaluate(save_path, logger, cat_id=cfg.cat_id)

if __name__ == "__main__":
    main()




