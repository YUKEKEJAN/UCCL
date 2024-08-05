import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch.distributed as dist

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from util.contrast_loss import StudentSegContrast
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from util.consistency import consistency_weight
import numpy as np
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.paltte import BuildPalette
import cv2

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, default="/home/user/HFPL/configs/cityscapes.yaml")
parser.add_argument('--labeled-id-path', type=str, default="/home/y212202015/Feature/Unimatch_noise/Unimatch/splits/coco/1_128/labeled.txt")
parser.add_argument('--unlabeled-id-path', type=str, default="/home/y212202015/Feature/Unimatch_noise/Unimatch/splits/coco/1_128/unlabeled.txt")
parser.add_argument('--save-path', type=str, default="/home/user/HFPL/Experiment_Single/Compare_Results/Cityscapes/Res50/744/best.pth")
parser.add_argument('--imagepath', type=str, default="/home/user/HFPL/Cityscapes_visual/ours")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=16107, type=int)

def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    rank, world_size = setup_distributed(port=args.port)

    palette = BuildPalette(dataset_type='cityscapes', num_classes=21)

    # if rank == 0:
    #     all_args = {**cfg, **vars(args), 'ngpus': world_size}
    #     logger.info('{}\n'.format(pprint.pformat(all_args)))
        
    #     writer = SummaryWriter(args.save_path)
        
    #     os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    local_rank = int(os.environ["LOCAL_RANK"])

    # 加载模型
    model = DeepLabV3Plus(cfg)

    if rank == 0:
        logger.info('Model params: {:.1f}M \n'.format(count_params(model)))
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)
    
    # 加载数据集
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=8,
                            drop_last=False,sampler=valsampler)
    
    # 加载权重文件
    checkpoint = torch.load(args.save_path)
    model.load_state_dict(checkpoint['model'])
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    model.eval()
    for img,mask,id in valloader:
        img = img.cuda()

        feature, pred = model(img)
        pred = pred.argmax(dim = 1)
        predd = pred.squeeze(0).cpu().numpy().astype(np.int32)
        maskk = np.zeros((predd.shape[0], predd.shape[1], 3), dtype=np.uint8)

        for clsid, color in enumerate(palette):
            maskk[predd == clsid, :] = np.array(color)[::-1]
        
        image = maskk
        image = image.astype(np.uint8)
        cv2.imwrite(os.path.join(args.imagepath,str(id).split(" ")[0].split("/")[-1]), image)

        intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    print(mIOU)



if __name__ == '__main__':
    main()
