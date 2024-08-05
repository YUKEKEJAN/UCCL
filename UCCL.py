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
from scipy.spatial.distance import cosine
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
import torch.distributed as dist
from prote import Protetype

parser = argparse.ArgumentParser(description='Uncertainty-Guided Context Consistency Learning for Semi-supervised Semantic Segmentation')
parser.add_argument('--config', type=str, default="/home/user/New_idea/HFPL/configs/pascal.yaml")
parser.add_argument('--labeled-id-path', type=str, default="/home/user/New_idea/HFPL/splits/pascal/1464/labeled.txt")
parser.add_argument('--unlabeled-id-path', type=str, default="/home/user/New_idea/HFPL/splits/pascal/1464/unlabeled.txt")
parser.add_argument('--save-path', type=str, default="/home/user/New_idea/HFPL/Test/1464_2")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=16107, type=int)



# 初始化内存库
# memory_bank = nn.Parameter(torch.zeros(21, 1, 256, dtype=torch.float), requires_grad=False)
# memory_bank = memory_bank.cuda()

def upsample(features,h,w):
    return F.interpolate(features, size=(h, w), mode="bilinear", align_corners=True)


#####  
def calculate_region(feature,label,cfg,ignore_mask,logger):
    batch_size,c_channel,h,w = feature.size()
    num_classes = label.size(1)
    feats_sl = torch.zeros(batch_size, h*w, c_channel).type_as(feature)
    max_logit = label.softmax(dim=1).max(dim=1)[0] # 取最大值
    # index = (max_logit <= cfg['conf_thresh']) # B H W
    index = (max_logit > 0)
    feature_temp = feature.clone()
    feature_temp = feature_temp.reshape(batch_size, h*w, c_channel)
    # logger.info("大于阈值的像素个数所占比例为: %s %s"%(index.sum()/(batch_size*h*w), index.sum()))
    for batch_idx in range(batch_size):
        ##### (C, H, W), (num_classes, H, W) --> (H*W, C), (H*W, num_classes)
        feats_iter, preds_iter = feature[batch_idx], label[batch_idx]
        feats_iter, preds_iter = feats_iter.reshape(c_channel, -1), preds_iter.reshape(num_classes, -1)
        feats_iter, preds_iter = feats_iter.permute(1, 0), preds_iter.permute(1, 0)
        index_batch = index[batch_idx] # H * W
        index_batch = index_batch.reshape(h*w)
        batch_ignore_index = ignore_mask[batch_idx] # H * W
        batch_ig = batch_ignore_index.reshape(h*w)
        argmax = preds_iter.argmax(1) # 直接 返回下标索引
        feature_temp_index = feature_temp[batch_idx]
        for clsid in range(num_classes):
            maskk = (argmax == clsid) # 为该类的掩码 H * 1 
            mask = (maskk == True) & (index_batch == True) & (batch_ig != 255) # 符合条件的像素个数
            if mask.sum() == 0: continue  # 没有这个类就放弃
            #logger.info("属于类%s的像素个数为:%s"%(clsid,maskk.sum()))
            #logger.info("满足条件的像素个数为:%s"%(mask.sum()))
            feats_iter_cls = feats_iter[mask] # (h*w) × C
            preds_iter_cls = preds_iter[:, clsid][mask] # 抽取特定类的最大值出来
            weight = F.softmax(preds_iter_cls, dim=0) # 进行softmax 
            feats_iter_cls = feats_iter_cls * weight.unsqueeze(-1) # 加权乘积等到一个类的区域特征
            feats_iter_cls = feats_iter_cls.sum(0)
            feats_sl[batch_idx][maskk] = feats_iter_cls # 得到对应的区域特征
        feats_sl[batch_idx][batch_ig == 255] =  feature_temp_index[batch_ig == 255]


    feats_sl = feats_sl.reshape(batch_size, h, w, c_channel)
    feats_sl = feats_sl.permute(0, 3, 1, 2).contiguous() # 返回B C H W

    return feats_sl

def cal_loss_divergence(Similarity,label_u_w,pred,ignore_mask_for_u,logger,cfg,criterion_u):
    ##########
    batch_size,h,w = Similarity.size()
    num_classes = label_u_w.size(1)
    max_logit = label_u_w.softmax(dim=1).max(dim=1)[0] # 取最大值
    index = (max_logit <= cfg['conf_thresh']) # B H W
    label = label_u_w.softmax(dim=1).max(dim=1)[1] # 取标签
    loss_all = torch.tensor(0)
    for batch_idx in range(batch_size):
        ###每个批处理的相似度
        Similarity_per_bacth_size = Similarity[batch_idx] * (-1) # (H × W)
        argmax = label[batch_idx]
        pred_per_batch = pred[batch_idx] # 
        index_batch = index[batch_idx] # H * W
        batch_ig = ignore_mask_for_u[batch_idx] # H W 
        index_batch = index_batch # H W
        #argmax = label_per_batch_size.argmax(1) # 直接 返回下标索引 (H × W)
        loss_current_all = torch.tensor(0)
        count = 0
        for clsid in range(num_classes):   # 目的是获得每个类的相似度softmax的值
            maskk = (argmax == clsid) # 为该类的掩码 H * W 
            mask = (maskk == True) & (batch_ig != 255)  & (index_batch == True) # # 符合条件的像素个数
            if mask.sum() == 0: continue
            Similarity_current_class = Similarity_per_bacth_size * mask   # 筛选出来的可靠的相似度 (H×W)
            Similarity_current_class = Similarity_current_class.reshape(h*w)
            Similarity_current_class = F.softmax(Similarity_current_class,dim=0)
            Similarity_current_class = Similarity_current_class.view(h,w)
            loss_simi_pro = criterion_u(pred_per_batch.unsqueeze(0),argmax.unsqueeze(0))
            loss_simi_pro = loss_simi_pro * (Similarity_current_class) * (batch_ig != 255)
            loss_simi_pro = loss_simi_pro.sum() #/ ((Similarity_current_class!=0) & (batch_ig != 255)).sum().item()
            #print("损失为:%s"%(loss_simi_pro))
            loss_current_all = loss_current_all + loss_simi_pro
            count = count + 1
        if count != 0:
            loss_all = loss_all + loss_current_all / count
        else:
            loss_all = loss_all + loss_current_all
          

    return loss_all / batch_size



    

def main():
    args =  parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    local_rank = int(os.environ["LOCAL_RANK"])

    model = DeepLabV3Plus(cfg)
    
    
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    
    
    if rank == 0:
        logger.info('Model params: {:.1f}M \n'.format(count_params(model)))

    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    cons_w_unsup = consistency_weight(final_w=cfg['unsupervised_w'], iters_per_epoch=len(trainloader_u),
                                        rampup_ends=int(cfg['epochs']))

    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'),map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    
    pro = Protetype()
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_loss_region = AverageMeter()
        total_loss_simi = AverageMeter()
        # loss_feature_ali_loss = AverageMeter()
        # loss_prote_all = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        
        # img_u_w_mix 是经正则化的弱增强图片，img_u_s1_mix 是经过强增强的图片，ignore_mask_mix 是忽略的像素 
        # cutmix_box1 cutmix_box2 是cutmix的坐标
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            img_1 = img_u_w.clone()
            img_2 = img_u_w.clone()
            b,c,h,w = img_1.size()
            img_s_s1 = img_u_s1.clone()
            ignore_mask_for_u_1 = ignore_mask.clone()
            ignore_mask_for_u_2 = ignore_mask.clone()
            ignore_mask_for_s_1 = ignore_mask.clone()
            ignore_mask_for_s_2 = ignore_mask.clone()

            iters = epoch * len(trainloader_u) + i

            weight_u = cons_w_unsup(epoch=epoch,curr_iter=i)

            with torch.no_grad():
                model.eval()

                featur_u_w_mix,pred_u_w_mix = model(img_u_w_mix) # 对增强图片的预测
                pred_u_w_mix = pred_u_w_mix.detach()
                probability_mix = F.softmax(pred_u_w_mix,dim=1)
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0] # 每个类的概率
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1) # 标签

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] # 强增强图片之间的cutmix
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            img_1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_w_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_w_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            ignore_mask_for_u_1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]
            ignore_mask_for_u_2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            ignore_mask_for_s_1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]
            ignore_mask_for_s_2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]
            
            
            # 产生的特征都是256维度的
            model.train()
            feature_1,label_u_w_1_large,label_u_w_1 = model(img_1,need_pre_logit = True) # 弱增强的第一个cutmix图片特征
            label_u_1 = label_u_w_1.clone()
            feature_2,label_u_w_2_large,label_u_w_2 = model(img_2,need_pre_logit = True) # 若增强的第二个cutmix图片特征
            label_u_2 = label_u_w_2.clone()

            

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            feature_weak, _ , pred_weak = model(img_u_w,need_pre_logit=True)
            probablity_weak = F.softmax(pred_weak,dim=1)

            label_weak_prote = probablity_weak.argmax(dim=1)
            probablity_weak = probablity_weak.max(dim=1)[0]
            feature_strong, _ , pred_strong = model(img_s_s1,need_pre_logit=True)

            _,preds, preds_fp = model(torch.cat((img_x, img_u_w)), need_fp=True) # 所有的预测结果 以及 扰动的预测结果  这里没有加入Cutmix
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])

            pred_u_w_fp = preds_fp[num_lb:] # 无标签数据的扰动特征

            feature_u_all, pred_u_all,pre_logit_all = model(torch.cat((img_u_s1, img_u_s2)),need_pre_logit = True) # 经过cutmix之后得到的 预测结果

            pred_u_s1, pred_u_s2 = pred_u_all.chunk(2)
            pre_logit_s1, pre_logit_s2 = pre_logit_all.chunk(2)

            cos_dis = nn.CosineSimilarity(dim=1, eps=1e-6)

            #像素级别的特征差异化损失
            feature_u_s1,feature_u_s2 = feature_u_all.chunk(2)

            # 区域之间的特征差异化损失
            # 先传入弱增强的图片特征 
            bb,cc,hh,ww = feature_u_s2.size()
            
            ## 相似度计算
            Similarity_1 = cos_dis(feature_1.clone().detach(),feature_u_s1) # B × H × W

            Similarity_2 = cos_dis(feature_2.clone().detach(),feature_u_s2)

            # 计算单张图像的区域损失
            ignore_mask_for_u_1 = F.interpolate(ignore_mask_for_u_1.float().unsqueeze(1), size=(hh, ww), mode="bilinear", align_corners=True).squeeze(1)
            ignore_mask_for_u_2 = F.interpolate(ignore_mask_for_u_2.float().unsqueeze(1), size=(hh, ww), mode="bilinear", align_corners=True).squeeze(1)
            feature_region_u_1 = calculate_region(feature_1,label_u_w_1,cfg,ignore_mask_for_u_1,logger)
            feature_region_u_2 = calculate_region(feature_2,label_u_w_2,cfg,ignore_mask_for_u_2,logger)

            feature_region_u_s1 = calculate_region(feature_u_s1,label_u_w_1,cfg,ignore_mask_for_u_1,logger)
            feature_region_u_s2 = calculate_region(feature_u_s2,label_u_w_2,cfg,ignore_mask_for_u_2,logger)

            ###########两个损失
            loss_region_1 = 1 - cos_dis(feature_region_u_1.clone().detach(),feature_region_u_s1).mean()
            loss_region_2 = 1 - cos_dis(feature_region_u_2.clone().detach(),feature_region_u_s2).mean()


            ####区域的总损失
            loss_region = (loss_region_1 * 0.02 + loss_region_2 * 0.02) / 2

            # #####接下来计算
            total_loss_region.update(loss_region.item())


            ###############相似度注入到损失函数中##################
            loss_similarity_1 = cal_loss_divergence(Similarity_1, label_u_1, pre_logit_s1, ignore_mask_for_u_1, logger, cfg, criterion_u)
            loss_similarity_2 = cal_loss_divergence(Similarity_2, label_u_2, pre_logit_s2, ignore_mask_for_u_2, logger, cfg, criterion_u)
            
            loss_sim = (loss_similarity_1 * 0.015 + loss_similarity_2 * 0.015) /2
            total_loss_simi.update(loss_sim.item())
            
            
            pred_u_w = pred_u_w.detach() # 弱增强图片的预测
            probability_u_w = F.softmax(pred_u_w,dim=1)
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            probability_clone_mix = probability_mix.clone()
            probability_clone_u_w_1 = probability_u_w.clone()
            probability_clone_u_w_2 = probability_u_w.clone()
            cutmix_box1_p1 = cutmix_box1.clone()
            cutmix_box2_p2 = cutmix_box2.clone()

            cutmix_box1_p1 = cutmix_box1_p1.unsqueeze(1).repeat(1,cfg['nclass'],1,1)
            cutmix_box2_p2 = cutmix_box2_p2.unsqueeze(1).repeat(1,cfg['nclass'],1,1)
            probability_clone_u_w_1[cutmix_box1_p1 == 1] = probability_clone_mix[cutmix_box1_p1 == 1]
            probability_clone_u_w_2[cutmix_box2_p2 == 1] = probability_clone_mix[cutmix_box2_p2 == 1]

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]


            loss_x = criterion_l(pred_x, mask_x) # 有标签的损失

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0  + loss_sim  + loss_region # + cfg['feat_ali_weight'] * (loss_feature_1 + loss_feature_2)/2.0

            # loss = loss + cfg['feat_nearest'] * loss_prote
            
            
        
            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            


            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            #loss_feature_ali_loss.update(cfg['feat_ali_weight'] * (loss_feature_1.item() + loss_feature_2.item())/2.0)
            #loss_prote_all.update(cfg['feat_nearest'] * loss_prote.item())
            
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss_s:{:.3f}, Loss_fp:{:.3f}, Loss_sim: {:.8f} loss_region: {:.5f} '.format(i, 
                                        total_loss.avg, total_loss_x.avg, total_loss_s.avg , total_loss_w_fp.avg , total_loss_simi.avg, total_loss_region.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
