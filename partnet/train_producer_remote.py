#!/usr/bin/env python
"""Train point cloud instance segmentation models"""

import sys
import os
import os.path as osp

sys.path.insert(0, osp.dirname(__file__) + '/..')
import argparse
import logging
import time

import torch
from torch import nn

from core.config import purge_cfg
from core.solver.build import build_optimizer, build_scheduler
from core.nn.freezer import Freezer
from core.utils.checkpoint import Checkpointer
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.tensorboard_logger import TensorboardLogger
from core.utils.torch_util import set_random_seed

from partnet.models.build import build_model
from partnet.data.build import build_dataloader
from IPython import embed
import numpy as np

import shaper.models.pointnet2.functions as _F
import torch.nn.functional as F
from partnet.models.pn2 import PointNetCls
from core.nn.functional import cross_entropy
from core.nn.functional import focal_loss
from core.nn.functional import l2_loss

from subprocess import Popen


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Learning Training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

def mask_to_xyz(pc, index, sample_num=1024):
    #pc:1 x 3 x num_points
    #index:num x num_points
    pc = pc.squeeze(0)
    parts_num = index.shape[0]
    parts_xyz = torch.zeros([parts_num, 3, sample_num]).cuda().type(torch.FloatTensor)
    parts_mean = torch.zeros([parts_num, 3]).cuda().type(torch.FloatTensor)
    for i in range(parts_num):
        part_pc = torch.masked_select(pc.squeeze(),mask=index[i].unsqueeze(0).byte()).reshape(3,-1)
        length = part_pc.shape[1]
        if length == 0:
            continue
        parts_mean[i] = torch.mean(part_pc, 1)
        initial_index = np.random.randint(length)
        parts_xyz[i] = part_pc[:,initial_index].unsqueeze(1).expand_as(parts_xyz[i])
        cur_sample_num = length if length < sample_num else sample_num
        parts_xyz[i,:,:cur_sample_num] = part_pc[:,torch.randperm(length)[:cur_sample_num]]
    return parts_xyz.cuda(), parts_mean.cuda().unsqueeze(-1)

def tile(tensor, dim, n):
    """Tile n times along the dim axis"""
    if dim == 0:
        return tensor.unsqueeze(0).transpose(0,1).repeat(1,n,1).view(-1,tensor.shape[1])
    else:
        return tensor.unsqueeze(0).transpose(0,1).repeat(1,1,n).view(tensor.shape[0], -1)
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

policy_update_bs = 64
xyz_pool1 = torch.zeros([0,3,1024]).float()
xyz_pool2 = torch.zeros([0,3,1024]).float()
context_xyz_pool1 = torch.zeros([0,3,1024]).float()
context_xyz_pool2 = torch.zeros([0,3,1024]).float()
context_context_xyz_pool = torch.zeros([0,3,2048]).float()
context_label_pool = torch.zeros([0]).float()
context_purity_pool = torch.zeros([0]).float()
label_pool = torch.zeros([0]).float()
purity_purity_pool = torch.zeros([0]).float()
purity_xyz_pool = torch.zeros([0,3,1024]).float()
policy_purity_pool = torch.zeros([0,policy_update_bs]).float()
policy_reward_pool = torch.zeros([0,policy_update_bs]).float()
policy_xyz_pool1 = torch.zeros([0,policy_update_bs,3,1024]).float()
policy_xyz_pool2 = torch.zeros([0,policy_update_bs,3,1024]).float()
old_iteration = 0

def train_one_epoch(model,
                    model_merge,
                    loss_fn,
                    metric,
                    dataloader,
                    cur_epoch,
                    optimizer,
                    optimizer_embed,
                    checkpointer_embed,
                    output_dir_merge,
                    max_grad_norm=0.0,
                    freezer=None,
                    log_period=-1):
    global xyz_pool1
    global xyz_pool2
    global context_xyz_pool1
    global context_xyz_pool2
    global context_context_xyz_pool
    global context_label_pool
    global context_purity_pool
    global label_pool
    global purity_purity_pool
    global purity_xyz_pool
    global policy_purity_pool
    global policy_reward_pool
    global policy_xyz_pool1
    global policy_xyz_pool2
    global old_iteration

    logger = logging.getLogger('shaper.train')
    meters = MetricLogger(delimiter='  ')
    metric.reset()
    meters.bind(metric)

    model.eval()
    loss_fn.eval()
    model_merge.eval()
    softmax = nn.Softmax()

    policy_total_bs = 8
    rnum = 1 if policy_total_bs-cur_epoch < 1 else policy_total_bs-cur_epoch
    end = time.time()

    buffer_txt = os.path.join(output_dir_merge, 'last_buffer')
    checkpoint_txt = os.path.join(output_dir_merge, 'last_checkpoint')
    if os.path.exists(checkpoint_txt):
        checkpoint_f = open(checkpoint_txt,'r')
        cur_checkpoint = checkpoint_f.read()
        checkpoint_f.close()
    else:
        cur_checkpoint = 'no_checkpoint'

    checkpointer_embed.load(None, resume=True)
    print('load checkpoint from %s'%cur_checkpoint)
    model_merge.eval()
    for iteration, data_batch in enumerate(dataloader):
        print('epoch: %d, iteration: %d, size of binary: %d, size of context: %d'%(cur_epoch, iteration, len(xyz_pool1), len(context_xyz_pool1)))
        sys.stdout.flush()
    #add conditions
        if os.path.exists(checkpoint_txt):
            checkpoint_f = open(checkpoint_txt,'r')
            new_checkpoint = checkpoint_f.read()
            checkpoint_f.close()
            if cur_checkpoint != new_checkpoint:
                checkpointer_embed.load(None, resume=True)
                cur_checkpoint = new_checkpoint
                print('load checkpoint from %s'%cur_checkpoint)
                model_merge.eval()

        data_time = time.time() - end

        data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        #predict box's coords
        with torch.no_grad():
            preds = model(data_batch)
            loss_dict = loss_fn(preds, data_batch)
        total_loss = sum(loss_dict.values())
        meters.update(loss=total_loss, **loss_dict)
        with torch.no_grad():
            # TODO add loss_dict hack
            metric.update_dict(preds, data_batch)

        #extraction box features
        batch_size, _, num_centroids, num_neighbours = data_batch['neighbour_xyz'].shape
        num_points = data_batch['points'].shape[-1]

        #batch_size, num_centroid, num_neighbor
        _, p = torch.max(preds['ins_logit'], 1)
        box_index_expand = torch.zeros((batch_size*num_centroids, num_points)).cuda()
        box_index_expand = box_index_expand.scatter_(dim=1, index=data_batch['neighbour_index'].reshape([-1, num_neighbours]), src=p.reshape([-1, num_neighbours]).float())
        centroid_label = data_batch['centroid_label'].reshape(-1)

        #remove proposal < minimum_num
        minimum_box_pc_num = 8
        minimum_overlap_pc_num = 8 #1/32 * num_neighbour
        gtmin_mask = (torch.sum(box_index_expand, dim=-1) > minimum_box_pc_num)

        #remove proposal whose purity score < 0.8
        box_label_expand = torch.zeros((batch_size*num_centroids, 200)).cuda()
        box_idx_expand = tile(data_batch['ins_id'],0,num_centroids).cuda()
        box_label_expand = box_label_expand.scatter_add_(dim=1, index=box_idx_expand, src=box_index_expand).float()
        maximum_label_num, maximum_label = torch.max(box_label_expand, 1)
        centroid_label = maximum_label
        total_num = torch.sum(box_label_expand, 1)
        box_purity = maximum_label_num / (total_num+1e-6)
        box_purity_mask = box_purity > 0.8
        box_purity_valid_mask = 1 - (box_purity < 0.8)*(box_purity > 0.6)
        box_purity_valid_mask *= gtmin_mask
        box_purity_valid_mask_l2 = gtmin_mask.long()#*data_batch['centroid_valid_mask'].reshape(-1).long()
        meters.update(purity_ratio = torch.sum(box_purity_mask).float()/box_purity_mask.shape[0], purity_valid_ratio=torch.sum(box_purity_valid_mask).float()/box_purity_mask.shape[0])
        meters.update(purity_pos_num = torch.sum(box_purity_mask), purity_neg_num = torch.sum(1-box_purity_mask), purity_neg_valid_num=torch.sum(box_purity<0.6))
        centroid_valid_mask = data_batch['centroid_valid_mask'].reshape(-1).long()
        meters.update(centroid_valid_purity_ratio = torch.sum(torch.index_select(box_purity_mask, dim=0, index=centroid_valid_mask.nonzero().squeeze())).float()/torch.sum(centroid_valid_mask),centroid_nonvalid_purity_ratio = torch.sum(torch.index_select(box_purity_mask, dim=0, index=(1-centroid_valid_mask).nonzero().squeeze())).float()/torch.sum(1-centroid_valid_mask))
        purity_pred = torch.zeros([0]).type(torch.FloatTensor).cuda()

        #update pool by valid_mask
        valid_mask = gtmin_mask.long() *  box_purity_mask.long() * (centroid_label!=0).long()
        centroid_label = torch.index_select(centroid_label, dim=0, index=valid_mask.nonzero().squeeze())

        box_num = torch.sum(valid_mask.reshape(batch_size, num_centroids),1)
        cumsum_box_num = torch.cumsum(box_num, dim=0)
        cumsum_box_num = torch.cat([torch.from_numpy(np.array(0)).cuda().unsqueeze(0),cumsum_box_num],dim=0)

        #initialization
        pc_all = data_batch['points']
        centroid_label_all = centroid_label.clone()
        sub_xyz_pool1 = torch.zeros([0,3,1024]).float().cuda()
        sub_xyz_pool2 = torch.zeros([0,3,1024]).float().cuda()
        sub_context_xyz_pool1 = torch.zeros([0,3,1024]).float().cuda()
        sub_context_xyz_pool2 = torch.zeros([0,3,1024]).float().cuda()
        sub_context_context_xyz_pool = torch.zeros([0,3,2048]).float().cuda()
        sub_context_label_pool = torch.zeros([0]).float().cuda()
        sub_context_purity_pool = torch.zeros([0]).float().cuda()
        sub_label_pool = torch.zeros([0]).float().cuda()
        sub_purity_pool = torch.zeros([0]).float().cuda()
        sub_purity_xyz_pool = torch.zeros([0,3,1024]).float().cuda()
        sub_policy_purity_pool = torch.zeros([0,policy_update_bs]).float().cuda()
        sub_policy_reward_pool = torch.zeros([0,policy_update_bs]).float().cuda()
        sub_policy_xyz_pool1 = torch.zeros([0,policy_update_bs,3,1024]).float().cuda()
        sub_policy_xyz_pool2 = torch.zeros([0,policy_update_bs,3,1024]).float().cuda()
        for i in range(pc_all.shape[0]):
            bs = policy_total_bs
            BS = policy_update_bs

            pc = pc_all[i].clone()
            cur_mask_pool = box_index_expand[cumsum_box_num[i]:cumsum_box_num[i+1]].clone()
            centroid_label = centroid_label_all[cumsum_box_num[i]:cumsum_box_num[i+1]].clone()
            cover_ratio = torch.unique(cur_mask_pool.nonzero()[:,1]).shape[0]/num_points
            cur_xyz_pool, xyz_mean = mask_to_xyz(pc, cur_mask_pool)
            init_pool_size = cur_xyz_pool.shape[0]
            meters.update(cover_ratio=cover_ratio, init_pool_size=init_pool_size)
            negative_num = 0
            positive_num = 0

            #intial adjacent matrix
            inter_matrix = torch.matmul(cur_mask_pool, cur_mask_pool.transpose(0, 1))
            inter_matrix_full = inter_matrix.clone()>minimum_overlap_pc_num
            inter_matrix[torch.eye(inter_matrix.shape[0]).byte()] = 0
            pair_idx = (inter_matrix.triu()>minimum_overlap_pc_num).nonzero()
            zero_pair = torch.ones([0,2]).long()

            small_flag = False
            remote_flag = False

            model_merge.eval()
            with torch.no_grad():
                while (pair_idx.shape[0] > 0) or (remote_flag == False):
                    if pair_idx.shape[0] == 0:
                        remote_flag = True
                        small_flag = False
                        inter_matrix = 20*torch.ones([cur_mask_pool.shape[0],cur_mask_pool.shape[0]]).cuda()
                        inter_matrix[zero_pair[:,0], zero_pair[:,1]] = 0
                        inter_matrix[torch.eye(inter_matrix.shape[0]).byte()] = 0
                        pair_idx = (inter_matrix.triu()>minimum_overlap_pc_num).nonzero()
                        if pair_idx.shape[0] == 0:
                            break
                    #when there are too few pairs, we calculate the policy score matrix on all pairs
                    if pair_idx.shape[0] <= BS and small_flag == False:
                        small_flag = True
                        purity_matrix = torch.zeros(inter_matrix.shape).cuda()
                        policy_matrix = torch.zeros(inter_matrix.shape).cuda()
                        bsp = 64
                        idx = torch.arange(pair_idx.shape[0]).cuda()
                        purity_pool = torch.zeros([0]).float().cuda()
                        policy_pool = torch.zeros([0]).float().cuda()
                        for k in range(int(np.ceil(idx.shape[0]/bsp))):
                            sub_part_idx = torch.index_select(pair_idx, dim=0, index=idx[k*bsp:(k+1)*bsp])
                            part_xyz1 = torch.index_select(cur_xyz_pool, dim=0, index=sub_part_idx[:,0])
                            part_xyz2 = torch.index_select(cur_xyz_pool, dim=0, index=sub_part_idx[:,1])
                            part_xyz = torch.cat([part_xyz1,part_xyz2],-1)
                            part_xyz -= torch.mean(part_xyz,-1).unsqueeze(-1)
                            part_norm = part_xyz.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                            part_xyz /= part_norm
                            logits_purity = model_merge(part_xyz, 'purity').squeeze()
                            if len(logits_purity.shape) == 0:
                                logits_purity = logits_purity.unsqueeze(0)
                            purity_pool = torch.cat([purity_pool, logits_purity], dim=0)

                            part_xyz11 = part_xyz1 - torch.mean(part_xyz1,-1).unsqueeze(-1)
                            part_xyz22 = part_xyz2 - torch.mean(part_xyz2,-1).unsqueeze(-1)
                            part_xyz11 /= part_norm
                            part_xyz22 /= part_norm
                            logits11 = model_merge(part_xyz11, 'policy')
                            logits22 = model_merge(part_xyz22, 'policy')
                            policy_scores = model_merge(torch.cat([logits11, logits22],dim=-1), 'policy_head').squeeze()
                            if len(policy_scores.shape) == 0:
                                policy_scores = policy_scores.unsqueeze(0)
                            policy_pool = torch.cat([policy_pool, policy_scores], dim=0)

                        purity_matrix[pair_idx[:,0],pair_idx[:,1]] = purity_pool
                        policy_matrix[pair_idx[:,0],pair_idx[:,1]] = policy_pool
                        score_matrix = torch.zeros(purity_matrix.shape).cuda()
                        score_matrix[pair_idx[:,0],pair_idx[:,1]] = softmax(purity_pool*policy_pool)

                    #if there are many pairs, we randomly sample a small batch of pairs and then compute the policy score matrix thereon to select pairs into the next stage
                    #else, we select a pair with highest policy score 
                    if pair_idx.shape[0] > BS and small_flag != True:
                        perm_idx = torch.randperm(pair_idx.shape[0]).cuda()
                        perm_idx_rnd = perm_idx[:bs]
                        sub_part_idx = torch.index_select(pair_idx, dim=0, index=perm_idx[:int(BS)])
                        part_xyz1 = torch.index_select(cur_xyz_pool, dim=0, index=sub_part_idx[:,0])
                        part_xyz2 = torch.index_select(cur_xyz_pool, dim=0, index=sub_part_idx[:,1])
                        part_xyz = torch.cat([part_xyz1,part_xyz2],-1)
                        part_xyz -= torch.mean(part_xyz,-1).unsqueeze(-1)
                        part_norm = part_xyz.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                        part_xyz /= part_norm
                        logits_purity = model_merge(part_xyz, 'purity').squeeze()
                        sub_policy_purity_pool = torch.cat([sub_policy_purity_pool, logits_purity.detach().unsqueeze(0).clone()],dim=0)

                        part_xyz11 = part_xyz1 - torch.mean(part_xyz1,-1).unsqueeze(-1)
                        part_xyz22 = part_xyz2 - torch.mean(part_xyz2,-1).unsqueeze(-1)
                        part_xyz11 /= part_norm
                        part_xyz22 /= part_norm
                        logits11 = model_merge(part_xyz11, 'policy')
                        logits22 = model_merge(part_xyz22, 'policy')
                        policy_scores = model_merge(torch.cat([logits11, logits22],dim=-1), 'policy_head').squeeze()
                        sub_policy_xyz_pool1 = torch.cat([sub_policy_xyz_pool1, part_xyz11.unsqueeze(0).clone()], dim=0)
                        sub_policy_xyz_pool2 = torch.cat([sub_policy_xyz_pool2, part_xyz22.unsqueeze(0).clone()], dim=0)
                        if sub_policy_xyz_pool1.shape[0] > 64:
                            policy_xyz_pool1 = torch.cat([policy_xyz_pool1, sub_policy_xyz_pool1.cpu().clone()], dim=0)
                            policy_xyz_pool2 = torch.cat([policy_xyz_pool2, sub_policy_xyz_pool2.cpu().clone()], dim=0)
                            sub_policy_xyz_pool1 = torch.zeros([0,policy_update_bs,3,1024]).float().cuda()
                            sub_policy_xyz_pool2 = torch.zeros([0,policy_update_bs,3,1024]).float().cuda()
                        score = softmax(logits_purity*policy_scores)

                        part_label1 = torch.index_select(centroid_label, dim=0, index=sub_part_idx[:,0])
                        part_label2 = torch.index_select(centroid_label, dim=0, index=sub_part_idx[:,1])
                        siamese_label_gt = (part_label1 == part_label2)*(1 - (part_label1 == -1))*(1 - (part_label2 == -1))*(logits_purity>0.8)
                        sub_policy_reward_pool = torch.cat([sub_policy_reward_pool, siamese_label_gt.unsqueeze(0).float().clone()], dim=0)
                        loss_policy = -torch.sum(score*(siamese_label_gt.float()))
                        meters.update(loss_policy =loss_policy)

                        #we also introduce certain random samples to encourage exploration
                        _, rank_idx = torch.topk(score,bs,largest=True,sorted=False)
                        perm_idx = perm_idx[rank_idx]
                        perm_idx = torch.cat([perm_idx[:policy_total_bs-rnum], perm_idx_rnd[:rnum]], dim=0)
                        if cur_epoch == 1 and iteration < 128:
                            perm_idx = torch.randperm(pair_idx.shape[0]).cuda()
                            perm_idx = perm_idx[:policy_total_bs]
                    else:
                        score = score_matrix[pair_idx[:,0],pair_idx[:,1]]
                        _, rank_idx = torch.topk(score,1,largest=True,sorted=False)
                        perm_idx = rank_idx
                        if len(perm_idx.shape) == 0:
                            perm_idx = perm_idx.unsqueeze(0)

                        if cur_epoch == 1 and iteration < 128:
                            perm_idx = torch.randperm(pair_idx.shape[0]).cuda()
                            perm_idx = perm_idx[:1]

                    #send the selected pairs into verification network
                    sub_part_idx = torch.index_select(pair_idx, dim=0, index=perm_idx[:bs])
                    part_xyz1 = torch.index_select(cur_xyz_pool, dim=0, index=sub_part_idx[:,0])
                    part_xyz2 = torch.index_select(cur_xyz_pool, dim=0, index=sub_part_idx[:,1])
                    part_mask11 = torch.index_select(cur_mask_pool, dim=0, index=sub_part_idx[:,0])
                    part_mask22 = torch.index_select(cur_mask_pool, dim=0, index=sub_part_idx[:,1])
                    part_label1 = torch.index_select(centroid_label, dim=0, index=sub_part_idx[:,0])
                    part_label2 = torch.index_select(centroid_label, dim=0, index=sub_part_idx[:,1])
                    new_part_mask = 1-(1-part_mask11)*(1-part_mask22)
                    box_label_expand = torch.zeros((new_part_mask.shape[0], 200)).cuda()
                    box_idx_expand = tile(data_batch['ins_id'][i].unsqueeze(0),0,new_part_mask.shape[0]).cuda()
                    box_label_expand = box_label_expand.scatter_add_(dim=1, index=box_idx_expand, src=new_part_mask).float()
                    maximum_label_num, maximum_label = torch.max(box_label_expand, 1)
                    total_num = torch.sum(box_label_expand, 1)
                    box_purity = maximum_label_num / (total_num+1e-6)
                    sub_purity_pool = torch.cat([sub_purity_pool, box_purity.clone()], dim=0)
                    purity_xyz, xyz_mean = mask_to_xyz(pc, new_part_mask)
                    purity_xyz -= xyz_mean
                    purity_xyz /=(purity_xyz+1e-6).norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                    sub_purity_xyz_pool = torch.cat([sub_purity_xyz_pool, purity_xyz.clone()],dim=0)

                    siamese_label_gt = (part_label1 == part_label2)*(1 - (part_label1 == -1))*(1 - (part_label2 == -1))*(box_purity > 0.8)
                    negative_num += torch.sum(siamese_label_gt == 0)
                    positive_num += torch.sum(siamese_label_gt == 1)

                    #save data
                    sub_xyz_pool1 = torch.cat([sub_xyz_pool1, part_xyz1.clone()], dim=0)
                    sub_xyz_pool2 = torch.cat([sub_xyz_pool2, part_xyz2.clone()], dim=0)
                    sub_label_pool = torch.cat([sub_label_pool, siamese_label_gt.clone().float()], dim=0)

                    #renorm
                    part_xyz = torch.cat([part_xyz1,part_xyz2],-1)
                    part_xyz -= torch.mean(part_xyz,-1).unsqueeze(-1)
                    part_xyz11 = part_xyz1 - torch.mean(part_xyz1,-1).unsqueeze(-1)
                    part_xyz22 = part_xyz2 - torch.mean(part_xyz2,-1).unsqueeze(-1)
                    part_xyz11 /=part_xyz11.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                    part_xyz22 /=part_xyz22.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                    part_xyz /=part_xyz.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

                    #save data
                    if remote_flag or cur_xyz_pool.shape[0] <= 32:
                        context_idx1 = torch.index_select(inter_matrix_full,dim=0,index=sub_part_idx[:,0])
                        context_idx2 = torch.index_select(inter_matrix_full,dim=0,index=sub_part_idx[:,1])
                        context_mask1 = (torch.matmul(context_idx1.float(), cur_mask_pool)>0).float()
                        context_mask2 = (torch.matmul(context_idx2.float(), cur_mask_pool)>0).float()
                        context_mask = ((context_mask1+context_mask2)>0).float()
                        context_xyz, xyz_mean = mask_to_xyz(pc, context_mask, sample_num=2048)
                        context_xyz = context_xyz - xyz_mean
                        context_xyz /= context_xyz.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                        sub_context_context_xyz_pool = torch.cat([sub_context_context_xyz_pool, context_xyz.clone()], dim=0)
                        sub_context_xyz_pool1 = torch.cat([sub_context_xyz_pool1, part_xyz1.clone()], dim=0)
                        sub_context_xyz_pool2 = torch.cat([sub_context_xyz_pool2, part_xyz2.clone()], dim=0)
                        sub_context_label_pool = torch.cat([sub_context_label_pool, siamese_label_gt.clone().float()], dim=0)
                        sub_context_purity_pool =  torch.cat([sub_context_purity_pool, box_purity.clone()], dim=0)

                    #at the very beginning, we group pairs according to ground-truth
                    if (cur_epoch == 1 and iteration < 128) or (cur_checkpoint == 'no_checkpoint'):
                        siamese_label = (part_label1 == part_label2)
                    #if we have many sub-parts in the pool, we use the binary branch to predict
                    elif remote_flag or cur_xyz_pool.shape[0] <= 32:
                        logits1 = model_merge(part_xyz11,'backbone')
                        logits2 = model_merge(part_xyz22,'backbone')
                        context_logits = model_merge(context_xyz,'backbone2')
                        merge_logits = model_merge(torch.cat([part_xyz, torch.cat([logits1.unsqueeze(-1).expand(-1,-1,part_xyz1.shape[-1]), logits2.unsqueeze(-1).expand(-1,-1,part_xyz2.shape[-1])], dim=-1), torch.cat([context_logits.unsqueeze(-1).expand(-1,-1,part_xyz.shape[-1])], dim=-1)], dim=1), 'head2')
                        _, p = torch.max(merge_logits, 1)
                        siamese_label = p
                    #if there are too few sub-parts in the pool, we use the context branch to predict
                    else:
                        logits1 = model_merge(part_xyz11,'backbone')
                        logits2 = model_merge(part_xyz22,'backbone')
                        merge_logits = model_merge(torch.cat([part_xyz, torch.cat([logits1.unsqueeze(-1).expand(-1,-1,part_xyz1.shape[-1]), logits2.unsqueeze(-1).expand(-1,-1,part_xyz2.shape[-1])], dim=-1)], dim=1), 'head')
                        _, p = torch.max(merge_logits, 1)
                        siamese_label = p


                    #group sub-parts according to the prediction
                    merge_idx1 = torch.index_select(sub_part_idx[:,0], dim=0, index=siamese_label.nonzero().squeeze())
                    merge_idx2 = torch.index_select(sub_part_idx[:,1], dim=0, index=siamese_label.nonzero().squeeze())
                    merge_idx = torch.unique(torch.cat([merge_idx1, merge_idx2], dim=0))
                    nonmerge_idx1 = torch.index_select(sub_part_idx[:,0], dim=0, index=(1-siamese_label).nonzero().squeeze())
                    nonmerge_idx2 = torch.index_select(sub_part_idx[:,1], dim=0, index=(1-siamese_label).nonzero().squeeze())
                    part_mask1 = torch.index_select(cur_mask_pool, dim=0, index=merge_idx1)
                    part_mask2 = torch.index_select(cur_mask_pool, dim=0, index=merge_idx2)
                    new_part_mask = 1-(1-part_mask1)*(1-part_mask2)
                    new_part_label = torch.index_select(part_label1, dim=0, index=siamese_label.nonzero().squeeze()).long()
                    new_part_label_invalid = torch.index_select(siamese_label_gt, dim=0, index=siamese_label.nonzero().squeeze()).long()
                    new_part_label = new_part_label*new_part_label_invalid + -1*(1-new_part_label_invalid)

                    #sometimes, we may obtain several identical sub-parts
                    #for those, we only keep one
                    equal_matrix = torch.matmul(new_part_mask,1-new_part_mask.transpose(0,1))+torch.matmul(1-new_part_mask,new_part_mask.transpose(0,1))
                    equal_matrix[torch.eye(equal_matrix.shape[0]).byte()]=1
                    fid = (equal_matrix==0).nonzero()
                    if fid.shape[0] > 0:
                        flag = torch.ones(equal_matrix.shape[0])
                        for k in range(flag.shape[0]):
                            if flag[k] != 0:
                                flag[fid[:,1][fid[:,0]==k]] = 0
                        new_part_mask = torch.index_select(new_part_mask, dim=0, index=flag.nonzero().squeeze().cuda())
                        new_part_label = torch.index_select(new_part_label, dim=0, index=flag.nonzero().squeeze().cuda())

                    new_part_xyz, xyz_mean = mask_to_xyz(pc, new_part_mask)

                    #when there are too few pairs, update the policy score matrix so that we do not need to calculate the whole matrix everytime
                    if small_flag and (new_part_mask.shape[0] > 0):
                        overlap_idx = (torch.matmul(cur_mask_pool, new_part_mask.transpose(0,1))>minimum_overlap_pc_num).nonzero().squeeze()
                        if overlap_idx.shape[0] > 0:
                            if len(overlap_idx.shape) == 1:
                                overlap_idx = overlap_idx.unsqueeze(0)
                            part_xyz1 = torch.index_select(cur_xyz_pool, dim=0, index=overlap_idx[:,0])
                            part_xyz2 = tile(new_part_xyz, 0, overlap_idx.shape[0])
                            part_xyz = torch.cat([part_xyz1,part_xyz2],-1)
                            part_xyz -= torch.mean(part_xyz,-1).unsqueeze(-1)
                            part_norm = part_xyz.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                            part_xyz /= part_norm
                            overlap_purity_scores = model_merge(part_xyz, 'purity').squeeze()

                            part_xyz11 = part_xyz1 - torch.mean(part_xyz1,-1).unsqueeze(-1)
                            part_xyz22 = part_xyz2 - torch.mean(part_xyz2,-1).unsqueeze(-1)
                            part_xyz11 /= part_norm
                            part_xyz22 /= part_norm
                            logits11 = model_merge(part_xyz11, 'policy')
                            logits22 = model_merge(part_xyz22, 'policy')
                            overlap_policy_scores = model_merge(torch.cat([logits11, logits22],dim=-1), 'policy_head').squeeze()

                            tmp_purity_arr = torch.zeros([purity_matrix.shape[0]]).cuda()
                            tmp_policy_arr = torch.zeros([policy_matrix.shape[0]]).cuda()
                            tmp_purity_arr[overlap_idx[:,0]] = overlap_purity_scores
                            tmp_policy_arr[overlap_idx[:,0]] = overlap_policy_scores
                            purity_matrix = torch.cat([purity_matrix,tmp_purity_arr.unsqueeze(1)],dim=1)
                            policy_matrix = torch.cat([policy_matrix,tmp_policy_arr.unsqueeze(1)],dim=1)
                            purity_matrix = torch.cat([purity_matrix,torch.zeros(purity_matrix.shape[1]).cuda().unsqueeze(0)])
                            policy_matrix = torch.cat([policy_matrix,torch.zeros(policy_matrix.shape[1]).cuda().unsqueeze(0)])
                        else:
                            purity_matrix = torch.cat([purity_matrix,torch.zeros(purity_matrix.shape[0]).cuda().unsqueeze(1)],dim=1)
                            policy_matrix = torch.cat([policy_matrix,torch.zeros(policy_matrix.shape[0]).cuda().unsqueeze(1)],dim=1)
                            purity_matrix = torch.cat([purity_matrix,torch.zeros(purity_matrix.shape[1]).cuda().unsqueeze(0)])
                            policy_matrix = torch.cat([policy_matrix,torch.zeros(policy_matrix.shape[1]).cuda().unsqueeze(0)])

                    #update cur_pool, add new parts, pick out merged input pairs
                    cur_mask_pool = torch.cat([cur_mask_pool, new_part_mask], dim=0)
                    cur_xyz_pool = torch.cat([cur_xyz_pool, new_part_xyz], dim=0)
                    centroid_label = torch.cat([centroid_label, new_part_label], dim=0)
                    cur_pool_size = cur_mask_pool.shape[0]
                    new_mask = torch.ones([cur_pool_size])
                    new_mask[merge_idx] = 0
                    new_idx = new_mask.nonzero().squeeze().cuda()
                    cur_xyz_pool = torch.index_select(cur_xyz_pool, dim=0, index=new_idx)
                    cur_mask_pool = torch.index_select(cur_mask_pool, dim=0, index=new_idx)
                    centroid_label = torch.index_select(centroid_label, dim=0, index=new_idx)
                    inter_matrix = torch.matmul(cur_mask_pool, cur_mask_pool.transpose(0, 1))
                    inter_matrix_full = inter_matrix.clone()>minimum_overlap_pc_num
                    if remote_flag:
                        inter_matrix = 20*torch.ones([cur_mask_pool.shape[0],cur_mask_pool.shape[0]]).cuda()
                    #update zero_matrix
                    zero_matrix = torch.zeros([cur_pool_size, cur_pool_size])
                    zero_matrix[zero_pair[:,0], zero_pair[:,1]] = 1
                    zero_matrix[nonmerge_idx1, nonmerge_idx2] = 1
                    zero_matrix[nonmerge_idx2, nonmerge_idx1] = 1
                    zero_matrix = torch.index_select(zero_matrix, dim=0, index=new_idx.cpu())
                    zero_matrix = torch.index_select(zero_matrix, dim=1, index=new_idx.cpu())
                    zero_pair = zero_matrix.nonzero()
                    inter_matrix[zero_pair[:,0], zero_pair[:,1]] = 0
                    inter_matrix[torch.eye(inter_matrix.shape[0]).byte()] = 0
                    pair_idx = (inter_matrix.triu()>minimum_overlap_pc_num).nonzero()
                    if small_flag == True:
                        purity_matrix = torch.index_select(purity_matrix, dim=0, index=new_idx)
                        purity_matrix = torch.index_select(purity_matrix, dim=1, index=new_idx)
                        policy_matrix = torch.index_select(policy_matrix, dim=0, index=new_idx)
                        policy_matrix = torch.index_select(policy_matrix, dim=1, index=new_idx)
                        score_matrix = torch.zeros(purity_matrix.shape).cuda()
                        score_idx = pair_idx
                        score_matrix[score_idx[:,0], score_idx[:,1]] = softmax(purity_matrix[score_idx[:,0], score_idx[:,1]] * policy_matrix[score_idx[:,0], score_idx[:,1]])
                final_pool_size = negative_num + positive_num
                meters.update(final_pool_size=final_pool_size,negative_num=negative_num, positive_num=positive_num)
        xyz_pool1 = torch.cat([xyz_pool1, sub_xyz_pool1.cpu().clone()],dim=0)
        xyz_pool2 = torch.cat([xyz_pool2, sub_xyz_pool2.cpu().clone()],dim=0)
        label_pool = torch.cat([label_pool, sub_label_pool.cpu().clone()], dim=0)
        context_context_xyz_pool = torch.cat([context_context_xyz_pool, sub_context_context_xyz_pool.cpu().clone()],dim=0)
        context_xyz_pool1 = torch.cat([context_xyz_pool1, sub_context_xyz_pool1.cpu().clone()],dim=0)
        context_xyz_pool2 = torch.cat([context_xyz_pool2, sub_context_xyz_pool2.cpu().clone()],dim=0)
        context_label_pool = torch.cat([context_label_pool, sub_context_label_pool.cpu().clone()], dim=0)
        context_purity_pool = torch.cat([context_purity_pool, sub_context_purity_pool.cpu().clone()], dim=0)
        purity_purity_pool = torch.cat([purity_purity_pool, sub_purity_pool.cpu().clone()], dim=0)
        purity_xyz_pool = torch.cat([purity_xyz_pool, sub_purity_xyz_pool.cpu().clone()], dim=0)
        policy_purity_pool = torch.cat([policy_purity_pool, sub_policy_purity_pool.cpu().clone()], dim=0)
        policy_reward_pool = torch.cat([policy_reward_pool, sub_policy_reward_pool.cpu().clone()], dim=0)
        policy_xyz_pool1 = torch.cat([policy_xyz_pool1, sub_policy_xyz_pool1.cpu().clone()], dim=0)
        policy_xyz_pool2 = torch.cat([policy_xyz_pool2, sub_policy_xyz_pool2.cpu().clone()], dim=0)
        produce_time = time.time() - end

        #condition
        if context_xyz_pool1.shape[0] > 10000:
            rbuffer = dict()
            rbuffer['xyz_pool1'] = xyz_pool1
            rbuffer['xyz_pool2'] = xyz_pool2
            rbuffer['context_xyz_pool1'] = context_xyz_pool1
            rbuffer['context_xyz_pool2'] = context_xyz_pool2
            rbuffer['context_context_xyz_pool'] = context_context_xyz_pool
            rbuffer['context_label_pool'] = context_label_pool
            rbuffer['context_purity_pool'] = context_purity_pool
            rbuffer['label_pool'] = label_pool
            rbuffer['purity_purity_pool'] = purity_purity_pool
            rbuffer['purity_xyz_pool'] = purity_xyz_pool
            rbuffer['policy_purity_pool'] = policy_purity_pool
            rbuffer['policy_reward_pool'] = policy_reward_pool
            rbuffer['policy_xyz_pool1'] = policy_xyz_pool1
            rbuffer['policy_xyz_pool2'] = policy_xyz_pool2
            torch.save(rbuffer, os.path.join(output_dir_merge, 'buffer', '%d_%d.pt'%(cur_epoch, iteration)))
            buffer_f = open(buffer_txt, 'w')
            buffer_f.write('%d_%d'%(cur_epoch, iteration))
            buffer_f.close()
            p = Popen('rm -rf %s'%(os.path.join(output_dir_merge, 'buffer', '%d_%d.pt'%(cur_epoch, old_iteration))), shell=True)
            old_iteration = iteration
            p = Popen('rm -rf %s_*'%(os.path.join(output_dir_merge, 'buffer', '%d'%(cur_epoch-1))), shell=True)

            xyz_pool1 = torch.zeros([0,3,1024]).float()
            xyz_pool2 = torch.zeros([0,3,1024]).float()
            context_xyz_pool1 = torch.zeros([0,3,1024]).float()
            context_xyz_pool2 = torch.zeros([0,3,1024]).float()
            context_context_xyz_pool = torch.zeros([0,3,2048]).float()
            context_label_pool = torch.zeros([0]).float()
            context_purity_pool = torch.zeros([0]).float()
            label_pool = torch.zeros([0]).float()
            purity_purity_pool = torch.zeros([0]).float()
            purity_xyz_pool = torch.zeros([0,3,1024]).float()
            policy_purity_pool = torch.zeros([0,policy_update_bs]).float()
            policy_reward_pool = torch.zeros([0,policy_update_bs]).float()
            policy_xyz_pool1 = torch.zeros([0,policy_update_bs,3,1024]).float()
            policy_xyz_pool2 = torch.zeros([0,policy_update_bs,3,1024]).float()



        batch_time = time.time() - end
        end = time.time()
        meters.update(data_time=data_time, produce_time=produce_time)

        if log_period > 0 and iteration % log_period == 0:
            logger.info(
                meters.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=iteration,
                    meters=str(meters),
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )
    return meters

def train(cfg, output_dir='', output_dir_merge='', output_dir_refine=''):
    output_dir_buffer = os.path.join(output_dir_merge, 'buffer')
    if not os.path.exists(output_dir_buffer):
        os.mkdir(output_dir_buffer)
    buffer_txt = os.path.join(output_dir_merge, 'last_buffer')
    buffer_file = open(buffer_txt, 'w')
    buffer_file.write('start')
    buffer_file.close()

    logger = logging.getLogger('shaper.train')

    # build model
    set_random_seed(cfg.RNG_SEED)
    model, loss_fn, train_metric, val_metric = build_model(cfg)
    logger.info('Build model:\n{}'.format(str(model)))
    model = nn.DataParallel(model).cuda()

    model_merge = nn.DataParallel(PointNetCls(in_channels=3, out_channels=128)).cuda()

    # build optimizer
    optimizer = build_optimizer(cfg, model)
    optimizer_embed = build_optimizer(cfg, model_merge)

    # build lr scheduler
    scheduler = build_scheduler(cfg, optimizer)
    scheduler_embed = build_scheduler(cfg, optimizer_embed)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                logger=logger)
    checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_embed = Checkpointer(model_merge,
                                optimizer=optimizer_embed,
                                scheduler=scheduler_embed,
                                save_dir=output_dir_merge,
                                logger=logger)
    checkpoint_data_embed = checkpointer_embed.load(cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)

    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build freezer
    if cfg.TRAIN.FROZEN_PATTERNS:
        freezer = Freezer(model, cfg.TRAIN.FROZEN_PATTERNS)
        freezer.freeze(verbose=True)  # sanity check
    else:
        freezer = None

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader = build_dataloader(cfg, mode='train')
    val_period = cfg.TRAIN.VAL_PERIOD
    val_dataloader = build_dataloader(cfg, mode='val') if val_period > 0 else None

    # build tensorboard logger (optionally by comment)
    tensorboard_logger = TensorboardLogger(output_dir_merge)

    # train
    max_epoch = 20000
    start_epoch = checkpoint_data_embed.get('epoch', 0)
    best_metric_name = 'best_{}'.format(cfg.TRAIN.VAL_METRIC)
    best_metric = checkpoint_data_embed.get(best_metric_name, None)
    logger.info('Start training from epoch {}'.format(start_epoch))
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch + 1
        scheduler_embed.step()
        start_time = time.time()
        train_meters = train_one_epoch(model,
                                       model_merge,
                                       loss_fn,
                                       train_metric,
                                       train_dataloader,
                                       cur_epoch,
                                       optimizer=optimizer,
                                       optimizer_embed=optimizer_embed,
                                       checkpointer_embed = checkpointer_embed,
                                       output_dir_merge = output_dir_merge,
                                       max_grad_norm=cfg.OPTIMIZER.MAX_GRAD_NORM,
                                       freezer=freezer,
                                       log_period=cfg.TRAIN.LOG_PERIOD,
                                       )

    logger.info('Best val-{} = {}'.format(cfg.TRAIN.VAL_METRIC, best_metric))
    return model

def main():
    print('begin program\n')
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from partnet.config.ins_seg_3d import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace('configs', 'outputs')
        output_dir_merge = output_dir.replace('@', config_path)+'_merge'
        os.makedirs(output_dir_merge, exist_ok=True)
        output_dir = osp.join('outputs/stage1/', cfg.DATASET.PartNetInsSeg.TRAIN.stage1)
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('shaper', output_dir_merge, prefix='train')
    logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
    logger.info(args)


    logger.info('Loaded configuration file {}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.TASK == 'ins_seg_3d'
    train(cfg, output_dir, output_dir_merge)


if __name__ == '__main__':
    main()
