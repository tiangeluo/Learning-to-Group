#!/usr/bin/env python
"""Test point cloud part segmentation models"""

from __future__ import division
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(__file__) + '/..')

import argparse
import logging
import time
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from core.config import purge_cfg
from core.utils.checkpoint import Checkpointer
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.torch_util import set_random_seed

from partnet.models.build import build_model
from partnet.data.build import build_dataloader, parse_augmentations
from shaper.data import transforms as T
from IPython import embed
import shaper.models.pointnet2.functions as _F
from partnet.models.pn2 import PointNetCls
import torch.nn.functional as F

from core.nn.functional import cross_entropy
from core.nn.functional import focal_loss, l2_loss
import copy
import h5py

import matplotlib.pyplot as plt
import matplotlib

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Learning Test')
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


def tensor2list(input_dict):
    return {k: v.detach().cpu().numpy()[0].tolist() for k, v in input_dict.items()}

def save_h5(fn, mask, valid, conf):
    fout = h5py.File(fn, 'w')
    fout.create_dataset('mask', data=mask, compression='gzip', compression_opts=4, dtype='bool')
    fout.create_dataset('valid', data=valid, compression='gzip', compression_opts=4, dtype='bool')
    fout.create_dataset('conf', data=conf, compression='gzip', compression_opts=4, dtype='float32')
    fout.close()

def save_shape(filename, pred_dict, data_dict):
    out_dict = dict()
    out_dict.update(tensor2list(data_dict))
    out_dict.update(tensor2list(pred_dict))
    #out_dict.update(tensor2list(pred_dict2))
    with open(filename, 'w') as f:
        json.dump(out_dict, f)
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

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)

def test(cfg, output_dir='', output_dir_merge='', output_dir_save=''):
    logger = logging.getLogger('shaper.test')

    # build model
    model, loss_fn, _, val_metric = build_model(cfg)
    model = nn.DataParallel(model).cuda()
    model_merge = nn.DataParallel(PointNetCls(in_channels=3, out_channels=128)).cuda()

    # build checkpointer
    checkpointer = Checkpointer(model, save_dir=output_dir, logger=logger)
    checkpointer_merge = Checkpointer(model_merge, save_dir=output_dir_merge, logger=logger)

    if cfg.TEST.WEIGHT:
        # load weight if specified
        weight_path = cfg.TEST.WEIGHT.replace('@', output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer.load(None, resume=True)
        checkpointer_merge.load(None, resume=True)
        #checkpointer_refine.load(None, resume=True)

    # build data loader
    test_dataloader = build_dataloader(cfg, mode='test')
    test_dataset = test_dataloader.dataset

    assert cfg.TEST.BATCH_SIZE == 1, '{} != 1'.format(cfg.TEST.BATCH_SIZE)
    save_fig_dir = osp.join(output_dir_save, 'test_fig')
    os.makedirs(save_fig_dir, exist_ok=True)
    save_fig_dir_size = osp.join(save_fig_dir, 'size')
    os.makedirs(save_fig_dir_size, exist_ok=True)
    save_fig_dir_gt = osp.join(save_fig_dir, 'gt')
    os.makedirs(save_fig_dir_gt, exist_ok=True)

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #
    model.eval()
    model_merge.eval()
    loss_fn.eval()
    softmax = nn.Softmax()
    set_random_seed(cfg.RNG_SEED)

    NUM_POINT = 10000
    n_shape = len(test_dataloader)
    NUM_INS = 200
    out_mask = np.zeros((n_shape, NUM_INS, NUM_POINT), dtype=np.bool)
    out_valid = np.zeros((n_shape, NUM_INS), dtype=np.bool)
    out_conf = np.ones((n_shape, NUM_INS), dtype=np.float32)

    meters = MetricLogger(delimiter='  ')
    meters.bind(val_metric)
    tot_purity_error_list = list()
    tot_purity_error_small_list = list()
    tot_purity_error_large_list = list()
    tot_pred_acc = list()
    tot_pred_small_acc = list()
    tot_pred_large_acc = list()
    tot_mean_rela_size_list = list()
    tot_mean_policy_label0 = list()
    tot_mean_label_policy0 = list()
    tot_mean_policy_label0_large = list()
    tot_mean_policy_label0_small = list()
    tot_mean_label_policy0_large = list()
    tot_mean_label_policy0_small = list()
    with torch.no_grad():
        start_time = time.time()
        end = start_time
        for iteration, data_batch in enumerate(test_dataloader):
            print(iteration)

            data_time = time.time() - end
            iter_start_time = time.time()

            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

            preds = model(data_batch)
            loss_dict = loss_fn(preds, data_batch)
            meters.update(**loss_dict)
            val_metric.update_dict(preds, data_batch)

            #extraction box features
            batch_size, _, num_centroids, num_neighbours = data_batch['neighbour_xyz'].shape
            num_points = data_batch['points'].shape[-1]

            #batch_size, num_centroid, num_neighbor
            _, p = torch.max(preds['ins_logit'], 1)
            box_index_expand = torch.zeros((batch_size*num_centroids, num_points)).cuda()
            box_index_expand = box_index_expand.scatter_(dim=1, index=data_batch['neighbour_index'].reshape([-1, num_neighbours]), src=p.reshape([-1, num_neighbours]).float())
            #centroid_label = data_batch['centroid_label'].reshape(-1)

            minimum_box_pc_num = 16
            minimum_overlap_pc_num = 16 #1/16 * num_neighbour
            gtmin_mask = (torch.sum(box_index_expand, dim=-1) > minimum_box_pc_num)

            #remove purity < 0.8
            box_label_expand = torch.zeros((batch_size*num_centroids, 200)).cuda()
            purity_pred = torch.zeros([0]).type(torch.LongTensor).cuda()
            purity_pred_float = torch.zeros([0]).type(torch.FloatTensor).cuda()

            for i in range(batch_size):
                cur_xyz_pool, xyz_mean = mask_to_xyz(data_batch['points'][i], box_index_expand.view(batch_size,num_centroids,num_points)[i], sample_num=512)
                cur_xyz_pool -= xyz_mean
                cur_xyz_pool /=(cur_xyz_pool+1e-6).norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                
                logits_purity = model_merge(cur_xyz_pool, 'purity')
                p = (logits_purity > 0.8).long().squeeze()
                purity_pred = torch.cat([purity_pred,p])
                purity_pred_float = torch.cat([purity_pred_float,logits_purity.squeeze()])

            p_thresh = 0.8
            purity_pred = purity_pred_float > p_thresh
            #in case remove too much
            while(torch.sum(purity_pred) < 48):
                p_thresh = p_thresh-0.01
                purity_pred = purity_pred_float > p_thresh
            valid_mask = gtmin_mask.long() *  purity_pred.long()
            box_index_expand = torch.index_select(box_index_expand, dim=0, index=valid_mask.nonzero().squeeze())

            box_num = torch.sum(valid_mask.reshape(batch_size, num_centroids),1)
            cumsum_box_num = torch.cumsum(box_num, dim=0)
            cumsum_box_num = torch.cat([torch.from_numpy(np.array(0)).cuda().unsqueeze(0),cumsum_box_num],dim=0)

            with torch.no_grad():
                pc_all = data_batch['points']
                xyz_pool1 = torch.zeros([0,3,1024]).float().cuda()
                xyz_pool2 = torch.zeros([0,3,1024]).float().cuda()
                label_pool = torch.zeros([0]).float().cuda()
                for i in range(pc_all.shape[0]):
                    bs = 1
                    pc = pc_all[i].clone()
                    cur_mask_pool = box_index_expand[cumsum_box_num[i]:cumsum_box_num[i+1]].clone()
                    cover_ratio = torch.unique(cur_mask_pool.nonzero()[:,1]).shape[0]/num_points
                    #print(iteration, cover_ratio)
                    cur_xyz_pool, xyz_mean = mask_to_xyz(pc, cur_mask_pool)
                    subpart_pool = cur_xyz_pool.clone()
                    subpart_mask_pool = cur_mask_pool.clone()
                    init_pool_size = cur_xyz_pool.shape[0]
                    meters.update(cover_ratio=cover_ratio, init_pool_size=init_pool_size) 
                    negative_num = 0
                    positive_num = 0

                    #remove I
                    inter_matrix = torch.matmul(cur_mask_pool, cur_mask_pool.transpose(0, 1))
                    inter_matrix_full = inter_matrix.clone()>minimum_overlap_pc_num
                    inter_matrix[torch.eye(inter_matrix.shape[0]).byte()] = 0
                    pair_idx = (inter_matrix.triu()>minimum_overlap_pc_num).nonzero()
                    zero_pair = torch.ones([0,2]).long()
                    purity_matrix = torch.zeros(inter_matrix.shape).cuda()
                    policy_matrix = torch.zeros(inter_matrix.shape).cuda()
                    bsp = 64
                    idx = torch.arange(pair_idx.shape[0]).cuda()
                    #calculate initial policy score matrix
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
                    meters.update(initial_pair_num = pair_idx.shape[0])
                    iteration_num = 0

                    #info
                    policy_list = []
                    purity_list = []
                    gt_purity_list = []
                    gt_label_list = []
                    pred_label_list = []
                    size_list=[]
                    relative_size_list=[]

                    while pair_idx.shape[0] > 0:
                        iteration_num += 1

                        #everytime select the pair with highest score
                        score_arr = score_matrix[pair_idx[:,0], pair_idx[:,1]]
                        highest_score, rank_idx = torch.topk(score_arr,1,largest=True,sorted=False)
                        perm_idx = rank_idx
                        assert highest_score == score_matrix[pair_idx[rank_idx,0],pair_idx[rank_idx,1]]

                        sub_part_idx = torch.index_select(pair_idx, dim=0, index=perm_idx[:bs])
                        purity_score = purity_matrix[sub_part_idx[:,0],sub_part_idx[:,1]]
                        policy_score = policy_matrix[sub_part_idx[:,0],sub_part_idx[:,1]]

                        #info
                        policy_list.append(policy_score.cpu().data.numpy()[0])
                        purity_list.append(purity_score.cpu().data.numpy()[0])

                        part_xyz1 = torch.index_select(cur_xyz_pool, dim=0, index=sub_part_idx[:,0])
                        part_xyz2 = torch.index_select(cur_xyz_pool, dim=0, index=sub_part_idx[:,1])
                        part_xyz = torch.cat([part_xyz1,part_xyz2],-1)
                        part_xyz -= torch.mean(part_xyz,-1).unsqueeze(-1)
                        part_xyz1 -= torch.mean(part_xyz1,-1).unsqueeze(-1)
                        part_xyz2 -= torch.mean(part_xyz2,-1).unsqueeze(-1)
                        part_xyz1 /=part_xyz1.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                        part_xyz2 /=part_xyz2.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                        part_xyz /=part_xyz.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                        part_mask11 = torch.index_select(cur_mask_pool, dim=0, index=sub_part_idx[:,0])
                        part_mask22 = torch.index_select(cur_mask_pool, dim=0, index=sub_part_idx[:,1])
                        context_idx1 = torch.index_select(inter_matrix_full,dim=0,index=sub_part_idx[:,0])
                        context_idx2 = torch.index_select(inter_matrix_full,dim=0,index=sub_part_idx[:,1])
                        context_mask1 = (torch.matmul(context_idx1.float(), cur_mask_pool)>0).float()
                        context_mask2 = (torch.matmul(context_idx2.float(), cur_mask_pool)>0).float()
                        context_mask = ((context_mask1+context_mask2)>0).float()
                        context_xyz, xyz_mean = mask_to_xyz(pc, context_mask, sample_num=2048)
                        context_xyz = context_xyz - xyz_mean
                        context_xyz /= context_xyz.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

                        if (cur_xyz_pool.shape[0] >= 32):
                            logits1 = model_merge(part_xyz1,'backbone')
                            logits2 = model_merge(part_xyz2,'backbone')
                            merge_logits = model_merge(torch.cat([part_xyz, torch.cat([logits1.unsqueeze(-1).expand(-1,-1,part_xyz1.shape[-1]), logits2.unsqueeze(-1).expand(-1,-1,part_xyz2.shape[-1])], dim=-1)], dim=1), 'head')
                        else:
                            logits1 = model_merge(part_xyz1,'backbone')
                            logits2 = model_merge(part_xyz2,'backbone')
                            context_logits = model_merge(context_xyz,'backbone2')
                            merge_logits = model_merge(torch.cat([part_xyz, torch.cat([logits1.unsqueeze(-1).expand(-1,-1,part_xyz1.shape[-1]), logits2.unsqueeze(-1).expand(-1,-1,part_xyz2.shape[-1])], dim=-1), torch.cat([context_logits.unsqueeze(-1).expand(-1,-1,part_xyz.shape[-1])], dim=-1)], dim=1), 'head2')

                        _, p = torch.max(merge_logits, 1)
                        siamese_label = p*((purity_score>p_thresh).long())
                        negative_num += torch.sum(siamese_label == 0)
                        positive_num += torch.sum(siamese_label == 1)
                        pred_label_list.append(siamese_label.cpu().data.numpy())

                        #info
                        new_part_mask = 1-(1-part_mask11)*(1-part_mask22)
                        size_list.append(torch.sum(new_part_mask).cpu().data.numpy())
                        size1 = torch.sum(part_mask11).cpu().data.numpy()
                        size2 = torch.sum(part_mask22).cpu().data.numpy()
                        relative_size_list.append(size1/size2+size2/size1)

                        #update info
                        merge_idx1 = torch.index_select(sub_part_idx[:,0], dim=0, index=siamese_label.nonzero().squeeze())
                        merge_idx2 = torch.index_select(sub_part_idx[:,1], dim=0, index=siamese_label.nonzero().squeeze())
                        merge_idx = torch.unique(torch.cat([merge_idx1, merge_idx2], dim=0))
                        nonmerge_idx1 = torch.index_select(sub_part_idx[:,0], dim=0, index=(1-siamese_label).nonzero().squeeze())
                        nonmerge_idx2 = torch.index_select(sub_part_idx[:,1], dim=0, index=(1-siamese_label).nonzero().squeeze())
                        part_mask1 = torch.index_select(cur_mask_pool, dim=0, index=merge_idx1)
                        part_mask2 = torch.index_select(cur_mask_pool, dim=0, index=merge_idx2)
                        new_part_mask = 1-(1-part_mask1)*(1-part_mask2)

                        equal_matrix = torch.matmul(new_part_mask,1-new_part_mask.transpose(0,1))+torch.matmul(1-new_part_mask,new_part_mask.transpose(0,1))
                        equal_matrix[torch.eye(equal_matrix.shape[0]).byte()]=1
                        fid = (equal_matrix==0).nonzero()
                        if fid.shape[0] > 0:
                            flag = torch.ones(merge_idx1.shape[0])
                            for k in range(flag.shape[0]):
                                if flag[k] != 0:
                                    flag[fid[:,1][fid[:,0]==k]] = 0
                            new_part_mask = torch.index_select(new_part_mask, dim=0, index=flag.nonzero().squeeze().cuda())

                        new_part_xyz, xyz_mean = mask_to_xyz(pc, new_part_mask)

                        #update purity and score, policy score matrix
                        if new_part_mask.shape[0] > 0:
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

                        cur_mask_pool = torch.cat([cur_mask_pool, new_part_mask], dim=0)
                        subpart_mask_pool = torch.cat([subpart_mask_pool, new_part_mask], dim=0)
                        cur_xyz_pool = torch.cat([cur_xyz_pool, new_part_xyz], dim=0)
                        subpart_pool = torch.cat([subpart_pool, new_part_xyz], dim=0)
                        cur_pool_size = cur_mask_pool.shape[0]
                        new_mask = torch.ones([cur_pool_size])
                        new_mask[merge_idx] = 0
                        new_idx = new_mask.nonzero().squeeze().cuda()
                        cur_xyz_pool = torch.index_select(cur_xyz_pool, dim=0, index=new_idx)
                        cur_mask_pool = torch.index_select(cur_mask_pool, dim=0, index=new_idx)
                        inter_matrix = torch.matmul(cur_mask_pool, cur_mask_pool.transpose(0, 1))
                        inter_matrix_full = inter_matrix.clone()>minimum_overlap_pc_num
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

                        purity_matrix = torch.index_select(purity_matrix, dim=0, index=new_idx)
                        purity_matrix = torch.index_select(purity_matrix, dim=1, index=new_idx)
                        policy_matrix = torch.index_select(policy_matrix, dim=0, index=new_idx)
                        policy_matrix = torch.index_select(policy_matrix, dim=1, index=new_idx)
                        score_matrix = torch.zeros(purity_matrix.shape).cuda()
                        score_idx = pair_idx
                        score_matrix[score_idx[:,0], score_idx[:,1]] = softmax(purity_matrix[score_idx[:,0], score_idx[:,1]] * policy_matrix[score_idx[:,0], score_idx[:,1]])
                    final_pool_size = subpart_pool.shape[0]
                    meters.update(final_pool_size=final_pool_size,negative_num=negative_num, positive_num=positive_num)
                    meters.update(iteration_num = iteration_num)
                    meters.update(iteration_time= time.time() - iter_start_time)


            t1 = torch.matmul(cur_mask_pool,1-cur_mask_pool.transpose(0,1))
            t1[torch.eye(t1.shape[0]).byte()] = 1
            t1_id = (t1==0).nonzero()
            final_idx = torch.ones(t1.shape[0])
            final_idx[t1_id[:,0]] = 0
            cur_mask_pool = torch.index_select(cur_mask_pool, dim=0, index=final_idx.nonzero().squeeze().cuda())
            
            pred_ins_label = torch.zeros(num_points).cuda()
            for k in range(cur_mask_pool.shape[0]):
                pred_ins_label[cur_mask_pool[k].byte()] = k+1
            valid_idx = torch.sum(cur_mask_pool,0)>0
            if torch.sum(1-valid_idx) != 0:
                valid_points = pc[:,valid_idx]
                invalid_points = pc[:,1-valid_idx]
                #perform knn to cover all points
                knn_index, _ = _F.knn_distance(invalid_points.unsqueeze(0), valid_points.unsqueeze(0), 5, False)
                invalid_pred,_ = pred_ins_label[valid_idx][knn_index.squeeze()].mode()
                pred_ins_label[1-valid_idx] = invalid_pred
            cur_mask_pool_new = torch.zeros([0,num_points]).cuda()
            for k in range(cur_mask_pool.shape[0]):
                if torch.sum(pred_ins_label==(k+1)) != 0:
                    cur_mask_pool_new = torch.cat([cur_mask_pool_new, ((pred_ins_label == (k+1)).float()).unsqueeze(0)], dim=0)
            out_mask[iteration, :cur_mask_pool_new.shape[0]] = copy.deepcopy(cur_mask_pool_new.cpu().data.numpy().astype(np.bool))
            out_valid[iteration, :cur_mask_pool_new.shape[0]] = np.sum(cur_mask_pool_new.cpu().data.numpy()) > 10

    test_time = time.time() - start_time
    logger.info('Test {}  test time: {:.2f}s'.format(meters.summary_str, test_time))
    for i in range(int(out_mask.shape[0]/1024) +1):
        save_h5(os.path.join(output_dir_save, 'test-%02d.h5'%(i)), out_mask[i*1024:(i+1)*1024], out_valid[i*1024:(i+1)*1024], out_conf[i*1024:(i+1)*1024])


def main():
    args = parse_args()

    from partnet.config.ins_seg_3d import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # Replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace('configs', 'outputs')
        output_dir_merge = output_dir.replace('@', config_path)+'_merge'
        os.makedirs(output_dir_merge, exist_ok=True)
        output_dir = osp.join('outputs/stage1/', cfg.DATASET.PartNetInsSeg.TRAIN.stage1)
        output_dir_save = './results/'+cfg.DATASET.PartNetInsSeg.TEST.shape
        os.makedirs(output_dir_save, exist_ok=True)
        output_dir_save = osp.join(output_dir_save,'Level_%d'%cfg.TEST.LEVEL)
        os.makedirs(output_dir_save, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('shaper', output_dir_save, prefix='test')
    logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
    logger.info(args)
    logger.info('Loaded configuration file {}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.TASK == 'ins_seg_3d'
    test(cfg, output_dir, output_dir_merge, output_dir_save)

if __name__ == '__main__':
    main()
