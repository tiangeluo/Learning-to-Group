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
from core.nn.functional import focal_loss
from core.nn.functional import l2_loss
import copy
import h5py
import matplotlib.pyplot as plt
import matplotlib
import json

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

    # build data loader
    test_dataloader = build_dataloader(cfg, mode='test')
    test_dataset = test_dataloader.dataset

    assert cfg.TEST.BATCH_SIZE == 1, '{} != 1'.format(cfg.TEST.BATCH_SIZE)
    output_dir = osp.join(output_dir_save, 'tree')
    os.makedirs(output_dir, exist_ok=True)
    save_shape_dir = osp.join(output_dir, 'shapes')
    os.makedirs(save_shape_dir, exist_ok=True)
    save_tree_dir = osp.join(output_dir, 'tree')
    os.makedirs(save_tree_dir, exist_ok=True)
    save_txt_dir = osp.join(output_dir, 'txt')
    os.makedirs(save_txt_dir, exist_ok=True)

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #
    model.eval()
    model_merge.eval()
    loss_fn.eval()
    set_random_seed(cfg.RNG_SEED)
    softmax = nn.Softmax()

    meters = MetricLogger(delimiter='  ')
    meters.bind(val_metric)
    with torch.no_grad():
        start_time = time.time()
        end = start_time
        for iteration, data_batch in enumerate(test_dataloader):
            print(iteration)
            data_time = time.time() - end

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
            centroid_label = data_batch['centroid_label'].reshape(-1)

            minimum_box_pc_num = 16
            minimum_overlap_pc_num = 16 #1/16 * num_neighbour
            gtmin_mask = (torch.sum(box_index_expand, dim=-1) > minimum_box_pc_num)

            #remove purity < 0.8
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
            box_purity_valid_mask_l2 = gtmin_mask.long()*data_batch['centroid_valid_mask'].reshape(-1).long()
            meters.update(purity_ratio = torch.sum(box_purity_mask).float()/box_purity_mask.shape[0], purity_valid_ratio=torch.sum(box_purity_valid_mask).float()/box_purity_mask.shape[0])
            meters.update(purity_pos_num = torch.sum(box_purity_mask), purity_neg_num = torch.sum(1-box_purity_mask), purity_neg_valid_num=torch.sum(box_purity<0.6))
            centroid_valid_mask = data_batch['centroid_valid_mask'].reshape(-1).long()
            meters.update(centroid_valid_purity_ratio = torch.sum(torch.index_select(box_purity_mask, dim=0, index=centroid_valid_mask.nonzero().squeeze())).float()/torch.sum(centroid_valid_mask),centroid_nonvalid_purity_ratio = torch.sum(torch.index_select(box_purity_mask, dim=0, index=(1-centroid_valid_mask).nonzero().squeeze())).float()/torch.sum(1-centroid_valid_mask))
            purity_pred = torch.zeros([0]).type(torch.FloatTensor).cuda()
            purity_score = torch.zeros([0]).cuda()

            for i in range(batch_size):
                cur_xyz_pool, xyz_mean = mask_to_xyz(data_batch['points'][i], box_index_expand.view(batch_size,num_centroids,num_points)[i], sample_num=512)
                cur_xyz_pool -= xyz_mean
                cur_xyz_pool /=(cur_xyz_pool+1e-6).norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

                logits_purity = model_merge(cur_xyz_pool, 'purity')
                purity_score = torch.cat([purity_score, logits_purity.squeeze()])
                cur_label = box_purity_mask.type(torch.LongTensor).view(batch_size,num_centroids)[i].cuda()
                cur_valid = box_purity_valid_mask.type(torch.LongTensor).view(batch_size, num_centroids)[i].cuda()
                cur_valid_l2 = box_purity_valid_mask_l2.type(torch.LongTensor).view(batch_size, num_centroids)[i].cuda()
                cur_label_l2 = box_purity.view(batch_size,num_centroids)[i].cuda()
                zero_term = (cur_label_l2 > 0.8).nonzero().squeeze()
                cur_valid_l2[zero_term[:int(zero_term.shape[0]/2)]] = 0
                loss_purity = l2_loss(logits_purity.squeeze(), cur_label_l2, cur_valid_l2.float())
                loss_dict_embed = {
                    'loss_purity': loss_purity,
                }
                meters.update(**loss_dict_embed)
                total_loss_embed = sum(loss_dict_embed.values())

                p = (logits_purity > 0.8).long().squeeze()
                purity_pred = torch.cat([purity_pred,logits_purity.squeeze()])
                purity_acc_arr = (p.float() == cur_label.float()).float()
                purity_acc = torch.mean(purity_acc_arr)
                purity_pos_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=(cur_label==1).nonzero().squeeze()))
                meters.update(purity_acc=purity_acc, purity_pos_acc=purity_pos_acc)
                if torch.sum(cur_label==0) != 0:
                    purity_neg_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=(cur_label==0).nonzero().squeeze()))
                    meters.update(purity_neg_acc=purity_neg_acc)
                if torch.sum((cur_label==0).long()*(cur_valid).long()) !=0:
                    purity_neg_valid_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=((cur_label==0).long()*cur_valid.long()).nonzero().squeeze()))
                    meters.update(purity_neg_valid_acc=purity_neg_valid_acc)

                if torch.sum(1-cur_valid) !=0:
                    purity_nonvalid_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=(1-cur_valid).nonzero().squeeze()))
                    meters.update(purity_nonvalid_acc = purity_nonvalid_acc)
                cur_centroid_valid_mask = data_batch['centroid_valid_mask'][i].long()
                if torch.sum((cur_label==1).long()*(cur_centroid_valid_mask)) != 0:
                    purity_valid_pos_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=((cur_label==1).long()*(cur_centroid_valid_mask)).nonzero().squeeze()))
                    meters.update(purity_valid_pos_acc=purity_valid_pos_acc)
                if torch.sum((cur_label==0).long()*(cur_centroid_valid_mask)) != 0:
                    purity_valid_neg_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=((cur_label==0).long()*(cur_centroid_valid_mask)).nonzero().squeeze()))
                    meters.update(purity_valid_neg_acc=purity_valid_neg_acc)
                if torch.sum((cur_label==1).long()*(1-cur_centroid_valid_mask)) != 0:
                    purity_nonvalid_pos_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=((cur_label==1).long()*(1-cur_centroid_valid_mask)).nonzero().squeeze()))
                    meters.update(purity_nonvalid_pos_acc=purity_nonvalid_pos_acc)
                if torch.sum((cur_label==0).long()*(1-cur_centroid_valid_mask)) != 0:
                    purity_nonvalid_neg_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=((cur_label==0).long()*(1-cur_centroid_valid_mask)).nonzero().squeeze()))
                    meters.update(purity_nonvalid_neg_acc=purity_nonvalid_neg_acc)

            #update pool by valid_mask
            valid_mask = gtmin_mask.long() *  (purity_pred>0.8).long()
            box_index_expand = torch.index_select(box_index_expand, dim=0, index=valid_mask.nonzero().squeeze())
            centroid_label = torch.index_select(centroid_label, dim=0, index=valid_mask.nonzero().squeeze())
            purity_pred = torch.index_select(purity_pred, dim=0, index=valid_mask.nonzero().squeeze())

            box_num = torch.sum(valid_mask.reshape(batch_size, num_centroids),1)
            cumsum_box_num = torch.cumsum(box_num, dim=0)
            cumsum_box_num = torch.cat([torch.from_numpy(np.array(0)).cuda().unsqueeze(0),cumsum_box_num],dim=0)

            #extraction part features
            gt_mask = data_batch['gt_mask']
            gt_valid = data_batch['gt_valid']
            valid_num = torch.sum(gt_valid, dim=1)
            cumsum_valid_num = torch.cumsum(valid_num, dim=0)
            cumsum_valid_num = torch.cat([torch.from_numpy(np.array(0)).cuda().unsqueeze(0),cumsum_valid_num],dim=0)
            parts_num = torch.sum(valid_num).cpu().data.numpy()
            parts_mask = torch.zeros([parts_num, 10000]).cuda().int()
            for i in range(batch_size):
                for j in range(valid_num[i]):
                    parts_mask[j] = gt_mask[i,j]

            #initialization
            with torch.no_grad():
                pc_all = data_batch['points']
                xyz_pool1 = torch.zeros([0,3,1024]).float().cuda()
                xyz_pool2 = torch.zeros([0,3,1024]).float().cuda()
                tree_mask_pool = torch.zeros([0,10000]).cuda()
                tree_trace = list()
                tree_num = 0
                label_pool = torch.zeros([0]).float().cuda()
                centroid_label_all = centroid_label.clone()
                for i in range(pc_all.shape[0]):
                    bs = 1
                    pc = pc_all[i].clone()
                    cur_mask_pool = box_index_expand[cumsum_box_num[i]:cumsum_box_num[i+1]].clone()
                    centroid_label = centroid_label_all[cumsum_box_num[i]:cumsum_box_num[i+1]].clone()
                    cover_ratio = torch.unique(cur_mask_pool.nonzero()[:,1]).shape[0]/num_points
                    cur_xyz_pool, xyz_mean = mask_to_xyz(pc, cur_mask_pool)
                    subpart_pool = cur_xyz_pool.clone()
                    subpart_mask_pool = cur_mask_pool.clone()

                    #Tree
                    tree_mask_pool = cur_mask_pool.clone()
                    tree_num = cur_mask_pool.shape[0]
                    print('line 329, tree_num: ', tree_num)
                    tree_num_arr = torch.range(0,tree_num-0.5).type(torch.LongTensor).cuda()
                    for j in range(tree_num):
                        tree_trace.append(())
                    purity_score = torch.index_select(purity_score, dim=0, index=valid_mask.nonzero().squeeze())
                    shape_txt_path = osp.join(save_txt_dir, '{:03d}'.format(iteration))
                    os.makedirs(shape_txt_path, exist_ok=True)
                    for j in range(tree_num):
                        node_txt_path = osp.join(shape_txt_path, '{:d}'.format(j)+'.txt')
                        node_txt_f = open(node_txt_path, 'w')
                        node_txt_f.write('id:%d, purity_score: %.2f' %(j, purity_score[j].cpu().data.numpy()))
                        node_txt_f.close()

                    init_pool_size = cur_xyz_pool.shape[0]
                    meters.update(cover_ratio=cover_ratio, init_pool_size=init_pool_size)
                    negative_num = 0
                    positive_num = 0

                    #remove I
                    inter_matrix = torch.matmul(cur_mask_pool, cur_mask_pool.transpose(0, 1))
                    inter_matrix[torch.eye(inter_matrix.shape[0]).byte()] = 0
                    pair_idx = (inter_matrix.triu()>minimum_overlap_pc_num).nonzero()
                    zero_pair = torch.ones([0,2]).long()

                    #p2
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
                    score_matrix = softmax(purity_matrix) * softmax(policy_matrix)

                    while pair_idx.shape[0] > 0:
                        score_arr = score_matrix[pair_idx[:,0], pair_idx[:,1]]
                        highest_score, rank_idx = torch.topk(score_arr,1,largest=True,sorted=False)
                        perm_idx = rank_idx

                        sub_part_idx = torch.index_select(pair_idx, dim=0, index=perm_idx[:bs])
                        purity_score = purity_matrix[sub_part_idx[:,0],sub_part_idx[:,1]]
                        policy_score = policy_matrix[sub_part_idx[:,0],sub_part_idx[:,1]]

                        part_xyz1 = torch.index_select(cur_xyz_pool, dim=0, index=sub_part_idx[:,0])
                        part_xyz2 = torch.index_select(cur_xyz_pool, dim=0, index=sub_part_idx[:,1])
                        part_xyz = torch.cat([part_xyz1,part_xyz2],-1)
                        part_xyz -= torch.mean(part_xyz,-1).unsqueeze(-1)
                        part_xyz1 -= torch.mean(part_xyz1,-1).unsqueeze(-1)
                        part_xyz2 -= torch.mean(part_xyz2,-1).unsqueeze(-1)
                        part_xyz1 /=part_xyz1.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                        part_xyz2 /=part_xyz2.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                        part_xyz /=part_xyz.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                        part_label1 = torch.index_select(centroid_label, dim=0, index=sub_part_idx[:,0])
                        part_label2 = torch.index_select(centroid_label, dim=0, index=sub_part_idx[:,1])

                        siamese_label_gt = (part_label1 == part_label2)*(1 - (part_label1 == -1))*(1 - (part_label2 == -1))
                        #siamese_label = (part_label1 == part_label2)
                        logits1 = model_merge(part_xyz1,'backbone')
                        logits2 = model_merge(part_xyz2,'backbone')
                        merge_logits = model_merge(torch.cat([part_xyz, torch.cat([logits1.unsqueeze(-1).expand(-1,-1,part_xyz1.shape[-1]), logits2.unsqueeze(-1).expand(-1,-1,part_xyz2.shape[-1])], dim=-1)], dim=1), 'head')
                        _, p = torch.max(merge_logits, 1)
                        siamese_label = p

                        negative_num += torch.sum(siamese_label == 0)
                        positive_num += torch.sum(siamese_label == 1)
                        ##predict

                        #update info
                        merge_idx1 = torch.index_select(sub_part_idx[:,0], dim=0, index=siamese_label.nonzero().squeeze())
                        merge_idx2 = torch.index_select(sub_part_idx[:,1], dim=0, index=siamese_label.nonzero().squeeze())
                        merge_idx = torch.unique(torch.cat([merge_idx1, merge_idx2], dim=0))
                        nonmerge_idx1 = torch.index_select(sub_part_idx[:,0], dim=0, index=(1-siamese_label).nonzero().squeeze())
                        nonmerge_idx2 = torch.index_select(sub_part_idx[:,1], dim=0, index=(1-siamese_label).nonzero().squeeze())
                        #nonmerge_idx = torch.unique(torch.cat([nonmerge_idx1, nonmerge_idx2], dim=0))
                        part_mask1 = torch.index_select(cur_mask_pool, dim=0, index=merge_idx1)
                        part_mask2 = torch.index_select(cur_mask_pool, dim=0, index=merge_idx2)
                        new_part_mask = 1-(1-part_mask1)*(1-part_mask2)
                        new_part_label = torch.index_select(part_label1, dim=0, index=siamese_label.nonzero().squeeze())
                        new_part_label_invalid = torch.index_select(siamese_label_gt, dim=0, index=siamese_label.nonzero().squeeze()).long()
                        new_part_label = new_part_label*new_part_label_invalid + -1*(1-new_part_label_invalid)
                        #remove totally the same term

                        equal_matrix = torch.matmul(new_part_mask,1-new_part_mask.transpose(0,1))+torch.matmul(1-new_part_mask,new_part_mask.transpose(0,1))
                        equal_matrix[torch.eye(equal_matrix.shape[0]).byte()]=1
                        fid = (equal_matrix==0).nonzero()
                        if fid.shape[0] > 0:
                            flag = torch.ones(merge_idx1.shape[0])
                            for k in range(flag.shape[0]):
                                if flag[k] != 0:
                                    flag[fid[:,1][fid[:,0]==k]] = 0
                            new_part_mask = torch.index_select(new_part_mask, dim=0, index=flag.nonzero().squeeze().cuda())
                            new_part_label = torch.index_select(new_part_label, dim=0, index=flag.nonzero().squeeze().cuda())

                        new_part_xyz, xyz_mean = mask_to_xyz(pc, new_part_mask)
                        refine_flag = 0

                        #p2, update purity and score
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
                        centroid_label = torch.cat([centroid_label, new_part_label], dim=0)
                        #update cur_pool, pick out merged
                        cur_pool_size = cur_mask_pool.shape[0]
                        new_mask = torch.ones([cur_pool_size])
                        new_mask[merge_idx] = 0
                        new_idx = new_mask.nonzero().squeeze().cuda()
                        cur_xyz_pool = torch.index_select(cur_xyz_pool, dim=0, index=new_idx)
                        cur_mask_pool = torch.index_select(cur_mask_pool, dim=0, index=new_idx)
                        centroid_label = torch.index_select(centroid_label, dim=0, index=new_idx)
                        #p1
                        inter_matrix = torch.matmul(cur_mask_pool, cur_mask_pool.transpose(0, 1))

                        #Tree
                        part_num1 = torch.index_select(tree_num_arr, dim=0, index=sub_part_idx[:,0])
                        part_num2 = torch.index_select(tree_num_arr, dim=0, index=sub_part_idx[:,1])
                        if siamese_label == 1:
                            if refine_flag == 1:
                                tree_mask_pool = torch.cat([tree_mask_pool, old_part_mask], dim=0)
                            else:
                                tree_mask_pool = torch.cat([tree_mask_pool, new_part_mask], dim=0)
                            tree_trace.append((part_num1.cpu().data.numpy()[0], part_num2.cpu().data.numpy()[0]))
                            tree_num_arr = torch.cat([tree_num_arr, torch.Tensor([tree_num]).long().cuda()], dim=0)
                            tree_num_arr = torch.index_select(tree_num_arr, dim=0, index=new_idx)
                            node_txt_path = osp.join(shape_txt_path, '{:d}'.format(tree_num)+'.txt')
                            node_txt_f = open(node_txt_path, 'w')

                            #gt_purity
                            new_part_mask_tree = tree_mask_pool[-1].unsqueeze(0)#1-(1-part_mask11)*(1-part_mask22)
                            box_label_expand = torch.zeros((new_part_mask_tree.shape[0], 200)).cuda()
                            box_idx_expand = tile(data_batch['ins_id'][i].unsqueeze(0),0,new_part_mask_tree.shape[0]).cuda()
                            box_label_expand = box_label_expand.scatter_add_(dim=1, index=box_idx_expand, src=new_part_mask_tree).float()
                            maximum_label_num, maximum_label = torch.max(box_label_expand, 1)
                            total_num = torch.sum(box_label_expand, 1)
                            box_purity = maximum_label_num / (total_num+1e-6)
                            node_txt_f.write('Merge: id:%d = [%d, %d], part labels: [%d, %d]' %(tree_num, part_num1, part_num2, part_label1.cpu().data.numpy(), part_label2.cpu().data.numpy()))
                            #More Info.
                            #node_txt_f.write('Merge: id:%d = [%d, %d], score %.2f = policy %.2f x purity %.2f, gt_purity %.2f, right: %d, part label: [%d, %d]' %(tree_num, part_num1, part_num2, highest_score.cpu().data.numpy(), policy_score.cpu().data.numpy(), purity_score.cpu().data.numpy(), box_purity.cpu().data.numpy(), siamese_label_gt.cpu().data.numpy(), part_label1.cpu().data.numpy(), part_label2.cpu().data.numpy()))
                            node_txt_f.close()
                            tree_num += 1

                        #p2, update purity and score
                        purity_matrix = torch.index_select(purity_matrix, dim=0, index=new_idx)
                        purity_matrix = torch.index_select(purity_matrix, dim=1, index=new_idx)
                        policy_matrix = torch.index_select(policy_matrix, dim=0, index=new_idx)
                        policy_matrix = torch.index_select(policy_matrix, dim=1, index=new_idx)
                        score_matrix = softmax(policy_matrix) * softmax(purity_matrix)

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
                    final_pool_size = subpart_pool.shape[0]
                    meters.update(final_pool_size=final_pool_size,negative_num=negative_num, positive_num=positive_num)

            pred_ins_label = torch.zeros(num_points).cuda()
            new_part_xyz, xyz_mean = mask_to_xyz(pc, cur_mask_pool)
            new_part_xyz -= xyz_mean
            new_part_xyz /=(new_part_xyz+1e-6).norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            purity_scores = model_merge(new_part_xyz, 'purity').squeeze()
            rank_idx = torch.argsort(purity_scores)
            if len(rank_idx.shape) == 0:
                rank_idx = rank_idx.unsqueeze(0)
            for k in range(cur_mask_pool.shape[0]):
                pred_ins_label[cur_mask_pool[rank_idx[k]].byte()] = k+1
            valid_idx = torch.sum(cur_mask_pool,0)>0
            tree_mask_pool = torch.cat([tree_mask_pool, pred_ins_label.unsqueeze(0)], dim=0)
            tree_trace.append(tree_num_arr.cpu().data.numpy())
            node_txt_path = osp.join(shape_txt_path, '{:d}'.format(tree_num)+'.txt')
            node_txt_f = open(node_txt_path, 'w')
            node_txt_f.write('id:%d = %s, Shape Point Clouds'%(tree_num, str(tree_num_arr.cpu().data.numpy())))
            node_txt_f.close()
            tree_num += 1

            tree_list = list()
            for x1 in range(tree_num):
                d = {'id':x1}
                son_pair = tree_trace[x1]
                if len(son_pair) > 0:
                    tmp = list()
                    for j in range(len(son_pair)):
                        tmp.append(tree_list[son_pair[j]])
                    d.update({'children':tmp})
                tree_list.append(d)

            tree_json_path = osp.join(save_tree_dir, '{:03d}.json'.format(iteration))
            with open(tree_json_path, 'w') as f:
                json.dump(tree_list[-1], f)

            shape_json_path = osp.join(save_shape_dir, '{:03d}.json'.format(iteration))
            save_dict = {
            'points': pc.unsqueeze(0),
            'mask_pool': tree_mask_pool.unsqueeze(0),
            }
            out_dict = tensor2list(save_dict)
            with open(shape_json_path, 'w') as f:
                #json.dump(save_dict, f)
                json.dump(out_dict, f)

            if cfg.TEST.LOG_PERIOD > 0 and iteration % cfg.TEST.LOG_PERIOD == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            'iter: {iter:4d}',
                            '{meters}',
                        ]
                    ).format(
                        iter=iteration,
                        meters=str(meters),
                    )
                )
    test_time = time.time() - start_time
    logger.info('Test {}  test time: {:.2f}s'.format(meters.summary_str, test_time))


def main():
    args = parse_args()

    # Load the configuration
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
        output_dir_merge = 'outputs/pn_stage2_fusion_l%d_merge'%cfg.TEST.LEVEL
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
