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
from partnet.data_merge.build import build_dataloader, parse_augmentations
from shaper.data import transforms as T
from IPython import embed
import shaper.models.pointnet2.functions as _F
#from partnet.pn_merge import PointNetCls
#from partnet.pn_merge_regress_hao import PointNetCls
#from partnet.pn_merge_regress_hao_policy_nosigmoid_2th_RND import PointNetCls
#from partnet.pn_merge_regress_hao_policy_context_2th_RND import PointNetCls
#from partnet.pn_merge_regress_hao_policy_context_2th_2RND import PointNetCls
#from partnet.pn_v6_2 import PointNetCls
#from partnet.pn_v6_2_split import PointNetCls
from partnet.pn2_v0 import PointNetCls
#from partnet.pn_v6_5 import PointNetCls
#from partnet.pn_v6_4 import PointNetCls
#from partnet.pn_v6_3 import PointNetCls
import torch.nn.functional as F

from core.nn.functional import cross_entropy
from core.nn.functional import focal_loss, l2_loss
import copy
import h5py

import matplotlib.pyplot as plt
import matplotlib

#def plot_curve(arr, name, fn)
#    matplotlib.use('Agg')
#    plot_fn = osp.join(fn, str(name)+'.png')
#    fig = plt.figure(figsize=(10,5))
#    x = np.arange(len(arr))
#    plt.plot(x, 
    

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
    #model_merge = nn.DataParallel(PointNetCls(in_channels=3, out_channels=256)).cuda()
    #model_merge = nn.DataParallel(PointNetCls(in_channels=3, out_channels=196)).cuda()
    #model_refine = nn.DataParallel(PointNetPartSeg()).cuda()

    # build checkpointer
    checkpointer = Checkpointer(model, save_dir=output_dir, logger=logger)
    checkpointer_merge = Checkpointer(model_merge, save_dir=output_dir_merge, logger=logger)
    #checkpointer_refine = Checkpointer(model_refine, save_dir=output_dir_save, logger=logger)

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
    #model_refine.eval()
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
            #centroid_label = data_batch['centroid_label'].reshape(-1)

            minimum_box_pc_num = 16
            minimum_overlap_pc_num = 16 #1/16 * num_neighbour
            gtmin_mask = (torch.sum(box_index_expand, dim=-1) > minimum_box_pc_num)

            #remove purity < 0.8
            box_label_expand = torch.zeros((batch_size*num_centroids, 200)).cuda()
            #box_idx_expand = tile(data_batch['ins_id'],0,num_centroids).cuda()
            #box_label_expand = box_label_expand.scatter_add_(dim=1, index=box_idx_expand, src=box_index_expand).float()
            #maximum_label_num, _ = torch.max(box_label_expand, 1)
            #total_num = torch.sum(box_label_expand, 1)
            #box_purity = maximum_label_num / total_num
            #box_purity_mask = box_purity > 0.85
            #box_purity_mask = box_purity > 0.8
            #box_purity_valid_mask = 1 - (box_purity < 0.8)*(box_purity > 0.6)
            #box_purity_valid_mask *= gtmin_mask
            #meters.update(purity_ratio = torch.sum(box_purity_mask).float()/box_purity_mask.shape[0], purity_valid_ratio=torch.sum(box_purity_valid_mask).float()/box_purity_mask.shape[0])
            #meters.update(purity_pos_num = torch.sum(box_purity_mask), purity_neg_num = torch.sum(1-box_purity_mask), purity_neg_valid_num=torch.sum(box_purity<0.6))
            #centroid_valid_mask = data_batch['centroid_valid_mask'].reshape(-1).long()
            #meters.update(centroid_valid_purity_ratio = torch.sum(torch.index_select(box_purity_mask, dim=0, index=centroid_valid_mask.nonzero().squeeze())).float()/torch.sum(centroid_valid_mask),centroid_nonvalid_purity_ratio = torch.sum(torch.index_select(box_purity_mask, dim=0, index=(1-centroid_valid_mask).nonzero().squeeze())).float()/torch.sum(1-centroid_valid_mask))
            purity_pred = torch.zeros([0]).type(torch.LongTensor).cuda()
            purity_pred_float = torch.zeros([0]).type(torch.FloatTensor).cuda()

            for i in range(batch_size):
                cur_xyz_pool, xyz_mean = mask_to_xyz(data_batch['points'][i], box_index_expand.view(batch_size,num_centroids,num_points)[i], sample_num=512)
                cur_xyz_pool -= xyz_mean
                cur_xyz_pool /=(cur_xyz_pool+1e-6).norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                
                logits_purity = model_merge(cur_xyz_pool, 'purity')
                #cur_label = box_purity_mask.type(torch.LongTensor).view(batch_size,num_centroids)[i].cuda()
                #cur_valid = box_purity_valid_mask.type(torch.LongTensor).view(batch_size, num_centroids)[i].cuda()
                #cur_label_l2 = box_purity.view(batch_size,num_centroids)[i].cuda()
                #loss_purity = torch.nn.CrossEntropyLoss()(logits_purity, cur_label)
                #loss_purity = focal_loss(logits_purity, cur_label, cur_valid.float())
                #loss_purity = cross_entropy(logits_purity, cur_label, cur_valid.float())
                #loss_purity = l2_loss(logits_purity.squeeze(), cur_label_l2)#, cur_valid_l2.float())
                #loss_dict_embed = {
                #    'loss_purity': loss_purity,
                #}
                #meters.update(**loss_dict_embed)
                #total_loss_embed = sum(loss_dict_embed.values())

                #_, p = torch.max(logits_purity,1)
                p = (logits_purity > 0.8).long().squeeze()
                purity_pred = torch.cat([purity_pred,p])
                purity_pred_float = torch.cat([purity_pred_float,logits_purity.squeeze()])
                #purity_acc_arr = (p.float() == cur_label.float()).float()
                #purity_acc = torch.mean(purity_acc_arr)
                #purity_pos_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=(cur_label==1).nonzero().squeeze()))
                #meters.update(purity_acc=purity_acc, purity_pos_acc=purity_pos_acc)
                #if torch.sum(cur_label==0) != 0:
                #    purity_neg_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=(cur_label==0).nonzero().squeeze()))
                #    meters.update(purity_neg_acc=purity_neg_acc)
                #if torch.sum((cur_label==0).long()*(cur_valid).long()) !=0:
                #    purity_neg_valid_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=((cur_label==0).long()*cur_valid.long()).nonzero().squeeze()))
                #    meters.update(purity_neg_valid_acc=purity_neg_valid_acc)

                #if torch.sum(1-cur_valid) !=0:
                #    purity_nonvalid_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=(1-cur_valid).nonzero().squeeze()))
                #    meters.update(purity_nonvalid_acc = purity_nonvalid_acc)
                #cur_centroid_valid_mask = data_batch['centroid_valid_mask'][i].long()
                #if torch.sum((cur_label==1).long()*(cur_centroid_valid_mask)) != 0:
                #    purity_valid_pos_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=((cur_label==1).long()*(cur_centroid_valid_mask)).nonzero().squeeze()))
                #    meters.update(purity_valid_pos_acc=purity_valid_pos_acc)
                #if torch.sum((cur_label==0).long()*(cur_centroid_valid_mask)) != 0:
                #    purity_valid_neg_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=((cur_label==0).long()*(cur_centroid_valid_mask)).nonzero().squeeze()))
                #    meters.update(purity_valid_neg_acc=purity_valid_neg_acc)
                #if torch.sum((cur_label==1).long()*(1-cur_centroid_valid_mask)) != 0:
                #    purity_nonvalid_pos_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=((cur_label==1).long()*(1-cur_centroid_valid_mask)).nonzero().squeeze()))
                #    meters.update(purity_nonvalid_pos_acc=purity_nonvalid_pos_acc)
                #if torch.sum((cur_label==0).long()*(1-cur_centroid_valid_mask)) != 0:
                #    purity_nonvalid_neg_acc = torch.mean(torch.index_select(purity_acc_arr, dim=0, index=((cur_label==0).long()*(1-cur_centroid_valid_mask)).nonzero().squeeze()))
                #    meters.update(purity_nonvalid_neg_acc=purity_nonvalid_neg_acc)

            #update pool by valid_mask
            #valid_mask = gtmin_mask.long() * data_batch['centroid_valid_mask'].reshape(-1).long() * box_purity_mask.long()
            p_thresh = 0.8
            purity_pred = purity_pred_float > p_thresh
            while(torch.sum(purity_pred) < 48):
                p_thresh = p_thresh-0.01
                purity_pred = purity_pred_float > p_thresh
            valid_mask = gtmin_mask.long() *  purity_pred.long()
            box_index_expand = torch.index_select(box_index_expand, dim=0, index=valid_mask.nonzero().squeeze())
            #centroid_label = torch.index_select(centroid_label, dim=0, index=valid_mask.nonzero().squeeze())

            box_num = torch.sum(valid_mask.reshape(batch_size, num_centroids),1)
            cumsum_box_num = torch.cumsum(box_num, dim=0)
            cumsum_box_num = torch.cat([torch.from_numpy(np.array(0)).cuda().unsqueeze(0),cumsum_box_num],dim=0)

            ##pull box feature near with its corresponding part
            #box_num = torch.sum(valid_mask.reshape(batch_size, num_centroids),1)
            #cumsum_box_num = torch.cumsum(box_num, dim=0)
            #cumsum_box_num = torch.cat([torch.from_numpy(np.array(0)).cuda().unsqueeze(0),cumsum_box_num],dim=0)
            #align_idx = torch.zeros(box_pc.shape[0]).cuda().type(torch.LongTensor)
            #for i in range(batch_size):
            #    for j in range(box_num[i]):
            #        ins_labels = centroid_label[cumsum_box_num[i] + j] - 1#-1 for other_mask
            #        align_idx[cumsum_box_num[i] + j] = ins_labels + cumsum_valid_num[i]

            ##extraction part features
            #gt_mask = data_batch['gt_mask']
            #gt_valid = data_batch['gt_valid']
            #valid_num = torch.sum(gt_valid, dim=1)
            #cumsum_valid_num = torch.cumsum(valid_num, dim=0)
            #cumsum_valid_num = torch.cat([torch.from_numpy(np.array(0)).cuda().unsqueeze(0),cumsum_valid_num],dim=0)
            #parts_num = torch.sum(valid_num).cpu().data.numpy()
            #parts_mask = torch.zeros([parts_num, 10000]).cuda().int()
            #for i in range(batch_size):
            #    for j in range(valid_num[i]):
            #        parts_mask[j] = gt_mask[i,j]

            #initialization
            with torch.no_grad():
                pc_all = data_batch['points']
                xyz_pool1 = torch.zeros([0,3,1024]).float().cuda()
                xyz_pool2 = torch.zeros([0,3,1024]).float().cuda()
                #mask_pool1 = torch.zeros([0,num_points]).float().cuda()
                #mask_pool2 = torch.zeros([0,num_points]).float().cuda()
                label_pool = torch.zeros([0]).float().cuda()
                #centroid_label_all = centroid_label.clone()
                for i in range(pc_all.shape[0]):
                    bs = 1
                    pc = pc_all[i].clone()
                    cur_mask_pool = box_index_expand[cumsum_box_num[i]:cumsum_box_num[i+1]].clone()
                    #centroid_label = centroid_label_all[cumsum_box_num[i]:cumsum_box_num[i+1]].clone()
                    cover_ratio = torch.unique(cur_mask_pool.nonzero()[:,1]).shape[0]/num_points
                    cur_xyz_pool, xyz_mean = mask_to_xyz(pc, cur_mask_pool)
                    #cur_xyz_pool -= xyz_mean
                    subpart_pool = cur_xyz_pool.clone()
                    subpart_mask_pool = cur_mask_pool.clone()
                    init_pool_size = cur_xyz_pool.shape[0]
                    meters.update(cover_ratio=cover_ratio, init_pool_size=init_pool_size) 
                    print(iteration, cover_ratio)
                    negative_num = 0
                    positive_num = 0

                    #remove I
                    inter_matrix = torch.matmul(cur_mask_pool, cur_mask_pool.transpose(0, 1))
                    inter_matrix_full = inter_matrix.clone()>minimum_overlap_pc_num
                    #zero_pair = torch.eye(inter_matrix.shape[0]).nonzero()
                    #inter_matrix[zero_pair[:,0], zero_pair[:,1]] = 0
                    inter_matrix[torch.eye(inter_matrix.shape[0]).byte()] = 0
                    pair_idx = (inter_matrix.triu()>minimum_overlap_pc_num).nonzero()
                    zero_pair = torch.ones([0,2]).long()
                    #equal_matrix = torch.matmul(cur_mask_pool,1-cur_mask_pool.transpose(0,1))+torch.matmul(1-cur_mask_pool,cur_mask_pool.transpose(0,1))
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
                    #score_matrix = softmax(purity_matrix) * softmax(policy_matrix)
                    score_matrix = torch.zeros(purity_matrix.shape).cuda()
                    score_matrix[pair_idx[:,0],pair_idx[:,1]] = softmax(purity_pool*policy_pool)
                    remote_flag = False

                    #info
                    policy_list = []
                    purity_list = []
                    gt_purity_list = []
                    gt_label_list = []
                    pred_label_list = []
                    size_list=[]
                    relative_size_list=[]

                    #while pair_idx.shape[0] > 0:
                    while (pair_idx.shape[0] > 0) or (remote_flag == False):
                        if pair_idx.shape[0] == 0:
                            remote_flag = True
                            inter_matrix = 20*torch.ones([cur_mask_pool.shape[0],cur_mask_pool.shape[0]]).cuda()
                            inter_matrix[zero_pair[:,0], zero_pair[:,1]] = 0
                            inter_matrix[torch.eye(inter_matrix.shape[0]).byte()] = 0
                            pair_idx = (inter_matrix.triu()>minimum_overlap_pc_num).nonzero()
                            if pair_idx.shape[0] == 0:
                                break
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
                        #if pair_idx.shape[0] < bs:
                        #    bs = pair_idx.shape[0]
                        #perm_idx = torch.randperm(pair_idx.shape[0]).cuda()
                        #if pair_idx.shape[0] > 2*bs:
                        #    size1 = torch.sum(torch.index_select(cur_mask_pool,dim=0,index=pair_idx[:,0]),-1)
                        #    size2 = torch.sum(torch.index_select(cur_mask_pool,dim=0,index=pair_idx[:,1]),-1)
                        #    ratio = size1/size2+size2/size1
                        #    _, rank_idx = torch.topk(ratio,2*bs,largest=False,sorted=False)
                        #    perm_idx = rank_idx
                        #elif pair_idx.shape[0] < bs:
                        #    bs = pair_idx.shape[0]
                        #    perm_idx = torch.randperm(pair_idx.shape[0]).cuda()
                        #else:
                        #    perm_idx = torch.randperm(pair_idx.shape[0]).cuda()

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
                        #part_label1 = torch.index_select(centroid_label, dim=0, index=sub_part_idx[:,0])
                        #part_label2 = torch.index_select(centroid_label, dim=0, index=sub_part_idx[:,1])
                        #siamese_label_gt = (part_label1 == part_label2)*(1 - (part_label1 == -1))*(1 - (part_label2 == -1))
                        context_idx1 = torch.index_select(inter_matrix_full,dim=0,index=sub_part_idx[:,0])
                        context_idx2 = torch.index_select(inter_matrix_full,dim=0,index=sub_part_idx[:,1])
                        context_mask1 = (torch.matmul(context_idx1.float(), cur_mask_pool)>0).float()
                        context_mask2 = (torch.matmul(context_idx2.float(), cur_mask_pool)>0).float()
                        context_mask = ((context_mask1+context_mask2)>0).float()
                        context_xyz, xyz_mean = mask_to_xyz(pc, context_mask, sample_num=2048)
                        context_xyz = context_xyz - xyz_mean
                        context_xyz /= context_xyz.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                        #siamese_label = (part_label1 == part_label2)
                        logits1 = model_merge(part_xyz1,'head_rnd')
                        logits2 = model_merge(part_xyz2,'head_rnd')
                        context_logits = model_merge(context_xyz,'head_rnd')
                        x1 = torch.cat([logits1.clone(), logits2.clone(), context_logits.clone()],dim=1)
                        logits1 = model_merge(part_xyz1,'head_pred')
                        logits2 = model_merge(part_xyz2,'head_pred')
                        context_logits = model_merge(context_xyz,'head_pred')
                        x2 = torch.cat([logits1, logits2, context_logits],dim=1)
                        logits_rnd = model_merge(x1, 'rnd', x2)
                        #print(logits_rnd)
                        meters.update(logits_rnd=logits_rnd)
                        #merge_logits = model_merge(torch.cat([part_xyz, torch.cat([logits1.unsqueeze(-1).expand(-1,-1,part_xyz1.shape[-1]), logits2.unsqueeze(-1).expand(-1,-1,part_xyz2.shape[-1])], dim=-1)], dim=1), 'head')
                        #if (remote_flag != True) and ((cur_xyz_pool.shape[0] >= 32) or (logits_rnd > 0.40)):
                        #if (remote_flag != True) and (cur_xyz_pool.shape[0] >= 32):# or (logits_rnd > 0.40)):
                        #if True:
                        #if (cur_xyz_pool.shape[0] > 32) or (logits_rnd > 0.40):
                        #if (cur_xyz_pool.shape[0] > 64):
                        if (cur_xyz_pool.shape[0] > 32):
                            #merge_logits = model_merge(torch.cat([logits1,logits2],-1),'head')
                            #merge_logits = model_merge(torch.cat([part_xyz1, part_xyz2, logits1.unsqueeze(-1).expand(-1,-1,part_xyz1.shape[-1]), logits2.unsqueeze(-1).expand(-1,-1,part_xyz2.shape[-1])], dim=1), 'head')
                            logits1 = model_merge(part_xyz1,'backbone')
                            logits2 = model_merge(part_xyz2,'backbone')
                            merge_logits = model_merge(torch.cat([part_xyz, torch.cat([logits1.unsqueeze(-1).expand(-1,-1,part_xyz1.shape[-1]), logits2.unsqueeze(-1).expand(-1,-1,part_xyz2.shape[-1])], dim=-1)], dim=1), 'head')
                        else:
                            #logits1 = model_merge(part_xyz1,'backbone_c')
                            #logits2 = model_merge(part_xyz2,'backbone_c')
                            logits1 = model_merge(part_xyz1,'backbone')
                            logits2 = model_merge(part_xyz2,'backbone')
                            context_logits = model_merge(context_xyz,'backbone2')
                            merge_logits = model_merge(torch.cat([part_xyz, torch.cat([logits1.unsqueeze(-1).expand(-1,-1,part_xyz1.shape[-1]), logits2.unsqueeze(-1).expand(-1,-1,part_xyz2.shape[-1])], dim=-1), torch.cat([context_logits.unsqueeze(-1).expand(-1,-1,part_xyz.shape[-1])], dim=-1)], dim=1), 'head2')

                        _, p = torch.max(merge_logits, 1)
                        #siamese_label = p*((purity_score>p_thresh).long())
                        if not remote_flag:
                            siamese_label = p*((purity_score>p_thresh).long())
                            #siamese_label = p#*((purity_score>p_thresh).long())
                        else:
                            siamese_label = p#*((purity_score>p_thresh).long())
                        negative_num += torch.sum(siamese_label == 0)
                        positive_num += torch.sum(siamese_label == 1)
                        pred_label_list.append(siamese_label.cpu().data.numpy())

                        #info
                        #gt_label_list.append(siamese_label_gt.cpu().data.numpy())
                        new_part_mask = 1-(1-part_mask11)*(1-part_mask22)
                        size_list.append(torch.sum(new_part_mask).cpu().data.numpy())
                        size1 = torch.sum(part_mask11).cpu().data.numpy()
                        size2 = torch.sum(part_mask22).cpu().data.numpy()
                        relative_size_list.append(size1/size2+size2/size1)
                        #box_label_expand = torch.zeros((new_part_mask.shape[0], 200)).cuda()
                        #box_idx_expand = tile(data_batch['ins_id'][i].unsqueeze(0),0,new_part_mask.shape[0]).cuda()
                        #box_label_expand = box_label_expand.scatter_add_(dim=1, index=box_idx_expand, src=new_part_mask).float()
                        #maximum_label_num, _ = torch.max(box_label_expand, 1)
                        #total_num = torch.sum(box_label_expand, 1)
                        #box_purity = maximum_label_num / (total_num+1e-6)
                        #gt_purity_list.append(box_purity.cpu().data.numpy())


                        #xyz_pool1 = torch.cat([xyz_pool1, part_xyz1.clone()], dim=0)
                        #xyz_pool2 = torch.cat([xyz_pool2, part_xyz2.clone()], dim=0)
                        #mask_pool1 = torch.cat([mask_pool1, part_mask11.clone()], dim=0)
                        #mask_pool2 = torch.cat([mask_pool2, part_mask22.clone()], dim=0)
                        #label_pool = torch.cat([label_pool, siamese_label.clone().float()], dim=0)
                        ##predict
                        #logits1 = model_merge(part_xyz1,'backbone')
                        #logits2 = model_merge(part_xyz2,'backbone')
                        #loss_sim += closs(logits1, logits2, siamese_label.float())

                        #merge_logits = model_merge(part_xyz1, part_xyz2)
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
                        #new_part_label = torch.index_select(part_label1, dim=0, index=siamese_label.nonzero().squeeze())
                        #new_part_label_invalid = torch.index_select(siamese_label_gt, dim=0, index=siamese_label.nonzero().squeeze()).long()
                        #new_part_label = new_part_label*new_part_label_invalid + -1*(1-new_part_label_invalid)
                        #remove totally the same term
                        #nonoverlap_mask = (torch.sum(new_part_mask,1)>torch.sum(part_mask1,1)) * (torch.sum(new_part_mask,1)>torch.sum(part_mask2,1))
                        #new_part_mask = torch.index_select(new_part_mask, dim=0, index=nonoverlap_mask.nonzero().squeeze())
                        #new_part_label = torch.index_select(new_part_label, dim=0, index=nonoverlap_mask.nonzero().squeeze())

                        equal_matrix = torch.matmul(new_part_mask,1-new_part_mask.transpose(0,1))+torch.matmul(1-new_part_mask,new_part_mask.transpose(0,1))
                        equal_matrix[torch.eye(equal_matrix.shape[0]).byte()]=1
                        fid = (equal_matrix==0).nonzero()
                        if fid.shape[0] > 0:
                            flag = torch.ones(merge_idx1.shape[0])
                            for k in range(flag.shape[0]):
                                if flag[k] != 0:
                                    flag[fid[:,1][fid[:,0]==k]] = 0
                            new_part_mask = torch.index_select(new_part_mask, dim=0, index=flag.nonzero().squeeze().cuda())
                            #new_part_label = torch.index_select(new_part_label, dim=0, index=flag.nonzero().squeeze().cuda())

                        new_part_xyz, xyz_mean = mask_to_xyz(pc, new_part_mask)

                        ##refine
                        #if (purity_score < 0.9) and torch.sum(new_part_mask) > 512:
                        ##if (purity_score < 1.9) and torch.sum(new_part_mask) > 2:
                        #    part_pc = torch.masked_select(pc,mask=new_part_mask.unsqueeze(0).byte()).reshape(3,-1).unsqueeze(0)
                        #    refine_logits = model_refine(part_pc)
                        #    refine_pred = refine_logits.argmax(1)
                        #    old_part_mask = new_part_mask.clone()
                        #    old_part_xyz = new_part_xyz.clone()

                        #    new_part_mask[new_part_mask==1] = refine_pred.float()
                        #    new_part_xyz, xyz_mean = mask_to_xyz(pc, new_part_mask)
                        #    part_xyz = new_part_xyz - xyz_mean
                        #    part_norm = part_xyz.norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                        #    part_xyz /= part_norm
                        #    refined_purity_score = model_merge(part_xyz, 'purity').squeeze()
                        #    if refined_purity_score <= purity_score:
                        #        new_part_xyz = old_part_xyz.clone()
                        #        new_part_mask = old_part_mask.clone()
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

                        #new_part_xyz -= xyz_mean
                        #add new part for cur_pool and global_pool
                        cur_mask_pool = torch.cat([cur_mask_pool, new_part_mask], dim=0)
                        subpart_mask_pool = torch.cat([subpart_mask_pool, new_part_mask], dim=0)
                        cur_xyz_pool = torch.cat([cur_xyz_pool, new_part_xyz], dim=0)
                        subpart_pool = torch.cat([subpart_pool, new_part_xyz], dim=0)
                        #centroid_label = torch.cat([centroid_label, new_part_label], dim=0)
                        #update cur_pool, pick out merged
                        cur_pool_size = cur_mask_pool.shape[0]
                        new_mask = torch.ones([cur_pool_size])
                        new_mask[merge_idx] = 0
                        new_idx = new_mask.nonzero().squeeze().cuda()
                        cur_xyz_pool = torch.index_select(cur_xyz_pool, dim=0, index=new_idx)
                        cur_mask_pool = torch.index_select(cur_mask_pool, dim=0, index=new_idx)
                        #centroid_label = torch.index_select(centroid_label, dim=0, index=new_idx)
                        inter_matrix = torch.matmul(cur_mask_pool, cur_mask_pool.transpose(0, 1))
                        inter_matrix_full = inter_matrix.clone()>minimum_overlap_pc_num
                        if remote_flag:
                            inter_matrix = 20*torch.ones([cur_mask_pool.shape[0],cur_mask_pool.shape[0]]).cuda()
                        #p2, update purity and score
                        purity_matrix = torch.index_select(purity_matrix, dim=0, index=new_idx)
                        purity_matrix = torch.index_select(purity_matrix, dim=1, index=new_idx)
                        policy_matrix = torch.index_select(policy_matrix, dim=0, index=new_idx)
                        policy_matrix = torch.index_select(policy_matrix, dim=1, index=new_idx)
                        #score_matrix = softmax(policy_matrix) * softmax(purity_matrix)
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

                        score_matrix = torch.zeros(purity_matrix.shape).cuda()
                        #score_idx = (policy_matrix!=0).nonzero()
                        score_idx = pair_idx
                        score_matrix[score_idx[:,0], score_idx[:,1]] = softmax(purity_matrix[score_idx[:,0], score_idx[:,1]] * policy_matrix[score_idx[:,0], score_idx[:,1]])
                    final_pool_size = subpart_pool.shape[0]
                    meters.update(final_pool_size=final_pool_size,negative_num=negative_num, positive_num=positive_num)

                    #size_arr = np.array(size_list)
                    #relative_size_arr = np.array(relative_size_list)
                    ##gt_label_arr = np.array(gt_label_list).squeeze()
                    #pred_label_arr = np.array(pred_label_list).squeeze()
                    ##gt_purity_arr = np.array(gt_purity_list)
                    #purity_arr = np.array(purity_list)
                    #policy_arr = np.array(policy_list)

                    ##x = np.arange(len(gt_label_list))

                    #matplotlib.use('Agg')
                    #plot_fn = osp.join(save_fig_dir_size, str(iteration)+'.png')
                    #fig = plt.figure(figsize=(20,10))
                    ##plt.plot(x, gt_label_arr, 'r*')
                    ##plt.plot(x, policy_arr, 'b*')
                    ##plt.plot(x, gt_purity_arr, 'r.')
                    ##plt.plot(x, purity_arr, 'b.')
                    #plt.plot(x, size_arr/10000+0.5, 'mo')
                    #plt.plot(x, relative_size_arr/20, 'ko')
                    ##plt.legend(['gt_label','pred_policy','gt_purity','pred_purity','size','rela_size'])
                    #plt.legend(['size','rela_size'])
                    #policy_arr = policy_arr>0.5
                    ##equal_arr = (policy_arr == gt_label_arr)
                    ##mean_policy_label0 = np.mean(equal_arr[gt_label_arr==0])
                    ##mean_policy_label0_large = np.mean(equal_arr[(gt_label_arr==0) * (size_arr > 512)])
                    ##mean_policy_label0_small = np.mean(equal_arr[(gt_label_arr==0) * (size_arr < 512)])
                    #mean_label_policy0 = np.mean(equal_arr[policy_arr==0])
                    #mean_label_policy0_large = np.mean(equal_arr[(policy_arr==0) * (size_arr > 512)])
                    #mean_label_policy0_small = np.mean(equal_arr[(policy_arr==0) * (size_arr < 512)])
                    ##plt.title('purity_error %.2f, mean_rela_size %.1f, mean_policy_label0 %.2f, mean_label_policy0 %.2f, mean_policy_label0_large %.2f, mean_label_policy0_large %.2f'%(np.mean(np.abs(purity_arr - gt_purity_arr)), np.mean(relative_size_arr), mean_policy_label0, mean_label_policy0, mean_policy_label0_large, mean_label_policy0_large))
                    #plt.title('mean_rela_size %.1f, mean_policy_label0 %.2f, mean_label_policy0 %.2f, mean_policy_label0_large %.2f, mean_label_policy0_large %.2f'%(np.mean(relative_size_arr), mean_policy_label0, mean_label_policy0, mean_policy_label0_large, mean_label_policy0_large))
                    ##plt.plot(out_rec, out_prec, 'b-')
                    ##plt.title('PR-Curve (AP: %4.2f%%)' % (ap*100))
                    #plt.xlabel('Iter')
                    ##plt.ylabel('Precision')
                    ##plt.ylim([-0.05, 1.05])
                    #fig.savefig(plot_fn)
                    #plt.close(fig)
                    #
                    #plot_fn = osp.join(save_fig_dir_gt, str(iteration)+'.png')
                    #fig = plt.figure(figsize=(20,10))
                    #plt.plot(x, gt_label_arr, 'r*')
                    #plt.xlabel('Iter')
                    #fig.savefig(plot_fn)
                    #plt.close(fig)

                    ##tot_purity_error_list.append(np.mean(np.abs(purity_arr - gt_purity_arr)))
                    ##tot_purity_error_small_list.append(np.mean(np.abs(purity_arr - gt_purity_arr)[size_arr < 512]))
                    ##tot_purity_error_large_list.append(np.mean(np.abs(purity_arr - gt_purity_arr)[size_arr > 512]))
                    #tot_pred_acc.append(np.mean(pred_label_arr==gt_label_arr))
                    #tot_pred_small_acc.append(np.mean((pred_label_arr==gt_label_arr)[size_arr<512]))
                    #tot_pred_large_acc.append(np.mean((pred_label_arr==gt_label_arr)[size_arr>512]))
                    #tot_mean_rela_size_list.append(np.mean(relative_size_arr))
                    #tot_mean_policy_label0.append(mean_policy_label0)
                    #if not np.isnan(mean_policy_label0_large):
                    #    tot_mean_policy_label0_large.append(mean_policy_label0_large)
                    #if not np.isnan(mean_policy_label0_small):
                    #    tot_mean_policy_label0_small.append(mean_policy_label0_small)
                    #tot_mean_label_policy0.append(mean_label_policy0)
                    #if not np.isnan(mean_label_policy0_large):
                    #    tot_mean_label_policy0_large.append(mean_label_policy0_large)
                    #if not np.isnan(mean_label_policy0_small):
                    #    tot_mean_label_policy0_small.append(mean_label_policy0_small)


            #perm_idx = torch.randperm(label_pool.shape[0]).cuda()
            #meters.update(pool_size=label_pool.shape[0])
            #for i in range(int(label_pool.shape[0]/bs2)):
            #    part_xyz1 = torch.index_select(xyz_pool1, dim=0, index=perm_idx[i*bs2:(i+1)*bs2])
            #    part_xyz2 = torch.index_select(xyz_pool2, dim=0, index=perm_idx[i*bs2:(i+1)*bs2])
            #    siamese_label = torch.index_select(label_pool, dim=0, index=perm_idx[i*bs2:(i+1)*bs2])
            #    logits1 = model_merge(part_xyz1,'backbone')
            #    logits2 = model_merge(part_xyz2,'backbone')
            #    loss_sim = closs(logits1, logits2, siamese_label.float())

            #    loss_dict_embed = {
            #        'loss_sim': loss_sim,
            #    }
            #    meters.update(**loss_dict_embed)
            #    total_loss_embed = sum(loss_dict_embed.values())

            #aligned_parts_mask = torch.index_select(parts_mask.float(), dim =0, index=centroid_label_pool-1)
            #n = torch.sum(subpart_mask_pool*aligned_parts_mask, 1)
            #u = torch.sum((1-(1-subpart_mask_pool)*(1- aligned_parts_mask)),1)
            #iou=n/u

            t1 = torch.matmul(cur_mask_pool,1-cur_mask_pool.transpose(0,1))
            t1[torch.eye(t1.shape[0]).byte()] = 1
            t1_id = (t1==0).nonzero()
            final_idx = torch.ones(t1.shape[0])
            final_idx[t1_id[:,0]] = 0
            cur_mask_pool = torch.index_select(cur_mask_pool, dim=0, index=final_idx.nonzero().squeeze().cuda())
            
            pred_ins_label = torch.zeros(num_points).cuda()
            #new_part_xyz, xyz_mean = mask_to_xyz(pc, cur_mask_pool)
            #new_part_xyz -= xyz_mean
            #new_part_xyz /=(new_part_xyz+1e-6).norm(dim=1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            #purity_scores = model_merge(new_part_xyz, 'purity').squeeze()
            #rank_idx = torch.argsort(purity_scores)
            #if len(rank_idx.shape) == 0:
            #    rank_idx = rank_idx.unsqueeze(0)
            for k in range(cur_mask_pool.shape[0]):
                #pred_ins_label[cur_mask_pool[rank_idx[k]].byte()] = k+1
                pred_ins_label[cur_mask_pool[k].byte()] = k+1
            valid_idx = torch.sum(cur_mask_pool,0)>0
            if torch.sum(1-valid_idx) != 0:
                valid_points = pc[:,valid_idx]
                invalid_points = pc[:,1-valid_idx]
                knn_index, _ = _F.knn_distance(invalid_points.unsqueeze(0), valid_points.unsqueeze(0), 5, False)
                invalid_pred,_ = pred_ins_label[valid_idx][knn_index.squeeze()].mode()
                pred_ins_label[1-valid_idx] = invalid_pred
            cur_mask_pool_new = torch.zeros([0,num_points]).cuda()
            for k in range(cur_mask_pool.shape[0]):
                if torch.sum(pred_ins_label==(k+1)) != 0:
                    cur_mask_pool_new = torch.cat([cur_mask_pool_new, ((pred_ins_label == (k+1)).float()).unsqueeze(0)], dim=0)
            #cur_mask_pool_new=cur_mask_pool

            #pred_ins_label = torch.zeros(num_points).cuda()
            #for k in range(cur_mask_pool.shape[0]):
            #    pred_ins_label[cur_mask_pool[k].byte()] = k+1
            #valid_idx = torch.sum(cur_mask_pool,0)>0
            #if torch.sum(1-valid_idx) != 0:
            #    valid_points = pc[:,valid_idx]
            #    invalid_points = pc[:,1-valid_idx]
            #    knn_index, _ = _F.knn_distance(invalid_points.unsqueeze(0), valid_points.unsqueeze(0), 5, False)
            #    invalid_pred,_ = pred_ins_label[valid_idx][knn_index.squeeze()].mode()
            #    pred_ins_label[1-valid_idx] = invalid_pred
            #cur_mask_pool_new = torch.zeros([0,num_points]).cuda()
            #for k in range(cur_mask_pool.shape[0]):
            #    if torch.sum(pred_ins_label==(k+1)) != 0:
            #        cur_mask_pool_new = torch.cat([cur_mask_pool_new, ((pred_ins_label == (k+1)).float()).unsqueeze(0)], dim=0)
            #more final
            #cur_mask_pool_new = torch.cat([cur_mask_pool_new, cur_mask_pool], dim=0)
            out_mask[iteration, :cur_mask_pool_new.shape[0]] = copy.deepcopy(cur_mask_pool_new.cpu().data.numpy().astype(np.bool))
            out_valid[iteration, :cur_mask_pool_new.shape[0]] = np.sum(cur_mask_pool_new.cpu().data.numpy()) > 10

    #tot_purity_error = np.array(tot_purity_error_list)
    #tot_purity_error_small = np.array(tot_purity_error_small_list)
    #tot_purity_error_large = np.array(tot_purity_error_large_list)
    #tot_pred_acc = np.array(tot_pred_acc)
    #tot_pred_small_acc = np.array(tot_pred_small_acc)
    #tot_pred_large_acc = np.array(tot_pred_large_acc)
    #tot_mean_rela_size = np.array(tot_mean_rela_size_list)
    #tot_mean_policy_label0 = np.array(tot_mean_policy_label0)
    #tot_mean_policy_label0_large = np.array(tot_mean_policy_label0_large)
    #tot_mean_policy_label0_small = np.array(tot_mean_policy_label0_small)
    #tot_mean_label_policy0 = np.array(tot_mean_label_policy0)
    #tot_mean_label_policy0_large = np.array(tot_mean_label_policy0_large)
    #tot_mean_label_policy0_small = np.array(tot_mean_label_policy0_small)

    #s = 'pred_acc %.2f, pred_small_acc %.2f, pred_large_acc %.2f, mean_rela_size %.1f, mean_policy_label0 %.2f, mean_policy_label0_large %.2f, mean_policy_label0_small %.2f, mean_label_policy0 %.2f, mean_label_policy0_large %.2f, mean_label_policy0_small %.2f'%(np.mean(tot_pred_acc), np.mean(tot_pred_small_acc), np.mean(tot_pred_large_acc), np.mean(tot_mean_rela_size), np.mean(tot_mean_policy_label0), np.mean(tot_mean_policy_label0_large),np.mean(tot_mean_policy_label0_small),np.mean(tot_mean_label_policy0),np.mean(tot_mean_label_policy0_large),np.mean(tot_mean_label_policy0_small))
    #print(s)
    #txt_fn = open(osp.join(save_fig_dir,'info.txt'),'w+')
    #txt_fn.write(s)
    #txt_fn.close()
    #
    #final_fig_dir = osp.join(save_fig_dir, 'final')
    #os.makedirs(final_fig_dir, exist_ok=True)

    #plot_fn = osp.join(final_fig_dir, 'all.png')
    #fig = plt.figure(figsize=(20,10))
    #x = np.arange(len(tot_purity_error))
    #plt.plot(x, tot_purity_error, 'ro')
    #plt.plot(x, tot_pred_acc, 'go')
    #plt.plot(x, tot_mean_rela_size, 'bo')
    #plt.plot(x, tot_mean_policy_label0, 'c*')
    #plt.plot(x, tot_mean_label_policy0, 'y*')
    #plt.legend(['purity_error','pred_acc','rela_size','policy_label0','label_policy0'])
    #plt.title('all')
    #plt.xlabel('Iter')
    #fig.savefig(plot_fn)
    #plt.close(fig)

    #plot_fn = osp.join(final_fig_dir, 'small.png')
    #fig = plt.figure(figsize=(20,10))
    #x = np.arange(len(tot_purity_error_small))
    #plt.plot(x, tot_purity_error_small, 'ro')
    #plt.plot(x, tot_pred_small_acc, 'go')
    ##plt.plot(x, tot_mean_policy_label0_small, 'c*')
    ##plt.plot(x, tot_mean_label_policy0_small, 'y*')
    #plt.legend(['purity_error','pred_acc'])
    #plt.title('small')
    #plt.xlabel('Iter')
    #fig.savefig(plot_fn)
    #plt.close(fig)
    #         
    #plot_fn = osp.join(final_fig_dir, 'large.png')
    #fig = plt.figure(figsize=(20,10))
    #x = np.arange(len(tot_purity_error_large))
    #plt.plot(x, tot_purity_error_large, 'ro')
    #plt.plot(x, tot_pred_large_acc, 'go')
    ##plt.plot(x, tot_mean_policy_label0_large, 'c*')
    ##plt.plot(x, tot_mean_label_policy0_large, 'y*')
    #plt.legend(['purity_error','pred_acc'])
    #plt.title('large')
    #plt.xlabel('Iter')
    #fig.savefig(plot_fn)
    #plt.close(fig)

    test_time = time.time() - start_time
    logger.info('Test {}  test time: {:.2f}s'.format(meters.summary_str, test_time))
    #save_shape_dir = os.path.join(save_shape_dir, cfg['DATASET']['PartNetInsSeg']['VAL']['shape'])
    #if not os.path.exists(save_shape_dir):
    #    os.mkdir(save_shape_dir)
    for i in range(int(out_mask.shape[0]/1024) +1):
        save_h5(os.path.join(output_dir_save, 'test-%02d.h5'%(i)), out_mask[i*1024:(i+1)*1024], out_valid[i*1024:(i+1)*1024], out_conf[i*1024:(i+1)*1024])


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
        output_dir_merge = output_dir.replace('@', config_path)+'_merge'
        #output_dir_merge = './outputs/pn_v7_fusionl%d_context_big_merge'%cfg.TEST.LEVEL
        #output_dir_merge = './outputs/pn_v8_fusionl%d_context_para_merge'%cfg.TEST.LEVEL
        #output_dir_merge = './outputs/pn_v8_fusionl%d_context_para_split_merge'%cfg.TEST.LEVEL
        #output_dir_merge = './outputs/pn_v8_fusionl%d_context_para_pn2_merge'%cfg.TEST.LEVEL
        output_dir_merge = './outputs/pn_v8_fusionl%d_context_para_pn2_final_merge'%cfg.TEST.LEVEL
        #output_dir_merge = './outputs/pn_v8_fusionl%d_context_para_pn2_bin_merge'%cfg.TEST.LEVEL
        #output_dir_merge = './outputs/pn_v8_fusionl%d_context_para_big_merge'%cfg.TEST.LEVEL
        #output_dir_merge = './outputs/pn_v8_fusionl%d_context_para_lr2_merge'%cfg.TEST.LEVEL
        #output_dir_merge = './outputs/pn_v7_fusionl%d_bin_merge'%cfg.TEST.LEVEL
        #output_dir_merge = './outputs/pn_v7_fusionl%d_context_big2_merge'%cfg.TEST.LEVEL
        #output_dir_merge = './outputs/pn_v8_fusionl%d_context_merge'%cfg.TEST.LEVEL
        #output_dir_merge = './outputs/pn_v8_fusionl%d_context_cs_merge'%cfg.TEST.LEVEL
        os.makedirs(output_dir_merge, exist_ok=True)
        #output_dir_save = output_dir.replace('@', config_path)+'_refine'
        output_dir_save = './results/'+cfg.DATASET.PartNetInsSeg.TEST.shape
        os.makedirs(output_dir_save, exist_ok=True)
        #output_dir_save = osp.join(output_dir_save,'Level_%d_fusion_v3'%cfg.TEST.LEVEL)
        #output_dir_save = osp.join(output_dir_save,'Level_%d_fusion_v3_1chair'%cfg.TEST.LEVEL)
        #os.makedirs(output_dir_save, exist_ok=True)
        #output_dir = osp.join('outputs/stage1/', cfg.DATASET.PartNetInsSeg.TRAIN.shape+'_'+str(cfg.DATASET.PartNetInsSeg.TRAIN.level))
        if (type(cfg.DATASET.PartNetInsSeg.TRAIN.shape).__name__ == 'list') == False:
            output_dir = osp.join('outputs/stage1/', cfg.DATASET.PartNetInsSeg.TRAIN.shape)
        else:
            if len(cfg.DATASET.PartNetInsSeg.TRAIN.shape) == 3:
                output_dir = osp.join('outputs/stage1/', 'fusion')
            else:
                output_dir = osp.join('outputs/stage1/', 'twoshape')
        #output_dir_merge = './outputs/pn_v6_fusionl%d_v3_merge'%cfg.TEST.LEVEL
        #output_dir = osp.join('outputs/stage1/', 'chair_storage')
        #output_dir_merge = output_dir.replace('@', config_path)+'_merge'

        #output_dir_save = osp.join(output_dir_save,'Level_%d_fusion_v3_1chair'%cfg.TEST.LEVEL)
        #output_dir_save = osp.join(output_dir_save,'Level_%d_fusion_v3'%cfg.TEST.LEVEL)
        output_dir_save = osp.join(output_dir_save,'Level_%d'%cfg.TEST.LEVEL)
        #output_dir_merge = output_dir.replace('@', config_path)+'_merge'
        #output_dir_merge = 'outputs/pn_v7_chair_three_merge'
        os.makedirs(output_dir_save, exist_ok=True)
        #output_dir = osp.join('outputs/stage1/', 'Chair')
        #output_dir = osp.join('outputs/stage1/', 'fusion')

        #output_dir_save = osp.join(output_dir_save,'Level_%d_chairlamp_1chair'%cfg.TEST.LEVEL)
        #os.makedirs(output_dir_save, exist_ok=True)
        #output_dir_merge = './outputs/pn_v6_chairlamp_v2_1chair_merge'
        #output_dir = osp.join('outputs/stage1/', 'Chair')

        #output_dir_save = osp.join(output_dir_save,'Level_%d_chairlamp_1two'%cfg.TEST.LEVEL)
        #os.makedirs(output_dir_save, exist_ok=True)
        #output_dir_merge = './outputs/pn_v6_chairlamp_v2_1two_merge'
        #output_dir = osp.join('outputs/stage1/', 'twoshape')
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('shaper', output_dir_save, prefix='test')
    logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
    logger.info(args)

    from core.utils.torch_util import collect_env_info
    logger.info('Collecting env info (might take some time)\n' + collect_env_info())

    logger.info('Loaded configuration file {}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.TASK == 'ins_seg_3d'
    test(cfg, output_dir, output_dir_merge, output_dir_save)

if __name__ == '__main__':
    main()
