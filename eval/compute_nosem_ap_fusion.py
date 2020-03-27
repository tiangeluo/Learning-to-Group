import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from eval_utils import eval_per_class_ap_nosem, eval_per_shape_mean_ap_nosem, eval_recall_iou_nosem_fusion
import numpy as np
from subprocess import call
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, help='Category name [default: Chair]')
parser.add_argument('--level_id', type=int, help='Level ID [default: 3]')
parser.add_argument('--pred_dir', default='../results/',type=str, help='log prediction directory [default: log]')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU Threshold [default: 0.5]')
parser.add_argument('--plot_dir', type=str, default=None, help='PR Curve Plot Output Directory [default: None, meaning no output]')
FLAGS = parser.parse_args()

gt_in_dir = '../data/partnet/ins_seg_h5_gt/%s-%d/' % (FLAGS.category, FLAGS.level_id)
pred_dir = FLAGS.pred_dir

recalls = eval_recall_iou_nosem_fusion(gt_in_dir, os.path.join(pred_dir,FLAGS.category), iou_threshold=FLAGS.iou_threshold, plot_dir=FLAGS.plot_dir)
print(recalls)
print('mRecall %f'%np.mean(recalls))
f = open('results.txt','a+')
f.write('%s-%d: %.3f\n'%(FLAGS.category,FLAGS.level_id,np.mean(recalls)))
f.close()

