import numpy as np
from scipy.special import softmax
import scipy.spatial.distance as sdist
import torch

from core.nn.functional import bpdist, bpdist2, pdist2
from core.nn.functional import encode_one_hot, smooth_cross_entropy


def test_bpdist():
    batch_size = 16
    channels = 64
    num_inst = 1024

    feature = np.random.rand(batch_size, channels, num_inst)
    feature_tensor = torch.from_numpy(feature)
    if torch.cuda.is_available():
        feature_tensor = feature_tensor.cuda()

    # check pairwise distance
    distance = np.stack([sdist.squareform(np.square(sdist.pdist(x.T))) for x in feature])

    distance_tensor = bpdist(feature_tensor)  # warm up
    np.testing.assert_allclose(distance, distance_tensor.cpu().numpy(), atol=1e-6)

    # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    #     bpdist(feature_tensor)
    # print(prof)


def test_bpdist2():
    batch_size = 16
    channels = 64
    num_inst1 = 1023
    num_inst2 = 1025

    feature1 = np.random.rand(batch_size, channels, num_inst1)
    feature2 = np.random.rand(batch_size, channels, num_inst2)
    feature1_tensor = torch.from_numpy(feature1)
    feature2_tensor = torch.from_numpy(feature2)
    if torch.cuda.is_available():
        feature1_tensor = feature1_tensor.cuda()
        feature2_tensor = feature2_tensor.cuda()

    # check pairwise distance
    distance = np.stack([np.square(sdist.cdist(x.T, y.T)) for x, y in zip(feature1, feature2)])

    distance_tensor = bpdist2(feature1_tensor, feature2_tensor)  # warm up
    np.testing.assert_allclose(distance, distance_tensor.cpu().numpy())

    # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    #     bpdist2(feature1_tensor, feature2_tensor)
    # print(prof)


def test_pdist2():
    channels = 64
    num_inst1 = 1023
    num_inst2 = 1025

    feature1 = np.random.rand(num_inst1, channels)
    feature2 = np.random.rand(num_inst2, channels)
    feature1_tensor = torch.from_numpy(feature1)
    feature2_tensor = torch.from_numpy(feature2)
    if torch.cuda.is_available():
        feature1_tensor = feature1_tensor.cuda()
        feature2_tensor = feature2_tensor.cuda()

    # check pairwise distance
    distance = np.square(sdist.cdist(feature1, feature2))

    distance_tensor = pdist2(feature1_tensor, feature2_tensor)  # warm up
    np.testing.assert_allclose(distance, distance_tensor.cpu().numpy())

    # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    #     pdist2(feature1_tensor, feature2_tensor)
    # print(prof)


def test_smooth_cross_entropy():
    num_samples = 2
    num_classes = 10
    label_smoothing = 0.1

    # numpy version
    target_np = np.random.randint(0, num_classes, [num_samples])
    one_hot_np = np.zeros([num_samples, num_classes])
    one_hot_np[np.arange(num_samples), target_np] = 1.0
    smooth_one_hot = one_hot_np * (1.0 - label_smoothing) + np.ones_like(one_hot_np) * label_smoothing / num_classes
    logit_np = np.random.randn(num_samples, num_classes)
    prob_np = softmax(logit_np, axis=-1)
    cross_entropy_np = - (smooth_one_hot * np.log(prob_np)).sum(1).mean()

    target = torch.from_numpy(target_np)
    logit = torch.from_numpy(logit_np)

    one_hot = encode_one_hot(target, num_classes)
    np.testing.assert_allclose(one_hot_np, one_hot.numpy())

    cross_entropy = smooth_cross_entropy(logit, target, label_smoothing)
    np.testing.assert_allclose(cross_entropy_np, cross_entropy.numpy())
