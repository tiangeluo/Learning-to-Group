import torch

from core.nn.functional import bpdist
from shaper.models.dgcnn.functions import construct_edge_feature_index, construct_edge_feature_gather, \
    construct_edge_feature


def generate_data(batch_size=16, num_points=1024, in_channels=64, k=20):
    with torch.no_grad():
        feature_tensor = torch.randn(batch_size, in_channels, num_points).float()
        feature_tensor = feature_tensor.cuda()
        distance = bpdist(feature_tensor)
        _, knn_inds = torch.topk(distance, k, largest=False)
        return feature_tensor, knn_inds


def test_construct_edge_feature():
    feature_tensor, knn_inds = generate_data()

    def forward_backward(fn):
        feature = feature_tensor.clone().detach()
        feature.requires_grad = True
        with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
            edge_feature = fn(feature, knn_inds)
            edge_feature.backward(torch.ones_like(edge_feature), retain_graph=True, create_graph=False)
        print(prof)
        return edge_feature.cpu(), feature.grad.cpu()

    o_index, g_index = forward_backward(construct_edge_feature_index)
    o_gather, g_gather = forward_backward(construct_edge_feature_gather)
    o_knn, g_knn = forward_backward(construct_edge_feature)

    # forward
    assert o_index.allclose(o_gather)
    assert o_gather.allclose(o_knn)

    # backward
    assert g_index.allclose(g_gather)
    assert g_gather.allclose(g_knn)
