import torch


class WeightSampler(object):

    def __init__(self, cls_weight, mode='all', ohem_frac=0.2):
        # TODO hack to eval the cls_weight
        self.cls_weight = cls_weight
        assert mode in ['all', 'ohem', 'balance']
        self.mode = mode
        self.ohem_frac = ohem_frac # only applied with ohem

    def __call__(self, ins_logit, ins_label, loss_fn=None):
        with torch.no_grad():
            weight = ins_logit.new_tensor(self.cls_weight).index_select(0, ins_label.view(-1)).view_as(ins_label)
            if self.mode == 'ohem':
                assert loss_fn is not None
                expect_num = int(self.ohem_frac * ins_label.numel())
                sample_weight = ins_logit.new_zeros(ins_label.size())
                ins_loss = loss_fn(ins_logit, ins_label, reduction='none')
                _, topk_loss_inds = ins_loss.view(-1).topk(expect_num)
                sample_weight.view(-1)[topk_loss_inds] = 1.
                weight = weight * sample_weight
            elif self.mode == 'balance':
                assert loss_fn is not None
                sample_weight = ins_logit.new_zeros(ins_label.size())
                ins_loss = loss_fn(ins_logit, ins_label, reduction='none')
                pos_num = ins_label.sum()
                if pos_num > ins_label.numel()//2:
                    mask = (ins_label == 1)
                    expect_num = ins_label.numel() - pos_num
                else:
                    mask = (ins_label == 0)
                    expect_num = pos_num
                sample_weight[1 - mask] = 1.
                _, topk_loss_inds = (ins_loss * mask.float()).view(-1).topk(expect_num)
                sample_weight.view(-1)[topk_loss_inds] = 1.
                weight = weight * sample_weight

            return weight
