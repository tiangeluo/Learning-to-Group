"""Build models

Notes:
    When a new model is implemented, please provide a builder to build the model with config,
    and register it in _MODEL_BUILDERS

    How to implement a model:
    1. Modularize the model
    2. Try to add in_channels, out_channels to all the modules' attributes
    3. For the complete model, like PointNetCls, output a non-nested dictionary instead of single tensor or tuples
    4. Implement loss module whose inputs are preds and labels. Both of inputs are dict.
    5. Implement metric module (or use a general module in 'metric.py')

"""

from .classifiers import build_classifier
from .metric import InsComAccuracy
from .sampler import WeightSampler
from .loss import PairInsSegLoss


def build_model(cfg):
    custom_cfg = cfg.MODEL[cfg.MODEL.TYPE]
    classifier_kwargs = custom_cfg.pop('classifier')
    loss_kwargs = custom_cfg.pop('loss')
    #sampler_kwargs = loss_kwargs.pop('sampler')
    model = build_classifier(classifier_kwargs)
    #sampler = WeightSampler(**sampler_kwargs)
    loss_fn = PairInsSegLoss(**loss_kwargs)#, sampler=sampler)
    train_metric = InsComAccuracy()
    val_metric = InsComAccuracy()

    return model, loss_fn, train_metric, val_metric
