from core.utils.metric_logger import Metric, MetricList


class DummyMetric(Metric):
    name = 'dummy_metric'

    def update_dict(self, preds, labels):
        if self.training:
            self.update(preds['a'], labels['a'])
        else:
            self.update(preds['b'], labels['b'])


def test_metric():
    metric = DummyMetric()

    # ---------------------------------------------------------------------------- #
    # Train mode
    # ---------------------------------------------------------------------------- #
    metric.reset()
    metric.train()
    # print('Train before update', str(metric))
    for i in range(1, 10):
        metric.update_dict({'a': i, 'b': 1}, {'a': 1, 'b': i})
        print('Train after {:d} update: '.format(i), str(metric), metric.summary_str)

    # ---------------------------------------------------------------------------- #
    # Eval mode
    # ---------------------------------------------------------------------------- #
    metric.reset()
    metric.eval()
    # print('Eval before update', str(metric))
    for i in range(1, 10):
        metric.update_dict({'a': i, 'b': 1}, {'a': 1, 'b': i})
        print('Eval after {:d} update: '.format(i), str(metric), metric.summary_str)

    # ---------------------------------------------------------------------------- #
    # MetricList
    # ---------------------------------------------------------------------------- #
    metric1 = DummyMetric()
    metric2 = DummyMetric()
    metric_list = MetricList([metric1, metric2])
    metric_list.reset()
    metric_list[0].train()
    metric_list[1].eval()
    for i in range(1, 10):
        metric_list.update_dict({'a': i, 'b': 1}, {'a': 1, 'b': i})
        print('(1) Eval after {:d} update: '.format(i), str(metric1), metric1.summary_str)
        print('(2) Eval after {:d} update: '.format(i), str(metric2), metric2.summary_str)
