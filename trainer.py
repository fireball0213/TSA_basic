import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from utils.metrics import mse, mae, mape, smape, mase
from utils.transforms import get_transform

class MLTrainer:
    def __init__(self, args, model, dataset):
        self.args = args
        self.model = model
        self.transform = get_transform(args)
        self.dataset = dataset
        self.period = args.period
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len


    def train(self):
        train_X = self.dataset.train_data
        t_X = self.transform.transform(train_X)
        self.model.fit(t_X)

    def evaluate(self, dataset):
        if dataset.type == 'm4':
            test_X = dataset.train_data
            test_Y = dataset.test_data
            pred_len = dataset.test_data.shape[-1]
        else:
            test_data = dataset.test_data
            test_data = self.transform.transform(test_data)
            subseries = np.concatenate(([sliding_window_view(v, (self.seq_len + self.pred_len, v.shape[-1])) for v in test_data]))
            test_X = subseries[:, 0, :self.seq_len, :]
            test_Y = subseries[:, 0, self.seq_len:, :]
        te_X = test_X
        fore = self.model.forecast(te_X)

        print('mse:', mse(fore, test_Y).round(5))
        # TODO:add other metrics


