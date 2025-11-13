from models.baselines import ZeroForecast, MeanForecast, LastPeriodForecast
from utils.transforms import IdentityTransform
from trainer import MLTrainer
from dataset.dataset import get_dataset
import argparse
import random
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument('--data_path', type=str, default='./dataset/ETT/ETTh1.csv')
    parser.add_argument('--train_data_path', type=str, default='./dataset/m4/Daily-train.csv')
    parser.add_argument('--test_data_path', type=str, default='./dataset/m4/Daily-test.csv')
    parser.add_argument('--dataset', type=str, default='ETT', help='dataset type, options: [M4, ETT, Custom]')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--ratio_train', type=int, default=0.6, help='train dataset length')
    parser.add_argument('--ratio_val', type=int, default=0.2, help='validate dataset length')
    parser.add_argument('--ratio_test', type=int, default=0.2, help='input sequence length')
    parser.add_argument('--frequency', type=str, default='h', help='frequency of time series data, options: [h, m]')
    parser.add_argument('--period', type=int, default=24, help='period used in datasets,ETT:24,illness:52')
    parser.add_argument('--channels', type=int, default=7, help='channels of multivariate data')

    # forcast task config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length in [96, 192, 336, 720]')

    # model define
    parser.add_argument('--model', type=str, required=True, default='MeanForecast', help='model name')
    parser.add_argument('--n_neighbors', type=int, default=1, help='number of neighbors used in TsfKNN')
    parser.add_argument('--distance', type=str, default='euclidean', help='distance used in TsfKNN')
    parser.add_argument('--msas', type=str, default='MIMO', help='multi-step ahead strategy used in TsfKNN, options: '
                                                                 '[MIMO, recursive]')

    # Dlinear define
    parser.add_argument('--individual', action='store_true', help='individual training for each channel')

    # transform define
    parser.add_argument('--transform', type=str, default='IdentityTransform')

    args = parser.parse_args()
    return args


def get_model(args):
    model_dict = {
        'ZeroForecast': ZeroForecast,
        'MeanForecast': MeanForecast,
        'LastPeriodForecast': LastPeriodForecast,
    }
    return model_dict[args.model](args)



if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    args = get_args()
    # load dataset
    dataset = get_dataset(args)
    # create model
    model = get_model(args)
    # create trainer
    trainer = MLTrainer(args=args, model=model, dataset=dataset)
    # train model
    trainer.train()
    # evaluate model
    trainer.evaluate(dataset)