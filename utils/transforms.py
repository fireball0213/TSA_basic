import numpy as np

def get_transform(args):
    transform_dict = {
        'IdentityTransform': IdentityTransform,
        # TODO:实现其他标准化方法
    }
    return transform_dict[args.transform](args)

class Transform:
    """
    Preprocess time series
    """

    def transform(self, data):
        """
        :param data: raw timeseries
        :return: transformed timeseries
        """
        raise NotImplementedError

    def inverse_transform(self, data):
        """
        :param data: raw timeseries
        :return: inverse_transformed timeseries
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self, args):
        pass

    def transform(self, data, update=False):
        return data

    def inverse_transform(self, data):
        return data

#TODO: implement other transforms such as Normalization, Standardization, etc.