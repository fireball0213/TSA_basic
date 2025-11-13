import numpy as np

def naive_forecast(y:np.array, season:int=1):
  "naive forecast: season-ahead step forecast, shift by season step ahead"
  return y[:-season]

def mse(predict, target):
    return np.mean((target - predict) ** 2)


def mae(predict, target):
    # TODO
    raise NotImplementedError


def mape(predict, target):
    # TODO
    raise NotImplementedError


def smape(predict, target):
    # TODO
    raise NotImplementedError


def mase(predict, target, season=24):
    # TODO
    raise NotImplementedError
