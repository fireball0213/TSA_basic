def get_decompose(args):
    decompose_dict = {
        'MA': moving_average_series,
        'Diff': differential_decomposition,
        'STL': STL_decomposition,
        'None': None,
    }
    return decompose_dict[args.decompose]

def moving_average_series(x,seasonal_period):
    """
    Moving Average Algorithm
    Args:
        x (numpy.ndarray): Input time series data
        seasonal_period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """
    # TODO
    raise NotImplementedError

#通过差分时间序列数据来分离趋势和季节性成分
def differential_decomposition(x):
    """
    Differential Decomposition Algorithm
    Args:
        x (numpy.ndarray): Input time series data
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """
    # TODO
    raise NotImplementedError


def STL_decomposition(x, seasonal_period):
    """
    STL Algorithm
    Args:
        x (numpy.ndarray): Input time series data
        seasonal_period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """
    # TODO
    raise NotImplementedError

