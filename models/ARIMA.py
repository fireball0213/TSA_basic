import numpy as np

from models.baselines import MLForecastModel


class ARIMA(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        # TODO
        raise NotImplementedError

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        # TODO
        raise NotImplementedError