from abc import ABC, abstractmethod
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
from fypy.model.FourierModel import FourierModel
from typing import Union
import numpy as np


class LevyModel(FourierModel, ABC):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 params: np.ndarray):
        """
        Base class for an exponential Levy model, which is a model for which the characteristic function is known, hence
        enabling pricing by Fourier methods. These models are defined uniquely by their Levy "symbol", which determines
        the chf by:   chf(T,xi) = exp(T*symbol(xi)),  for all T>=0

        :param forwardCurve: Forward curve term structure
        """
        super().__init__(discountCurve=discountCurve,
                         forwardCurve=forwardCurve)
        self._forwardCurve = forwardCurve  # Overrides base forward
        self._params = params

    def chf(self, T: float, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Characteristic function
        :param T: float, time to maturity
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, characteristic function evaluated at input points in frequency domain
        """
        return np.exp(T * self.symbol(xi))

    @abstractmethod
    def symbol(self, xi: Union[float, np.ndarray]):
        """
        Levy symbol, uniquely defines Characteristic Function via: chf(T,xi) = exp(T * symbol(xi)), for all T>=0
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, symbol evaluated at input points in frequency domain
        """
        raise NotImplementedError

    @abstractmethod
    def convexity_correction(self) -> float:
        """
        Computes the convexity correction for the Levy model, added to log process drift to ensure
        risk neutrality
        """
        raise NotImplementedError

    def risk_neutral_log_drift(self) -> float:
        """ Compute the risk-neutral drift of log process """
        return self.forwardCurve.drift(0, 1) + self.convexity_correction()

    def set_params(self, params: np.ndarray):
        self._params = params
        return self

    def get_params(self) -> np.ndarray:
        return self._params
