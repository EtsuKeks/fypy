import warnings
warnings.filterwarnings('ignore')

import numpy as np

from fypy.model.levy.BlackScholes import BlackScholes
from fypy.model.sv.Heston import Heston
from fypy.model.slv.Sabr import Sabr
from fypy.pricing.analytical.black_scholes import black76_price_strikes
from fypy.pricing.analytical.SabrHaganOblojPricer import SabrHaganOblojPricer
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.pricing.fourier.CarrMadanEuropeanPricer import CarrMadanEuropeanPricer
from fypy.pricing.fourier.GilPeleazEuropeanPricer import GilPeleazEuropeanPricer
from fypy.pricing.fourier.LewisEuropeanPricer import LewisEuropeanPricer
from fypy.pricing.fourier.HilbertEuropeanPricer import HilbertEuropeanPricer

def predict_black_scholes_analytical(model: BlackScholes, surface) -> np.ndarray:
    prices = []
    for ttm, slc in surface.slices.items():
        pred_prices = black76_price_strikes(F=slc.F, K=slc.strikes, is_calls=slc.is_calls, vol=model.vol, disc=slc.disc, T=ttm)
        prices.append(pred_prices)

    return np.concatenate(prices)


def predict_heston_fourier(model: Heston, surface, pricer_name) -> np.ndarray:
    prices = []
    for _, slc in surface.slices.items():
        if pricer_name == "proj":
            pricer = ProjEuropeanPricer(model=model, N=2**12, L=20)
        elif pricer_name == "lewis":
            pricer = LewisEuropeanPricer(model=model)
        elif pricer_name == "gil":
            pricer = GilPeleazEuropeanPricer(model=model)
        elif pricer_name == "carr":
            pricer = CarrMadanEuropeanPricer(model=model)
        elif pricer_name == "hilbert":
            pricer = HilbertEuropeanPricer(model=model)
        
        pred_prices = pricer.price_strikes(T=slc.T, K=slc.strikes, is_calls=slc.is_calls)
        prices.append(pred_prices)

    return np.concatenate(prices)


def predict_sabr_analytical(model: Sabr, surface) -> np.ndarray:
    prices = []
    for _, slc in surface.slices.items():
        pred_prices = SabrHaganOblojPricer(model).price_strikes(T=slc.T, K=slc.strikes, is_calls=slc.is_calls)
        prices.append(pred_prices)

    return np.concatenate(prices)
