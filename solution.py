import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tqdm import tqdm

from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward
from fypy.market.MarketSlice import MarketSlice
from fypy.market.MarketSurface import MarketSurface
from fypy.model.levy.BlackScholes import BlackScholes
from fypy.model.sv.Heston import Heston
from fypy.model.slv.Sabr import Sabr
from fypy.calibrate.FourierModelCalibrator import FourierModelCalibrator
from fypy.calibrate.SabrModelCalibrator import SabrModelCalibrator
from fypy.pricing.analytical.black_scholes import black76_price_strikes
from fypy.pricing.analytical.SabrHaganOblojPricer import SabrHaganOblojPricer
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.pricing.fourier.CarrMadanEuropeanPricer import CarrMadanEuropeanPricer
from fypy.pricing.fourier.GilPeleazEuropeanPricer import GilPeleazEuropeanPricer
from fypy.pricing.fourier.LewisEuropeanPricer import LewisEuropeanPricer
from fypy.pricing.fourier.HilbertEuropeanPricer import HilbertEuropeanPricer
from fypy.volatility.implied.ImpliedVolCalculator import ImpliedVolCalculator_Black76

def build_surface_from_hourly_data(df_hour, r) -> MarketSurface:
    if df_hour.empty:
        return MarketSurface()

    S0 = float(df_hour['underlying_price'].iloc[0])
    disc_curve = DiscountCurve_ConstRate(rate=r); fwd_curve = EquityForward.from_rates(S0=S0, r=r, q=0.0)
    surface = MarketSurface(forward_curve=fwd_curve, discount_curve=disc_curve)

    for ttm, grp in df_hour.groupby('ttm'):
        strikes, is_calls, mid_prices = grp['strike'].to_numpy(), grp['is_call'].to_numpy(), grp['close'].to_numpy()
        mkt_slice = MarketSlice(T=ttm, F=fwd_curve(ttm), disc=disc_curve(ttm), strikes=strikes, is_calls=is_calls, mid_prices=mid_prices)
        surface.add_slice(ttm, mkt_slice)

    surface.fill_implied_vols(ImpliedVolCalculator_Black76(fwd_curve=fwd_curve, disc_curve=disc_curve))
    return surface

def calibrate_black_scholes(surface, prev_model) -> BlackScholes:
    if prev_model is not None:
        model = BlackScholes(forwardCurve=surface.forward_curve, discountCurve=surface.discount_curve, sigma=prev_model.vol)
    else:
        model = BlackScholes(forwardCurve=surface.forward_curve, discountCurve=surface.discount_curve, sigma=0.2)
    FourierModelCalibrator(surface=surface, do_vega_weight=False).calibrate(model=model)
    return model

def calibrate_heston(surface, prev_model) -> Heston:
    if prev_model is not None:
        model = Heston(forwardCurve=surface.forward_curve, discountCurve=surface.discount_curve,
                      v_0=prev_model.v_0, theta=prev_model.theta, kappa=prev_model.kappa, 
                      sigma_v=prev_model.sigma_v, rho=prev_model.rho)
    else:
        model = Heston(forwardCurve=surface.forward_curve, discountCurve=surface.discount_curve)
    FourierModelCalibrator(surface=surface, do_vega_weight=False).calibrate(model=model)
    return model

def calibrate_sabr(surface, prev_model):
    if prev_model is not None:
        model = Sabr(forwardCurve=surface.forward_curve, discountCurve=surface.discount_curve,
                    v_0=prev_model.v_0, alpha=prev_model.alpha, beta=prev_model.beta, rho=prev_model.rho)
    else:
        model = Sabr(forwardCurve=surface.forward_curve, discountCurve=surface.discount_curve)
    _, pricer, _, _ = SabrModelCalibrator(surface=surface).calibrate(model=model)
    return model, pricer

def predict_black_scholes_analytical(model: BlackScholes, surface) -> np.ndarray:
    prices = []
    for ttm, slc in surface.slices.items():
        pred_prices = black76_price_strikes(F=slc.F, K=slc.strikes, is_calls=slc.is_calls, vol=model.vol, disc=slc.disc, T=ttm)
        prices.append(pred_prices)

    if len(prices) == 0:
        return np.array([])
    return np.concatenate(prices)

def predict_heston_fourier(model: Heston, surface, pricer_class, pricer_name) -> np.ndarray:
    prices = []
    for _, slc in surface.slices.items():
        if pricer_name == "proj":
            pricer = pricer_class(model=model, N=2**12, L=20)
        elif pricer_name == "lewis":
            pricer = pricer_class(model=model, N=2**12, limit=200)
        elif pricer_name == "gil":
            pricer = pricer_class(model=model, limit=1000)
        elif pricer_name == "carr":
            pricer = pricer_class(model=model, alpha=0.75, eta=0.1, N=2**9)
        elif pricer_name == "hilbert":
            pricer = pricer_class(model=model, alpha=0.75, eta=0.1, N=2**9, Nh=2**5)
        
        pred_prices = pricer.price_strikes(T=slc.T, K=slc.strikes, is_calls=slc.is_calls)
        prices.append(pred_prices)

    if len(prices) == 0:
        return np.array([])
    return np.concatenate(prices)

def predict_sabr_analytical(model: Sabr, pricer: SabrHaganOblojPricer, surface) -> np.ndarray:
    prices = []
    for _, slc in surface.slices.items():
        pred_prices = pricer.price_strikes(T=slc.T, K=slc.strikes, is_calls=slc.is_calls)
        prices.append(pred_prices)

    if len(prices) == 0:
        return np.array([])
    return np.concatenate(prices)

def process_hourly_data(df, r) -> pd.DataFrame:
    results = []
    hourly_groups = df.groupby('current_time'); timestamps = list(hourly_groups.groups.keys())

    heston_pricers = [
        (ProjEuropeanPricer, "proj"),
        (CarrMadanEuropeanPricer, "carr"),
        (GilPeleazEuropeanPricer, "gil"),
        (LewisEuropeanPricer, "lewis"),
        (HilbertEuropeanPricer, "hilbert")
    ]

    bs_model, heston_model, sabr_model = None, None, None
    for timestamp in tqdm(timestamps, desc="Processing hours", unit="hour"):
        current_hour_data = hourly_groups.get_group(timestamp)
        if len(current_hour_data) < 2:
            results.append(add_nan_columns(current_hour_data, heston_pricers))
            continue

        surface = build_surface_from_hourly_data(current_hour_data, r)
        current_hour_data = add_nan_columns(current_hour_data, heston_pricers)

        if len(surface.slices) == 0:
            print(f"No valid slices for timestamp {timestamp}")
            results.append(current_hour_data)
            continue

        bs_model = calibrate_black_scholes(surface, bs_model)
        current_hour_data['close_bs'] = predict_black_scholes_analytical(bs_model, surface)

        heston_model = calibrate_heston(surface, heston_model)
        for pricer_class, pricer_name in heston_pricers:
            current_hour_data[f'close_heston_{pricer_name}'] = predict_heston_fourier(heston_model, surface, pricer_class, pricer_name)

        # sabr_model, sabr_pricer = calibrate_sabr(surface, sabr_model)
        # current_hour_data['close_sabr'] = predict_sabr_analytical(sabr_model, sabr_pricer, surface)

        results.append(current_hour_data)

    return pd.concat(results, ignore_index=True)

def add_nan_columns(df_hour, heston_pricers) -> pd.DataFrame:
    df_hour['close_bs'] = np.nan
    df_hour['close_sabr'] = np.nan
    for _, pricer_name in heston_pricers:
        df_hour[f'close_heston_{pricer_name}'] = np.nan

    return df_hour

df = pd.read_csv("~/Downloads/binance_dump_with_valid_defaults.csv")
df_result = process_hourly_data(df, r=0.05)
df_result.to_csv("~/Downloads/binance_dump_with_valid_defaults_and_models_preds.csv", index=False)