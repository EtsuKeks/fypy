import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward
from fypy.market.MarketSlice import MarketSlice
from fypy.market.MarketSurface import MarketSurface
from fypy.volatility.implied.ImpliedVolCalculator import ImpliedVolCalculator_Black76

def build_surface_from_hourly_data(df_hour, r) -> MarketSurface:
    if df_hour.empty:
        return MarketSurface()

    S0 = float(df_hour['underlying_price'].iloc[0])
    disc_curve = DiscountCurve_ConstRate(rate=r); fwd_curve = EquityForward.from_rates(S0=S0, r=r, q=0.0)
    surface = MarketSurface(forward_curve=fwd_curve, discount_curve=disc_curve)

    for ttm, grp in df_hour.groupby('ttm'):
        strikes, is_calls, mid_prices = grp['strike'].to_numpy(), grp['is_call'].to_numpy(), grp['close'].to_numpy()
        surface.add_slice(ttm, MarketSlice(T=ttm, F=fwd_curve(ttm), disc=disc_curve(ttm), strikes=strikes, is_calls=is_calls, mid_prices=mid_prices))

    surface.fill_implied_vols(ImpliedVolCalculator_Black76(fwd_curve=fwd_curve, disc_curve=disc_curve))
    return surface


def add_nan_columns(df_hour, heston_pricers) -> pd.DataFrame:
    df_hour['close_bs'], df_hour['close_sabr'] = np.nan, np.nan
    for pricer_name in heston_pricers:
        df_hour[f'close_heston_{pricer_name}'] = np.nan

    return df_hour
