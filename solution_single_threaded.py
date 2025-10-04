import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from tqdm import tqdm

from src.utils import build_surface_from_hourly_data, add_nan_columns
from src.predict import predict_black_scholes_analytical, predict_heston_fourier, predict_sabr_analytical
from src.calibrate import calibrate_black_scholes, calibrate_heston, calibrate_sabr


def process_chunk(dfs, r, pbar):
    results = []
    heston_pricers = ["proj", "carr", "gil", "lewis", "hilbert"]
    bs_model, heston_model, sabr_model = None, None, None

    for i in range(len(dfs) - 1):
        current_hour_data = dfs[i]
        next_hour_data = add_nan_columns(dfs[i+1], heston_pricers)
        current_surface = build_surface_from_hourly_data(current_hour_data, r)
        next_surface = build_surface_from_hourly_data(next_hour_data, r)

        bs_model = calibrate_black_scholes(current_surface, bs_model)
        next_hour_data['close_bs'] = predict_black_scholes_analytical(bs_model, next_surface)

        heston_model = calibrate_heston(current_surface, heston_model)
        for pricer_name in heston_pricers:
            next_hour_data[f'close_heston_{pricer_name}'] = predict_heston_fourier(heston_model, next_surface, pricer_name)

        sabr_model = calibrate_sabr(current_surface, sabr_model)
        next_hour_data['close_sabr'] = predict_sabr_analytical(sabr_model, next_surface)

        results.append(next_hour_data)
        pbar.update(1)

    return pd.concat(results)


def process_hourly_data_parallel(df, r):
    dfs = [group for _, group in list(df.groupby('current_time'))]
    with tqdm(total=len(dfs) - 1, desc="Overall progress") as pbar:
        return process_chunk(dfs, r, pbar)

if __name__ == '__main__':
    df = pd.read_csv("~/Downloads/binance_dump_with_valid_volumes_arbitrage_free_test.csv")
    df_result = process_hourly_data_parallel(df, r=0.00)
    df_result.to_csv("~/Downloads/binance_dump_with_models_preds_test.csv", index=False)
