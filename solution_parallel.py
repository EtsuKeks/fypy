import warnings
warnings.filterwarnings('ignore')

import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from src.utils import build_surface_from_hourly_data, add_nan_columns
from src.predict import predict_black_scholes_analytical, predict_heston_fourier, predict_sabr_analytical
from src.calibrate import calibrate_black_scholes, calibrate_heston, calibrate_sabr

global_progress_counter = None

def init_pool_processes(counter):
    global global_progress_counter
    global_progress_counter = counter

def process_chunk(chunk, r):
    results = []
    heston_pricers = ["proj", "carr", "gil", "lewis", "hilbert"]
    bs_model, heston_model, sabr_model = None, None, None

    for i in range(len(chunk) - 1):
        current_hour_data = chunk[i]
        next_hour_data = add_nan_columns(chunk[i+1], heston_pricers)
        current_surface = build_surface_from_hourly_data(current_hour_data, r)
        next_surface = build_surface_from_hourly_data(next_hour_data, r)

        bs_model = calibrate_black_scholes(current_surface, bs_model)
        next_hour_data['close_bs'] = predict_black_scholes_analytical(bs_model, next_surface)

        heston_model = calibrate_heston(current_surface, heston_model)
        for pricer_name in heston_pricers:
            next_hour_data[f'close_heston_{pricer_name}'] = predict_heston_fourier(heston_model, next_surface, pricer_name)

        # sabr_model = calibrate_sabr(current_surface, sabr_model)
        # next_hour_data['close_sabr'] = predict_sabr_analytical(sabr_model, next_surface)

        results.append(next_hour_data)

        with global_progress_counter.get_lock():
            global_progress_counter.value += 1

    return pd.concat(results)

def process_hourly_data_parallel(df, r, n_workers):
    df.reset_index(drop=True, inplace=True)
    groups_list = [group for _, group in list(df.groupby('current_time'))]
    chunks = [[groups_list[i] for i in chunk_idx] for chunk_idx in np.array_split(np.arange(len(groups_list)), n_workers)]

    total_iterations = sum(len(chunk) - 1 for chunk in chunks)
    progress_counter = multiprocessing.Value('i', 0)
    results, futures = [], []
    with tqdm(total=total_iterations, desc="Overall progress") as pbar:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_pool_processes, initargs=(progress_counter,)) as exe:
            for chunk in chunks:
                futures.append(exe.submit(process_chunk, chunk, r))

            last_value = 0
            while len(results) < len(futures):
                current_value = progress_counter.value
                pbar.update(current_value - last_value)
                last_value = current_value

                done_futures = [f for f in futures if f.done() and f not in results]
                for future in done_futures:
                    results.append(future)

                time.sleep(0.1)

            pbar.n = total_iterations
            pbar.refresh()

    out = pd.concat([future.result() for future in results]).sort_values('current_time')
    return out

if __name__ == '__main__':
    df = pd.read_csv("~/Downloads/binance_dump_with_valid_volumes_arbitrage_free.csv")
    df_result = process_hourly_data_parallel(df, r=0.00, n_workers=10)
    df_result.to_csv("~/Downloads/binance_dump_with_models_preds.csv", index=False)
