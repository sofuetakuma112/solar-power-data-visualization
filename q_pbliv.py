import argparse
import datetime
from utils.phase_diff import (
    calculate_phase_diffs,
    generate_time_array,
    perform_frequency_analysis,
    plot_calced_q_graph,
    plot_phase_graph,
)
from utils.init_matplotlib import figsize_px_to_inch
import matplotlib.pyplot as plt
import numpy as np
from utils.init_matplotlib import figsize_px_to_inch, init_rcParams
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from utils.q import Q

from scipy.signal import find_peaks

FONT_SIZE = 14


# > python3 phase_difference/q_simulation.py -max_shift 1000
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", type=str)  # グラフ描画したい日付のリスト
    parser.add_argument(
        "-model", type=str, default="isotropic"
    )  # 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'
    parser.add_argument("-surface_tilt", type=int, default=22)
    parser.add_argument("-surface_azimuth", type=float, default=185.0)
    parser.add_argument("-shift_offset", type=int, default=0)
    parser.add_argument("-max_shift", type=int, default=10)
    parser.add_argument("-plot_calced_q", action="store_true")
    return parser.parse_args()


def find_peak_times(t, q_all):
    peaks, _ = find_peaks(q_all)
    return t[peaks]

# 指定したtarget_durationの長さになるように左右を均等に切り落とす
# FIXME: target_durationが奇数秒の場合には対応していない
def drop_data_points(time_series, target_duration):
    num_points = len(time_series)
    target_points = int(target_duration.total_seconds())
    drop_points = num_points - target_points

    if drop_points <= 0:
        return time_series

    drop_per_side = drop_points // 2
    remaining_points = num_points - drop_per_side * 2

    dropped_series = time_series[drop_per_side:drop_per_side + remaining_points]

    return dropped_series


def main(args):
    figsize_inch = figsize_px_to_inch(np.array([1280, 720]))
    plt.rcParams = init_rcParams(plt.rcParams, FONT_SIZE, figsize_inch)

    year, month, day = args.dt.split("/")
    from_dt = datetime.datetime(
        int(year),
        int(month),
        int(day),
    )

    seconds_in_day = 24 * 60 * 60
    sampling_interval = 1

    t = generate_time_array(seconds_in_day, sampling_interval)
    dt_all = np.array([from_dt + datetime.timedelta(seconds=s.item()) for s in t])

    q = Q()
    q_all = q.calc_qs_kw_v2(
        dt_all,
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=args.surface_tilt,
        surface_azimuth=args.surface_azimuth,
        model=args.model,
    )

    dt_all_next_day = np.array(
        [
            from_dt + datetime.timedelta(seconds=s.item()) + datetime.timedelta(days=1)
            for s in t
        ]
    )
    q_all_next_day = q.calc_qs_kw_v2(
        dt_all_next_day,
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=args.surface_tilt,
        surface_azimuth=args.surface_azimuth,
        model=args.model,
    )

    peak_time = find_peak_times(dt_all, q_all)[0]
    peak_time_next_day = find_peak_times(dt_all_next_day, q_all_next_day)[0]

    period = peak_time_next_day - peak_time
    period_seconds = period.total_seconds()

    T = period_seconds

    q_all_dropped = drop_data_points(q_all, period)
    dt_all_dropped = drop_data_points(dt_all, period)

    q_all_dropped_original = q_all_dropped.copy()
    q_all_dropped_copy = q_all_dropped.copy()

    sampling_rate = 1 / sampling_interval
    
    xf_measured, yf_measured, fundamental_frequency = perform_frequency_analysis(q_all_dropped, sampling_rate)

    print(f"基本周波数: {fundamental_frequency}")
    print(f"時系列データの周期の逆数: {1 / T}")
    print(f"基本周波数 == 時系列データのサンプリングレート: {fundamental_frequency == 1 / T }")
    
    shifts = np.arange(args.shift_offset, args.max_shift)
    phase_diffs = calculate_phase_diffs(shifts, q_all_dropped_copy, yf_measured, sampling_rate)
    
    slope, intercept = np.polyfit(shifts, phase_diffs, 1)
    print(f"1sずれるごとに増えるラジアン: {slope} [rad]")
    print(f"phase_diffs[0]: {phase_diffs[0]} [rad]")
    print(f"phase_diffs[-1]: {phase_diffs[-1]} [rad]")

    plot_phase_graph(shifts, phase_diffs)

    if args.plot_calced_q:
        plot_calced_q_graph(dt_all_dropped, q_all_dropped_original)

    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
