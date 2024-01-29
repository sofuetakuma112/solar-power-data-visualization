import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.phase_diff import calculate_phase_diffs, create_dataframe, generate_time_array, perform_frequency_analysis, plot_calced_q_graph, plot_phase_graph
from utils.init_matplotlib import figsize_px_to_inch, init_rcParams
# import japanize_matplotlib
import matplotlib_fontja

FONT_SIZE = 14

# > python3 q_simulation.py -max_shift 1000
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", type=str)  # グラフ描画したい日付のリスト
    parser.add_argument("-model", type=str, default="isotropic")  # 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'
    parser.add_argument("-surface_tilt", type=int, default=22)
    parser.add_argument("-surface_azimuth", type=float, default=185.0)
    parser.add_argument("-shift_offset", type=int, default=0)
    parser.add_argument("-max_shift", type=int, default=10)
    parser.add_argument("-plot_calced_q", action="store_true")
    return parser.parse_args()

def calculate_output(t, seconds_in_day):
    # (t - 21600) を1日の中心時間からの経過時間として表現しています。
    # 具体的には、午前6時（6 * 60 * 60秒）を基準として、指定された時間からの経過時間を計算しています。
    # この経過時間を (2 * np.pi / seconds_in_day) で割ることで、1日の周期に対する角度を求めています。
    # そして、np.sin 関数を適用することで、周期的な振動を持つ値を計算しています。
    #
    # y(t) = A * sin((2π/T) * t + φ)
    #
    # y(t): 時刻 t における振動の値（正弦波の出力）
    # A: 振幅（振動の最大値から最小値までの距離）
    # np.sin: サイン関数（角度の値に対応する正弦の値を返す関数）
    # (2π/T): 角周波数（周期 T における角度の変化量）
    # t: 時刻（時間の経過を表す）
    # φ(phi): 位相（振動の開始位置をずらすためのオフセット）
    #
    # ここで、(2π/T) は周期 T における角周波数を表しています。周期 T は振動が一つの完全なサイクルを完了するまでにかかる時間を表します。
    # この式では、時間 t が進むにつれて正弦波が周期的に振動し、値は正と負の範囲を繰り返します。振幅 A は振動の最大振幅を制御し、位相 φ は振動の開始位置を調整します。
    # 周期 T を使用することで、振動の周期を直接指定できるため、時間の単位に依存せずに正弦波を表現することができます。
    T = seconds_in_day
    phi = 60 * 60 * 6
    output = np.sin((t - phi) * (2 * np.pi / T))
    output[output < 0] = 0
    return output

def main(args):
    figsize_inch = figsize_px_to_inch(np.array([1280, 720]))
    plt.rcParams = init_rcParams(plt.rcParams, FONT_SIZE, figsize_inch)
    
    seconds_in_day = 24 * 60 * 60
    sampling_interval = 1
    
    t = generate_time_array(seconds_in_day, sampling_interval)
    output = calculate_output(t, seconds_in_day)
    df = create_dataframe(t, output)
    
    q_all = df["q"].to_numpy()

    q_all_original = q_all.copy()
    calced_q_all = q_all.copy()

    dt_all = df["dt"].to_numpy()
    
    sampling_rate = 1 / sampling_interval
    
    xf_measured, yf_measured, _ = perform_frequency_analysis(q_all, sampling_rate)
    
    shifts = np.arange(args.shift_offset, args.max_shift)
    phase_diffs = calculate_phase_diffs(shifts, calced_q_all, yf_measured, sampling_rate)
    
    slope, intercept = np.polyfit(shifts, phase_diffs, 1)
    print(f"1sずれるごとに増えるラジアン: {slope} [rad]")
    print(f"phase_diffs[0]: {phase_diffs[0]} [rad]")
    print(f"phase_diffs[-1]: {phase_diffs[-1]} [rad]")
    
    plot_phase_graph(shifts, phase_diffs)

    if args.plot_calced_q:
        plot_calced_q_graph(dt_all, q_all_original)

    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
