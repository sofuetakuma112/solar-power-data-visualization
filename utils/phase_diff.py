from utils.frequency_analysis import find_phase_difference, perform_fft
from utils.colors import colorlist
from matplotlib import dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_time_array(seconds_in_day, sampling_interval):
    return np.arange(0, seconds_in_day, sampling_interval)


def create_dataframe(t, q_all, period=1):
    df = pd.DataFrame({"time": pd.to_timedelta(t * period, unit="s"), "q": q_all})
    date = pd.to_datetime("2023-01-01")
    df["dt"] = pd.to_datetime(df["time"].dt.total_seconds(), unit="s", origin=date)
    return df


def perform_frequency_analysis(q_all, sampling_rate):
    q_all -= np.mean(q_all)
    xf_measured, yf_measured = perform_fft(q_all, sampling_rate)
    return xf_measured, yf_measured

def calculate_phase_diff(shift, calced_q_all, yf_measured, sampling_rate):
    calced_q_all_rolled = np.roll(calced_q_all, shift)
    calced_q_all_rolled -= np.mean(calced_q_all_rolled)
    xf_calculated, yf_calculated = perform_fft(calced_q_all_rolled, sampling_rate)
    phase_diff = find_phase_difference(yf_measured, yf_calculated, len(yf_measured))
    return phase_diff

def calculate_phase_diffs(shifts, calced_q_all, yf_measured, sampling_rate):
    phase_diffs = []
    for shift in shifts:
        phase_diff = calculate_phase_diff(shift, calced_q_all, yf_measured, sampling_rate)
        phase_diffs.append(phase_diff)
    return np.array(phase_diffs)


def plot_phase_graph(shifts, phase_diffs):
    fig, ax = plt.subplots()
    ax.plot(shifts, phase_diffs, label="ずれ時間とそれに対応する位相差")
    ax.set_xlabel("ずれ時間 [s]")
    ax.set_ylabel("位相差 [rad]")
    ax.legend()


def plot_calced_q_graph(dt_all, q_all):
    fig, ax = plt.subplots()
    ax.plot(
        dt_all,
        q_all,
        label=f"自作関数で生成した日射量",
        color=colorlist[0],
    )
    ax.set_xlabel("時刻")
    ax.set_ylabel("日射量 [kW/m$^2$]")
    ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    ax.legend()
