import datetime
import sys
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.es.load import load_q_and_dt_for_period
import json
import numpy as np
from utils.q import calc_q_kw
from utils.correlogram import (
    unify_deltas_between_dts_v2,
)

if __name__ == "__main__":
    json_open = open("data/json/sorted_diffs.json", "r")
    diff_and_dates = json.load(json_open)

    for diff_and_date in diff_and_dates:
        from_dt = datetime.datetime.strptime(diff_and_date[-1], "%Y-%m-%d")
        to_dt = from_dt + datetime.timedelta(days=1)

        diff_days = 1.0
        dt_all, q_all = load_q_and_dt_for_period(from_dt, diff_days)
        dt_all, q_all = unify_deltas_between_dts_v2(dt_all, q_all)

        dt_all = np.array(dt_all)
        Q_all = np.array(Q_all)
        q_calc_all = np.vectorize(calc_q_kw)(dt_all)

        corr = np.correlate(Q_all, q_calc_all, "full")
        estimated_delay = corr.argmax() - (len(q_calc_all) - 1)

        print(f"{from_dt}: {estimated_delay}")
