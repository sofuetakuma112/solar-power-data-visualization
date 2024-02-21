import datetime
from utils.date import get_total_seconds
import numpy as np


def test_equality_delta_between_dts(dts, delta=1.0):
    """
    日時データのdeltaが1[s]で統一されているかテストする
    """
    for i in range(len(dts)):
        if len(dts) - 1 == i:
            break
        diff = dts[i + 1] - dts[i]
        if diff.total_seconds() != delta:  # 日時のデルタが1sではない
            print(f"i: {i}")
            print(f"dts[i + 1]: {dts[i + 1]}")
            print(f"dts[i]: {dts[i]}")
            print(diff.total_seconds())
            raise ValueError("error!")


def unify_deltas_between_dts_v2(dts, qs):
    """
    補完した日時データとそれに対応した日射量のリストを取得する
    """
    # 指定した間隔で補完
    first_date = dts[0]
    last_date = dts[-1]

    offset_date = datetime.datetime(
        int(first_date.year), int(first_date.month), int(first_date.day)
    )

    total_seconds_between_dts = (last_date - first_date).total_seconds()

    equally_spaced_dt = np.vectorize(
        lambda s: offset_date + datetime.timedelta(seconds=float(s))
    )(np.arange(0, total_seconds_between_dts, 1))
    completed_q = np.interp(
        np.vectorize(get_total_seconds)(equally_spaced_dt),
        np.vectorize(get_total_seconds)(dts),
        qs,
    )  # 新しいx列に対応するy列を計算

    # 時系列データの点間が全て1.0[s]かテストする
    test_equality_delta_between_dts(equally_spaced_dt)

    return equally_spaced_dt, completed_q
