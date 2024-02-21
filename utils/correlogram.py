import datetime
from utils.date import get_total_seconds
import numpy as np


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

    return equally_spaced_dt, completed_q
