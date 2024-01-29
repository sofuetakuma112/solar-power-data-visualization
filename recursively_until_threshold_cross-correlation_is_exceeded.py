import datetime
import os
from matplotlib import dates
import numpy as np
# import japanize_matplotlib
import matplotlib_fontja
import math

import pandas as pd
from utils.correlogram import (
    NotEnoughLengthErr,
    unify_deltas_between_dts_v2,
    calc_dts_for_q_calc,
    slides_q_calc_for_corr,
    calc_ratios,
)
from utils.es.load import NotEnoughDocErr, load_q_and_dt_for_period
import csv
import argparse
from utils.q import calc_q_kw
import matplotlib.pyplot as plt

import time

MODE_DYNAMIC = "dynamic"
MODE_FIXED = "fixed"

MODE_REPLACE_ZERO = "replace_zero"
MODE_AVG = "avg"

INIT_TOP_N = -1


def dt_to_ymd(dt):
    return format(dt, "%Y-%m-%d")


class CorrSearch:
    def __init__(
        self,
        from_dt_str,
        fixed_days_len,
        dynamic_days_len,
        threshold,
        percentage_of_data,
        span_update_strategy,
        q_modification_strategy,
        allow_duplicate,
        should_log,
        should_fix_position,
        no_missing_data_err,
        top_n=-1,
    ):
        self.from_dt_str = from_dt_str
        self.fixed_days_len = fixed_days_len
        self.dynamic_days_len = dynamic_days_len
        self.threshold = threshold
        self.percentage_of_data = percentage_of_data
        self.top_n = top_n
        self.span_update_strategy = span_update_strategy
        self.q_modification_strategy = q_modification_strategy
        self.allow_duplicate = allow_duplicate
        self.should_log = should_log
        self.should_fix_position = should_fix_position
        self.no_missing_data_err = no_missing_data_err

    # スレショルドを超えるまで再帰的に実行する関数
    def search_optimal_lag(self):
        fixed_days_len_copied = self.fixed_days_len
        dynamic_days_len_copied = self.dynamic_days_len

        from_dt = datetime.datetime(
            int(self.from_dt_str[0]), int(self.from_dt_str[1]), int(self.from_dt_str[2])
        )
        to_dt = from_dt + datetime.timedelta(days=math.ceil(self.fixed_days_len))

        if self.should_log:
            log = self.init_logger(
                from_dt,
            )

        corr_cached = np.array([0])
        while True:
            print(from_dt, to_dt, self.fixed_days_len, self.dynamic_days_len)

            if datetime.datetime.now().timestamp() < from_dt.timestamp():
                break

            # no_filtering_actual_data = -1
            corr_max_per_one_time_unit_or_err, lag = self.calc_corr(
                from_dt,
                # to_dt,
            )

            corr_max_per_one_time_unit = None
            if isinstance(corr_max_per_one_time_unit_or_err, NotEnoughLengthErr):
                print(corr_max_per_one_time_unit_or_err.message)

                if self.should_fix_position:
                    break

                corr_cached = np.array([0])
                (
                    self.fixed_days_len,
                    self.dynamic_days_len,
                    from_dt,
                    to_dt,
                ) = self.update_days_span_length_and_position(
                    fixed_days_len_copied=fixed_days_len_copied,
                    dynamic_days_len_copied=dynamic_days_len_copied,
                    from_dt=from_dt,
                    to_dt=to_dt,
                    only_change_position=True,
                )
                continue
            else:
                corr_max_per_one_time_unit = corr_max_per_one_time_unit_or_err

            corr_cached = corr_cached.append(corr_cached, corr_max_per_one_time_unit)

            if self.should_log:
                log(
                    from_dt,
                    self.fixed_days_len,
                    self.dynamic_days_len,
                    corr_max_per_one_time_unit,
                    lag,
                )

            if corr_max_per_one_time_unit > self.threshold:  # しきい値を超えた
                break
            elif corr_max_per_one_time_unit != corr_cached.max():
                # これまでの相互相関の最大値を下回ったので次の期間に移動する
                corr_cached = np.array([0])
                (
                    self.fixed_days_len,
                    self.dynamic_days_len,
                    from_dt,
                    to_dt,
                ) = self.update_days_span_length_and_position(
                    fixed_days_len_copied=fixed_days_len_copied,
                    dynamic_days_len_copied=dynamic_days_len_copied,
                    from_dt=from_dt,
                    to_dt=to_dt,
                    only_change_position=True,
                )
                continue
            else:
                print()
                # スパンの更新
                (
                    self.fixed_days_len,
                    self.dynamic_days_len,
                    from_dt,
                    to_dt,
                ) = self.update_days_span_length_and_position(
                    fixed_days_len_copied,
                    dynamic_days_len_copied,
                    from_dt,
                    to_dt,
                )

        print(
            f"結果: from_dt: {from_dt}, fixed_days_len: {self.fixed_days_len}, dynamic_days_len: {self.dynamic_days_len}, lag: {lag}, 相互相関の平均の最大値: {corr_max_per_one_time_unit}"
        )

    # 固定長とダイナミック長のスパンを更新する
    def update_days_span_length_and_position(
        self,
        fixed_days_len_copied,
        dynamic_days_len_copied,
        from_dt,
        to_dt,
        only_change_position=False,
    ):
        def _update_span_length(
            fixed_days_len,
            dynamic_days_len,
            fixed_days_len_copied,
            dynamic_days_len_copied,
        ):
            if self.span_update_strategy == MODE_FIXED:
                # 固定長とダイナミック長の長さは常に固定
                fixed_days_len = fixed_days_len_copied
                dynamic_days_len = dynamic_days_len_copied
            else:
                # 固定長とダイナミック長の長さを伸ばす
                fixed_days_len += 1
                dynamic_days_len += 1

            return fixed_days_len, dynamic_days_len

        def _update_span_position(from_dt, to_dt, fixed_days_len):
            if self.allow_duplicate:
                # 重複ありでずらす場合
                from_dt += datetime.timedelta(days=1)
                to_dt += datetime.timedelta(days=1)
            else:
                # 重複無しでずらす場合
                from_dt = to_dt
                to_dt = from_dt + datetime.timedelta(days=math.ceil(fixed_days_len))

            return from_dt, to_dt

        if self.should_fix_position:
            # 開始日付は常に固定モード
            fixed_days_len, dynamic_days_len = _update_span_length(
                fixed_days_len,
                dynamic_days_len,
                fixed_days_len_copied,
                dynamic_days_len_copied,
            )
            # to_dtのみ変える
            to_dt = from_dt + datetime.timedelta(days=math.ceil(fixed_days_len))
        else:
            if only_change_position:
                # 位置のみを変える(相互相関計算時にエラー、相互相関の右肩上がりが止まった)
                from_dt, to_dt = _update_span_position(from_dt, to_dt, fixed_days_len)
            else:
                fixed_days_len, dynamic_days_len = _update_span_length(
                    fixed_days_len,
                    dynamic_days_len,
                    fixed_days_len_copied,
                    dynamic_days_len_copied,
                )
                from_dt, to_dt = _update_span_position(from_dt, to_dt, fixed_days_len)

        return fixed_days_len, dynamic_days_len, from_dt, to_dt

    def calc_corr(self, from_dt):
        """
        相互相関の平均の最大値とその時のラグを返す

        > python3 recursively_until_threshold_cross-correlation_is_exceeded.py 2022/04/01 2.5 2 0.27
        """
        print(f"from_dt: {from_dt}")
        print(f"self.fixed_days_len: {self.fixed_days_len}")
        (
            dt_all,
            q_all,
        ) = load_q_and_dt_for_period(  # 計算値をスライドさせるため、固定側は動的側より長いリスト長が必要となる
            from_dt, self.fixed_days_len
        )  # 与えた期間の日射量と計測日時をファイルから読み込む(dtでソート済み)

        print(f"after load: dt_all[-1]: {dt_all[-1]}")

        dt_all, q_all = unify_deltas_between_dts_v2(dt_all, q_all)  # 時系列データのデルタを均一にする

        print(f"after unify: dt_all[-1]: {dt_all[-1]}")

        def calc_daily_diff(dt_and_q_list_per_day):
            SHOULD_MATCH_MAX_VALUE = True
            if SHOULD_MATCH_MAX_VALUE:

                def _calc_q_kw(l):
                    dt, _ = l
                    q_calc = calc_q_kw(dt)
                    return q_calc  # ユークリッド距離

                qs_calc = np.apply_along_axis(_calc_q_kw, 1, dt_and_q_list_per_day)

                q_calc_max = np.max(qs_calc)

                qs = dt_and_q_list_per_day[:, 1]
                q_max = np.max(qs)

                scaled_qs_calc = qs_calc

                # if q_max == 0.0:
                #     scaled_qs_calc = qs_calc
                # else:
                #     scaled_qs_calc = qs_calc * (q_max / q_calc_max)

                diff_with_scaled_q_calc_per_day = np.square(scaled_qs_calc - qs)
                diff_per_day = diff_with_scaled_q_calc_per_day
            else:

                def calc_diff(l):
                    dt, q, _ = l
                    q_calc = calc_q_kw(dt)
                    return np.square(q_calc - q)  # ユークリッド距離

                diff_per_day = np.apply_along_axis(calc_diff, 1, dt_and_q_list_per_day)

            diff_square_sum_sqrt = np.sqrt(np.sum(diff_per_day) / len(diff_per_day))

            return diff_square_sum_sqrt  # 戻り値はスカラ??

        def aggr_diffs_on_daily(group):
            print("aggr_diffs_on_daily")
            dts_per_day = np.array(group.index.to_pydatetime(), dtype=datetime.datetime)
            qs_per_day = group["q"].to_numpy()
            dt_and_q_list_per_day = np.column_stack([dts_per_day, qs_per_day])

            diff = calc_daily_diff(dt_and_q_list_per_day)
            date = dts_per_day[0]

            return [date, diff]

            # diffs = np.append(diffs, diff)
            # dates = np.append(dates, date)

        # top_nを使って上位n日のみを残すフィルタリング処理
        # 日毎に計算値と実測値の差分の総和を求める
        if self.top_n != INIT_TOP_N:
            df = pd.DataFrame(
                data=np.column_stack([dt_all, q_all]), columns=["dt", "q"]
            )
            df.set_index("dt", inplace=True)

            diff_and_date = np.array(
                df.groupby(pd.Grouper(freq="1D")).apply(aggr_diffs_on_daily).to_list()
            )

            sorted_row_indexes = diff_and_date[:, 1].astype(np.float32).argsort() # diffでソート

            sorted_dates = diff_and_date[sorted_row_indexes, 0] # ソート後の日付列
            top_n_dts_datetime = sorted_dates[: self.top_n] # 日付列の内、上位n件

            top_n_dts_date = np.vectorize(lambda top_dt: top_dt.date())( # datetime => dateに変換
                top_n_dts_datetime
            )
            # top_n_dts_datetimeのそれぞれのyear, month, dayの組み合わせのいずれかに一致するdtのみにフィルタリングするmaskを作る
            mask_by_top_n_dts = np.vectorize(lambda dt: dt.date() in top_n_dts_date)(
                dt_all
            )

            dt_all = dt_all[mask_by_top_n_dts]
            q_all = q_all[mask_by_top_n_dts]

            # axes = [plt.subplots()[1] for _ in range(1)]
            # axes[0].plot(
            #     dt_all,
            #     q_all,
            #     label=f"実測値",
            # )
            # axes[0].set_xlabel("時刻")
            # axes[0].set_ylabel("日射量[kW/m^2]")
            # axes[0].xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
            # axes[0].legend()
            # plt.show()

        # (実測値 / 理論値)を各日時ごとに計算して、ソートして上から何割かだけの日射量を採用して残りは0にする
        ratios = calc_ratios(dt_all, q_all)
        diffs_between_ratio_and_one_ndarray = np.abs(1 - ratios)
        total_len = len(diffs_between_ratio_and_one_ndarray)
        last_idx = int(
            total_len * np.abs(1 - self.percentage_of_data)
        )  # 全体のうち何%のデータをそのまま変化加えないとするか(小数表記)
        should_replace_zero_idxes_ndarray = np.argsort(
            -diffs_between_ratio_and_one_ndarray
        )[:last_idx]

        should_replace_zero_idxes_ndarray = np.sort(should_replace_zero_idxes_ndarray)

        # TODO: numpy対応する
        # for i in should_replace_zero_idxes_ndarray:
        #     if q_modification_strategy == MODE_REPLACE_ZERO:
        #         q_all[i] = 0
        #     elif q_modification_strategy == MODE_AVG:
        #         if i - 1 >= 0 and i < len(should_replace_zero_idxes_ndarray) - 1:
        #             # iの左右に最低1つはデータ点がある
        #             q_all[i] = (q_all[i - 1] + q_all[i + 1]) / 2
        #         else:
        #             # 左右の端
        #             q_all[i] = 0
        #     else:
        #         raise ValueError("不正なq_modification_strategy")

        dts_for_q_calc_or_err = calc_dts_for_q_calc(dt_all, self.dynamic_days_len)

        dts_for_q_calc = np.array([])
        if isinstance(dts_for_q_calc_or_err, NotEnoughLengthErr):
            return dts_for_q_calc_or_err, None
        else:
            dts_for_q_calc = dts_for_q_calc_or_err

        # Q_calc_allの時系列データを実測値の時系列データより進める
        # fixed_days_len: 2.5, dynamic_days_len: 2.0で差が0.5日の場合、計算用の日時列は6時間進んだ状態に変化する
        # 計算値の日時をスライドさせる全量の半分だけ進めた状態で相互相関を求めることで、全量の半分スライドさせたときに相互相関が最大となる
        Q_calc_all_applied_lag = slides_q_calc_for_corr(
            dts_for_q_calc, self.fixed_days_len, self.dynamic_days_len
        )

        corr = np.correlate(list(q_all), list(Q_calc_all_applied_lag))
        largest_lag_sec = 6 * 60 * 60 - corr.argmax()

        # estimated_delay = corr.argmax() - (len(q_all) - 1)
        # print("estimated delay is " + str(estimated_delay))
        # largest_lag_sec = estimated_delay

        # 時系列データの1単位あたりの相互相関の値を返す
        return corr.max() / len(Q_calc_all_applied_lag), largest_lag_sec

    # 渡された条件で一度だけ計算する
    def calc_corr_at_once(self):
        from_dt = datetime.datetime(
            int(self.from_dt_str[0]), int(self.from_dt_str[1]), int(self.from_dt_str[2])
        )
        to_dt = from_dt + datetime.timedelta(days=math.ceil(self.fixed_days_len))

        print(from_dt, to_dt, self.fixed_days_len, self.dynamic_days_len)

        if datetime.datetime.now().timestamp() < from_dt.timestamp():
            print("存在しない未来を範囲に含んでいる")
            return

        corr_max_per_one_time_unit_or_err, lag = self.calc_corr(
            from_dt,
        )

        corr_max_per_one_time_unit = None
        if isinstance(
            corr_max_per_one_time_unit_or_err, (NotEnoughLengthErr, NotEnoughDocErr)
        ):
            print(corr_max_per_one_time_unit_or_err.message)
            return
        else:
            corr_max_per_one_time_unit = corr_max_per_one_time_unit_or_err

        print(
            f"結果: from_dt: {from_dt}, fixed_days_len: {self.fixed_days_len}, dynamic_days_len: {self.dynamic_days_len}, lag: {lag}, 相互相関の平均の最大値: {corr_max_per_one_time_unit}"
        )

    def init_logger(self, from_dt):
        dir_path = f"data/csv/{dt_to_ymd(from_dt)}"

        if self.allow_duplicate:
            duplicate = "allow_duplicate"
        else:
            duplicate = "no_duplicate"

        if self.should_fix_position:
            position = "fix_position"
        else:
            position = "dynamic_position"

        file_name = f"{self.fixed_days_len},{self.dynamic_days_len},{self.percentage_of_data},{self.span_update_strategy},{self.q_modification_strategy},{duplicate},{position}"

        os.makedirs(dir_path, exist_ok=True)

        def _log(*args):
            with open(f"{dir_path}/{file_name}.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(args)

        return _log


# 相互相関の平均値の最大が指定したしきい値を超えるまで再帰的に相互相関を求める
# python3 recursively_until_threshold_cross-correlation_is_exceeded.py -f 2022/04/01 -fd 2.5 -dd 2 -th 0.4 -p 0.95 -rz

# 一つのfrom_dt, fixed, dynamicの組み合わせでのみ相互相関を計算する（更にTOP4件の実測波形のみを残して残りは落とす）
# python3 recursively_until_threshold_cross-correlation_is_exceeded.py -f 2022/09/28 -fd 26.5 -dd 26 -th 0.6 -p 1 -tp 4 -rz
def main():
    parser = argparse.ArgumentParser(description="add two integer")
    parser.add_argument("-f", type=str)  # from date str format
    parser.add_argument("-fd", type=float, default=2.5)  # fixed day length
    parser.add_argument("-dd", type=float, default=2.0)  # dynamic day length
    parser.add_argument("-th", type=float, default=0.3)  # threshold
    parser.add_argument("-p", type=float, default=1.0)  # percentage of data
    parser.add_argument(
        "-tp", type=int, default=INIT_TOP_N
    )  # 相互相関を計算する際に理論値の波形に近い波形を上位何日残すか
    parser.add_argument("-ss", "--span_strategy", action="store_true")
    parser.add_argument("-ai", "--auto_increment", action="store_true")
    parser.add_argument("-ms", "--modification_strategy", action="store_true")
    parser.add_argument(
        "-ad", "--allow_duplicate", action="store_true"
    )  # 期間をスライドさせる際に重複を許可するか
    parser.add_argument("-sl", "--should_log", action="store_true")  # ログを取るか
    parser.add_argument("-fp", "--fix_position", action="store_true")  # 開始日を固定して伸ばしていくか
    parser.add_argument(
        "-nmde", "--no_missing_data_err", action="store_true"
    )  # データがないデータ点の日射量を0扱いにしてエラーとして扱わないか
    args = parser.parse_args()

    from_dt_str = args.f.split("/")
    fixed_days_len = float(args.fd)
    dynamic_days_len = float(args.dd)
    threshold = float(args.th)
    percentage_of_data = float(args.p)
    top_n = int(args.tp)

    if args.span_strategy:
        span_update_strategy = MODE_DYNAMIC  # ループごとに固定長とダイナミック長を1日ずつ伸ばすモード
    else:
        span_update_strategy = MODE_FIXED  # 固定長とダイナミック長を固定するモード

    if args.modification_strategy:
        modification_strategy = MODE_REPLACE_ZERO
    else:
        modification_strategy = MODE_AVG

    should_fix_position = args.fix_position
    no_missing_data_err = args.no_missing_data_err

    corr_search = CorrSearch(
        from_dt_str,
        fixed_days_len,
        dynamic_days_len,
        threshold,
        percentage_of_data,
        span_update_strategy,
        modification_strategy,
        args.allow_duplicate,
        args.should_log,
        should_fix_position,
        no_missing_data_err,
        top_n,
    )

    if args.auto_increment:
        # 再帰的に計算する
        corr_search.search_optimal_lag()
    else:
        # 引数の条件でのみ計算する
        corr_search.calc_corr_at_once()


if __name__ == "__main__":
    main()
