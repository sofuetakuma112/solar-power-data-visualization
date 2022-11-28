import datetime
import os
from utils.es.fetch import fetchDocsByPeriod
import numpy as np
import japanize_matplotlib
import math
from utils.correlogram import (
    NotEnoughLengthErr,
    testEqualityDeltaBetweenDts,
    unifyDeltasBetweenDts,
    calc_dts_for_q_calc,
    slidesQCalcForCorr,
    calcRatios,
)
from utils.es.load import NotEnoughDocErr, loadQAndDtForPeriod
import csv
import argparse
import itertools
from utils.q import calcQ
import matplotlib.pyplot as plt

MODE_DYNAMIC = "dynamic"
MODE_FIXED = "fixed"

MODE_REPLACE_ZERO = "replace_zero"
MODE_AVG = "avg"

INIT_TOP_N = -1


def dt_to_ymd(dt):
    return format(dt, "%Y-%m-%d")


# 実測値同士で相互相関を計算する際に、スライドさせる側の実測値リストの開始インデックスを求める
def calc_start_idx_for_slide(dt_all, fixedSpanLen, dynamicSpanLen):
    # 進める時間を計算する
    dtStartLag_float, dtStartLag_int = math.modf((fixedSpanLen - dynamicSpanLen) / 2)
    additional_lag = datetime.timedelta(days=dtStartLag_int) + datetime.timedelta(
        hours=dtStartLag_float * 24
    )
    # dts[0] + 進める時間に対応するQ_allのインデックスを求める
    slide_q_start_dt = dt_all[0] + additional_lag
    print(f"additional_lag: {additional_lag}")
    start_idx = -1
    for i, dt in enumerate(dt_all):
        if dt >= slide_q_start_dt:
            start_idx = i
            print(f"dt[0]: {dt_all[0]}")
            print(f"dt_all[start_idx]: {dt}")
            break

    return start_idx


def calc_corr(
    fromDt,
    toDt,
    fixedSpanLen,
    dynamicSpanLen,
    percentageOfData,
    q_modification_strategy,
    top_n=INIT_TOP_N,
    no_missing_data_err=False,
):
    """
    相互相関の平均の最大値とその時のラグを返す

    > python3 recursively_until_threshold_cross-correlation_is_exceeded.py 2022/04/01 2.5 2 0.27
    """
    fetchDocsByPeriod(fromDt, toDt)  # pickleファイルがない場合、取得する
    dt_all_or_err, Q_all = loadQAndDtForPeriod(  # 計算値をスライドさせるため、固定側は動的側より長いリスト長が必要となる
        fromDt, fixedSpanLen, no_missing_data_err
    )  # 与えた期間の日射量と計測日時をファイルから読み込む(dtでソート済み)

    dt_all = None
    if isinstance(dt_all_or_err, NotEnoughDocErr):
        return dt_all_or_err, None
    else:
        dt_all = dt_all_or_err

    dt_all, Q_all = unifyDeltasBetweenDts(dt_all, Q_all)  # 時系列データのデルタを均一にする
    testEqualityDeltaBetweenDts(dt_all)  # 時系列データの点間が全て1.0[s]かテストする

    # 【デバッグ】
    # Q_all_cached = copy.deepcopy(Q_all)

    print(f"dt_all[-1]: {dt_all[-1]}")

    # TODO: top_nを使って上位n日のみを残すフィルタリング処理を実装する
    # 日毎に計算値と実測値の差分の総和を求める
    if top_n != INIT_TOP_N:
        dt_and_q_list = np.array(
            list(
                zip(
                    np.array(dt_all),
                    np.array(Q_all),
                    np.vectorize(dt_to_ymd)(np.array(dt_all)),
                )
            )
        )

        ymds = dt_and_q_list[:, 2]
        unique_ymds = np.unique(ymds)

        def set_unique_ymd(unique_ymd):
            return np.vectorize(lambda ymd: ymd == unique_ymd)

        def calc_daily_diff(get_mask_func):
            def _calc_daily_diff(dt_and_q_list_per_day):
                def calc_diff(l):
                    dt, q, _ = l
                    q_calc = max(calcQ(dt, 33.82794, 132.75093), 0) / 1000
                    return np.square(q_calc - q)  # ユークリッド距離

                diff_square_sum_sqrt = np.sqrt(
                    np.sum(np.apply_along_axis(calc_diff, 1, dt_and_q_list_per_day))
                )

                return diff_square_sum_sqrt

            masked = dt_and_q_list[get_mask_func(ymds)]
            return _calc_daily_diff(masked)

        def get_ymd(get_mask_func):
            masked = dt_and_q_list[get_mask_func(ymds)]
            return masked[0][-1]

        mask_funcs = np.vectorize(set_unique_ymd)(unique_ymds)

        # FIXME: マスクの作成が2重で走ってるからリファクタする
        dates = np.vectorize(get_ymd)(mask_funcs)
        diffs = np.vectorize(calc_daily_diff)(
            mask_funcs
        )  # (単位時間あたりの計算値と実測値の差, YYYY/MM/DD)

        diff_euclid_distances = np.concatenate(
            [diffs.reshape([-1, 1]), dates.reshape([-1, 1])], 1
        )

        print(f"diff_euclid_distances: {diff_euclid_distances}")

        sorted_row_indexes = diff_euclid_distances[:, 0].astype(np.float32).argsort()

        sorted_dates = np.apply_along_axis(
            lambda row: row[1], 1, diff_euclid_distances[sorted_row_indexes, :]
        )

        top_n_dts = sorted_dates[:top_n]

        print(f"top_n_dts: {top_n_dts}")

        def filter_q_by_top_n(l):
            dt, q, ymd = l
            if ymd in top_n_dts:
                return [dt, q, ymd]
            else:
                return [dt, 0, ymd]

        # top_n_dtsに含まれている日付のみになるようフィルタリングする
        filtered_dt_and_q_list = np.apply_along_axis(
            filter_q_by_top_n, 1, dt_and_q_list
        )

        dt_all = filtered_dt_and_q_list[:, 0]
        Q_all = filtered_dt_and_q_list[:, 1]

    # 【デバッグ】
    axes = [plt.subplots()[1] for i in range(1)]
    axes[0].plot(dt_all, Q_all, label="実測値")  # 実データをプロット
    axes[0].set_xlabel("日時")
    axes[0].set_ylabel("日射量[kW/m^2]")
    axes[0].set_xlim(fromDt, toDt)
    plt.show()

    dt_all = list(dt_all)
    Q_all = list(Q_all)

    # (実測値 / 理論値)を各日時ごとに計算して、ソートして上から何割かだけの日射量を採用して残りは0にする
    ratios = calcRatios(dt_all, Q_all)
    diffs_between_ratio_and_one = [  # 比が1からどれだけ離れているか
        (i, np.abs(1 - ratio)) for i, ratio in enumerate(ratios)
    ]
    total_len = len(diffs_between_ratio_and_one)
    last_idx = int(
        total_len * np.abs(1 - percentageOfData)
    )  # 全体のうち何%のデータをそのまま変化加えないとするか(小数表記)
    should_replace_zero_idxes = list(
        map(
            lambda idx_and_diff_ratio: idx_and_diff_ratio[0],
            sorted(diffs_between_ratio_and_one, key=lambda x: x[1], reverse=True)[
                :last_idx
            ],
        )
    )
    should_replace_zero_idxes = sorted(
        should_replace_zero_idxes
    )  # インデックスの大小関係がUNIX時間の大小関係と等しい

    for i in should_replace_zero_idxes:
        if q_modification_strategy == MODE_REPLACE_ZERO:
            Q_all[i] = 0
        elif q_modification_strategy == MODE_AVG:
            if i - 1 >= 0 and i < len(diffs_between_ratio_and_one) - 1:
                # iの左右に最低1つはデータ点がある
                Q_all[i] = (Q_all[i - 1] + Q_all[i + 1]) / 2
            else:
                # 左右の端
                Q_all[i] = 0
        else:
            raise ValueError("不正なq_modification_strategy")

    dts_for_q_calc_or_err = calc_dts_for_q_calc(dt_all, dynamicSpanLen)

    dts_for_q_calc = []
    if isinstance(dts_for_q_calc_or_err, NotEnoughLengthErr):
        return dts_for_q_calc_or_err, None
    else:
        dts_for_q_calc = dts_for_q_calc_or_err

    # Q_calc_allの時系列データを実測値の時系列データより進める
    # fixedSpanLen: 2.5, dynamicSpanLen: 2.0で差が0.5日の場合、計算用の日時列は6時間進んだ状態に変化する
    # 計算値の日時をスライドさせる全量の半分だけ進めた状態で相互相関を求めることで、全量の半分スライドさせたときに相互相関が最大となる
    Q_calc_all_applied_lag = slidesQCalcForCorr(
        dts_for_q_calc, fixedSpanLen, dynamicSpanLen
    )

    # 【デバッグ】
    # start_idx = calc_start_idx_for_slide(dt_all, fixedSpanLen, dynamicSpanLen)
    # 【デバッグ】
    # corr = np.correlate(Q_all, Q_all_cached[start_idx:-start_idx])

    corr = np.correlate(Q_all, Q_calc_all_applied_lag)
    largest_lag_sec = 6 * 60 * 60 - corr.argmax()

    # 時系列データの1単位あたりの相互相関の値を返す
    return corr.max() / len(Q_calc_all_applied_lag), largest_lag_sec


# 渡された条件で一度だけ計算する
def calc_corr_at_once(
    q_modification_strategy,
    fromDtStr,
    fixedDaysLen,
    dynamicDaysLen,
    percentageOfData,
    top_n,
    no_missing_data_err,
):
    fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
    toDt = fromDt + datetime.timedelta(days=math.ceil(fixedDaysLen))

    # if should_log:
    #     log = init_logger(
    #         fromDt,
    #         fixedDaysLen,
    #         dynamicDaysLen,
    #         percentageOfData,
    #         span_update_strategy,
    #         q_modification_strategy,
    #         allow_duplicate,
    #         should_fix_position,
    #     )

    print(fromDt, toDt, fixedDaysLen, dynamicDaysLen)

    if datetime.datetime.now().timestamp() < fromDt.timestamp():
        print("存在しない未来を範囲に含んでいる")
        return

    corr_max_per_one_time_unit_or_err, lag = calc_corr(
        fromDt,
        toDt,
        fixedDaysLen,
        dynamicDaysLen,
        percentageOfData,
        q_modification_strategy,
        top_n,
        no_missing_data_err,
    )

    corr_max_per_one_time_unit = None
    if isinstance(
        corr_max_per_one_time_unit_or_err, (NotEnoughLengthErr, NotEnoughDocErr)
    ):
        print(corr_max_per_one_time_unit_or_err.message)
        return
    else:
        corr_max_per_one_time_unit = corr_max_per_one_time_unit_or_err

    # if should_log:
    #     log(fromDt, fixedDaysLen, dynamicDaysLen, corr_max_per_one_time_unit, lag)

    print(
        f"結果: fromDt: {fromDt}, fixedDaysLen: {fixedDaysLen}, dynamicDaysLen: {dynamicDaysLen}, lag: {lag}, 相互相関の平均の最大値: {corr_max_per_one_time_unit}"
    )


# スレショルドを超えるまで再帰的に実行する関数
def search_optimal_lag(
    span_update_strategy,
    q_modification_strategy,
    allow_duplicate,
    should_fix_position,
    fromDtStr,
    fixedDaysLen,
    dynamicDaysLen,
    threshold,
    percentageOfData,
    no_missing_data_err,
    should_log=False,
):
    fixed_days_len_copied = fixedDaysLen
    dynamic_days_len_copied = dynamicDaysLen

    fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
    toDt = fromDt + datetime.timedelta(days=math.ceil(fixedDaysLen))

    if should_log:
        log = init_logger(
            fromDt,
            fixedDaysLen,
            dynamicDaysLen,
            percentageOfData,
            span_update_strategy,
            q_modification_strategy,
            allow_duplicate,
            should_fix_position,
        )

    # 固定長とダイナミック長のスパンを更新する
    def _update_days_span_length_and_position(
        fixedDaysLen,
        dynamicDaysLen,
        fixed_days_len_copied,
        dynamic_days_len_copied,
        fromDt,
        toDt,
        only_change_position=False,
    ):
        def _update_span_length(
            fixedDaysLen, dynamicDaysLen, fixed_days_len_copied, dynamic_days_len_copied
        ):
            if span_update_strategy == MODE_FIXED:
                # 固定長とダイナミック長の長さは常に固定
                fixedDaysLen = fixed_days_len_copied
                dynamicDaysLen = dynamic_days_len_copied
            else:
                # 固定長とダイナミック長の長さを伸ばす
                fixedDaysLen += 1
                dynamicDaysLen += 1

            return fixedDaysLen, dynamicDaysLen

        def _update_span_position(fromDt, toDt, fixedDaysLen):
            if allow_duplicate:
                # 重複ありでずらす場合
                fromDt += datetime.timedelta(days=1)
                toDt += datetime.timedelta(days=1)
            else:
                # 重複無しでずらす場合
                fromDt = toDt
                toDt = fromDt + datetime.timedelta(days=math.ceil(fixedDaysLen))

            return fromDt, toDt

        if should_fix_position:
            # 開始日付は常に固定モード
            fixedDaysLen, dynamicDaysLen = _update_span_length(
                fixedDaysLen,
                dynamicDaysLen,
                fixed_days_len_copied,
                dynamic_days_len_copied,
            )
            # toDtのみ変える
            toDt = fromDt + datetime.timedelta(days=math.ceil(fixedDaysLen))
        else:
            if only_change_position:
                # 位置のみを変える(相互相関計算時にエラー、相互相関の右肩上がりが止まった)
                fromDt, toDt = _update_span_position(fromDt, toDt, fixedDaysLen)
            else:
                fixedDaysLen, dynamicDaysLen = _update_span_length(
                    fixedDaysLen,
                    dynamicDaysLen,
                    fixed_days_len_copied,
                    dynamic_days_len_copied,
                )
                fromDt, toDt = _update_span_position(fromDt, toDt, fixedDaysLen)

        return fixedDaysLen, dynamicDaysLen, fromDt, toDt

    corr_cached = [0]
    while True:
        print(fromDt, toDt, fixedDaysLen, dynamicDaysLen)

        if datetime.datetime.now().timestamp() < fromDt.timestamp():
            break

        no_filtering_actual_data = -1
        corr_max_per_one_time_unit_or_err, lag = calc_corr(
            fromDt,
            toDt,
            fixedDaysLen,
            dynamicDaysLen,
            percentageOfData,
            q_modification_strategy,
            no_filtering_actual_data,
            no_missing_data_err,
        )

        corr_max_per_one_time_unit = None
        if isinstance(
            corr_max_per_one_time_unit_or_err, (NotEnoughLengthErr, NotEnoughDocErr)
        ):
            print(corr_max_per_one_time_unit_or_err.message)

            if should_fix_position:
                break

            corr_cached = [0]
            (
                fixedDaysLen,
                dynamicDaysLen,
                fromDt,
                toDt,
            ) = _update_days_span_length_and_position(
                fixedDaysLen,
                dynamicDaysLen,
                fixed_days_len_copied,
                dynamic_days_len_copied,
                fromDt,
                toDt,
                True,
            )
            continue
        else:
            corr_max_per_one_time_unit = corr_max_per_one_time_unit_or_err

        corr_cached.append(corr_max_per_one_time_unit)

        if should_log:
            log(fromDt, fixedDaysLen, dynamicDaysLen, corr_max_per_one_time_unit, lag)

        if corr_max_per_one_time_unit > threshold:  # しきい値を超えた
            break
        elif corr_max_per_one_time_unit != max(corr_cached):
            # これまでの相互相関の最大値を下回ったので次の期間に移動する
            corr_cached = [0]
            (
                fixedDaysLen,
                dynamicDaysLen,
                fromDt,
                toDt,
            ) = _update_days_span_length_and_position(
                fixedDaysLen,
                dynamicDaysLen,
                fixed_days_len_copied,
                dynamic_days_len_copied,
                fromDt,
                toDt,
                True,
            )
            continue
        else:
            print()
            # スパンの更新
            (
                fixedDaysLen,
                dynamicDaysLen,
                fromDt,
                toDt,
            ) = _update_days_span_length_and_position(
                fixedDaysLen,
                dynamicDaysLen,
                fixed_days_len_copied,
                dynamic_days_len_copied,
                fromDt,
                toDt,
            )

    print(
        f"結果: fromDt: {fromDt}, fixedDaysLen: {fixedDaysLen}, dynamicDaysLen: {dynamicDaysLen}, lag: {lag}, 相互相関の平均の最大値: {corr_max_per_one_time_unit}"
    )


def init_logger(
    fromDt,
    fixedDaysLen,
    dynamicDaysLen,
    percentageOfData,
    span_update_strategy,
    q_modification_strategy,
    allow_duplicate,
    should_fix_position,
):
    dir_path = f"data/csv/{dt_to_ymd(fromDt)}"

    if allow_duplicate:
        duplicate = "allow_duplicate"
    else:
        duplicate = "no_duplicate"

    if should_fix_position:
        position = "fix_position"
    else:
        position = "dynamic_position"

    file_name = f"{fixedDaysLen},{dynamicDaysLen},{percentageOfData},{span_update_strategy},{q_modification_strategy},{duplicate},{position}"

    os.makedirs(dir_path, exist_ok=True)

    def _log(*args):
        with open(f"{dir_path}/{file_name}.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(args)

    return _log


# 相互相関の平均値の最大が指定したしきい値を超えるまで再帰的に相互相関を求める
# python3 recursively_until_threshold_cross-correlation_is_exceeded.py -f 2022/04/01 -fd 2.5 -dd 2 -th 0.4 -p 0.95 -rz

# 一つのfromDt, fixed, dynamicの組み合わせでのみ相互相関を計算する（更にTOP4件の実測波形のみを残して残りは落とす）
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
    parser.add_argument("-dm", "--dynamic_mode", action="store_true")
    parser.add_argument("-ai", "--auto_increment", action="store_true")
    parser.add_argument("-rz", "--replace_zero", action="store_true")
    parser.add_argument(
        "-ad", "--allow_duplicate", action="store_true"
    )  # 期間をスライドさせる際に重複を許可するか
    parser.add_argument("-sl", "--should_log", action="store_true")  # ログを取るか
    parser.add_argument("-fp", "--fix_position", action="store_true")  # 開始日を固定して伸ばしていくか
    parser.add_argument(
        "-nmde", "--no_missing_data_err", action="store_true"
    )  # データがないデータ点の日射量を0扱いにしてエラーとして扱わないか
    args = parser.parse_args()

    fromDtStr = args.f.split("/")
    fixedDaysLen = float(args.fd)
    dynamicDaysLen = float(args.dd)
    threshold = float(args.th)
    percentageOfData = float(args.p)
    top_n = int(args.tp)

    if args.dynamic_mode:
        span_update_strategy = MODE_DYNAMIC  # ループごとに固定長とダイナミック長を1日ずつ伸ばすモード
    else:
        span_update_strategy = MODE_FIXED  # 固定長とダイナミック長を固定するモード

    if args.replace_zero:
        q_modification_strategy = MODE_REPLACE_ZERO
    else:
        q_modification_strategy = MODE_AVG

    should_fix_position = args.fix_position
    no_missing_data_err = args.no_missing_data_err

    if args.auto_increment:
        # 再帰的に計算する
        search_optimal_lag(
            span_update_strategy,
            q_modification_strategy,
            args.allow_duplicate,
            should_fix_position,
            fromDtStr,
            fixedDaysLen,
            dynamicDaysLen,
            threshold,
            percentageOfData,
            no_missing_data_err,
            args.should_log,
        )
    else:
        # 引数の条件でのみ計算する
        calc_corr_at_once(
            q_modification_strategy,
            fromDtStr,
            fixedDaysLen,
            dynamicDaysLen,
            percentageOfData,
            top_n,
            no_missing_data_err,
        )


if __name__ == "__main__":
    main()
