import datetime
import time


def getYear(dt):
    return dt.year


def getMonth(dt):
    return dt.month


def getDay(dt):
    return dt.day


def getHour(dt):
    return dt.hour


def getMinute(dt):
    return dt.minute


def getSecond(dt):
    return dt.second


def get_dt_without_second(dt):
    return datetime.datetime(
        getYear(dt), getMonth(dt), getDay(dt), getHour(dt), getMinute(dt)
    )


def get_dt_without_milisecond(dt):
    return datetime.datetime(
        getYear(dt), getMonth(dt), getDay(dt), getHour(dt), getMinute(dt), getSecond(dt)
    )


def getSecondAndMiliSecond(dt):
    return float(dt.strftime("%S.%f"))


# 補完する点の数を返す
def getPointsNumberBeCompleted(dt_past, dt_future):
    dt_past_nosecond = get_dt_without_second(dt_past)
    dt_future_nosecond = get_dt_without_second(dt_future)
    diff = dt_future_nosecond - dt_past_nosecond
    diff_seconds = diff.total_seconds()

    second_past = getSecond(dt_past)
    second_future = getSecond(dt_future)

    pointsNumber = second_future + diff_seconds - second_past
    return int(pointsNumber)


def dt_to_hours(dt):
    return dt.days * 24 + (dt.seconds + dt.microseconds / 1000000) / 60 / 60


def time_to_seconds(t):
    return (t.hour * 60 + t.minute) * 60 + t.second


def datetime_to_miliseconds(dt):
    return int(time.mktime(dt.timetuple()) * 1000) + int(dt.microsecond / 1000)


# 2点間の距離における補完する点の(始点からみた際の)位置の割合？を返す
# 秒の足し合わせはtimedeltaに任せて、割合の計算をメインで行う
# dt_pastが12s, dt_futureが14.2の場合は13, 14秒を補完する
# dt_pastが11.3s, dt_futureが12の場合は12秒を補完する
def get_relative_position_between_two_dts(dt_past, dt_future):
    second_with_mili_past = getSecondAndMiliSecond(dt_past)
    second_with_mili_future = getSecondAndMiliSecond(dt_future)

    dt_past_nosecond = get_dt_without_second(dt_past)
    dt_future_nosecond = get_dt_without_second(dt_future)
    diff = dt_future_nosecond - dt_past_nosecond
    diff_seconds = diff.total_seconds()

    pointsNum = getPointsNumberBeCompleted(dt_past, dt_future)

    second_past = getSecond(dt_past)
    total_diff_second = (  # TODO: dt_pastとdt_futureのdiffでも同じ？
        second_with_mili_future + diff_seconds - second_with_mili_past
    )

    dt_past_nomilisecond = get_dt_without_milisecond(dt_past)
    dt_and_rates = []
    for i in range(pointsNum):
        add_second = i + 1
        secondsToComplete = second_past + add_second
        # print(f"secondsToComplete: {secondsToComplete}")
        left_diff_second = secondsToComplete - second_with_mili_past
        percentage_increase_from_starting_point = left_diff_second / total_diff_second
        # print(f"delta yに掛ける係数: {percentage_increase_from_starting_point}")
        dt_comp = dt_past_nomilisecond + datetime.timedelta(seconds=add_second)
        dt_and_rates.append(
            [
                dt_comp,
                percentage_increase_from_starting_point,
            ]
        )
    return list(filter(lambda dt_and_coef: dt_and_coef[1] != 1.0, dt_and_rates))


def get_total_seconds(dt):
    epoch_time = datetime.datetime(1970, 1, 1)
    return (dt - epoch_time).total_seconds()


def mask_from_into_dt(mask_from, y, m, d):
    mask_from_hour, mask_from_minute = mask_from.split(":")
    return datetime.datetime(
        int(y),
        int(m),
        int(d),
        int(mask_from_hour),
        int(mask_from_minute),
        int(0),
    )


def mask_to_into_dt(mask_to, y, m, d):
    mask_to_hour, mask_to_minute = mask_to.split(":")
    if int(mask_to_hour) == 24:
        return datetime.datetime(
            int(y),
            int(m),
            int(d),
            int(0),
            int(mask_to_minute),
            int(0),
        ) + datetime.timedelta(days=1)
    else:
        return datetime.datetime(
            int(y),
            int(m),
            int(d),
            int(mask_to_hour),
            int(mask_to_minute),
            int(0),
        )


if __name__ == "__main__":
    dt1 = datetime.datetime(2018, 12, 31, 5, 0, 30, 500000)
    dt2 = datetime.datetime(2018, 12, 31, 5, 0, 31, 200000)
    dt3 = datetime.datetime(2018, 12, 31, 5, 0, 59, 500000)
    dt4 = datetime.datetime(2018, 12, 31, 5, 1, 1, 700000)

    dt5 = datetime.datetime(2018, 12, 31, 5, 1, 11, 300000)
    dt6 = datetime.datetime(2018, 12, 31, 5, 1, 12)
    dt7 = datetime.datetime(2018, 12, 31, 5, 1, 12)
    dt8 = datetime.datetime(2018, 12, 31, 5, 1, 14, 200000)

    dt9 = datetime.datetime(2018, 12, 31, 5, 1, 59, 300000)
    dt10 = datetime.datetime(2018, 12, 31, 5, 2, 1)
    dt11 = datetime.datetime(2018, 12, 31, 5, 3, 1)
    dt12 = datetime.datetime(2018, 12, 31, 5, 4, 3, 200000)

    # print(f"getPointsNumberBeCompleted(dt1, dt2): {getPointsNumberBeCompleted(dt1, dt2)}")
    # print(f"getPointsNumberBeCompleted(dt3, dt4): {getPointsNumberBeCompleted(dt3, dt4)}")
    # print(f"getPointsNumberBeCompleted(dt5, dt6): {getPointsNumberBeCompleted(dt5, dt6)}")
    # print(f"getPointsNumberBeCompleted(dt7, dt8): {getPointsNumberBeCompleted(dt7, dt8)}")
    print(
        f"getPointsNumberBeCompleted(dt9, dt10): {getPointsNumberBeCompleted(dt9, dt10)}"
    )  # 期待: 2点補完する
    print(
        f"getPointsNumberBeCompleted(dt11, dt12): {getPointsNumberBeCompleted(dt11, dt12)}"
    )

    print("\n", end="")

    # get_relative_position_between_two_dts(dt1, dt2)
    # get_relative_position_between_two_dts(dt3, dt4)
    # get_relative_position_between_two_dts(dt5, dt6)
    # get_relative_position_between_two_dts(dt7, dt8)
    # get_relative_position_between_two_dts(dt9, dt10)
    get_relative_position_between_two_dts(dt11, dt12)

    # print(get_relative_position_between_two_dts(dt9, dt10))
