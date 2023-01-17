import numpy as np
import datetime
import pvlib

# 単位はワット
def calcQ(dt, lat_deg, lng_deg):
    dt_new_year = datetime.datetime(dt.year, 1, 1)
    dt_delta = dt - dt_new_year  # 元旦からの通し日数

    dn = dt_delta.days + 1
    theta = 2 * np.pi * (dn - 1) / 365  # TODO: うるう年に対応できるか調査

    # 太陽赤緯(単位はラジアン)
    delta = (
        0.006918
        - (0.399912 * np.cos(theta))
        + (0.070257 * np.sin(theta))
        - (0.006758 * np.cos(2 * theta))
        + (0.000907 * np.sin(2 * theta))
        - (0.002697 * np.cos(3 * theta))
        + (0.001480 * np.sin(3 * theta))
    )
    # 地心太陽距離のルートの中身
    geocentri_distance_like = (
        1.000110
        + 0.034221 * np.cos(theta)
        + 0.001280 * np.sin(theta)
        + 0.000719 * np.cos(2 * theta)
        + 0.000077 * np.sin(2 * theta)
    )
    eq = (  # 均時差
        0.000075
        + 0.001868 * np.cos(theta)
        - 0.032077 * np.sin(theta)
        - 0.014615 * np.cos(2 * theta)
        - 0.040849 * np.sin(2 * theta)
    )
    # lng_rad = lng_deg * np.pi / 180
    phi = lat_deg * np.pi / 180

    lng_diff = (lng_deg - 135) / 180 * np.pi  # 経度差

    def calc_h(dt, lng_diff, eq):
        # return (dt.hour + dt.minute / 60 - 12) / 12 * np.pi + lng_diff + eq
        return (
            (dt.hour + dt.minute / 60 + dt.second / (60 * 60) - 12) / 12 * np.pi
            + lng_diff
            + eq
        )

    # 太陽高度のarcsinの引数になる値
    def calc_sun_altitude_like(h, delta, phi):
        return np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(h)

    h = calc_h(dt, lng_diff, eq)  # 時角

    sin_alpha = calc_sun_altitude_like(h, delta, phi)
    return 1367 * geocentri_distance_like * sin_alpha


# 単位はキロワット
def calc_q_kw(dt, lng=33.82794, lat=132.75093):
    return max(calcQ(dt, lng, lat), 0) / 1000


def calc_q_v2(date, latitude, longitude):
    # 太陽高度角、方位角を計算
    solpos = pvlib.solarposition.get_solarposition(date, latitude, longitude)

    print(solpos)

    # 地上に到達する日射量を計算

    # pvlib.irradiance.aoiの引数
    # surface_tilt : 数値
    #     パネルの水平方向からの傾き。
    # surface_azimuth : 数値
    #     北からのパネルの方位角。
    # solar_zenith : 数値
    #     太陽天頂角。
    # solar_azimuth : 数値
    #     太陽の方位角。
    aoi = pvlib.irradiance.aoi(0, 0, solpos["apparent_zenith"], solpos["azimuth"])
    dni_extra = pvlib.irradiance.get_extra_radiation(date)

    # 太陽高度角、方位角、日射量指数から全球水平面放射照度を計算
    total_irrad = pvlib.irradiance.get_total_irradiance(
        0,
        0,
        solpos["apparent_zenith"],
        solpos["azimuth"],
        airmass=pvlib.atmosphere.relativeairmass(solpos["apparent_zenith"]),
        dni=pvlib.irradiance.extraradiation(date),
        ghi=None,
        dhi=None,
        dni_extra=None,
        model="haydavies",
    )

    # 全球水平面放射照度を表示
    print(total_irrad)
    dni = pvlib.irradiance.dni(aoi, dni_extra, solpos["zenith"])

    print(f"aoi: {aoi.iloc[-1]}")

    # # 太陽高度角、方位角から直達日射量を計算
    # print(f"solpos['zenith']: {type(solpos['zenith'])}")
    # dni = pvlib.irradiance.dni(solpos['apparent_zenith'], solpos['azimuth'], solpos["zenith"])

    return dni


def calc_q_kw_v2(dt, lng=33.82794, lat=132.75093):
    return max(calc_q_v2(dt, lng, lat), 0) / 1000


if __name__ == "__main__":
    print(calcQ(datetime.datetime(2022, 5, 17, 17, 53), 33.82794, 132.75093))
    print(
        f"dni: {calc_q_v2(datetime.datetime(2022, 5, 17, 17, 53), 33.82794, 132.75093)}"
    )
