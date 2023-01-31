import numpy as np
import datetime
import pandas as pd
import pvlib
from pvlib import clearsky

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


def calc_qs_kw_v2(dts, latitude, longitude, altitude, surface_tilt, surface_azimuth):
    times = pd.DatetimeIndex(dts, tz="Asia/Tokyo")

    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    apparent_zenith = solpos["apparent_zenith"]
    azimuth = solpos["azimuth"]
    airmass = pvlib.atmosphere.get_relative_airmass(apparent_zenith)
    pressure = pvlib.atmosphere.alt2pres(altitude)
    airmass = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
    linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)

    ineichen = clearsky.ineichen(
        apparent_zenith, airmass, linke_turbidity, altitude, dni_extra
    )

    ghi = ineichen["ghi"]  # 全天日射量
    dhi = ineichen["dhi"]  # 散乱日射量
    dni = ineichen["dni"]  # 直達日射量

    df_poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt,
        surface_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        solar_zenith=apparent_zenith,
        solar_azimuth=azimuth,
        model="isotropic",
    )

    goa_global = df_poa.loc[:, ["poa_global"]].to_numpy().flatten() / 1000

    return goa_global


# def calc_q_kw_v2(dt, lng=33.82794, lat=132.75093):
#     return max(calc_q_v2(dt, lng, lat), 0) / 1000


if __name__ == "__main__":
    print(calcQ(datetime.datetime(2022, 5, 17, 17, 53), 33.82794, 132.75093))
