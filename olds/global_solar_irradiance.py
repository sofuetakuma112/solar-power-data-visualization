import numpy as np
import datetime

dt = datetime.datetime(2022, 5, 17, 17, 53)
dt_new_year = datetime.datetime(dt.year, 1, 1)
dt_delta = dt - dt_new_year  # 元旦からの通し日数

dn = dt_delta.days + 1
theta = 2 * np.pi * (dn - 1) / 365  # OK

# 太陽赤緯(単位はラジアン) OK
sun_declination = (
    0.006918
    - (0.399912 * np.cos(theta))
    + (0.070257 * np.sin(theta))
    - (0.006758 * np.cos(2 * theta))
    + (0.000907 * np.sin(2 * theta))
    - (0.002697 * np.cos(3 * theta))
    + (0.001480 * np.sin(3 * theta))
)

# 地心太陽距離
# 1/{1.000110+0.034221cos(θo)+0.001280sin(θo)+0.000719cos(2θo)+0.000077sin(2θo)}^0.5
geocentri_distance = 1 / np.sqrt(
    (
        1.000110
        + 0.034221 * np.cos(theta)
        + 0.001280 * np.sin(theta)
        + 0.000719 * np.cos(2 * theta)
        + 0.000077 * np.sin(2 * theta)
    )
)

# 地心太陽距離のルートの中身
# 1.000110+0.034221cos(θo)+0.001280sin(θo)+0.000719cos(2θo)+0.000077sin(2θo)
geocentri_distance_like = (  # OK
    1.000110
    + 0.034221 * np.cos(theta)
    + 0.001280 * np.sin(theta)
    + 0.000719 * np.cos(2 * theta)
    + 0.000077 * np.sin(2 * theta)
)

equation_of_time = (  # 均時差(時間に依らない) OK
    0.000075
    + 0.001868 * np.cos(theta)
    - 0.032077 * np.sin(theta)
    - 0.014615 * np.cos(2 * theta)
    - 0.040849 * np.sin(2 * theta)
)

def dd2dms(dd):
    deg = np.floor(dd)
    f = dd - deg
    minute = np.floor(f * 60)
    second = np.floor((f - minute / 60) * 3600)
    return [deg, minute, second]

lng_deg = 132.75093  # 任意の経度(longitude)
lng_rad = lng_deg * np.pi / 180

lat_deg = 33.82794  # 任意の緯度(latitude)
lat_rad = lat_deg * np.pi / 180

lng_diff = (lng_deg - 135) / 180 * np.pi # 経度差

# 太陽の時角を求める
# http://www.es.ris.ac.jp/~nakagawa/met_cal/solar.html
def calc_sun_hour_angle1(hour, lng_diff, equation_time):
    return ((hour - 12) * np.pi / 12) + lng_diff + equation_time


# 太陽の時角を求める
# https://environmental-engineering.work/archives/245
def calc_sun_hour_angle2(hour, lng_diff, equation_time):
    pacific_sun_time = hour + (lng_diff) * 4 + equation_time
    return (pacific_sun_time - 12) * 15


def calc_sun_hour_angle3(dt, lng_diff, equation_of_time):
    return (dt.hour + dt.minute / 60 - 12) / 12 * np.pi + lng_diff + equation_of_time


# 太陽高度
# arcsin{sin(φ)sin(δ)+cos(φ)cos(δ)cos(h)}
def calc_sun_altitude(hour_angle, sun_declination, lat_rad):
    return np.arcsin(
        np.sin(lat_rad) * np.sin(sun_declination)
        + np.cos(lat_rad) * np.cos(sun_declination) * np.cos(hour_angle)
    )


# 太陽高度のarcsinの引数になる値
# sin(φ)sin(δ)+cos(φ)cos(δ)cos(h)の値を求める
def calc_sun_altitude_like(hour_angle, sun_declination, lat_rad):
    return np.sin(lat_rad) * np.sin(sun_declination) + np.cos(lat_rad) * np.cos(
        sun_declination
    ) * np.cos(hour_angle)


# 太陽方位
def calc_sun_azimuths(hour_angle, sun_declination, altitude, lat_rad):
    return np.arctan(  # 太陽方位
        np.cos(lat_rad)
        * np.cos(sun_declination)
        * np.sin(hour_angle)
        / (np.sin(lat_rad) * np.sin(altitude) - np.sin(sun_declination))
    )


hour_angle = calc_sun_hour_angle3(dt, lng_diff, equation_of_time)  # 時角

sun_altitude = calc_sun_altitude(hour_angle, sun_declination, lat_rad)
sun_azimuths = calc_sun_azimuths(hour_angle, sun_declination, sun_altitude, lat_rad)
sin_alpha = calc_sun_altitude_like(hour_angle, sun_declination, lat_rad)
print(f"日射量[kW/m^2]: {1367 * geocentri_distance_like * sin_alpha / 1000}")  # 日射量の式を変形したもの

