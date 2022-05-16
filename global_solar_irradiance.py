import numpy as np

dn = 0  # 元旦からの通し日数
theta = np.pi * (dn - 1) / 365

delta = (  # 当該日の太陽赤緯(時間に依らない)
    0.006918
    - 0.399912 * np.cos(theta)
    + 0.070257 * np.sin(theta)
    - 0.006758 * np.cos(2 * theta)
    + 0.000907 * np.sin(2 * theta)
    - 0.002697 * np.cos(3 * theta)
    + 0.001480 * np.sin(3 * theta)
)

geocentri_distance = (  # 地心太陽距離
    1
    / (
        1.000110
        + 0.034221 * np.cos(theta)
        + 0.001280 * np.sin(theta)
        + 0.000719 * np.cos(2 * theta)
        + 0.000077 * np.sin(2 * theta)
    )
    ^ 0.5
)

equation_of_time = (  # 均時差(時間に依らない)
    0.000075
    + 0.001868 * np.cos(theta)
    - 0.032077 * np.sin(theta)
    - 0.014615 * np.cos(2 * theta)
    - 0.040849 * np.sin(2 * theta)
)

lng = 0  # 任意の経度(longitude)
lat = 0 # 任意の緯度(latitude)

JST = 9
lng_diff_from_standard_meridian = 135 - lng  # 標準子午線からの経度差

# 時間の関数？？
# 均時差(equation_of_time)は時間に依らない
# 日本標準時間JST
# TODO: 任意の時刻から計算できるように修正する
h = (  # 太陽の時角
    (JST - 12) * np.pi / 12 + lng_diff_from_standard_meridian + equation_of_time
)
# deltaは時間に依らない
alpha = np.arcsin(  # 高度(時間によって変化する)
    np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.cos(h)
)
# 太陽方位は日射量の計算に不要？
psi = np.arctan(  # 太陽方位
    np.cos(lat)
    * np.cos(delta)
    * np.sin(h)
    / (np.sin(lat) * np.sin(alpha) - np.sin(delta))
)

q = 1367(1 / geocentri_distance) ^ 2 * np.sin(alpha)
