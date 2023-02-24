from matplotlib import pyplot as plt
import numpy as np
import datetime
import pandas as pd
import pvlib
from pvlib import clearsky
import requests
import sqlite3
import japanize_matplotlib

from scipy.interpolate import interp1d
import zoneinfo

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


class Q:
    def __init__(self):
        dbname = "SOLAR.db"
        self.conn = sqlite3.connect(dbname)
        self.cur = self.conn.cursor()

        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS elevations(id INTEGER PRIMARY KEY AUTOINCREMENT, latitude REAL, longitude REAL, elevation REAL)"
        )
        self.conn.commit()

    def fetch_altitude(self, latitude, longitude):
        # 既にDBにキャッシュ済みの組み合わせかチェック
        self.cur.execute(
            "SELECT * FROM elevations WHERE latitude = ? AND longitude = ?",
            (latitude, longitude),
        )
        self.conn.commit()
        result = self.cur.fetchone()

        if result:
            elevation = result[3]
            return elevation

        url = f"http://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php?lon={longitude}&lat={latitude}&outtype=JSON"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        elevation = data["elevation"]

        self.cur.execute(
            "INSERT INTO elevations(latitude, longitude, elevation) values(?, ?, ?);",
            (latitude, longitude, elevation),
        )
        self.conn.commit()

        return elevation

    def calc_qs_kw_v2(
        self, dts, latitude, longitude, surface_tilt, surface_azimuth, model
    ):
        times = pd.DatetimeIndex(dts, tz="Asia/Tokyo")

        solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

        apparent_zenith = solpos["apparent_zenith"]  # 太陽天頂角
        azimuth = solpos["azimuth"]  # 太陽方位角
        airmass = pvlib.atmosphere.get_relative_airmass(apparent_zenith)  # 相対風量

        altitude = self.fetch_altitude(latitude, longitude)  # 海抜高度
        pressure = pvlib.atmosphere.alt2pres(altitude)  # 圧力？
        airmass = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)  # 絶対風量
        linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(
            times, latitude, longitude
        )  # 気候学的濁度値
        dni_extra = pvlib.irradiance.get_extra_radiation(times)  # 大気外日射量？

        # GHI, DHI, DNIを求める
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
            model=model,
        )

        goa_global = df_poa.loc[:, ["poa_global"]].to_numpy().flatten() / 1000

        return goa_global

    def calc_qs_kw_by_real_ghi(
        self, dts, latitude, longitude, surface_tilt, surface_azimuth, model
    ):
        times = pd.DatetimeIndex(dts, tz="Asia/Tokyo")

        solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

        apparent_zenith = solpos["apparent_zenith"]  # 太陽天頂角
        azimuth = solpos["azimuth"]  # 太陽方位角
        airmass = pvlib.atmosphere.get_relative_airmass(apparent_zenith)  # 相対風量

        altitude = self.fetch_altitude(latitude, longitude)  # 海抜高度
        pressure = pvlib.atmosphere.alt2pres(altitude)  # 圧力？
        airmass = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)  # 絶対風量
        linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(
            times, latitude, longitude
        )  # 気候学的濁度値
        dni_extra = pvlib.irradiance.get_extra_radiation(times)  # 大気外日射量？

        # GHI, DHI, DNIを求める
        ineichen = clearsky.ineichen(
            apparent_zenith, airmass, linke_turbidity, altitude, dni_extra
        )

        # CSVからGHIを読み込む
        df = pd.read_csv(
            "/home/sofue/apps/solar-power-data-visualization/data/csv/japan_meteorological _agency/2022_01_01-2022_12_31.csv"
        )
        df["年月日時"] = pd.to_datetime(df["年月日時"])

        # 引数で渡されたdtsの期間に対応する行だけにフィルタリングする
        # HACK: 一日しか対応できない
        dt_first = dts[0]
        next_day = datetime.datetime(
            dt_first.year, dt_first.month, dt_first.day
        ) + datetime.timedelta(days=1)
        df = df[(dt_first < df["年月日時"]) & (df["年月日時"] <= next_day)]

        # naive => awareに変更
        for index, row in df.iterrows():
            df.at[index, "年月日時"] = (
                row["年月日時"] + datetime.timedelta(minutes=-30)
            ).tz_localize("Asia/Tokyo")

        df_raw = df  # 行を追加する前の生データ

        # dfの頭と末尾に00:00:00, 23:59:59を差し込む
        df_first = pd.DataFrame(
            [
                [
                    datetime.datetime(
                        dt_first.year,
                        dt_first.month,
                        dt_first.day,
                        tzinfo=zoneinfo.ZoneInfo(key="Asia/Tokyo"),
                    ),
                    0.0,
                    0,
                    0,
                ]
            ],
            columns=df.columns,
        )
        df_last = pd.DataFrame(
            [
                [
                    datetime.datetime(
                        dt_first.year,
                        dt_first.month,
                        dt_first.day,
                        23,
                        59,
                        59,
                        tzinfo=zoneinfo.ZoneInfo(key="Asia/Tokyo"),
                    ),
                    0.0,
                    0,
                    0,
                ]
            ],
            columns=df.columns,
        )

        df = pd.concat([df_first, df], axis=0)
        df = pd.concat([df, df_last], axis=0)

        # 線形補完前
        ghi_real = df["日射量(MJ/㎡)"].to_numpy() * 277.84

        elapsed_times_from_dt_start = np.vectorize(
            lambda dt: (
                dt - pd.to_datetime(dt_first).tz_localize("Asia/Tokyo")
            ).total_seconds()
        )(df["年月日時"])

        myfunc = interp1d(elapsed_times_from_dt_start, ghi_real, kind="cubic")
        ghi_complemented = myfunc(np.arange(0, dts.size, 1))

        # minus_indexes = np.where(ghi_complemented < 0)
        # print(f"minus_indexes: {minus_indexes}")

        # 生データでDNIを推定する
        # ghi_raw = df_raw["日射量(MJ/㎡)"].to_numpy() * 277.84
        dts_utc_from_df = np.vectorize(
            lambda ts: ts.to_pydatetime().astimezone(datetime.timezone.utc)
        )(df["年月日時"].to_numpy())

        dt_index = pd.DatetimeIndex(dts_utc_from_df, tz="Asia/Tokyo")

        solar_zeniths_by_df_timestamp = pvlib.solarposition.get_solarposition(
            dt_index,
            latitude,
            longitude,
        )["apparent_zenith"]

        # 生のGNIでDNIを推定する
        dni_calc_with_raw_ghi = pvlib.irradiance.dirint(
            ghi_real,
            solar_zeniths_by_df_timestamp,
            dt_index,
            pressure,
        ).fillna(0)

        print(f"dni_calc_with_raw_ghi.size: {dni_calc_with_raw_ghi.size}")
        print(f"elapsed_times_from_dt_start.size: {elapsed_times_from_dt_start.size}")

        myfunc = interp1d(
            elapsed_times_from_dt_start, dni_calc_with_raw_ghi, kind="cubic"
        )
        dni_complemented = myfunc(np.arange(0, dts.size, 1))

        print(dni_calc_with_raw_ghi)

        # 補間したGHIでDNIを推定する
        dni_calc_with_comp_ghi = pvlib.irradiance.dirint(
            ghi_complemented,
            apparent_zenith,
            times,
            pressure,
        )

        ghi = ineichen["ghi"]  # 全天日射量
        dhi = ineichen["dhi"]  # 散乱日射量
        dni = ineichen["dni"]  # 直達日射量

        axes = [plt.subplots()[1] for _ in range(2)]
        axes[0].plot(
            times,
            ghi,
            label="ghi(推定)",
        )
        axes[0].plot(
            np.vectorize(lambda dt: pd.to_datetime(dt).tz_localize("Asia/Tokyo"))(dts),
            ghi_complemented,
            label="ghi(実測 + 線形補間)",
            linestyle="dashed",
        )
        axes[0].plot(
            df["年月日時"],
            ghi_real,
            label="ghi(実測)",
            linestyle="dashdot",
        )
        axes[0].set_xlabel("日時")
        axes[0].set_ylabel("GHI W/m^2")
        axes[0].legend()

        axes[1].plot(
            times,
            dni,
            label="dni(推定)",
        )
        axes[1].plot(
            df["年月日時"],
            dni_calc_with_raw_ghi,
            label="dni(DHI実測値から推定)",
            linestyle="dashed",
        )
        axes[1].plot(
            times,
            dni_complemented,
            label="補間済みDNI(DHI実測値から推定)",
            linestyle="dotted",
        )
        # axes[1].plot(
        #     times,
        #     dni_calc_with_comp_ghi,
        #     label="dni(DHI実測値(補間)から推定)",
        #     linestyle="dashdot",
        # )
        axes[1].set_xlabel("日時")
        axes[1].set_ylabel("DNI W/m^2")
        axes[1].legend()

        plt.show()

        # df_poa = pvlib.irradiance.get_total_irradiance(
        #     surface_tilt,
        #     surface_azimuth,
        #     dni=dni,
        #     ghi=ghi,
        #     dhi=dhi,
        #     dni_extra=dni_extra,
        #     solar_zenith=apparent_zenith,
        #     solar_azimuth=azimuth,
        #     model=model,
        # )

        # goa_global = df_poa.loc[:, ["poa_global"]].to_numpy().flatten() / 1000

        # return goa_global


if __name__ == "__main__":
    # print(calcQ(datetime.datetime(2022, 5, 17, 17, 53), 33.82794, 132.75093))
    q = Q()
    dts = np.vectorize(
        lambda s: datetime.datetime(2022, 4, 8) + datetime.timedelta(seconds=int(s))
    )(np.arange(0, 86400, 1))
    q.calc_qs_kw_by_real_ghi(dts, 33.82794, 132.75093, 26, 180, "isotropic")
