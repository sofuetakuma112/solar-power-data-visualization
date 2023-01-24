import pvlib
import pandas as pd
import matplotlib.pyplot as plt


coordinates = [
    (32.2, -111.0, "Tucson", 700, "Etc/GMT+7"),
    (35.1, -106.6, "Albuquerque", 1500, "Etc/GMT+7"),
    (37.8, -122.4, "San Francisco", 10, "Etc/GMT+8"),
    (52.5, 13.4, "Berlin", 34, "Etc/GMT-1"),
]


sandia_modules = pvlib.pvsystem.retrieve_sam("SandiaMod")

sapm_inverters = pvlib.pvsystem.retrieve_sam("cecinverter")

module = sandia_modules["Canadian_Solar_CS5P_220M___2009_"]

inverter = sapm_inverters["ABB__MICRO_0_25_I_OUTD_US_208__208V_"]

temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
    "open_rack_glass_glass"
]

# 日射量、気温、風速を含むTMY（Typical Meteorological Year）
tmys = []

for location in coordinates:
    latitude, longitude, name, altitude, timezone = location
    weather = pvlib.iotools.get_pvgis_tmy(latitude, longitude, map_variables=True)[0]
    weather.index.name = "utc_time"
    tmys.append(weather)

system = {"module": module, "inverter": inverter, "surface_azimuth": 180}


energies = {}

for location, weather in zip(coordinates, tmys):
    latitude, longitude, name, altitude, timezone = location
    system["surface_tilt"] = latitude
    solpos = pvlib.solarposition.get_solarposition(
        time=weather.index,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        temperature=weather["temp_air"],
        pressure=pvlib.atmosphere.alt2pres(altitude),
    )
    dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)  # 曜日から地球外放射量を求める。
    airmass = pvlib.atmosphere.get_relative_airmass(
        solpos["apparent_zenith"]
    )  # 海抜高度における相対的な（気圧調整されていない）風量を計算する。
    pressure = pvlib.atmosphere.alt2pres(altitude)  # 高度から気圧？を決定する
    am_abs = pvlib.atmosphere.get_absolute_airmass(
        airmass, pressure
    )  # 相対風量と気圧から絶対風量（気圧調整済み）を求める。
    aoi = pvlib.irradiance.aoi(  # 太陽ベクトルの表面への入射角度(太陽ベクトルと表面法線の間の角度)を計算
        system["surface_tilt"],
        system["surface_azimuth"],
        solpos["apparent_zenith"],
        solpos["azimuth"],
    )
    total_irradiance = pvlib.irradiance.get_total_irradiance(  # 指定された天空拡散放射照度モデルを用いて、面内全放射照度とそのビーム成分、天空拡散成分、地上反射成分を測定する。
        system["surface_tilt"],
        system["surface_azimuth"],
        solpos["apparent_zenith"], # 太陽高度角
        solpos["azimuth"], # 太陽方位角
        weather["dni"], # 直達日射量
        weather["ghi"], # 全天日射量(DNI x cos(θ) + DHI)
        weather["dhi"], # 散乱日射量
        dni_extra=dni_extra,
        model="haydavies",
    )
    print(f"total_irradiance: \n{total_irradiance}")
    cell_temperature = (
        pvlib.temperature.sapm_cell(  # Sandia Array Performance Modelに基づき、セル温度を計算する。
            total_irradiance["poa_global"],
            weather["temp_air"],
            weather["wind_speed"],
            **temperature_model_parameters,
        )
    )
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance( # SAPM分光損失関数とSAPM入射角損失関数を用いて、SAPM実効照度を算出します。
        total_irradiance["poa_direct"],
        total_irradiance["poa_diffuse"],
        am_abs,
        aoi,
        module,
    )
    dc = pvlib.pvsystem.sapm(effective_irradiance, cell_temperature, module)
    ac = pvlib.inverter.sandia(dc["v_mp"], dc["p_mp"], inverter)
    annual_energy = ac.sum()
    energies[name] = annual_energy

energies = pd.Series(energies)
print(energies)

energies.plot(kind="bar", rot=0)
plt.ylabel("Yearly energy yield (W hr)")
