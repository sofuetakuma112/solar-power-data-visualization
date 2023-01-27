import matplotlib.pyplot as plt
import pandas as pd
import pvlib
from pvlib import clearsky

latitude, longitude, tz, altitude = (
    33.82794,
    132.75093,
    "Asia/Tokyo",
    25.720,
)
times = pd.date_range(
    start="2022-10-30", end="2022-10-31", freq="S", tz=tz, inclusive="left"
)

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
    surface_tilt=20,  # tilted 20 degrees from horizontal
    surface_azimuth=180,  # facing South
    dni=dni,
    ghi=ghi,
    dhi=dhi,
    solar_zenith=apparent_zenith,
    solar_azimuth=azimuth,
    model="isotropic",
)

# print(df_poa)

print(df_poa["poa_global"].values)

goa_global = df_poa.loc[:, ["poa_global"]]

goa_global.plot()
plt.show()
