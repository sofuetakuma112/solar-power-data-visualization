import pvlib
import pandas as pd  # for data wrangling
import matplotlib.pyplot as plt  # for visualization
import pathlib  # for finding the example dataset

print(pvlib.__version__)

DATA_DIR = pathlib.Path(pvlib.__file__).parent / "data"
df_tmy, metadata = pvlib.iotools.read_tmy3(DATA_DIR / "723170TYA.CSV", coerce_year=1990)

# make a Location object corresponding to this TMY
location = pvlib.location.Location(
    latitude=metadata["latitude"], longitude=metadata["longitude"]
)

# 注意: TMYデータセットは1時間ごとの区間が右ラベルで表示されている,
# 例えば10AMから11AMの区間は11と表示されている.
# 太陽位置はこの区間の真ん中(10:30)で計算する必要があるので、30分差し引く:
times = df_tmy.index - pd.Timedelta("30min")
solar_position = location.get_solarposition(times)
# しかし、TMYのデータに合わせるために、インデックスを後ろにずらすことを忘れないでください。
solar_position.index += pd.Timedelta("30min")

print(solar_position.head())

df_poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=20,  # tilted 20 degrees from horizontal
    surface_azimuth=180,  # facing South
    dni=df_tmy["DNI"],
    ghi=df_tmy["GHI"],
    dhi=df_tmy["DHI"],
    solar_zenith=solar_position["apparent_zenith"],
    solar_azimuth=solar_position["azimuth"],
    model="isotropic",
)

df = pd.DataFrame(
    {
        "ghi": df_tmy["GHI"],
        "poa": df_poa["poa_global"],
    }
)
df_monthly = df.resample("M").sum()
df_monthly.plot.bar()
plt.ylabel("Monthly Insolation [W h/m$^2$]")

plt.show()
