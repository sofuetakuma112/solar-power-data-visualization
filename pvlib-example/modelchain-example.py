import pandas as pd
import numpy as np
import pvlib
from pvlib.pvsystem import PVSystem, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS["sapm"][
    "open_rack_glass_glass"
]


sandia_modules = pvlib.pvsystem.retrieve_sam("SandiaMod")

cec_inverters = pvlib.pvsystem.retrieve_sam("cecinverter")

sandia_module = sandia_modules["Canadian_Solar_CS5P_220M___2009_"]

cec_inverter = cec_inverters["ABB__MICRO_0_25_I_OUTD_US_208__208V_"]

location = Location(latitude=32.2, longitude=-110.9)

system = PVSystem(
    surface_tilt=20,
    surface_azimuth=200,
    module_parameters=sandia_module,
    inverter_parameters=cec_inverter,
    temperature_model_parameters=temperature_model_parameters,
)


mc = ModelChain(system, location)
print(mc)

weather = pd.DataFrame(
    [[1050, 1000, 100, 30, 5]],
    columns=["ghi", "dni", "dhi", "temp_air", "wind_speed"],
    index=[pd.Timestamp("20170401 1200", tz="US/Arizona")],
)


mc.run_model(weather)
print()

print(mc.results.aoi)
print()
print(mc.results.cell_temperature)
print()
print(mc.results.dc)
print()
print(mc.results.ac)


sapm_system = PVSystem(
    module_parameters=sandia_module,
    inverter_parameters=cec_inverter,
    temperature_model_parameters=temperature_model_parameters,
)


mc = ModelChain(sapm_system, location)

print(mc)
