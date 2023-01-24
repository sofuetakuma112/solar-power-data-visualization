from pvlib import pvsystem

module_parameters = {"pdc0": 5000, "gamma_pdc": -0.004}
inverter_parameters = {"pdc0": 5000, "eta_inv_nom": 0.96}

system = pvsystem.PVSystem(
    inverter_parameters=inverter_parameters, module_parameters=module_parameters
)

# PVシステム自体のデータはオブジェクトの属性に格納されている
print(system.inverter_parameters)

# 外部データは、PVシステムのメソッドの引数に渡されます。
pdc = system.pvwatts_dc(g_poa_effective=1000, temp_cell=30)

print(pdc)
