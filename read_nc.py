# pip install netcdf4

import netCDF4 as nc

path_to_file = "ERA5/10m_u_component_of_wind/10m_u_component_of_wind_1979_5.625deg.nc"
d = nc.Dataset(path_to_file)
u10 = d["u10"]
