using NCDatasets

function read_temperature(year::Int, return_others::Bool=false)
    @assert 1979 ≤ year ≤ 2018
    d = NCDataset("ERA5/temperature_850/temperature_850hPa_$(year)_5.625deg.nc")
    temperature = d[:t][:, :, :]
    return temperature # (lon, lat, 1, time) shaped.
    if return_others
        time = d[:time][:]
        lon = d[:lon][:]
        lat = d[:lat][:]
        return temperature, lon, lat, time
    end
end
